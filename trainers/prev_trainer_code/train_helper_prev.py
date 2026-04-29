import numpy as np
import torch
import torch.nn.functional as F



def _orthonormalize_columns(basis: torch.Tensor) -> torch.Tensor:
    """Orthonormalize columns for [D,R] or [C,D,R] basis tensors."""
    if basis is None or basis.numel() == 0:
        return basis
    if basis.dim() == 2:
        q, _ = torch.linalg.qr(basis, mode="reduced")
        if q.size(1) < basis.size(1):
            pad = torch.zeros(q.size(0), basis.size(1) - q.size(1), device=q.device, dtype=q.dtype)
            q = torch.cat([q, pad], dim=1)
        return q[:, : basis.size(1)]
    if basis.dim() == 3:
        return torch.stack([_orthonormalize_columns(b) for b in basis], dim=0)
    raise ValueError(f"basis must be [D,R] or [C,D,R], got {tuple(basis.shape)}")


def _complete_orthonormal_basis(active_basis: torch.Tensor, rank: int) -> torch.Tensor:
    """Return [D, rank] basis without zero columns."""
    d = int(active_basis.size(0))
    rank = min(int(rank), d)
    device, dtype = active_basis.device, active_basis.dtype

    if active_basis.numel() > 0 and active_basis.size(1) > 0:
        basis = _orthonormalize_columns(active_basis[:, :rank])
    else:
        basis = torch.zeros(d, 0, device=device, dtype=dtype)

    if basis.size(1) >= rank:
        return basis[:, :rank]

    cols = [basis[:, i] for i in range(basis.size(1))]
    eye = torch.eye(d, device=device, dtype=dtype)

    for j in range(d):
        v = eye[:, j].clone()
        for u in cols:
            v = v - torch.dot(v, u) * u
        n = v.norm()
        if n > 1e-6:
            cols.append(v / n)
        if len(cols) >= rank:
            break

    while len(cols) < rank:
        v = torch.randn(d, device=device, dtype=dtype)
        for u in cols:
            v = v - torch.dot(v, u) * u
        cols.append(v / v.norm().clamp_min(1e-6))

    return torch.stack(cols[:rank], dim=1)


class TrainerHelper:
    # ============================================================
    # Utils
    # ============================================================
    def _zero(self, ref=None):
        if isinstance(ref, torch.Tensor):
            return torch.tensor(0.0, device=ref.device, dtype=ref.dtype)
        return torch.tensor(0.0, device=self.device)

    def _safe_get_subspace_bank(self):
        return self.model.get_subspace_bank()

    def _snapshot_old_bank(self, old_class_count: int):
        """Snapshot old geometry safely, including reliability/active rank and inter-class matrices."""
        old_class_count = int(old_class_count)
        sb = self._safe_get_subspace_bank()
        snap = {}

        for k, v in sb.items():
            if not isinstance(v, torch.Tensor) or v.numel() == 0:
                snap[k] = v
                continue

            if v.dim() == 0:
                snap[k] = v.detach().clone()
            elif v.dim() == 1:
                snap[k] = v[:old_class_count].detach().clone()
            elif v.dim() == 2 and k.startswith("inter_"):
                snap[k] = v[:old_class_count, :old_class_count].detach().clone()
            elif v.dim() >= 2:
                snap[k] = v[:old_class_count].detach().clone()
            else:
                snap[k] = v.detach().clone()

        return snap

    def _snapshot_old_token_memory(self, old_class_count: int):
        snap = {}
        for c in range(int(old_class_count)):
            if c in self.token_memory:
                snap[c] = {k: v.detach().clone() for k, v in self.token_memory[c].items()}
        return snap

    def _projector_from_basis(self, basis: torch.Tensor) -> torch.Tensor:
        if basis.dim() == 2:
            return basis @ basis.t()
        return torch.matmul(basis, basis.transpose(-1, -2))

    def _safe_log_variances(self, variances: torch.Tensor) -> torch.Tensor:
        return torch.log(variances.clamp_min(max(self.geom_var_floor, 1e-6)))

    # ============================================================
    # Accuracy helpers
    # ============================================================
    def _labels_are_local(self, y: torch.Tensor, new_class_ids):
        if y.numel() == 0:
            return False
        allowed_local = set(range(len(new_class_ids)))
        unique_y = set(int(v) for v in y.detach().cpu().unique().tolist())
        return unique_y.issubset(allowed_local)

    def _incremental_accuracy_with_count(self, logits, y, new_class_ids):
        if y.numel() == 0:
            return 0, 0

        class_ids = torch.tensor(new_class_ids, device=logits.device, dtype=torch.long)

        if class_ids.numel() > 0:
            max_id = int(class_ids.max().item())
            if max_id >= logits.size(1):
                raise RuntimeError(
                    f"Classifier output size mismatch: max requested class id {max_id}, "
                    f"but logits only have {logits.size(1)} classes."
                )

        masked_logits = logits.index_select(1, class_ids)
        pred_local = masked_logits.argmax(dim=1)

        if self._labels_are_local(y, new_class_ids):
            valid = torch.ones_like(y, dtype=torch.bool)
            correct = int((pred_local == y).sum().item())
            return correct, int(valid.sum().item())

        y_local = torch.full_like(y, fill_value=-1)
        for local_idx, global_cls in enumerate(class_ids):
            y_local[y == global_cls] = local_idx

        valid = y_local >= 0
        if not valid.any():
            return 0, 0

        correct = int((pred_local[valid] == y_local[valid]).sum().item())
        return correct, int(valid.sum().item())

    def _masked_weighted_ce_new(self, logits, y, new_class_ids):
        class_ids = torch.tensor(new_class_ids, device=logits.device, dtype=torch.long)

        if class_ids.numel() > 0:
            max_id = int(class_ids.max().item())
            if max_id >= logits.size(1):
                raise RuntimeError(
                    f"Classifier output size mismatch: max requested class id {max_id}, "
                    f"but logits only have {logits.size(1)} classes. "
                    f"Current phase classes were not bootstrapped/expanded before training."
                )

        logits_new = logits.index_select(1, class_ids)

        allowed_local = set(range(len(new_class_ids)))
        unique_y = set(int(v) for v in y.detach().cpu().unique().tolist())

        if unique_y.issubset(allowed_local):
            y_local = y
        else:
            y_local = torch.full_like(y, fill_value=-1)
            for local_idx, global_cls in enumerate(class_ids):
                y_local[y == global_cls] = local_idx

        valid = y_local >= 0
        logits_new = logits_new[valid]
        y_local = y_local[valid]

        if y_local.numel() == 0:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        counts = torch.bincount(y_local, minlength=len(new_class_ids)).float()
        weights = counts.sum() / counts.clamp_min(1.0)
        weights = weights / weights.mean().clamp_min(1e-8)

        return F.cross_entropy(logits_new, y_local, weight=weights)

    # ============================================================
    # Memory building: current-phase classes only
    # ============================================================
    @torch.no_grad()
    def _extract_backbone_outputs_for_class(self, cls: int, split: str = "train"):
        patches_np = self.dataset.get_class_patches(cls, split=split)
        x = torch.from_numpy(patches_np).float().to(self.device)

        was_training = self.model.training
        self.model.eval()

        feats, band_weights_all, spectral_all = [], [], []
        spectral_tokens_all, spatial_tokens_all, fused_tokens_all = [], [], []

        bs = int(getattr(self.args, "subspace_extract_batch_size", 256))

        for start in range(0, x.size(0), bs):
            xb = x[start:start + bs]
            out = self.model.extract_backbone_outputs(xb)

            feats.append(out["features"].detach())
            band_weights_all.append(out["band_weights"].detach())
            spectral_all.append(out["spectral_summary"].detach())

            if out.get("spectral_tokens") is not None:
                spectral_tokens_all.append(out["spectral_tokens"].detach())
            if out.get("spatial_tokens") is not None:
                spatial_tokens_all.append(out["spatial_tokens"].detach())
            if out.get("fused_tokens") is not None:
                fused_tokens_all.append(out["fused_tokens"].detach())

        if was_training:
            self.model.train()

        return {
            "features": torch.cat(feats, dim=0),
            "band_weights": torch.cat(band_weights_all, dim=0),
            "spectral_summary": torch.cat(spectral_all, dim=0),
            "spectral_tokens": torch.cat(spectral_tokens_all, dim=0) if len(spectral_tokens_all) > 0 else None,
            "spatial_tokens": torch.cat(spatial_tokens_all, dim=0) if len(spatial_tokens_all) > 0 else None,
            "fused_tokens": torch.cat(fused_tokens_all, dim=0) if len(fused_tokens_all) > 0 else None,
        }

    @torch.no_grad()
    def _extract_feature_guided_concepts(self, cls: int, split: str = "train", num_concepts=None):
        num_concepts = int(num_concepts or self.num_concepts_per_class)
        outs = self._extract_backbone_outputs_for_class(cls, split=split)
        feat_mat = outs["features"]

        if feat_mat.size(0) == 1:
            return feat_mat

        joint = feat_mat.detach().cpu().numpy().astype("float32")
        k = max(1, min(int(num_concepts), joint.shape[0]))
        centers = self.dataset._kmeans_numpy(joint, k=k, seed=self.dataset.seed + int(cls))
        return torch.from_numpy(centers).float().to(self.device)

    @torch.no_grad()
    def _extract_class_geometry(self, cls: int, split: str = "train", rank=None):
        """Extract reliability-aware low-rank class geometry."""
        rank = int(rank or self.subspace_rank)
        outs = self._extract_backbone_outputs_for_class(cls, split=split)

        feat_mat = torch.nan_to_num(outs["features"], nan=0.0, posinf=0.0, neginf=0.0)
        band_weights = outs["band_weights"]
        spectral_summary = outs["spectral_summary"]

        mean = feat_mat.mean(dim=0)
        centered = feat_mat - mean.unsqueeze(0)
        n, d = centered.shape

        var_floor = float(getattr(self.args, "geom_var_floor", getattr(self, "geom_var_floor", 1e-4)))
        shrink = float(getattr(self.args, "geometry_variance_shrinkage", 0.10))
        shrink = max(0.0, min(shrink, 1.0))
        max_ratio = float(getattr(self.args, "geometry_max_variance_ratio", 50.0))

        total_var = centered.pow(2).sum(dim=1).mean().clamp_min(var_floor)
        data_floor = torch.maximum(
            torch.tensor(var_floor, device=self.device, dtype=feat_mat.dtype),
            (total_var / max(d, 1)) * 1e-3,
        )

        # Correct covariance rank: at most n-1. Anything else is fake geometry.
        q = min(rank, max(n - 1, 0), d)

        if q <= 0:
            active_basis = torch.zeros(d, 0, device=self.device, dtype=feat_mat.dtype)
            active_eig = torch.empty(0, device=self.device, dtype=feat_mat.dtype)
            res_var = (total_var / max(d, 1)).clamp_min(data_floor)
        else:
            try:
                _, s, vh = torch.linalg.svd(centered, full_matrices=False)
                active_basis = vh[:q].t().contiguous()
                active_eig = (s[:q] ** 2) / max(n - 1, 1)
            except RuntimeError:
                cov = centered.t().mm(centered) / max(n - 1, 1)
                evals_all, evecs = torch.linalg.eigh(cov)
                idx = torch.argsort(evals_all, descending=True)[:q]
                active_basis = evecs[:, idx]
                active_eig = evals_all[idx]

            active_basis = _orthonormalize_columns(active_basis)
            avg_var = (total_var / max(d, 1)).clamp_min(data_floor)
            active_eig = (1.0 - shrink) * active_eig + shrink * avg_var
            active_eig = active_eig.clamp(
                min=float(data_floor.item()),
                max=float((data_floor * max_ratio).item()),
            )

            proj = centered.mm(active_basis)
            recon = proj.mm(active_basis.t())
            residual = centered - recon
            residual_energy = residual.pow(2).sum(dim=1).mean()
            res_var = (residual_energy / max(d - q, 1)).clamp_min(data_floor)

        basis = _complete_orthonormal_basis(active_basis, rank)
        eigvals = torch.full((rank,), float(var_floor), device=self.device, dtype=feat_mat.dtype)
        if q > 0:
            eigvals[:q] = active_eig[:q]
        if q < rank:
            eigvals[q:] = res_var

        spectral_proto = spectral_summary.mean(dim=0)
        band_importance = band_weights.mean(dim=0)
        band_importance = band_importance / band_importance.sum().clamp_min(1e-8)

        sample_rel = n / float(n + 5)
        rank_rel = q / float(max(rank, 1))
        reliability = min(1.0, max(0.05, 0.7 * sample_rel + 0.3 * rank_rel))

        return (
            mean,
            basis[:, :rank],
            eigvals[:rank],
            res_var,
            spectral_proto,
            band_importance,
            torch.tensor(q, device=self.device, dtype=torch.long),
            torch.tensor(reliability, device=self.device, dtype=feat_mat.dtype),
        )

    def _extract_class_geometry(self, cls: int, split: str = "train", rank=None):
        rank = int(rank or self.subspace_rank)
        outs = self._extract_backbone_outputs_for_class(cls, split=split)

        feat_mat = outs["features"]
        band_weights = outs["band_weights"]
        spectral_summary = outs["spectral_summary"]

        mean = feat_mat.mean(dim=0)
        centered = feat_mat - mean.unsqueeze(0)
        n, d = centered.shape

        if n <= 1:
            basis = torch.zeros(d, rank, device=self.device, dtype=feat_mat.dtype)
            eigvals = torch.full((rank,), self.geom_var_floor, device=self.device, dtype=feat_mat.dtype)
            res_var = torch.tensor(self.geom_var_floor, device=self.device, dtype=feat_mat.dtype)
        else:
            q = min(rank, min(n, d))
            try:
                _, s, v = torch.pca_lowrank(centered, q=q, center=False)
                basis = v[:, :q]
                eigvals = (s[:q] ** 2) / max(n - 1, 1)
            except RuntimeError:
                cov = centered.t() @ centered / max(n - 1, 1)
                evals_all, evecs = torch.linalg.eigh(cov)
                idx = torch.argsort(evals_all, descending=True)[:q]
                basis = evecs[:, idx]
                eigvals = evals_all[idx]

            if basis.size(1) < rank:
                pad = torch.zeros(d, rank - basis.size(1), device=basis.device, dtype=basis.dtype)
                basis = torch.cat([basis, pad], dim=1)

            eigvals = eigvals.clamp_min(self.geom_var_floor)
            if eigvals.numel() < rank:
                pad = torch.full((rank - eigvals.numel(),), self.geom_var_floor, device=eigvals.device, dtype=eigvals.dtype)
                eigvals = torch.cat([eigvals, pad], dim=0)

            residual = centered - (centered @ basis[:, :q]) @ basis[:, :q].t() if q > 0 else centered
            if d - q > 0:
                res_var = residual.pow(2).sum(dim=1).mean() / max(d - q, 1)
            else:
                res_var = residual.pow(2).sum(dim=1).mean()
            res_var = res_var.clamp_min(self.geom_var_floor)

        spectral_proto = spectral_summary.mean(dim=0)
        band_importance = band_weights.mean(dim=0)
        band_importance = band_importance / band_importance.sum().clamp_min(1e-8)

        return mean, basis[:, :rank], eigvals[:rank], res_var, spectral_proto, band_importance

    @torch.no_grad()
    def _extract_class_token_relations(self, cls: int, split: str = "train"):
        outs = self._extract_backbone_outputs_for_class(cls, split=split)
        spectral_tokens = outs["spectral_tokens"]
        spatial_tokens = outs["spatial_tokens"]

        if spectral_tokens is None or spatial_tokens is None:
            return None

        rel = self.model.semantic_encoder.summarize_class_token_relations(
            spectral_tokens=spectral_tokens,
            spatial_tokens=spatial_tokens,
        )
        return {k: v.detach().cpu() for k, v in rel.items()}

    @torch.no_grad()
    def _refresh_class_token_memory(self, cls: int, split: str = "train"):
        rel = self._extract_class_token_relations(cls, split=split)
        if rel is not None:
            self.token_memory[int(cls)] = rel

    @torch.no_grad()
    def _build_class_memory_from_current_phase(self, cls: int, split: str = "train"):
        concepts = self._extract_feature_guided_concepts(
            cls,
            split=split,
            num_concepts=self.num_concepts_per_class,
        )

        if int(cls) >= self.model.current_num_classes:
            self.model.add_new_class_concepts(concepts)
        else:
            self.model.refresh_class_concepts(int(cls), concepts, reset_delta=True)

        geom = self._extract_class_geometry(cls, split=split, rank=self.subspace_rank)
        if len(geom) == 8:
            mean, basis, eigvals, res_var, spectral_proto, band_importance, active_rank, reliability = geom
        else:
            mean, basis, eigvals, res_var, spectral_proto, band_importance = geom
            active_rank = torch.tensor(self.subspace_rank, device=self.device, dtype=torch.long)
            reliability = torch.tensor(1.0, device=self.device, dtype=mean.dtype)

        try:
            self.model.refresh_class_subspace(
                int(cls),
                mean,
                basis,
                eigvals,
                res_var,
                spectral_proto=spectral_proto,
                band_importance=band_importance,
                active_rank=active_rank,
                reliability=reliability,
            )
        except TypeError:
            self.model.refresh_class_subspace(
                int(cls),
                mean,
                basis,
                eigvals,
                res_var,
                spectral_proto=spectral_proto,
                band_importance=band_importance,
            )
            gb = getattr(self.model, "geometry_bank", None)
            if gb is not None:
                if hasattr(gb, "active_ranks") and int(cls) < gb.active_ranks.numel():
                    gb.active_ranks[int(cls)] = active_rank.to(gb.active_ranks.device)
                if hasattr(gb, "reliability") and int(cls) < gb.reliability.numel():
                    gb.reliability[int(cls)] = reliability.to(gb.reliability.device, dtype=gb.reliability.dtype)

        gb = getattr(self.model, "geometry_bank", None)
        if gb is not None and hasattr(gb, "refresh_inter_class_geometry"):
            gb.refresh_inter_class_geometry()

        self._refresh_class_token_memory(int(cls), split=split)

    def _build_class_memory_from_current_phase(self, cls: int, split: str = "train"):
        concepts = self._extract_feature_guided_concepts(
            cls,
            split=split,
            num_concepts=self.num_concepts_per_class,
        )

        if int(cls) >= self.model.current_num_classes:
            self.model.add_new_class_concepts(concepts)
        else:
            self.model.refresh_class_concepts(int(cls), concepts, reset_delta=True)

        mean, basis, eigvals, res_var, spectral_proto, band_importance = self._extract_class_geometry(
            cls,
            split=split,
            rank=self.subspace_rank,
        )

        self.model.refresh_class_subspace(
            int(cls),
            mean,
            basis,
            eigvals,
            res_var,
            spectral_proto=spectral_proto,
            band_importance=band_importance,
        )

        self._refresh_class_token_memory(int(cls), split=split)

    @torch.no_grad()
    def _bootstrap_phase_classes(self, phase: int, split: str = "train"):
        phase = int(phase)

        with self.dataset.memory_build_context(phase):
            for cls in self.dataset.phase_to_classes[phase]:
                if int(cls) >= self.model.current_num_classes:
                    self._build_class_memory_from_current_phase(int(cls), split=split)

    @torch.no_grad()
    def _finalize_phase_memory(self, phase: int, split: str = "train"):
        with self.dataset.memory_build_context(phase):
            for cls in self.dataset.phase_to_classes[int(phase)]:
                self._build_class_memory_from_current_phase(int(cls), split=split)

        self.dataset.finalize_phase(phase)

    # ============================================================
    # Base geometry shaping
    # ============================================================
    def _compute_class_means(self, features, labels):
        return {int(c): features[labels == c].mean(dim=0) for c in labels.unique()}

    def _compute_class_bases(self, features, labels):
        bases = {}
        for c in labels.unique():
            z = features[labels == c]
            if z.size(0) < 2:
                continue

            zc = z - z.mean(0, keepdim=True)
            q = min(self.subspace_rank, max(zc.size(0) - 1, 0), zc.size(1))
            if q <= 0:
                continue

            try:
                _, _, vh = torch.linalg.svd(zc, full_matrices=False)
                active = vh[:q].t().contiguous()
            except RuntimeError:
                _, _, v = torch.pca_lowrank(zc, q=q, center=False)
                active = v[:, :q]

            bases[int(c)] = _complete_orthonormal_basis(active, self.subspace_rank)
        return bases

    def _base_geometry_loss(self, features, labels):
        means = self._compute_class_means(features, labels)
        bases = self._compute_class_bases(features, labels)

        compact = self._zero(features)
        sep = self._zero(features)
        ortho = self._zero(features)
        center_norm = self._zero(features)
        radius = self._zero(features)

        classes = list(means.keys())

        for c in classes:
            z = features[labels == c]
            residual = z - means[c]
            compact += residual.pow(2).mean()
            radius += residual.norm(dim=1).mean()
            center_norm += means[c].pow(2).mean()

        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                d = torch.norm(means[classes[i]] - means[classes[j]])
                sep += F.relu(self.base_margin - d).pow(2)

                if classes[i] in bases and classes[j] in bases:
                    ortho += (bases[classes[i]].T @ bases[classes[j]]).pow(2).mean()

        denom = max(len(classes), 1)
        return compact, sep, ortho, center_norm / denom, radius / denom

    # ============================================================
    # Replay / alignment / geometry scoring
    # ============================================================
    def _sample_replay_from_snapshot(self, old_bank_snapshot, old_class_count):
        if (
            old_class_count <= 0
            or self.replay_weight <= 0.0
            or self.replay_per_class <= 0
            or old_bank_snapshot is None
        ):
            return None, None

        means = old_bank_snapshot["means"][:old_class_count].to(self.device)
        bases = old_bank_snapshot["bases"][:old_class_count].to(self.device)
        vars_ = old_bank_snapshot["variances"][:old_class_count].to(self.device)

        active_ranks = old_bank_snapshot.get("active_ranks", None)
        reliability = old_bank_snapshot.get("reliability", None)
        active_ranks = active_ranks[:old_class_count].to(self.device) if isinstance(active_ranks, torch.Tensor) and active_ranks.numel() > 0 else None
        reliability = reliability[:old_class_count].to(self.device) if isinstance(reliability, torch.Tensor) and reliability.numel() > 0 else None

        residual_scale = float(getattr(self.args, "replay_residual_scale", 0.35))
        subspace_scale = float(getattr(self.args, "replay_subspace_scale", 0.8))
        min_rel = float(getattr(self.args, "replay_min_reliability", 0.05))

        feats, labels = [], []

        for c in range(old_class_count):
            mu, u, var = means[c], bases[c], vars_[c]
            eig = var[:-1].clamp_min(self.geom_var_floor)
            res = var[-1].clamp_min(self.geom_var_floor)

            q = int(active_ranks[c].item()) if active_ranks is not None else u.size(1)
            q = max(0, min(q, u.size(1)))
            rel = float(reliability[c].detach().clamp(min_rel, 1.0).item()) if reliability is not None else 1.0

            class_subspace_scale = subspace_scale * (0.5 + 0.5 * rel)
            class_residual_scale = residual_scale * (0.25 + 0.75 * rel)

            if q > 0:
                u_active = u[:, :q]
                eig_active = eig[:q]
                eps = torch.randn(self.replay_per_class, q, device=self.device, dtype=mu.dtype)
                feat_parallel = mu.unsqueeze(0) + class_subspace_scale * (eps * torch.sqrt(eig_active).unsqueeze(0)) @ u_active.T
            else:
                u_active = torch.empty(mu.numel(), 0, device=self.device, dtype=mu.dtype)
                feat_parallel = mu.unsqueeze(0).expand(self.replay_per_class, -1)

            iso = torch.randn(self.replay_per_class, mu.numel(), device=self.device, dtype=mu.dtype)
            iso = iso * torch.sqrt(res) * class_residual_scale
            if u_active.numel() > 0:
                iso = iso - (iso @ u_active) @ u_active.T

            feats.append(feat_parallel + iso)
            labels.append(torch.full((self.replay_per_class,), c, device=self.device, dtype=torch.long))

        if len(feats) == 0:
            return None, None

        return torch.cat(feats, dim=0), torch.cat(labels, dim=0)

    def _geometry_alignment_losses(self, old_bank_snapshot, old_class_count):
        if old_class_count <= 0 or old_bank_snapshot is None:
            z = self._zero()
            return z, z, z, z

        sb = self._safe_get_subspace_bank()
        cur_means = sb.get("means", None)
        cur_bases = sb.get("bases", None)
        cur_vars = sb.get("variances", None)

        old_means = old_bank_snapshot.get("means", None)
        old_bases = old_bank_snapshot.get("bases", None)
        old_vars = old_bank_snapshot.get("variances", None)

        if any(v is None for v in [cur_means, cur_bases, cur_vars, old_means, old_bases, old_vars]):
            z = self._zero()
            return z, z, z, z

        if cur_means.numel() == 0 or old_means.numel() == 0:
            z = self._zero()
            return z, z, z, z

        cur_means = cur_means[:old_class_count].to(self.device)
        cur_bases = cur_bases[:old_class_count].to(self.device)
        cur_vars = cur_vars[:old_class_count].to(self.device)

        old_means = old_means[:old_class_count].to(self.device)
        old_bases = old_bases[:old_class_count].to(self.device)
        old_vars = old_vars[:old_class_count].to(self.device)

        mean_loss = F.mse_loss(cur_means, old_means)
        proj_cur = self._projector_from_basis(cur_bases)
        proj_old = self._projector_from_basis(old_bases)
        basis_loss = F.mse_loss(proj_cur, proj_old)
        var_loss = F.mse_loss(self._safe_log_variances(cur_vars), self._safe_log_variances(old_vars))

        return mean_loss, basis_loss, var_loss, self._zero(mean_loss)

    def _geometry_distance(self, z, mu, u, var):
        if z is None or z.numel() == 0:
            return self._zero(z)
        diff = z - mu.unsqueeze(0)
        eig = var[:-1].clamp_min(self.geom_var_floor)
        res = var[-1].clamp_min(self.geom_var_floor)

        proj = diff @ u
        parallel = (proj.pow(2) / eig.unsqueeze(0)).sum(dim=1)
        recon = proj @ u.T
        residual = diff - recon
        residual_term = residual.pow(2).sum(dim=1) / res
        return (parallel + residual_term) / max(diff.size(1), 1)

    def _geometry_distance_matrix(self, z, means, bases, vars_):
        if z is None or z.numel() == 0:
            return torch.empty(0, 0, device=self.device)
        if means is None or bases is None or vars_ is None or means.numel() == 0:
            return torch.empty(z.size(0), 0, device=z.device, dtype=z.dtype)
        all_d = []
        for c in range(means.size(0)):
            d = self._geometry_distance(z, means[c], bases[c], vars_[c])
            all_d.append(d.unsqueeze(1))
        return torch.cat(all_d, dim=1) if len(all_d) > 0 else torch.empty(z.size(0), 0, device=z.device, dtype=z.dtype)

    def _replay_geometry_loss(self, replay_x, replay_y, old_bank_snapshot=None):
        if replay_x is None or replay_y is None or replay_x.numel() == 0:
            return self._zero(replay_x)

        if old_bank_snapshot is None:
            return self._zero(replay_x)

        means = old_bank_snapshot["means"].to(self.device)
        bases = old_bank_snapshot["bases"].to(self.device)
        vars_ = old_bank_snapshot["variances"].to(self.device)

        total = self._zero(replay_x)
        count = 0

        for cls in replay_y.unique():
            cls = int(cls.item())
            z = replay_x[replay_y == cls]
            if z.numel() == 0:
                continue
            d = self._geometry_distance(z, means[cls], bases[cls], vars_[cls])
            total = total + d.mean()
            count += 1

        if count == 0:
            return self._zero(replay_x)

        return total / count

    def _geometry_separation_loss(self, features, labels, old_class_count, new_class_ids):
        """New -> old geometry-energy separation."""
        if old_class_count <= 0 or features is None or features.numel() == 0:
            return self._zero(features)

        new_class_ids = [int(c) for c in new_class_ids]
        new_mask = torch.zeros_like(labels, dtype=torch.bool)
        for c in new_class_ids:
            new_mask |= (labels == c)

        if not new_mask.any():
            return self._zero(features)

        z_new = features[new_mask]
        y_new = labels[new_mask]

        calibrated_old = self.model.get_calibrated_old_subspace_bank(old_class_count)
        if calibrated_old["means"].numel() == 0:
            return self._zero(features)

        old_means = calibrated_old["means"].to(features.device)
        old_bases = calibrated_old["bases"].to(features.device)
        old_vars = calibrated_old["variances"].to(features.device)

        cur_bank = self.model.get_subspace_bank()
        cur_means = cur_bank["means"].to(features.device)
        cur_bases = cur_bank["bases"].to(features.device)
        cur_vars = cur_bank["variances"].to(features.device)

        losses = []
        margin = float(getattr(self.args, "insert_margin", 5.0))

        for cls in new_class_ids:
            if int(cls) >= cur_means.size(0):
                continue
            cls_mask = y_new == int(cls)
            if not cls_mask.any():
                continue

            z_cls = z_new[cls_mask]
            own_d = self._geometry_distance(z_cls, cur_means[int(cls)], cur_bases[int(cls)], cur_vars[int(cls)])
            old_d_mat = self._geometry_distance_matrix(z_cls, old_means, old_bases, old_vars)
            if old_d_mat.numel() == 0:
                continue
            closest_old_d = old_d_mat.min(dim=1).values
            losses.append(F.relu(margin + own_d - closest_old_d).mean())

        if len(losses) == 0:
            return self._zero(features)

        return torch.stack(losses).mean()

    def _symmetric_geometry_separation_loss(
        self,
        new_features,
        new_labels,
        replay_features,
        replay_labels,
        old_class_count,
        new_class_ids,
        old_bank_snapshot,
    ):
        """Symmetric old/new energy separation to reduce new-class dominance."""
        if old_class_count <= 0:
            return self._zero(new_features)

        losses = [self._geometry_separation_loss(new_features, new_labels, old_class_count, new_class_ids)]

        if replay_features is None or replay_labels is None or replay_features.numel() == 0 or old_bank_snapshot is None:
            return torch.stack(losses).mean()

        cur_bank = self.model.get_subspace_bank()
        cur_means = cur_bank.get("means", None)
        cur_bases = cur_bank.get("bases", None)
        cur_vars = cur_bank.get("variances", None)
        if cur_means is None or cur_bases is None or cur_vars is None or cur_means.numel() == 0:
            return torch.stack(losses).mean()

        old_means = old_bank_snapshot["means"].to(self.device)
        old_bases = old_bank_snapshot["bases"].to(self.device)
        old_vars = old_bank_snapshot["variances"].to(self.device)

        new_ids = torch.tensor([int(c) for c in new_class_ids], device=self.device, dtype=torch.long)
        new_ids = new_ids[new_ids < cur_means.size(0)]
        if new_ids.numel() == 0:
            return torch.stack(losses).mean()

        new_means = cur_means.to(self.device).index_select(0, new_ids)
        new_bases = cur_bases.to(self.device).index_select(0, new_ids)
        new_vars = cur_vars.to(self.device).index_select(0, new_ids)

        margin = float(getattr(self.args, "old_new_energy_margin", getattr(self, "insert_margin", 5.0)))
        old_terms = []

        for cls in replay_labels.unique():
            cls_int = int(cls.item())
            if cls_int >= old_means.size(0):
                continue
            z = replay_features[replay_labels == cls_int]
            if z.numel() == 0:
                continue

            own_old_d = self._geometry_distance(z, old_means[cls_int], old_bases[cls_int], old_vars[cls_int])
            new_d = self._geometry_distance_matrix(z, new_means, new_bases, new_vars)
            if new_d.numel() == 0:
                continue
            closest_new_d = new_d.min(dim=1).values
            old_terms.append(F.relu(margin + own_old_d - closest_new_d).mean())

        if len(old_terms) > 0:
            losses.append(torch.stack(old_terms).mean())

        return torch.stack(losses).mean()

    def _new_class_volume_loss(self, features, labels, new_class_ids):
        if self.new_volume_weight <= 0.0 or features is None or features.numel() == 0:
            return self._zero(features)

        cur_bank = self.model.get_subspace_bank() if hasattr(self.model, "get_subspace_bank") else {}
        old_class_count = int(getattr(self.model, "old_class_count", 0))
        old_vars = cur_bank.get("variances", None) if isinstance(cur_bank, dict) else None

        relative_target = None
        if isinstance(old_vars, torch.Tensor) and old_vars.numel() > 0 and old_class_count > 0:
            old_trace = old_vars[:old_class_count, :-1].sum(dim=1) + old_vars[:old_class_count, -1]
            if old_trace.numel() > 0:
                margin = float(getattr(self.args, "new_volume_relative_margin", 1.5))
                relative_target = old_trace.detach().median().to(features.device, features.dtype) * margin

        losses = []
        for cls in new_class_ids:
            z = features[labels == int(cls)]
            if z.size(0) < 2:
                continue

            centered = z - z.mean(dim=0, keepdim=True)
            trace_proxy = centered.pow(2).sum(dim=1).mean()
            target = relative_target if relative_target is not None else torch.tensor(
                float(self.new_volume_target), device=features.device, dtype=features.dtype
            )
            losses.append(F.relu(trace_proxy - target))

        if len(losses) == 0:
            return self._zero(features)

        return torch.stack(losses).mean()

    # ============================================================
    # Spectral guidance
    # ============================================================
    def _spectral_guidance_losses(self, out, y, old_class_count):
        ref = out["features"] if isinstance(out, dict) and "features" in out else None
        z = self._zero(ref)

        if old_class_count <= 0:
            return z, z

        spectral_summary = out.get("spectral_summary", None)
        spectral_ref = out.get("spectral_ref", None)
        band_weights = out.get("band_weights", None)
        band_importance_ref = out.get("band_importance_ref", None)

        spec_loss = z
        band_loss = z

        if (
            spectral_summary is not None
            and spectral_ref is not None
            and torch.is_tensor(spectral_summary)
            and torch.is_tensor(spectral_ref)
            and spectral_summary.numel() > 0
            and spectral_ref.numel() > 0
            and spectral_summary.shape == spectral_ref.shape
        ):
            spec_loss = F.mse_loss(spectral_summary, spectral_ref)

        if (
            band_weights is not None
            and band_importance_ref is not None
            and torch.is_tensor(band_weights)
            and torch.is_tensor(band_importance_ref)
            and band_weights.numel() > 0
            and band_importance_ref.numel() > 0
            and band_weights.shape == band_importance_ref.shape
        ):
            bw = band_weights.clamp_min(1e-8)
            br = band_importance_ref.clamp_min(1e-8)

            bw = bw / bw.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            br = br / br.sum(dim=-1, keepdim=True).clamp_min(1e-8)

            band_loss = F.kl_div(bw.log(), br, reduction="batchmean")

        return spec_loss, band_loss

    # ============================================================
    # Token manifold preservation
    # ============================================================
    def _match_old_classes_by_geometry(self, features, old_bank_snapshot, use_calibrated_old=True):
        if features is None or features.numel() == 0 or old_bank_snapshot is None:
            return None, None

        if use_calibrated_old and hasattr(self.model, "get_calibrated_old_subspace_bank"):
            old_count = int(old_bank_snapshot["means"].size(0))
            calibrated = self.model.get_calibrated_old_subspace_bank(old_count)
            means = calibrated.get("means", None)
            bases = calibrated.get("bases", None)
            vars_ = calibrated.get("variances", None)
        else:
            means = old_bank_snapshot.get("means", None)
            bases = old_bank_snapshot.get("bases", None)
            vars_ = old_bank_snapshot.get("variances", None)

        if means is None or bases is None or vars_ is None or means.numel() == 0:
            return None, None

        means = means.to(features.device)
        bases = bases.to(features.device)
        vars_ = vars_.to(features.device)

        dist_mat = self._geometry_distance_matrix(features, means, bases, vars_)
        min_dist, min_idx = dist_mat.min(dim=1)
        return min_idx, min_dist

    def _token_manifold_loss(self, out, y, old_class_count, old_token_snapshot):
        ref = out["features"] if isinstance(out, dict) and "features" in out else None
        z = self._zero(ref)

        if old_class_count <= 0 or old_token_snapshot is None or len(old_token_snapshot) == 0:
            return z, {"spectral": z, "spatial": z, "cross": z, "fused": z}

        token_relations = out.get("token_relations", None)
        features = out.get("features", None)
        if token_relations is None or features is None:
            return z, {"spectral": z, "spatial": z, "cross": z, "fused": z}

        old_bank_snapshot = self._snapshot_old_bank(old_class_count)
        matched_cls, matched_dist = self._match_old_classes_by_geometry(features, old_bank_snapshot, use_calibrated_old=True)
        if matched_cls is None:
            return z, {"spectral": z, "spatial": z, "cross": z, "fused": z}

        gate_threshold = float(getattr(self.args, "token_match_distance_threshold", 1.5))
        valid = matched_dist < gate_threshold

        old_reliability = old_bank_snapshot.get("reliability", None)
        if isinstance(old_reliability, torch.Tensor) and old_reliability.numel() > 0:
            old_reliability = old_reliability.to(features.device, dtype=features.dtype)
            rel = old_reliability[matched_cls.clamp_max(old_reliability.numel() - 1)]
            rel_threshold = float(getattr(self.args, "token_reliability_threshold", 0.35))
            valid = valid & (rel > rel_threshold)
        else:
            old_reliability = None

        if not valid.any():
            return z, {"spectral": z, "spatial": z, "cross": z, "fused": z}

        spec_losses, spat_losses, cross_losses, fused_losses = [], [], [], []
        cur_spec = token_relations.get("spectral_affinity", None)
        cur_spat = token_relations.get("spatial_affinity", None)
        cur_cross = token_relations.get("cross_affinity", None)
        cur_fused = token_relations.get("fused_affinity", None)

        valid_idx = torch.where(valid)[0]
        for idx in valid_idx.tolist():
            cls = int(matched_cls[idx].item())
            if cls not in old_token_snapshot:
                continue

            tgt = old_token_snapshot[cls]
            if old_reliability is not None and cls < old_reliability.numel():
                w = old_reliability[cls].detach().clamp(0.05, 1.0)
            else:
                w = torch.tensor(1.0, device=features.device, dtype=features.dtype)

            if cur_spec is not None and "spectral_affinity" in tgt:
                spec_losses.append(w * F.mse_loss(cur_spec[idx], tgt["spectral_affinity"].to(cur_spec.device, cur_spec.dtype)))
            if cur_spat is not None and "spatial_affinity" in tgt:
                spat_losses.append(w * F.mse_loss(cur_spat[idx], tgt["spatial_affinity"].to(cur_spat.device, cur_spat.dtype)))
            if cur_cross is not None and "cross_affinity" in tgt:
                cross_losses.append(w * F.mse_loss(cur_cross[idx], tgt["cross_affinity"].to(cur_cross.device, cur_cross.dtype)))
            if cur_fused is not None and "fused_affinity" in tgt:
                fused_losses.append(w * F.mse_loss(cur_fused[idx], tgt["fused_affinity"].to(cur_fused.device, cur_fused.dtype)))

        spec_loss = torch.stack(spec_losses).mean() if len(spec_losses) > 0 else z
        spat_loss = torch.stack(spat_losses).mean() if len(spat_losses) > 0 else z
        cross_loss = torch.stack(cross_losses).mean() if len(cross_losses) > 0 else z
        fused_loss = torch.stack(fused_losses).mean() if len(fused_losses) > 0 else z

        w_spec = float(getattr(self.args, "token_spectral_weight", 0.25))
        w_spat = float(getattr(self.args, "token_spatial_weight", 0.25))
        w_cross = float(getattr(self.args, "token_cross_weight", 0.50))
        w_fused = float(getattr(self.args, "token_fused_weight", 0.0))

        total = w_spec * spec_loss + w_spat * spat_loss + w_cross * cross_loss + w_fused * fused_loss
        return total, {"spectral": spec_loss, "spatial": spat_loss, "cross": cross_loss, "fused": fused_loss}






# import numpy as np
# import torch
# import torch.nn.functional as F



# def _orthonormalize_columns(basis: torch.Tensor) -> torch.Tensor:
#     """Orthonormalize columns for [D,R] or [C,D,R] basis tensors."""
#     if basis is None or basis.numel() == 0:
#         return basis
#     if basis.dim() == 2:
#         q, _ = torch.linalg.qr(basis, mode="reduced")
#         if q.size(1) < basis.size(1):
#             pad = torch.zeros(q.size(0), basis.size(1) - q.size(1), device=q.device, dtype=q.dtype)
#             q = torch.cat([q, pad], dim=1)
#         return q[:, : basis.size(1)]
#     if basis.dim() == 3:
#         return torch.stack([_orthonormalize_columns(b) for b in basis], dim=0)
#     raise ValueError(f"basis must be [D,R] or [C,D,R], got {tuple(basis.shape)}")


# def _complete_orthonormal_basis(active_basis: torch.Tensor, rank: int) -> torch.Tensor:
#     """Return [D, rank] basis without zero columns."""
#     d = int(active_basis.size(0))
#     rank = min(int(rank), d)
#     device, dtype = active_basis.device, active_basis.dtype

#     if active_basis.numel() > 0 and active_basis.size(1) > 0:
#         basis = _orthonormalize_columns(active_basis[:, :rank])
#     else:
#         basis = torch.zeros(d, 0, device=device, dtype=dtype)

#     if basis.size(1) >= rank:
#         return basis[:, :rank]

#     cols = [basis[:, i] for i in range(basis.size(1))]
#     eye = torch.eye(d, device=device, dtype=dtype)

#     for j in range(d):
#         v = eye[:, j].clone()
#         for u in cols:
#             v = v - torch.dot(v, u) * u
#         n = v.norm()
#         if n > 1e-6:
#             cols.append(v / n)
#         if len(cols) >= rank:
#             break

#     while len(cols) < rank:
#         v = torch.randn(d, device=device, dtype=dtype)
#         for u in cols:
#             v = v - torch.dot(v, u) * u
#         cols.append(v / v.norm().clamp_min(1e-6))

#     return torch.stack(cols[:rank], dim=1)


# class TrainerHelper:
#     # ============================================================
#     # Utils
#     # ============================================================
#     def _zero(self, ref=None):
#         if isinstance(ref, torch.Tensor):
#             return torch.tensor(0.0, device=ref.device, dtype=ref.dtype)
#         return torch.tensor(0.0, device=self.device)

#     def _safe_get_subspace_bank(self):
#         return self.model.get_subspace_bank()

#     def _snapshot_old_bank(self, old_class_count: int):
#         """Snapshot old geometry safely, including reliability/active rank and inter-class matrices."""
#         old_class_count = int(old_class_count)
#         sb = self._safe_get_subspace_bank()
#         snap = {}

#         for k, v in sb.items():
#             if not isinstance(v, torch.Tensor) or v.numel() == 0:
#                 snap[k] = v
#                 continue

#             if v.dim() == 0:
#                 snap[k] = v.detach().clone()
#             elif v.dim() == 1:
#                 snap[k] = v[:old_class_count].detach().clone()
#             elif v.dim() == 2 and k.startswith("inter_"):
#                 snap[k] = v[:old_class_count, :old_class_count].detach().clone()
#             elif v.dim() >= 2:
#                 snap[k] = v[:old_class_count].detach().clone()
#             else:
#                 snap[k] = v.detach().clone()

#         return snap

#     def _snapshot_old_token_memory(self, old_class_count: int):
#         snap = {}
#         for c in range(int(old_class_count)):
#             if c in self.token_memory:
#                 snap[c] = {k: v.detach().clone() for k, v in self.token_memory[c].items()}
#         return snap

#     def _projector_from_basis(self, basis: torch.Tensor) -> torch.Tensor:
#         if basis.dim() == 2:
#             return basis @ basis.t()
#         return torch.matmul(basis, basis.transpose(-1, -2))

#     def _safe_log_variances(self, variances: torch.Tensor) -> torch.Tensor:
#         return torch.log(variances.clamp_min(max(self.geom_var_floor, 1e-6)))

#     # ============================================================
#     # Accuracy helpers
#     # ============================================================
#     def _labels_are_local(self, y: torch.Tensor, new_class_ids):
#         if y.numel() == 0:
#             return False
#         allowed_local = set(range(len(new_class_ids)))
#         unique_y = set(int(v) for v in y.detach().cpu().unique().tolist())
#         return unique_y.issubset(allowed_local)

#     def _incremental_accuracy_with_count(self, logits, y, new_class_ids):
#         if y.numel() == 0:
#             return 0, 0

#         class_ids = torch.tensor(new_class_ids, device=logits.device, dtype=torch.long)

#         if class_ids.numel() > 0:
#             max_id = int(class_ids.max().item())
#             if max_id >= logits.size(1):
#                 raise RuntimeError(
#                     f"Classifier output size mismatch: max requested class id {max_id}, "
#                     f"but logits only have {logits.size(1)} classes."
#                 )

#         masked_logits = logits.index_select(1, class_ids)
#         pred_local = masked_logits.argmax(dim=1)

#         if self._labels_are_local(y, new_class_ids):
#             valid = torch.ones_like(y, dtype=torch.bool)
#             correct = int((pred_local == y).sum().item())
#             return correct, int(valid.sum().item())

#         y_local = torch.full_like(y, fill_value=-1)
#         for local_idx, global_cls in enumerate(class_ids):
#             y_local[y == global_cls] = local_idx

#         valid = y_local >= 0
#         if not valid.any():
#             return 0, 0

#         correct = int((pred_local[valid] == y_local[valid]).sum().item())
#         return correct, int(valid.sum().item())

#     def _masked_weighted_ce_new(self, logits, y, new_class_ids):
#         class_ids = torch.tensor(new_class_ids, device=logits.device, dtype=torch.long)

#         if class_ids.numel() > 0:
#             max_id = int(class_ids.max().item())
#             if max_id >= logits.size(1):
#                 raise RuntimeError(
#                     f"Classifier output size mismatch: max requested class id {max_id}, "
#                     f"but logits only have {logits.size(1)} classes. "
#                     f"Current phase classes were not bootstrapped/expanded before training."
#                 )

#         logits_new = logits.index_select(1, class_ids)

#         allowed_local = set(range(len(new_class_ids)))
#         unique_y = set(int(v) for v in y.detach().cpu().unique().tolist())

#         if unique_y.issubset(allowed_local):
#             y_local = y
#         else:
#             y_local = torch.full_like(y, fill_value=-1)
#             for local_idx, global_cls in enumerate(class_ids):
#                 y_local[y == global_cls] = local_idx

#         valid = y_local >= 0
#         logits_new = logits_new[valid]
#         y_local = y_local[valid]

#         if y_local.numel() == 0:
#             return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

#         counts = torch.bincount(y_local, minlength=len(new_class_ids)).float()
#         weights = counts.sum() / counts.clamp_min(1.0)
#         weights = weights / weights.mean().clamp_min(1e-8)

#         return F.cross_entropy(logits_new, y_local, weight=weights)

#     # ============================================================
#     # Memory building: current-phase classes only
#     # ============================================================
#     @torch.no_grad()
#     def _extract_backbone_outputs_for_class(self, cls: int, split: str = "train"):
#         patches_np = self.dataset.get_class_patches(cls, split=split)
#         x = torch.from_numpy(patches_np).float().to(self.device)

#         was_training = self.model.training
#         self.model.eval()

#         feats, band_weights_all, spectral_all = [], [], []
#         spectral_tokens_all, spatial_tokens_all, fused_tokens_all = [], [], []

#         bs = int(getattr(self.args, "subspace_extract_batch_size", 256))

#         for start in range(0, x.size(0), bs):
#             xb = x[start:start + bs]
#             out = self.model.extract_backbone_outputs(xb)

#             feats.append(out["features"].detach())
#             band_weights_all.append(out["band_weights"].detach())
#             spectral_all.append(out["spectral_summary"].detach())

#             if out.get("spectral_tokens") is not None:
#                 spectral_tokens_all.append(out["spectral_tokens"].detach())
#             if out.get("spatial_tokens") is not None:
#                 spatial_tokens_all.append(out["spatial_tokens"].detach())
#             if out.get("fused_tokens") is not None:
#                 fused_tokens_all.append(out["fused_tokens"].detach())

#         if was_training:
#             self.model.train()

#         return {
#             "features": torch.cat(feats, dim=0),
#             "band_weights": torch.cat(band_weights_all, dim=0),
#             "spectral_summary": torch.cat(spectral_all, dim=0),
#             "spectral_tokens": torch.cat(spectral_tokens_all, dim=0) if len(spectral_tokens_all) > 0 else None,
#             "spatial_tokens": torch.cat(spatial_tokens_all, dim=0) if len(spatial_tokens_all) > 0 else None,
#             "fused_tokens": torch.cat(fused_tokens_all, dim=0) if len(fused_tokens_all) > 0 else None,
#         }

#     @torch.no_grad()
#     def _extract_feature_guided_concepts(self, cls: int, split: str = "train", num_concepts=None):
#         num_concepts = int(num_concepts or self.num_concepts_per_class)
#         outs = self._extract_backbone_outputs_for_class(cls, split=split)
#         feat_mat = outs["features"]

#         if feat_mat.size(0) == 1:
#             return feat_mat

#         joint = feat_mat.detach().cpu().numpy().astype("float32")
#         k = max(1, min(int(num_concepts), joint.shape[0]))
#         centers = self.dataset._kmeans_numpy(joint, k=k, seed=self.dataset.seed + int(cls))
#         return torch.from_numpy(centers).float().to(self.device)

#     @torch.no_grad()
#     def _extract_class_geometry(self, cls: int, split: str = "train", rank=None):
#         """Extract reliability-aware low-rank class geometry."""
#         rank = int(rank or self.subspace_rank)
#         outs = self._extract_backbone_outputs_for_class(cls, split=split)

#         feat_mat = torch.nan_to_num(outs["features"], nan=0.0, posinf=0.0, neginf=0.0)
#         band_weights = outs["band_weights"]
#         spectral_summary = outs["spectral_summary"]

#         mean = feat_mat.mean(dim=0)
#         centered = feat_mat - mean.unsqueeze(0)
#         n, d = centered.shape

#         var_floor = float(getattr(self.args, "geom_var_floor", getattr(self, "geom_var_floor", 1e-4)))
#         shrink = float(getattr(self.args, "geometry_variance_shrinkage", 0.10))
#         shrink = max(0.0, min(shrink, 1.0))
#         max_ratio = float(getattr(self.args, "geometry_max_variance_ratio", 50.0))

#         total_var = centered.pow(2).sum(dim=1).mean().clamp_min(var_floor)
#         data_floor = torch.maximum(
#             torch.tensor(var_floor, device=self.device, dtype=feat_mat.dtype),
#             (total_var / max(d, 1)) * 1e-3,
#         )

#         # Correct covariance rank: at most n-1. Anything else is fake geometry.
#         q = min(rank, max(n - 1, 0), d)

#         if q <= 0:
#             active_basis = torch.zeros(d, 0, device=self.device, dtype=feat_mat.dtype)
#             active_eig = torch.empty(0, device=self.device, dtype=feat_mat.dtype)
#             res_var = (total_var / max(d, 1)).clamp_min(data_floor)
#         else:
#             try:
#                 _, s, vh = torch.linalg.svd(centered, full_matrices=False)
#                 active_basis = vh[:q].t().contiguous()
#                 active_eig = (s[:q] ** 2) / max(n - 1, 1)
#             except RuntimeError:
#                 cov = centered.t().mm(centered) / max(n - 1, 1)
#                 evals_all, evecs = torch.linalg.eigh(cov)
#                 idx = torch.argsort(evals_all, descending=True)[:q]
#                 active_basis = evecs[:, idx]
#                 active_eig = evals_all[idx]

#             active_basis = _orthonormalize_columns(active_basis)
#             avg_var = (total_var / max(d, 1)).clamp_min(data_floor)
#             active_eig = (1.0 - shrink) * active_eig + shrink * avg_var
#             active_eig = active_eig.clamp(
#                 min=float(data_floor.item()),
#                 max=float((data_floor * max_ratio).item()),
#             )

#             proj = centered.mm(active_basis)
#             recon = proj.mm(active_basis.t())
#             residual = centered - recon
#             residual_energy = residual.pow(2).sum(dim=1).mean()
#             res_var = (residual_energy / max(d - q, 1)).clamp_min(data_floor)

#         basis = _complete_orthonormal_basis(active_basis, rank)
#         eigvals = torch.full((rank,), float(var_floor), device=self.device, dtype=feat_mat.dtype)
#         if q > 0:
#             eigvals[:q] = active_eig[:q]
#         if q < rank:
#             eigvals[q:] = res_var

#         spectral_proto = spectral_summary.mean(dim=0)
#         band_importance = band_weights.mean(dim=0)
#         band_importance = band_importance / band_importance.sum().clamp_min(1e-8)

#         sample_rel = n / float(n + 5)
#         rank_rel = q / float(max(rank, 1))
#         reliability = min(1.0, max(0.05, 0.7 * sample_rel + 0.3 * rank_rel))

#         return (
#             mean,
#             basis[:, :rank],
#             eigvals[:rank],
#             res_var,
#             spectral_proto,
#             band_importance,
#             torch.tensor(q, device=self.device, dtype=torch.long),
#             torch.tensor(reliability, device=self.device, dtype=feat_mat.dtype),
#         )

#     def _extract_class_geometry(self, cls: int, split: str = "train", rank=None):
#         rank = int(rank or self.subspace_rank)
#         outs = self._extract_backbone_outputs_for_class(cls, split=split)

#         feat_mat = outs["features"]
#         band_weights = outs["band_weights"]
#         spectral_summary = outs["spectral_summary"]

#         mean = feat_mat.mean(dim=0)
#         centered = feat_mat - mean.unsqueeze(0)
#         n, d = centered.shape

#         if n <= 1:
#             basis = torch.zeros(d, rank, device=self.device, dtype=feat_mat.dtype)
#             eigvals = torch.full((rank,), self.geom_var_floor, device=self.device, dtype=feat_mat.dtype)
#             res_var = torch.tensor(self.geom_var_floor, device=self.device, dtype=feat_mat.dtype)
#         else:
#             q = min(rank, min(n, d))
#             try:
#                 _, s, v = torch.pca_lowrank(centered, q=q, center=False)
#                 basis = v[:, :q]
#                 eigvals = (s[:q] ** 2) / max(n - 1, 1)
#             except RuntimeError:
#                 cov = centered.t() @ centered / max(n - 1, 1)
#                 evals_all, evecs = torch.linalg.eigh(cov)
#                 idx = torch.argsort(evals_all, descending=True)[:q]
#                 basis = evecs[:, idx]
#                 eigvals = evals_all[idx]

#             if basis.size(1) < rank:
#                 pad = torch.zeros(d, rank - basis.size(1), device=basis.device, dtype=basis.dtype)
#                 basis = torch.cat([basis, pad], dim=1)

#             eigvals = eigvals.clamp_min(self.geom_var_floor)
#             if eigvals.numel() < rank:
#                 pad = torch.full((rank - eigvals.numel(),), self.geom_var_floor, device=eigvals.device, dtype=eigvals.dtype)
#                 eigvals = torch.cat([eigvals, pad], dim=0)

#             residual = centered - (centered @ basis[:, :q]) @ basis[:, :q].t() if q > 0 else centered
#             if d - q > 0:
#                 res_var = residual.pow(2).sum(dim=1).mean() / max(d - q, 1)
#             else:
#                 res_var = residual.pow(2).sum(dim=1).mean()
#             res_var = res_var.clamp_min(self.geom_var_floor)

#         spectral_proto = spectral_summary.mean(dim=0)
#         band_importance = band_weights.mean(dim=0)
#         band_importance = band_importance / band_importance.sum().clamp_min(1e-8)

#         return mean, basis[:, :rank], eigvals[:rank], res_var, spectral_proto, band_importance

#     @torch.no_grad()
#     def _extract_class_token_relations(self, cls: int, split: str = "train"):
#         outs = self._extract_backbone_outputs_for_class(cls, split=split)
#         spectral_tokens = outs["spectral_tokens"]
#         spatial_tokens = outs["spatial_tokens"]

#         if spectral_tokens is None or spatial_tokens is None:
#             return None

#         rel = self.model.semantic_encoder.summarize_class_token_relations(
#             spectral_tokens=spectral_tokens,
#             spatial_tokens=spatial_tokens,
#         )
#         return {k: v.detach().cpu() for k, v in rel.items()}

#     @torch.no_grad()
#     def _refresh_class_token_memory(self, cls: int, split: str = "train"):
#         rel = self._extract_class_token_relations(cls, split=split)
#         if rel is not None:
#             self.token_memory[int(cls)] = rel

#     @torch.no_grad()
#     def _build_class_memory_from_current_phase(self, cls: int, split: str = "train"):
#         concepts = self._extract_feature_guided_concepts(
#             cls,
#             split=split,
#             num_concepts=self.num_concepts_per_class,
#         )

#         if int(cls) >= self.model.current_num_classes:
#             self.model.add_new_class_concepts(concepts)
#         else:
#             self.model.refresh_class_concepts(int(cls), concepts, reset_delta=True)

#         geom = self._extract_class_geometry(cls, split=split, rank=self.subspace_rank)
#         if len(geom) == 8:
#             mean, basis, eigvals, res_var, spectral_proto, band_importance, active_rank, reliability = geom
#         else:
#             mean, basis, eigvals, res_var, spectral_proto, band_importance = geom
#             active_rank = torch.tensor(self.subspace_rank, device=self.device, dtype=torch.long)
#             reliability = torch.tensor(1.0, device=self.device, dtype=mean.dtype)

#         try:
#             self.model.refresh_class_subspace(
#                 int(cls),
#                 mean,
#                 basis,
#                 eigvals,
#                 res_var,
#                 spectral_proto=spectral_proto,
#                 band_importance=band_importance,
#                 active_rank=active_rank,
#                 reliability=reliability,
#             )
#         except TypeError:
#             self.model.refresh_class_subspace(
#                 int(cls),
#                 mean,
#                 basis,
#                 eigvals,
#                 res_var,
#                 spectral_proto=spectral_proto,
#                 band_importance=band_importance,
#             )
#             gb = getattr(self.model, "geometry_bank", None)
#             if gb is not None:
#                 if hasattr(gb, "active_ranks") and int(cls) < gb.active_ranks.numel():
#                     gb.active_ranks[int(cls)] = active_rank.to(gb.active_ranks.device)
#                 if hasattr(gb, "reliability") and int(cls) < gb.reliability.numel():
#                     gb.reliability[int(cls)] = reliability.to(gb.reliability.device, dtype=gb.reliability.dtype)

#         gb = getattr(self.model, "geometry_bank", None)
#         if gb is not None and hasattr(gb, "refresh_inter_class_geometry"):
#             gb.refresh_inter_class_geometry()

#         self._refresh_class_token_memory(int(cls), split=split)

#     def _build_class_memory_from_current_phase(self, cls: int, split: str = "train"):
#         concepts = self._extract_feature_guided_concepts(
#             cls,
#             split=split,
#             num_concepts=self.num_concepts_per_class,
#         )

#         if int(cls) >= self.model.current_num_classes:
#             self.model.add_new_class_concepts(concepts)
#         else:
#             self.model.refresh_class_concepts(int(cls), concepts, reset_delta=True)

#         mean, basis, eigvals, res_var, spectral_proto, band_importance = self._extract_class_geometry(
#             cls,
#             split=split,
#             rank=self.subspace_rank,
#         )

#         self.model.refresh_class_subspace(
#             int(cls),
#             mean,
#             basis,
#             eigvals,
#             res_var,
#             spectral_proto=spectral_proto,
#             band_importance=band_importance,
#         )

#         self._refresh_class_token_memory(int(cls), split=split)

#     @torch.no_grad()
#     def _bootstrap_phase_classes(self, phase: int, split: str = "train"):
#         phase = int(phase)

#         with self.dataset.memory_build_context(phase):
#             for cls in self.dataset.phase_to_classes[phase]:
#                 if int(cls) >= self.model.current_num_classes:
#                     self._build_class_memory_from_current_phase(int(cls), split=split)

#     @torch.no_grad()
#     def _finalize_phase_memory(self, phase: int, split: str = "train"):
#         with self.dataset.memory_build_context(phase):
#             for cls in self.dataset.phase_to_classes[int(phase)]:
#                 self._build_class_memory_from_current_phase(int(cls), split=split)

#         self.dataset.finalize_phase(phase)

#     # ============================================================
#     # Base geometry shaping
#     # ============================================================
#     def _compute_class_means(self, features, labels):
#         return {int(c): features[labels == c].mean(dim=0) for c in labels.unique()}

#     def _compute_class_bases(self, features, labels):
#         bases = {}
#         for c in labels.unique():
#             z = features[labels == c]
#             if z.size(0) < 2:
#                 continue

#             zc = z - z.mean(0, keepdim=True)
#             q = min(self.subspace_rank, max(zc.size(0) - 1, 0), zc.size(1))
#             if q <= 0:
#                 continue

#             try:
#                 _, _, vh = torch.linalg.svd(zc, full_matrices=False)
#                 active = vh[:q].t().contiguous()
#             except RuntimeError:
#                 _, _, v = torch.pca_lowrank(zc, q=q, center=False)
#                 active = v[:, :q]

#             bases[int(c)] = _complete_orthonormal_basis(active, self.subspace_rank)
#         return bases

#     def _base_geometry_loss(self, features, labels):
#         means = self._compute_class_means(features, labels)
#         bases = self._compute_class_bases(features, labels)

#         compact = self._zero(features)
#         sep = self._zero(features)
#         ortho = self._zero(features)
#         center_norm = self._zero(features)
#         radius = self._zero(features)

#         classes = list(means.keys())

#         for c in classes:
#             z = features[labels == c]
#             residual = z - means[c]
#             compact += residual.pow(2).mean()
#             radius += residual.norm(dim=1).mean()
#             center_norm += means[c].pow(2).mean()

#         for i in range(len(classes)):
#             for j in range(i + 1, len(classes)):
#                 d = torch.norm(means[classes[i]] - means[classes[j]])
#                 sep += F.relu(self.base_margin - d).pow(2)

#                 if classes[i] in bases and classes[j] in bases:
#                     ortho += (bases[classes[i]].T @ bases[classes[j]]).pow(2).mean()

#         denom = max(len(classes), 1)
#         return compact, sep, ortho, center_norm / denom, radius / denom

#     # ============================================================
#     # Replay / alignment / geometry scoring
#     # ============================================================
#     def _sample_replay_from_snapshot(self, old_bank_snapshot, old_class_count):
#         if (
#             old_class_count <= 0
#             or self.replay_weight <= 0.0
#             or self.replay_per_class <= 0
#             or old_bank_snapshot is None
#         ):
#             return None, None

#         means = old_bank_snapshot["means"][:old_class_count].to(self.device)
#         bases = old_bank_snapshot["bases"][:old_class_count].to(self.device)
#         vars_ = old_bank_snapshot["variances"][:old_class_count].to(self.device)

#         active_ranks = old_bank_snapshot.get("active_ranks", None)
#         reliability = old_bank_snapshot.get("reliability", None)
#         active_ranks = active_ranks[:old_class_count].to(self.device) if isinstance(active_ranks, torch.Tensor) and active_ranks.numel() > 0 else None
#         reliability = reliability[:old_class_count].to(self.device) if isinstance(reliability, torch.Tensor) and reliability.numel() > 0 else None

#         residual_scale = float(getattr(self.args, "replay_residual_scale", 0.35))
#         subspace_scale = float(getattr(self.args, "replay_subspace_scale", 0.8))
#         min_rel = float(getattr(self.args, "replay_min_reliability", 0.05))

#         feats, labels = [], []

#         for c in range(old_class_count):
#             mu, u, var = means[c], bases[c], vars_[c]
#             eig = var[:-1].clamp_min(self.geom_var_floor)
#             res = var[-1].clamp_min(self.geom_var_floor)

#             q = int(active_ranks[c].item()) if active_ranks is not None else u.size(1)
#             q = max(0, min(q, u.size(1)))
#             rel = float(reliability[c].detach().clamp(min_rel, 1.0).item()) if reliability is not None else 1.0

#             class_subspace_scale = subspace_scale * (0.5 + 0.5 * rel)
#             class_residual_scale = residual_scale * (0.25 + 0.75 * rel)

#             if q > 0:
#                 u_active = u[:, :q]
#                 eig_active = eig[:q]
#                 eps = torch.randn(self.replay_per_class, q, device=self.device, dtype=mu.dtype)
#                 feat_parallel = mu.unsqueeze(0) + class_subspace_scale * (eps * torch.sqrt(eig_active).unsqueeze(0)) @ u_active.T
#             else:
#                 u_active = torch.empty(mu.numel(), 0, device=self.device, dtype=mu.dtype)
#                 feat_parallel = mu.unsqueeze(0).expand(self.replay_per_class, -1)

#             iso = torch.randn(self.replay_per_class, mu.numel(), device=self.device, dtype=mu.dtype)
#             iso = iso * torch.sqrt(res) * class_residual_scale
#             if u_active.numel() > 0:
#                 iso = iso - (iso @ u_active) @ u_active.T

#             feats.append(feat_parallel + iso)
#             labels.append(torch.full((self.replay_per_class,), c, device=self.device, dtype=torch.long))

#         if len(feats) == 0:
#             return None, None

#         return torch.cat(feats, dim=0), torch.cat(labels, dim=0)

#     def _geometry_alignment_losses(self, old_bank_snapshot, old_class_count):
#         if old_class_count <= 0 or old_bank_snapshot is None:
#             z = self._zero()
#             return z, z, z, z

#         sb = self._safe_get_subspace_bank()
#         cur_means = sb.get("means", None)
#         cur_bases = sb.get("bases", None)
#         cur_vars = sb.get("variances", None)

#         old_means = old_bank_snapshot.get("means", None)
#         old_bases = old_bank_snapshot.get("bases", None)
#         old_vars = old_bank_snapshot.get("variances", None)

#         if any(v is None for v in [cur_means, cur_bases, cur_vars, old_means, old_bases, old_vars]):
#             z = self._zero()
#             return z, z, z, z

#         if cur_means.numel() == 0 or old_means.numel() == 0:
#             z = self._zero()
#             return z, z, z, z

#         cur_means = cur_means[:old_class_count].to(self.device)
#         cur_bases = cur_bases[:old_class_count].to(self.device)
#         cur_vars = cur_vars[:old_class_count].to(self.device)

#         old_means = old_means[:old_class_count].to(self.device)
#         old_bases = old_bases[:old_class_count].to(self.device)
#         old_vars = old_vars[:old_class_count].to(self.device)

#         mean_loss = F.mse_loss(cur_means, old_means)
#         proj_cur = self._projector_from_basis(cur_bases)
#         proj_old = self._projector_from_basis(old_bases)
#         basis_loss = F.mse_loss(proj_cur, proj_old)
#         var_loss = F.mse_loss(self._safe_log_variances(cur_vars), self._safe_log_variances(old_vars))

#         return mean_loss, basis_loss, var_loss, self._zero(mean_loss)

#     def _geometry_distance(self, z, mu, u, var):
#         if z is None or z.numel() == 0:
#             return self._zero(z)
#         diff = z - mu.unsqueeze(0)
#         eig = var[:-1].clamp_min(self.geom_var_floor)
#         res = var[-1].clamp_min(self.geom_var_floor)

#         proj = diff @ u
#         parallel = (proj.pow(2) / eig.unsqueeze(0)).sum(dim=1)
#         recon = proj @ u.T
#         residual = diff - recon
#         residual_term = residual.pow(2).sum(dim=1) / res
#         return (parallel + residual_term) / max(diff.size(1), 1)

#     def _geometry_distance_matrix(self, z, means, bases, vars_):
#         if z is None or z.numel() == 0:
#             return torch.empty(0, 0, device=self.device)
#         if means is None or bases is None or vars_ is None or means.numel() == 0:
#             return torch.empty(z.size(0), 0, device=z.device, dtype=z.dtype)
#         all_d = []
#         for c in range(means.size(0)):
#             d = self._geometry_distance(z, means[c], bases[c], vars_[c])
#             all_d.append(d.unsqueeze(1))
#         return torch.cat(all_d, dim=1) if len(all_d) > 0 else torch.empty(z.size(0), 0, device=z.device, dtype=z.dtype)

#     def _replay_geometry_loss(self, replay_x, replay_y, old_bank_snapshot=None):
#         if replay_x is None or replay_y is None or replay_x.numel() == 0:
#             return self._zero(replay_x)

#         if old_bank_snapshot is None:
#             return self._zero(replay_x)

#         means = old_bank_snapshot["means"].to(self.device)
#         bases = old_bank_snapshot["bases"].to(self.device)
#         vars_ = old_bank_snapshot["variances"].to(self.device)

#         total = self._zero(replay_x)
#         count = 0

#         for cls in replay_y.unique():
#             cls = int(cls.item())
#             z = replay_x[replay_y == cls]
#             if z.numel() == 0:
#                 continue
#             d = self._geometry_distance(z, means[cls], bases[cls], vars_[cls])
#             total = total + d.mean()
#             count += 1

#         if count == 0:
#             return self._zero(replay_x)

#         return total / count

#     def _geometry_separation_loss(self, features, labels, old_class_count, new_class_ids):
#         """New -> old geometry-energy separation."""
#         if old_class_count <= 0 or features is None or features.numel() == 0:
#             return self._zero(features)

#         new_class_ids = [int(c) for c in new_class_ids]
#         new_mask = torch.zeros_like(labels, dtype=torch.bool)
#         for c in new_class_ids:
#             new_mask |= (labels == c)

#         if not new_mask.any():
#             return self._zero(features)

#         z_new = features[new_mask]
#         y_new = labels[new_mask]

#         calibrated_old = self.model.get_calibrated_old_subspace_bank(old_class_count)
#         if calibrated_old["means"].numel() == 0:
#             return self._zero(features)

#         old_means = calibrated_old["means"].to(features.device)
#         old_bases = calibrated_old["bases"].to(features.device)
#         old_vars = calibrated_old["variances"].to(features.device)

#         cur_bank = self.model.get_subspace_bank()
#         cur_means = cur_bank["means"].to(features.device)
#         cur_bases = cur_bank["bases"].to(features.device)
#         cur_vars = cur_bank["variances"].to(features.device)

#         losses = []
#         margin = float(getattr(self.args, "insert_margin", 5.0))

#         for cls in new_class_ids:
#             if int(cls) >= cur_means.size(0):
#                 continue
#             cls_mask = y_new == int(cls)
#             if not cls_mask.any():
#                 continue

#             z_cls = z_new[cls_mask]
#             own_d = self._geometry_distance(z_cls, cur_means[int(cls)], cur_bases[int(cls)], cur_vars[int(cls)])
#             old_d_mat = self._geometry_distance_matrix(z_cls, old_means, old_bases, old_vars)
#             if old_d_mat.numel() == 0:
#                 continue
#             closest_old_d = old_d_mat.min(dim=1).values
#             losses.append(F.relu(margin + own_d - closest_old_d).mean())

#         if len(losses) == 0:
#             return self._zero(features)

#         return torch.stack(losses).mean()

#     def _symmetric_geometry_separation_loss(
#         self,
#         new_features,
#         new_labels,
#         replay_features,
#         replay_labels,
#         old_class_count,
#         new_class_ids,
#         old_bank_snapshot,
#     ):
#         """Symmetric old/new energy separation to reduce new-class dominance."""
#         if old_class_count <= 0:
#             return self._zero(new_features)

#         losses = [self._geometry_separation_loss(new_features, new_labels, old_class_count, new_class_ids)]

#         if replay_features is None or replay_labels is None or replay_features.numel() == 0 or old_bank_snapshot is None:
#             return torch.stack(losses).mean()

#         cur_bank = self.model.get_subspace_bank()
#         cur_means = cur_bank.get("means", None)
#         cur_bases = cur_bank.get("bases", None)
#         cur_vars = cur_bank.get("variances", None)
#         if cur_means is None or cur_bases is None or cur_vars is None or cur_means.numel() == 0:
#             return torch.stack(losses).mean()

#         old_means = old_bank_snapshot["means"].to(self.device)
#         old_bases = old_bank_snapshot["bases"].to(self.device)
#         old_vars = old_bank_snapshot["variances"].to(self.device)

#         new_ids = torch.tensor([int(c) for c in new_class_ids], device=self.device, dtype=torch.long)
#         new_ids = new_ids[new_ids < cur_means.size(0)]
#         if new_ids.numel() == 0:
#             return torch.stack(losses).mean()

#         new_means = cur_means.to(self.device).index_select(0, new_ids)
#         new_bases = cur_bases.to(self.device).index_select(0, new_ids)
#         new_vars = cur_vars.to(self.device).index_select(0, new_ids)

#         margin = float(getattr(self.args, "old_new_energy_margin", getattr(self, "insert_margin", 5.0)))
#         old_terms = []

#         for cls in replay_labels.unique():
#             cls_int = int(cls.item())
#             if cls_int >= old_means.size(0):
#                 continue
#             z = replay_features[replay_labels == cls_int]
#             if z.numel() == 0:
#                 continue

#             own_old_d = self._geometry_distance(z, old_means[cls_int], old_bases[cls_int], old_vars[cls_int])
#             new_d = self._geometry_distance_matrix(z, new_means, new_bases, new_vars)
#             if new_d.numel() == 0:
#                 continue
#             closest_new_d = new_d.min(dim=1).values
#             old_terms.append(F.relu(margin + own_old_d - closest_new_d).mean())

#         if len(old_terms) > 0:
#             losses.append(torch.stack(old_terms).mean())

#         return torch.stack(losses).mean()

#     def _new_class_volume_loss(self, features, labels, new_class_ids):
#         if self.new_volume_weight <= 0.0 or features is None or features.numel() == 0:
#             return self._zero(features)

#         cur_bank = self.model.get_subspace_bank() if hasattr(self.model, "get_subspace_bank") else {}
#         old_class_count = int(getattr(self.model, "old_class_count", 0))
#         old_vars = cur_bank.get("variances", None) if isinstance(cur_bank, dict) else None

#         relative_target = None
#         if isinstance(old_vars, torch.Tensor) and old_vars.numel() > 0 and old_class_count > 0:
#             old_trace = old_vars[:old_class_count, :-1].sum(dim=1) + old_vars[:old_class_count, -1]
#             if old_trace.numel() > 0:
#                 margin = float(getattr(self.args, "new_volume_relative_margin", 1.5))
#                 relative_target = old_trace.detach().median().to(features.device, features.dtype) * margin

#         losses = []
#         for cls in new_class_ids:
#             z = features[labels == int(cls)]
#             if z.size(0) < 2:
#                 continue

#             centered = z - z.mean(dim=0, keepdim=True)
#             trace_proxy = centered.pow(2).sum(dim=1).mean()
#             target = relative_target if relative_target is not None else torch.tensor(
#                 float(self.new_volume_target), device=features.device, dtype=features.dtype
#             )
#             losses.append(F.relu(trace_proxy - target))

#         if len(losses) == 0:
#             return self._zero(features)

#         return torch.stack(losses).mean()

#     # ============================================================
#     # Spectral guidance
#     # ============================================================
#     def _spectral_guidance_losses(self, out, y, old_class_count):
#         ref = out["features"] if isinstance(out, dict) and "features" in out else None
#         z = self._zero(ref)

#         if old_class_count <= 0:
#             return z, z

#         spectral_summary = out.get("spectral_summary", None)
#         spectral_ref = out.get("spectral_ref", None)
#         band_weights = out.get("band_weights", None)
#         band_importance_ref = out.get("band_importance_ref", None)

#         spec_loss = z
#         band_loss = z

#         if (
#             spectral_summary is not None
#             and spectral_ref is not None
#             and torch.is_tensor(spectral_summary)
#             and torch.is_tensor(spectral_ref)
#             and spectral_summary.numel() > 0
#             and spectral_ref.numel() > 0
#             and spectral_summary.shape == spectral_ref.shape
#         ):
#             spec_loss = F.mse_loss(spectral_summary, spectral_ref)

#         if (
#             band_weights is not None
#             and band_importance_ref is not None
#             and torch.is_tensor(band_weights)
#             and torch.is_tensor(band_importance_ref)
#             and band_weights.numel() > 0
#             and band_importance_ref.numel() > 0
#             and band_weights.shape == band_importance_ref.shape
#         ):
#             bw = band_weights.clamp_min(1e-8)
#             br = band_importance_ref.clamp_min(1e-8)

#             bw = bw / bw.sum(dim=-1, keepdim=True).clamp_min(1e-8)
#             br = br / br.sum(dim=-1, keepdim=True).clamp_min(1e-8)

#             band_loss = F.kl_div(bw.log(), br, reduction="batchmean")

#         return spec_loss, band_loss

#     # ============================================================
#     # Token manifold preservation
#     # ============================================================
#     def _match_old_classes_by_geometry(self, features, old_bank_snapshot, use_calibrated_old=True):
#         if features is None or features.numel() == 0 or old_bank_snapshot is None:
#             return None, None

#         if use_calibrated_old and hasattr(self.model, "get_calibrated_old_subspace_bank"):
#             old_count = int(old_bank_snapshot["means"].size(0))
#             calibrated = self.model.get_calibrated_old_subspace_bank(old_count)
#             means = calibrated.get("means", None)
#             bases = calibrated.get("bases", None)
#             vars_ = calibrated.get("variances", None)
#         else:
#             means = old_bank_snapshot.get("means", None)
#             bases = old_bank_snapshot.get("bases", None)
#             vars_ = old_bank_snapshot.get("variances", None)

#         if means is None or bases is None or vars_ is None or means.numel() == 0:
#             return None, None

#         means = means.to(features.device)
#         bases = bases.to(features.device)
#         vars_ = vars_.to(features.device)

#         dist_mat = self._geometry_distance_matrix(features, means, bases, vars_)
#         min_dist, min_idx = dist_mat.min(dim=1)
#         return min_idx, min_dist

#     def _token_manifold_loss(self, out, y, old_class_count, old_token_snapshot):
#         ref = out["features"] if isinstance(out, dict) and "features" in out else None
#         z = self._zero(ref)

#         if old_class_count <= 0 or old_token_snapshot is None or len(old_token_snapshot) == 0:
#             return z, {"spectral": z, "spatial": z, "cross": z, "fused": z}

#         token_relations = out.get("token_relations", None)
#         features = out.get("features", None)
#         if token_relations is None or features is None:
#             return z, {"spectral": z, "spatial": z, "cross": z, "fused": z}

#         old_bank_snapshot = self._snapshot_old_bank(old_class_count)
#         matched_cls, matched_dist = self._match_old_classes_by_geometry(features, old_bank_snapshot, use_calibrated_old=True)
#         if matched_cls is None:
#             return z, {"spectral": z, "spatial": z, "cross": z, "fused": z}

#         gate_threshold = float(getattr(self.args, "token_match_distance_threshold", 1.5))
#         valid = matched_dist < gate_threshold

#         old_reliability = old_bank_snapshot.get("reliability", None)
#         if isinstance(old_reliability, torch.Tensor) and old_reliability.numel() > 0:
#             old_reliability = old_reliability.to(features.device, dtype=features.dtype)
#             rel = old_reliability[matched_cls.clamp_max(old_reliability.numel() - 1)]
#             rel_threshold = float(getattr(self.args, "token_reliability_threshold", 0.35))
#             valid = valid & (rel > rel_threshold)
#         else:
#             old_reliability = None

#         if not valid.any():
#             return z, {"spectral": z, "spatial": z, "cross": z, "fused": z}

#         spec_losses, spat_losses, cross_losses, fused_losses = [], [], [], []
#         cur_spec = token_relations.get("spectral_affinity", None)
#         cur_spat = token_relations.get("spatial_affinity", None)
#         cur_cross = token_relations.get("cross_affinity", None)
#         cur_fused = token_relations.get("fused_affinity", None)

#         valid_idx = torch.where(valid)[0]
#         for idx in valid_idx.tolist():
#             cls = int(matched_cls[idx].item())
#             if cls not in old_token_snapshot:
#                 continue

#             tgt = old_token_snapshot[cls]
#             if old_reliability is not None and cls < old_reliability.numel():
#                 w = old_reliability[cls].detach().clamp(0.05, 1.0)
#             else:
#                 w = torch.tensor(1.0, device=features.device, dtype=features.dtype)

#             if cur_spec is not None and "spectral_affinity" in tgt:
#                 spec_losses.append(w * F.mse_loss(cur_spec[idx], tgt["spectral_affinity"].to(cur_spec.device, cur_spec.dtype)))
#             if cur_spat is not None and "spatial_affinity" in tgt:
#                 spat_losses.append(w * F.mse_loss(cur_spat[idx], tgt["spatial_affinity"].to(cur_spat.device, cur_spat.dtype)))
#             if cur_cross is not None and "cross_affinity" in tgt:
#                 cross_losses.append(w * F.mse_loss(cur_cross[idx], tgt["cross_affinity"].to(cur_cross.device, cur_cross.dtype)))
#             if cur_fused is not None and "fused_affinity" in tgt:
#                 fused_losses.append(w * F.mse_loss(cur_fused[idx], tgt["fused_affinity"].to(cur_fused.device, cur_fused.dtype)))

#         spec_loss = torch.stack(spec_losses).mean() if len(spec_losses) > 0 else z
#         spat_loss = torch.stack(spat_losses).mean() if len(spat_losses) > 0 else z
#         cross_loss = torch.stack(cross_losses).mean() if len(cross_losses) > 0 else z
#         fused_loss = torch.stack(fused_losses).mean() if len(fused_losses) > 0 else z

#         w_spec = float(getattr(self.args, "token_spectral_weight", 0.25))
#         w_spat = float(getattr(self.args, "token_spatial_weight", 0.25))
#         w_cross = float(getattr(self.args, "token_cross_weight", 0.50))
#         w_fused = float(getattr(self.args, "token_fused_weight", 0.0))

#         total = w_spec * spec_loss + w_spat * spat_loss + w_cross * cross_loss + w_fused * fused_loss
#         return total, {"spectral": spec_loss, "spatial": spat_loss, "cross": cross_loss, "fused": fused_loss}

