import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Iterable, Optional, Tuple


# ============================================================
# Generic helpers
# ============================================================
def normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1, eps=1e-6)


def safe_zero_like(
    ref: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if isinstance(ref, torch.Tensor):
        return torch.tensor(0.0, device=ref.device, dtype=ref.dtype)
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32
    return torch.tensor(0.0, device=device, dtype=dtype)


def projector_from_basis(basis: torch.Tensor) -> torch.Tensor:
    if basis.dim() == 2:
        return basis @ basis.t()
    if basis.dim() == 3:
        return torch.matmul(basis, basis.transpose(-1, -2))
    raise ValueError(f"basis must be 2D or 3D, got {tuple(basis.shape)}")


def safe_log_variances(variances: torch.Tensor, var_floor: float) -> torch.Tensor:
    return torch.log(variances.clamp_min(max(float(var_floor), 1e-8)))


def _as_class_tensor(class_ids: Iterable[int], device: torch.device) -> torch.Tensor:
    ids = [int(c) for c in class_ids]
    if len(ids) == 0:
        return torch.empty(0, device=device, dtype=torch.long)
    return torch.tensor(ids, device=device, dtype=torch.long)


# ============================================================
# Geometry distance utilities
# ============================================================
def geometry_distance(
    z: torch.Tensor,
    mu: torch.Tensor,
    basis: torch.Tensor,
    var: torch.Tensor,
    var_floor: float = 1e-4,
    active_rank: Optional[torch.Tensor] = None,
    normalize_by_dim: bool = True,
) -> torch.Tensor:
    """
    Low-rank Gaussian/Mahalanobis-style geometry distance.

    Args:
        z:           [B, D]
        mu:          [D]
        basis:       [D, R]
        var:         [R+1] = [eigvals..., residual_var]
        active_rank: scalar, optional. Inactive PCA directions are ignored.
    Returns:
        [B] distance. Lower is better.
    """
    if z is None or z.numel() == 0:
        return safe_zero_like(z).view(1)

    z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    mu = mu.to(device=z.device, dtype=z.dtype)
    basis = basis.to(device=z.device, dtype=z.dtype)
    var = var.to(device=z.device, dtype=z.dtype)

    diff = z - mu.unsqueeze(0)

    eig = var[:-1].clamp_min(var_floor)
    res = var[-1].clamp_min(var_floor)

    coeff = diff @ basis

    if active_rank is not None:
        ar = int(torch.as_tensor(active_rank).detach().cpu().item())
        ar = max(0, min(ar, basis.size(1)))
        if ar <= 0:
            parallel = torch.zeros(z.size(0), device=z.device, dtype=z.dtype)
            recon = torch.zeros_like(diff)
        else:
            coeff_active = coeff[:, :ar]
            basis_active = basis[:, :ar]
            eig_active = eig[:ar]
            parallel = (coeff_active.pow(2) / eig_active.unsqueeze(0)).sum(dim=1)
            recon = coeff_active @ basis_active.t()
    else:
        parallel = (coeff.pow(2) / eig.unsqueeze(0)).sum(dim=1)
        recon = coeff @ basis.t()

    residual = diff - recon
    residual_term = residual.pow(2).sum(dim=1) / res

    dist = parallel + residual_term
    if normalize_by_dim:
        dist = dist / max(z.size(1), 1)

    return torch.nan_to_num(dist, nan=1e6, posinf=1e6, neginf=1e6)


def geometry_distance_matrix(
    z: torch.Tensor,
    means: torch.Tensor,
    bases: torch.Tensor,
    variances: torch.Tensor,
    var_floor: float = 1e-4,
    active_ranks: Optional[torch.Tensor] = None,
    reliability: Optional[torch.Tensor] = None,
    reliability_penalty: float = 0.05,
    normalize_by_dim: bool = True,
) -> torch.Tensor:
    """
    Args:
        z:          [B, D]
        means:      [C, D]
        bases:      [C, D, R]
        variances:  [C, R+1]
    Returns:
        [B, C] geometry distances.
    """
    if z is None or z.numel() == 0:
        return torch.empty(0, 0, device=means.device if means is not None else torch.device("cpu"))

    if means is None or bases is None or variances is None or means.numel() == 0:
        return torch.empty(z.size(0), 0, device=z.device, dtype=z.dtype)

    means = means.to(device=z.device, dtype=z.dtype)
    bases = bases.to(device=z.device, dtype=z.dtype)
    variances = variances.to(device=z.device, dtype=z.dtype)

    all_d = []
    for c in range(means.size(0)):
        ar = None
        if active_ranks is not None and torch.is_tensor(active_ranks) and active_ranks.numel() > c:
            ar = active_ranks[c]

        d = geometry_distance(
            z,
            means[c],
            bases[c],
            variances[c],
            var_floor=var_floor,
            active_rank=ar,
            normalize_by_dim=normalize_by_dim,
        )

        if reliability is not None and torch.is_tensor(reliability) and reliability.numel() > c:
            rel = reliability[c].to(device=z.device, dtype=z.dtype).clamp(0.05, 1.0)
            d = d + float(reliability_penalty) * (-torch.log(rel))

        all_d.append(d.unsqueeze(1))

    return torch.cat(all_d, dim=1)


# ============================================================
# Logit / concept auxiliary losses
# ============================================================
class GlobalLogitMargin(nn.Module):
    """
    Ensure GT logit exceeds strongest wrong-class logit by a margin.
    """

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = float(margin)

    def forward(self, logits: Optional[torch.Tensor], labels: Optional[torch.Tensor]) -> torch.Tensor:
        if logits is None or not torch.is_tensor(logits) or logits.numel() == 0:
            device = labels.device if labels is not None and torch.is_tensor(labels) else torch.device("cpu")
            return torch.tensor(0.0, device=device)

        if labels is None or not torch.is_tensor(labels) or labels.numel() == 0:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        labels = labels.view(-1).long()
        valid = (labels >= 0) & (labels < logits.size(1))
        if not valid.any() or logits.size(1) <= 1:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        logits = logits[valid]
        labels = labels[valid]

        gt = logits.gather(1, labels.view(-1, 1)).squeeze(1)
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask.scatter_(1, labels.view(-1, 1), False)
        impostor = logits.masked_fill(~mask, -1e9).max(dim=1).values

        return F.relu(self.margin - (gt - impostor)).mean()


class ConceptSeparation(nn.Module):
    """
    Push class concept means apart. Weak auxiliary only.
    """

    def __init__(self, max_cosine: float = 0.25):
        super().__init__()
        self.max_cosine = float(max_cosine)

    def forward(self, concept_bank: Optional[torch.Tensor]) -> torch.Tensor:
        if concept_bank is None or not torch.is_tensor(concept_bank) or concept_bank.numel() == 0:
            return torch.tensor(0.0)

        class_means = normalize(concept_bank.mean(dim=1))
        sim = torch.matmul(class_means, class_means.t())

        C = sim.size(0)
        if C <= 1:
            return torch.tensor(0.0, device=sim.device, dtype=sim.dtype)

        eye = torch.eye(C, device=sim.device, dtype=torch.bool)
        offdiag = sim[~eye]

        return F.relu(offdiag - self.max_cosine).mean()


class FeatureConceptCompactness(nn.Module):
    """
    Pull feature toward its class concepts. Weak in incremental phases.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = float(temperature)

    def forward(
        self,
        features: Optional[torch.Tensor],
        concept_bank: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if features is None or not torch.is_tensor(features) or features.numel() == 0:
            return torch.tensor(0.0)

        if concept_bank is None or not torch.is_tensor(concept_bank) or concept_bank.numel() == 0:
            return torch.tensor(0.0, device=features.device, dtype=features.dtype)

        if labels is None or not torch.is_tensor(labels) or labels.numel() == 0:
            return torch.tensor(0.0, device=features.device, dtype=features.dtype)

        labels = labels.view(-1).long()
        valid = (labels >= 0) & (labels < concept_bank.size(0))
        if not valid.any():
            return torch.tensor(0.0, device=features.device, dtype=features.dtype)

        f = normalize(features[valid])
        cb = normalize(concept_bank)

        target_cb = cb[labels[valid]]
        sims = torch.einsum("bd,bkd->bk", f, target_cb)
        weights = F.softmax(sims / max(self.temperature, 1e-6), dim=-1)
        target = torch.einsum("bk,bkd->bd", weights, target_cb)

        cos = (f * target).sum(dim=-1)
        return (1.0 - cos).mean()


# ============================================================
# Geometry preservation and calibration losses
# ============================================================
class GeometryDriftLoss(nn.Module):
    """
    Preserve old class geometry using frozen memory snapshots.
    Uses projector distance for bases to avoid sign/rotation ambiguity.
    """

    def __init__(
        self,
        var_floor: float = 1e-4,
        mean_weight: float = 1.0,
        basis_weight: float = 1.0,
        var_weight: float = 1.0,
        reliability_weighted: bool = True,
    ):
        super().__init__()
        self.var_floor = float(var_floor)
        self.mean_weight = float(mean_weight)
        self.basis_weight = float(basis_weight)
        self.var_weight = float(var_weight)
        self.reliability_weighted = bool(reliability_weighted)

    def forward(
        self,
        cur_means: Optional[torch.Tensor],
        cur_bases: Optional[torch.Tensor],
        cur_vars: Optional[torch.Tensor],
        old_means: Optional[torch.Tensor],
        old_bases: Optional[torch.Tensor],
        old_vars: Optional[torch.Tensor],
        reliability: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if any(v is None for v in [cur_means, cur_bases, cur_vars, old_means, old_bases, old_vars]):
            z = safe_zero_like(cur_means)
            return {"total": z, "mean": z, "basis": z, "var": z}

        if cur_means.numel() == 0 or old_means.numel() == 0:
            z = safe_zero_like(cur_means)
            return {"total": z, "mean": z, "basis": z, "var": z}

        n = min(cur_means.size(0), old_means.size(0))
        cur_means, old_means = cur_means[:n], old_means[:n]
        cur_bases, old_bases = cur_bases[:n], old_bases[:n]
        cur_vars, old_vars = cur_vars[:n], old_vars[:n]

        if reliability is not None and self.reliability_weighted:
            w = reliability[:n].to(device=cur_means.device, dtype=cur_means.dtype).clamp(0.05, 1.0)
            w = w / w.mean().clamp_min(1e-6)
            mean_loss = (w * (cur_means - old_means).pow(2).mean(dim=1)).mean()
            basis_per = (projector_from_basis(cur_bases) - projector_from_basis(old_bases)).pow(2).flatten(1).mean(dim=1)
            basis_loss = (w * basis_per).mean()
            var_per = (safe_log_variances(cur_vars, self.var_floor) - safe_log_variances(old_vars, self.var_floor)).pow(2).mean(dim=1)
            var_loss = (w * var_per).mean()
        else:
            mean_loss = F.mse_loss(cur_means, old_means)
            basis_loss = F.mse_loss(projector_from_basis(cur_bases), projector_from_basis(old_bases))
            var_loss = F.mse_loss(
                safe_log_variances(cur_vars, self.var_floor),
                safe_log_variances(old_vars, self.var_floor),
            )

        total = self.mean_weight * mean_loss + self.basis_weight * basis_loss + self.var_weight * var_loss
        return {"total": total, "mean": mean_loss, "basis": basis_loss, "var": var_loss}


class GeometryCalibrationRegularizationLoss(nn.Module):
    """
    Keep calibrated old geometry close to frozen old geometry.
    Includes optional projector-basis regularization.
    """

    def __init__(
        self,
        var_floor: float = 1e-4,
        mean_weight: float = 1.0,
        var_weight: float = 1.0,
        basis_weight: float = 0.2,
    ):
        super().__init__()
        self.var_floor = float(var_floor)
        self.mean_weight = float(mean_weight)
        self.var_weight = float(var_weight)
        self.basis_weight = float(basis_weight)

    def forward(
        self,
        raw_means: Optional[torch.Tensor],
        raw_variances: Optional[torch.Tensor],
        calibrated_means: Optional[torch.Tensor],
        calibrated_variances: Optional[torch.Tensor],
        raw_bases: Optional[torch.Tensor] = None,
        calibrated_bases: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if any(v is None for v in [raw_means, raw_variances, calibrated_means, calibrated_variances]):
            z = safe_zero_like(raw_means)
            return {"total": z, "mean": z, "basis": z, "var": z}

        if raw_means.numel() == 0 or calibrated_means.numel() == 0:
            z = safe_zero_like(raw_means)
            return {"total": z, "mean": z, "basis": z, "var": z}

        mean_reg = F.mse_loss(calibrated_means, raw_means)

        raw_logv = safe_log_variances(raw_variances, self.var_floor)
        cal_logv = safe_log_variances(calibrated_variances, self.var_floor)
        var_reg = F.mse_loss(cal_logv, raw_logv)

        if raw_bases is not None and calibrated_bases is not None and raw_bases.numel() > 0:
            basis_reg = F.mse_loss(projector_from_basis(calibrated_bases), projector_from_basis(raw_bases))
        else:
            basis_reg = safe_zero_like(raw_means)

        total = self.mean_weight * mean_reg + self.var_weight * var_reg + self.basis_weight * basis_reg
        return {"total": total, "mean": mean_reg, "basis": basis_reg, "var": var_reg}


# ============================================================
# Geometry separation losses
# ============================================================
class GeometrySeparationLoss(nn.Module):
    """
    One-sided new-vs-old geometry separation.

    For each new sample z of class y:
        relu(margin + d(z, G_y_new) - min_c_old d(z, G_c_old))
    """

    def __init__(self, margin: float = 5.0, var_floor: float = 1e-4):
        super().__init__()
        self.margin = float(margin)
        self.var_floor = float(var_floor)

    def forward(
        self,
        features: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        new_class_ids: Iterable[int],
        cur_means: Optional[torch.Tensor],
        cur_bases: Optional[torch.Tensor],
        cur_variances: Optional[torch.Tensor],
        calibrated_old_means: Optional[torch.Tensor],
        calibrated_old_bases: Optional[torch.Tensor],
        calibrated_old_variances: Optional[torch.Tensor],
        cur_active_ranks: Optional[torch.Tensor] = None,
        old_active_ranks: Optional[torch.Tensor] = None,
        old_reliability: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if features is None or labels is None or not torch.is_tensor(features) or features.numel() == 0:
            z = safe_zero_like(features)
            return {"total": z, "own": z, "old": z}

        if any(v is None for v in [cur_means, cur_bases, cur_variances, calibrated_old_means, calibrated_old_bases, calibrated_old_variances]):
            z = safe_zero_like(features)
            return {"total": z, "own": z, "old": z}

        if calibrated_old_means.numel() == 0:
            z = safe_zero_like(features)
            return {"total": z, "own": z, "old": z}

        labels = labels.view(-1).long()
        new_class_ids = [int(c) for c in new_class_ids]

        new_mask = torch.zeros_like(labels, dtype=torch.bool)
        for c in new_class_ids:
            new_mask |= labels == c

        if not new_mask.any():
            z = safe_zero_like(features)
            return {"total": z, "own": z, "old": z}

        z_new = features[new_mask]
        y_new = labels[new_mask]

        old_d_mat_all = geometry_distance_matrix(
            z_new,
            calibrated_old_means,
            calibrated_old_bases,
            calibrated_old_variances,
            var_floor=self.var_floor,
            active_ranks=old_active_ranks,
            reliability=old_reliability,
        )

        own_terms, old_terms, total_terms = [], [], []
        for cls in new_class_ids:
            cls_mask = y_new == int(cls)
            if not cls_mask.any():
                continue

            z_cls = z_new[cls_mask]
            ar = cur_active_ranks[int(cls)] if cur_active_ranks is not None and cur_active_ranks.numel() > int(cls) else None

            own_d = geometry_distance(
                z_cls,
                cur_means[int(cls)],
                cur_bases[int(cls)],
                cur_variances[int(cls)],
                var_floor=self.var_floor,
                active_rank=ar,
            )

            old_d = old_d_mat_all[cls_mask].min(dim=1).values
            total_terms.append(F.relu(self.margin + own_d - old_d).mean())
            own_terms.append(own_d.mean())
            old_terms.append(old_d.mean())

        if len(total_terms) == 0:
            z = safe_zero_like(features)
            return {"total": z, "own": z, "old": z}

        return {
            "total": torch.stack(total_terms).mean(),
            "own": torch.stack(own_terms).mean(),
            "old": torch.stack(old_terms).mean(),
        }


class SymmetricGeometrySeparationLoss(nn.Module):
    """
    Symmetric old/new geometry-energy separation.

    New sample:
        d(z_new, own_new) + margin < min_old d(z_new, old)

    Old replay sample:
        d(z_old, own_old) + margin < min_new d(z_old, new)
    """

    def __init__(self, margin: float = 5.0, var_floor: float = 1e-4):
        super().__init__()
        self.margin = float(margin)
        self.var_floor = float(var_floor)

    def forward(
        self,
        new_features: Optional[torch.Tensor],
        new_labels: Optional[torch.Tensor],
        replay_features: Optional[torch.Tensor],
        replay_labels: Optional[torch.Tensor],
        new_class_ids: Iterable[int],
        old_means: Optional[torch.Tensor],
        old_bases: Optional[torch.Tensor],
        old_variances: Optional[torch.Tensor],
        cur_means: Optional[torch.Tensor],
        cur_bases: Optional[torch.Tensor],
        cur_variances: Optional[torch.Tensor],
        old_active_ranks: Optional[torch.Tensor] = None,
        cur_active_ranks: Optional[torch.Tensor] = None,
        old_reliability: Optional[torch.Tensor] = None,
        cur_reliability: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        ref = new_features if isinstance(new_features, torch.Tensor) else replay_features
        z0 = safe_zero_like(ref)

        if any(v is None for v in [old_means, old_bases, old_variances, cur_means, cur_bases, cur_variances]):
            return {"total": z0, "new_old": z0, "old_new": z0}

        losses = []
        new_old_loss = z0
        old_new_loss = z0

        # New -> old separation.
        if new_features is not None and new_labels is not None and new_features.numel() > 0:
            one_sided = GeometrySeparationLoss(margin=self.margin, var_floor=self.var_floor)
            out = one_sided(
                new_features,
                new_labels,
                new_class_ids,
                cur_means,
                cur_bases,
                cur_variances,
                old_means,
                old_bases,
                old_variances,
                cur_active_ranks=cur_active_ranks,
                old_active_ranks=old_active_ranks,
                old_reliability=old_reliability,
            )
            new_old_loss = out["total"]
            losses.append(new_old_loss)

        # Old replay -> new separation.
        if replay_features is not None and replay_labels is not None and replay_features.numel() > 0:
            new_ids = _as_class_tensor(new_class_ids, replay_features.device)
            if new_ids.numel() > 0:
                new_means = cur_means.index_select(0, new_ids)
                new_bases = cur_bases.index_select(0, new_ids)
                new_vars = cur_variances.index_select(0, new_ids)

                new_ar = cur_active_ranks.index_select(0, new_ids) if cur_active_ranks is not None and cur_active_ranks.numel() > int(new_ids.max().item()) else None
                new_rel = cur_reliability.index_select(0, new_ids) if cur_reliability is not None and cur_reliability.numel() > int(new_ids.max().item()) else None

                terms = []
                for cls in replay_labels.unique():
                    cls_int = int(cls.item())
                    z_cls = replay_features[replay_labels == cls_int]
                    if z_cls.numel() == 0 or cls_int >= old_means.size(0):
                        continue

                    ar = old_active_ranks[cls_int] if old_active_ranks is not None and old_active_ranks.numel() > cls_int else None

                    own_old = geometry_distance(
                        z_cls,
                        old_means[cls_int],
                        old_bases[cls_int],
                        old_variances[cls_int],
                        var_floor=self.var_floor,
                        active_rank=ar,
                    )

                    new_d = geometry_distance_matrix(
                        z_cls,
                        new_means,
                        new_bases,
                        new_vars,
                        var_floor=self.var_floor,
                        active_ranks=new_ar,
                        reliability=new_rel,
                    ).min(dim=1).values

                    terms.append(F.relu(self.margin + own_old - new_d).mean())

                if len(terms) > 0:
                    old_new_loss = torch.stack(terms).mean()
                    losses.append(old_new_loss)

        total = torch.stack(losses).mean() if len(losses) > 0 else z0
        return {"total": total, "new_old": new_old_loss, "old_new": old_new_loss}


# ============================================================
# HSI spectral / token structure losses
# ============================================================
class SpectralGuidanceLoss(nn.Module):
    """
    Preserve spectral summary and band-importance consistency.

    This is HSI-specific. It should be weak: the geometry bank is the main memory,
    spectral guidance only stabilizes band-level drift.
    """

    def __init__(
        self,
        band_loss_type: str = "kl",
        spectral_weight: float = 1.0,
        band_weight: float = 1.0,
    ):
        super().__init__()
        self.band_loss_type = str(band_loss_type).lower()
        self.spectral_weight = float(spectral_weight)
        self.band_weight = float(band_weight)

    def forward(
        self,
        spectral_summary: Optional[torch.Tensor],
        spectral_ref: Optional[torch.Tensor],
        band_weights: Optional[torch.Tensor],
        band_importance_ref: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        z = safe_zero_like(spectral_summary)

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
            spec_loss = F.smooth_l1_loss(spectral_summary, spectral_ref)

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

            if self.band_loss_type == "kl":
                band_loss = 0.5 * (
                    F.kl_div(bw.log(), br, reduction="batchmean")
                    + F.kl_div(br.log(), bw, reduction="batchmean")
                )
            else:
                band_loss = F.mse_loss(bw, br)

        total = self.spectral_weight * spec_loss + self.band_weight * band_loss
        return {"total": total, "spectral": spec_loss, "band": band_loss}


class TokenManifoldPreservationLoss(nn.Module):
    """
    Preserve token-relation structure against stored class memory.

    Supports optional reliability weighting. Use this as a weak auxiliary loss,
    not as the main classifier memory.
    """

    def __init__(
        self,
        spectral_weight: float = 1.0,
        spatial_weight: float = 1.0,
        cross_weight: float = 1.0,
        fused_weight: float = 0.0,
        loss_type: str = "smooth_l1",
    ):
        super().__init__()
        self.spectral_weight = float(spectral_weight)
        self.spatial_weight = float(spatial_weight)
        self.cross_weight = float(cross_weight)
        self.fused_weight = float(fused_weight)
        self.loss_type = str(loss_type).lower()

    def _relation_loss(self, cur: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        tgt = tgt.to(device=cur.device, dtype=cur.dtype)

        # Allow class-level target [N,N] against sample-level current [B,N,N].
        if cur.dim() == 3 and tgt.dim() == 2:
            tgt = tgt.unsqueeze(0).expand(cur.size(0), -1, -1)

        if cur.shape != tgt.shape:
            return safe_zero_like(cur)

        if self.loss_type in {"mse", "l2"}:
            return F.mse_loss(cur, tgt)
        return F.smooth_l1_loss(cur, tgt)

    def forward(
        self,
        current_relations: Optional[Dict[str, torch.Tensor]],
        target_relations: Optional[Dict[str, torch.Tensor]],
        reliability_weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if current_relations is None or target_relations is None:
            z = torch.tensor(0.0)
            return {"total": z, "spectral": z, "spatial": z, "cross": z, "fused": z}

        device = None
        dtype = None
        for v in current_relations.values():
            if torch.is_tensor(v):
                device = v.device
                dtype = v.dtype
                break

        z = safe_zero_like(device=device, dtype=dtype)

        losses = {"spectral": z, "spatial": z, "cross": z, "fused": z}

        mapping = {
            "spectral": "spectral_affinity",
            "spatial": "spatial_affinity",
            "cross": "cross_affinity",
            "fused": "fused_affinity",
        }

        for name, key in mapping.items():
            if key in current_relations and key in target_relations:
                losses[name] = self._relation_loss(current_relations[key], target_relations[key])

        total = (
            self.spectral_weight * losses["spectral"]
            + self.spatial_weight * losses["spatial"]
            + self.cross_weight * losses["cross"]
            + self.fused_weight * losses["fused"]
        )

        if reliability_weight is not None and torch.is_tensor(reliability_weight):
            w = reliability_weight.to(device=device, dtype=dtype).clamp(0.05, 1.0).mean()
            total = total * w

        return {
            "total": total,
            "spectral": losses["spectral"],
            "spatial": losses["spatial"],
            "cross": losses["cross"],
            "fused": losses["fused"],
        }


# ============================================================
# Classifier adaptation regularizer
# ============================================================
class ClassifierAdaptationRegularization(nn.Module):
    """
    Prevent classifier calibration/debias parameters from becoming the real model.
    Use only with a small weight.
    """

    def __init__(self, bias_weight: float = 1.0, temp_weight: float = 0.25, alpha_weight: float = 0.01):
        super().__init__()
        self.bias_weight = float(bias_weight)
        self.temp_weight = float(temp_weight)
        self.alpha_weight = float(alpha_weight)

    def forward(self, classifier: nn.Module) -> Dict[str, torch.Tensor]:
        device = next(classifier.parameters()).device
        dtype = next(classifier.parameters()).dtype
        z = torch.tensor(0.0, device=device, dtype=dtype)

        bias_loss = z
        temp_loss = z
        alpha_loss = z

        if hasattr(classifier, "old_bias_offset") and hasattr(classifier, "new_bias_offset"):
            bias_loss = classifier.old_bias_offset.pow(2).mean() + classifier.new_bias_offset.pow(2).mean()

        if hasattr(classifier, "old_temp_offset") and hasattr(classifier, "new_temp_offset"):
            temp_loss = classifier.old_temp_offset.pow(2).mean() + classifier.new_temp_offset.pow(2).mean()
            if hasattr(classifier, "geom_temperature"):
                temp_loss = temp_loss + classifier.geom_temperature.pow(2).mean()

        if hasattr(classifier, "alpha"):
            alpha_loss = classifier.alpha.pow(2).mean()

        total = self.bias_weight * bias_loss + self.temp_weight * temp_loss + self.alpha_weight * alpha_loss
        return {"total": total, "bias": bias_loss, "temp": temp_loss, "alpha": alpha_loss}
