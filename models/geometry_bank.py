import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


# ============================================================
# Standalone helpers
# ============================================================
def _orthonormalize_columns(basis: torch.Tensor) -> torch.Tensor:
    """
    Orthonormalize basis columns.
    Accepts [D,R] or [C,D,R]; returns same shape with orthonormal columns.
    """
    if basis is None or basis.numel() == 0:
        return basis

    if basis.dim() == 2:
        q, _ = torch.linalg.qr(basis, mode="reduced")
        if q.size(1) < basis.size(1):
            pad = torch.zeros(
                q.size(0),
                basis.size(1) - q.size(1),
                device=q.device,
                dtype=q.dtype,
            )
            q = torch.cat([q, pad], dim=1)
        return q[:, : basis.size(1)]

    if basis.dim() == 3:
        return torch.stack([_orthonormalize_columns(b) for b in basis], dim=0)

    raise ValueError(f"basis must be [D,R] or [C,D,R], got {tuple(basis.shape)}")


def _complete_orthonormal_basis(active_basis: torch.Tensor, rank: int) -> torch.Tensor:
    """
    Return [D, rank] orthonormal-ish basis. If active_basis has fewer columns,
    fill the remaining columns with deterministic orthogonal complement vectors.

    This avoids zero basis columns, which silently create degenerate geometry.
    """
    if active_basis.dim() != 2:
        raise ValueError(f"active_basis must be [D,Q], got {tuple(active_basis.shape)}")

    d = int(active_basis.size(0))
    q = int(active_basis.size(1))
    device, dtype = active_basis.device, active_basis.dtype
    rank = min(int(rank), d)

    if q > 0:
        basis = _orthonormalize_columns(active_basis[:, :q])
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

    # Fallback should almost never run, but avoids hard failure under numerical degeneracy.
    while len(cols) < rank:
        v = torch.randn(d, device=device, dtype=dtype)
        for u in cols:
            v = v - torch.dot(v, u) * u
        n = v.norm().clamp_min(1e-6)
        cols.append(v / n)

    return torch.stack(cols[:rank], dim=1)


class GeometryBank(nn.Module):
    """
    Reliability-aware Spectral-Guided Low-Rank Geometry Bank for NECIL-HSI.

    Per class c, stores:
        mean:              [D]
        basis:             [D, R]
        eigvals:           [R]
        res_var:           scalar residual variance
        active_rank:       reliable PCA rank, <= n - 1
        reliability:       scalar memory reliability in [0, 1]
        spectral_proto:    [S] optional spectral prototype
        band_importance:   [S] optional band-importance distribution

    Why this version is safer:
        - does not invent fake PCA rank for low-sample classes
        - avoids zero basis columns by completing an orthonormal complement
        - applies variance shrinkage and robust floors
        - stores active rank and reliability for safe replay/token preservation
        - preserves the existing public API used by NECILModel and TrainerHelper
    """

    def __init__(
        self,
        d_model: int,
        rank: int,
        device: str = "cpu",
        variance_floor: float = 1e-4,
        variance_shrinkage: float = 0.10,
        max_variance_ratio: float = 50.0,
        min_reliability: float = 0.05,
        adjacency_temperature: float = 1.0,
        energy_temperature: float = 1.0,
        volume_temperature: float = 1.0,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.rank = min(int(rank), self.d_model)

        self.variance_floor = float(variance_floor)
        self.variance_shrinkage = float(variance_shrinkage)
        self.max_variance_ratio = float(max_variance_ratio)
        self.min_reliability = float(max(0.0, min(float(min_reliability), 1.0)))
        self.adjacency_temperature = float(adjacency_temperature)
        self.energy_temperature = float(energy_temperature)
        self.volume_temperature = float(volume_temperature)

        dev = torch.device(device)

        self.register_buffer("means", torch.empty((0, self.d_model), device=dev))
        self.register_buffer("bases", torch.empty((0, self.d_model, self.rank), device=dev))
        self.register_buffer("eigvals", torch.empty((0, self.rank), device=dev))
        self.register_buffer("res_vars", torch.empty((0,), device=dev))

        # Reliability-aware geometry metadata.
        self.register_buffer("active_ranks", torch.empty((0,), dtype=torch.long, device=dev))
        self.register_buffer("reliability", torch.empty((0,), device=dev))
        self.register_buffer("sample_counts", torch.empty((0,), device=dev))
        self.register_buffer("geometry_volumes", torch.empty((0,), device=dev))
        self.register_buffer("class_dispersions", torch.empty((0,), device=dev))
        self.register_buffer("class_risk", torch.empty((0,), device=dev))

        # Spectral metadata may be unknown initially; allow dynamic expansion.
        self.register_buffer("spectral_protos", torch.empty((0, 0), device=dev))
        self.register_buffer("band_importances", torch.empty((0, 0), device=dev))
        self.register_buffer("_spectral_dim", torch.tensor(0, dtype=torch.long, device=dev))

        # Inter-class geometry diagnostics / optional regularization support.
        self.register_buffer("inter_center_dist", torch.empty((0, 0), device=dev))
        self.register_buffer("inter_subspace_overlap", torch.empty((0, 0), device=dev))
        self.register_buffer("inter_spectral_dist", torch.empty((0, 0), device=dev))
        self.register_buffer("inter_adjacency", torch.empty((0, 0), device=dev))

    # ============================================================
    # Compatibility properties
    # ============================================================
    @property
    def device(self) -> torch.device:
        return self.means.device

    def __len__(self) -> int:
        return int(self.means.size(0))

    @property
    def resvars(self) -> torch.Tensor:
        return self.res_vars

    @resvars.setter
    def resvars(self, value: torch.Tensor) -> None:
        self.res_vars = value

    # ============================================================
    # Internal helpers
    # ============================================================
    def _dtype(self) -> torch.dtype:
        return self.means.dtype if self.means.numel() > 0 else torch.float32

    def _default_spectral_proto(self, spectral_dim: int, dtype: torch.dtype):
        if spectral_dim <= 0:
            return torch.empty((0,), device=self.device, dtype=dtype)
        return torch.zeros((spectral_dim,), device=self.device, dtype=dtype)

    def _default_band_importance(self, spectral_dim: int, dtype: torch.dtype):
        if spectral_dim <= 0:
            return torch.empty((0,), device=self.device, dtype=dtype)
        return torch.full(
            (spectral_dim,),
            1.0 / float(spectral_dim),
            device=self.device,
            dtype=dtype,
        )

    def _robust_variance_floor(self, total_var: torch.Tensor, d: int) -> torch.Tensor:
        data_floor = (total_var / max(d, 1)) * 1e-3
        return torch.maximum(
            torch.tensor(self.variance_floor, device=total_var.device, dtype=total_var.dtype),
            data_floor,
        )

    def _shrink_eigvals(self, eig: torch.Tensor, total_var: torch.Tensor, d: int) -> torch.Tensor:
        if eig.numel() == 0:
            return eig

        avg_var = (total_var / max(d, 1)).clamp_min(self.variance_floor)
        shrink = float(max(0.0, min(self.variance_shrinkage, 1.0)))
        eig = (1.0 - shrink) * eig + shrink * avg_var

        floor = self._robust_variance_floor(total_var, d)
        ceil = floor * self.max_variance_ratio
        return eig.clamp(min=float(floor.item()), max=float(ceil.item()))

    def _infer_spectral_dim(
        self,
        spectral_proto: Optional[torch.Tensor],
        band_importance: Optional[torch.Tensor],
    ) -> int:
        dims = []
        current = int(self._spectral_dim.item())
        if current > 0:
            dims.append(current)

        if spectral_proto is not None:
            spectral_proto = torch.as_tensor(spectral_proto)
            if spectral_proto.numel() > 0:
                dims.append(int(spectral_proto.numel()))

        if band_importance is not None:
            band_importance = torch.as_tensor(band_importance)
            if band_importance.numel() > 0:
                dims.append(int(band_importance.numel()))

        if len(dims) == 0:
            return current

        if len(set(dims)) != 1:
            raise ValueError(f"Inconsistent spectral dimensions detected: {dims}")

        return dims[0]

    def _ensure_spectral_dim(self, spectral_dim: int, dtype: torch.dtype):
        spectral_dim = int(spectral_dim)
        current = int(self._spectral_dim.item())

        if spectral_dim == current:
            return
        if current > 0 and spectral_dim == 0:
            return
        if current > 0 and spectral_dim > 0 and current != spectral_dim:
            raise ValueError(
                f"GeometryBank spectral dim mismatch: existing {current}, requested {spectral_dim}"
            )

        if current == 0 and spectral_dim > 0:
            n = len(self)
            if n == 0:
                self.spectral_protos = torch.empty((0, spectral_dim), device=self.device, dtype=dtype)
                self.band_importances = torch.empty((0, spectral_dim), device=self.device, dtype=dtype)
            else:
                self.spectral_protos = torch.zeros((n, spectral_dim), device=self.device, dtype=dtype)
                self.band_importances = torch.full(
                    (n, spectral_dim),
                    1.0 / float(spectral_dim),
                    device=self.device,
                    dtype=dtype,
                )
            self._spectral_dim = torch.tensor(spectral_dim, device=self.device, dtype=torch.long)

    def _prepare_mean(self, mean) -> torch.Tensor:
        mean = torch.as_tensor(mean, device=self.device, dtype=self._dtype()).flatten()
        if mean.numel() != self.d_model:
            raise ValueError(f"mean must have {self.d_model} values, got {mean.numel()}")
        return torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)

    def _sanitize_basis(self, basis: torch.Tensor) -> torch.Tensor:
        """
        Enforce shape [D, rank] and return non-degenerate orthonormal columns.
        """
        basis = torch.as_tensor(basis, device=self.device, dtype=self._dtype())

        if basis.dim() != 2:
            raise ValueError(f"basis must be 2D, got {tuple(basis.shape)}")

        # Allow accidental transpose.
        if basis.shape[0] == self.rank and basis.shape[1] == self.d_model:
            basis = basis.transpose(0, 1)

        if basis.shape[0] != self.d_model:
            raise ValueError(
                f"basis first dim must equal d_model={self.d_model}, got {tuple(basis.shape)}"
            )

        if basis.shape[1] > self.rank:
            basis = basis[:, : self.rank]

        col_norms = basis.norm(dim=0) if basis.numel() > 0 else torch.empty(0, device=self.device)
        active = col_norms > 1e-8
        active_basis = basis[:, active] if active.numel() > 0 and active.any() else torch.zeros(
            self.d_model, 0, device=self.device, dtype=basis.dtype
        )

        return _complete_orthonormal_basis(active_basis, self.rank)

    def _prepare_eigvals(self, eigvals, fallback_resvar: Optional[torch.Tensor] = None) -> torch.Tensor:
        eigvals = torch.as_tensor(eigvals, device=self.device, dtype=self._dtype()).flatten()
        floor = float(self.variance_floor)

        if eigvals.numel() > self.rank:
            eigvals = eigvals[: self.rank]
        elif eigvals.numel() < self.rank:
            fill = floor
            if fallback_resvar is not None:
                fill = float(torch.as_tensor(fallback_resvar).detach().clamp_min(floor).item())
            pad = torch.full(
                (self.rank - eigvals.numel(),),
                fill,
                device=self.device,
                dtype=eigvals.dtype,
            )
            eigvals = torch.cat([eigvals, pad], dim=0)

        return torch.nan_to_num(eigvals, nan=floor, posinf=floor, neginf=floor).clamp_min(floor)

    def _prepare_res_var(self, res_var) -> torch.Tensor:
        res_var = torch.as_tensor(res_var, device=self.device, dtype=self._dtype()).reshape(())
        return torch.nan_to_num(res_var, nan=self.variance_floor, posinf=self.variance_floor, neginf=self.variance_floor).clamp_min(self.variance_floor)

    def _geometry_volume_and_dispersion(
        self,
        eigvals: torch.Tensor,
        res_var: torch.Tensor,
        active_rank: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        active_rank_i = int(torch.as_tensor(active_rank).detach().cpu().item())
        active_rank_i = max(0, min(active_rank_i, self.rank))

        if active_rank_i > 0:
            active = eigvals[:active_rank_i].clamp_min(self.variance_floor)
            log_volume = torch.log(active).mean()
            dispersion = active.mean()
        else:
            log_volume = torch.log(res_var.clamp_min(self.variance_floor))
            dispersion = res_var.clamp_min(self.variance_floor)

        return log_volume, dispersion

    def _prepare_spectral_pair(
        self,
        spectral_proto=None,
        band_importance=None,
        dtype=torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        spectral_dim = self._infer_spectral_dim(spectral_proto, band_importance)
        self._ensure_spectral_dim(spectral_dim, dtype=dtype)

        if spectral_dim <= 0:
            return (
                torch.empty((0,), device=self.device, dtype=dtype),
                torch.empty((0,), device=self.device, dtype=dtype),
            )

        if spectral_proto is None or torch.as_tensor(spectral_proto).numel() == 0:
            spectral_proto = self._default_spectral_proto(spectral_dim, dtype=dtype)
        else:
            spectral_proto = torch.as_tensor(spectral_proto, device=self.device, dtype=dtype).flatten()
            if spectral_proto.numel() != spectral_dim:
                raise ValueError(
                    f"spectral_proto dim mismatch: expected {spectral_dim}, got {spectral_proto.numel()}"
                )
            spectral_proto = torch.nan_to_num(spectral_proto, nan=0.0, posinf=0.0, neginf=0.0)

        if band_importance is None or torch.as_tensor(band_importance).numel() == 0:
            band_importance = self._default_band_importance(spectral_dim, dtype=dtype)
        else:
            band_importance = torch.as_tensor(band_importance, device=self.device, dtype=dtype).flatten()
            if band_importance.numel() != spectral_dim:
                raise ValueError(
                    f"band_importance dim mismatch: expected {spectral_dim}, got {band_importance.numel()}"
                )
            band_importance = torch.nan_to_num(band_importance, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
            s = band_importance.sum()
            if not torch.isfinite(s) or s.abs() <= 0:
                band_importance = self._default_band_importance(spectral_dim, dtype=dtype)
            else:
                band_importance = band_importance / s.clamp_min(1e-8)

        return spectral_proto, band_importance

    def _resize_inter_class_buffers(self, n: int, dtype: torch.dtype):
        z = torch.zeros(n, n, device=self.device, dtype=dtype)

        def copy_old(old_tensor: torch.Tensor) -> torch.Tensor:
            if old_tensor is not None and old_tensor.numel() > 0:
                old = min(old_tensor.size(0), n)
                z_new = z.clone()
                z_new[:old, :old] = old_tensor[:old, :old].to(device=self.device, dtype=dtype)
                return z_new
            return z.clone()

        self.inter_center_dist = copy_old(self.inter_center_dist)
        self.inter_subspace_overlap = copy_old(self.inter_subspace_overlap)
        self.inter_spectral_dist = copy_old(self.inter_spectral_dist)
        self.inter_adjacency = copy_old(self.inter_adjacency)

    # ============================================================
    # Geometry extraction from features
    # ============================================================
    @torch.no_grad()
    def extract_geometry(
        self,
        feat: torch.Tensor,
        labels: torch.Tensor,
        spectral_summary: Optional[torch.Tensor] = None,
        band_weights: Optional[torch.Tensor] = None,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Extract robust low-rank class geometry from a feature batch.
        This is optional but useful if trainer code wants the bank to compute geometry directly.
        """
        geo: Dict[int, Dict[str, torch.Tensor]] = {}
        if feat is None or labels is None or feat.numel() == 0 or labels.numel() == 0:
            return geo

        feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        device, dtype = feat.device, feat.dtype

        for cid_t in torch.unique(labels):
            cid = int(cid_t.item())
            mask = labels == cid_t
            z = feat[mask]
            if z.size(0) == 0:
                continue

            n, d = z.shape
            mean = z.mean(dim=0)
            centered = z - mean.unsqueeze(0)
            total_var = centered.pow(2).sum(dim=1).mean().clamp_min(self.variance_floor)

            # Correct covariance rank: cannot exceed n - 1.
            q = min(self.rank, max(n - 1, 0), d)

            if q <= 0:
                active_basis = torch.zeros(d, 0, device=device, dtype=dtype)
                active_eig = torch.empty(0, device=device, dtype=dtype)
                res_var = (total_var / max(d, 1)).clamp_min(self.variance_floor)
            else:
                try:
                    _, s, vh = torch.linalg.svd(centered, full_matrices=False)
                    active_basis = vh[:q].t().contiguous()
                    active_eig = (s[:q] ** 2) / max(n - 1, 1)
                except RuntimeError:
                    cov = centered.t().mm(centered) / max(n - 1, 1)
                    evals, evecs = torch.linalg.eigh(cov)
                    idx = torch.argsort(evals, descending=True)[:q]
                    active_basis = evecs[:, idx]
                    active_eig = evals[idx]

                active_basis = _orthonormalize_columns(active_basis)
                active_eig = self._shrink_eigvals(active_eig, total_var, d)

                proj = centered.mm(active_basis)
                recon = proj.mm(active_basis.t())
                residual = centered - recon
                residual_energy = residual.pow(2).sum(dim=1).mean()
                res_var = (residual_energy / max(d - q, 1)).clamp_min(
                    self._robust_variance_floor(total_var, d)
                )

            basis = _complete_orthonormal_basis(active_basis, self.rank)
            eigvals = torch.full((self.rank,), self.variance_floor, device=device, dtype=dtype)
            if q > 0:
                eigvals[:q] = active_eig[:q]
            if q < self.rank:
                eigvals[q:] = res_var

            spectral_proto = None
            if spectral_summary is not None and spectral_summary.numel() > 0:
                spectral_proto = spectral_summary[mask].mean(dim=0)

            band_importance = None
            if band_weights is not None and band_weights.numel() > 0:
                band_importance = band_weights[mask].mean(dim=0)
                band_importance = band_importance / band_importance.sum().clamp_min(1e-8)

            sample_rel = n / float(n + 5)
            rank_rel = q / float(max(self.rank, 1))
            reliability = torch.tensor(
                min(1.0, max(self.min_reliability, 0.7 * sample_rel + 0.3 * rank_rel)),
                device=device,
                dtype=dtype,
            )

            geo[cid] = {
                "mean": mean.detach(),
                "basis": basis.detach(),
                "eigvals": eigvals.detach(),
                "res_var": res_var.detach(),
                "resvar": res_var.detach(),
                "spectral_proto": spectral_proto.detach() if spectral_proto is not None else None,
                "band_importance": band_importance.detach() if band_importance is not None else None,
                "active_rank": torch.tensor(q, device=device, dtype=torch.long),
                "reliability": reliability.detach(),
            }

        return geo

    # ============================================================
    # Getters
    # ============================================================
    def get_means(self):
        return self.means

    def get_bases(self):
        return self.bases

    def get_eigvals(self):
        return self.eigvals

    def get_res_vars(self):
        return self.res_vars

    def get_active_ranks(self):
        if self.active_ranks.numel() == 0:
            return torch.empty((0,), device=self.device, dtype=torch.long)
        return self.active_ranks

    def get_reliability(self):
        if self.reliability.numel() == 0:
            return torch.empty((0,), device=self.device, dtype=self.means.dtype)
        return self.reliability

    def get_variances(self):
        if self.eigvals.numel() == 0:
            return torch.empty((0, self.rank + 1), device=self.device, dtype=self.means.dtype)
        return torch.cat([self.eigvals, self.res_vars.unsqueeze(-1)], dim=-1)

    def get_projectors(self):
        if self.bases.numel() == 0:
            return torch.empty((0, self.d_model, self.d_model), device=self.device, dtype=self.means.dtype)
        return torch.einsum("cdr,cer->cde", self.bases, self.bases)

    def get_spectral_protos(self):
        return self.spectral_protos

    def get_band_importances(self):
        return self.band_importances

    def get_bank(self) -> Dict[str, torch.Tensor]:
        spectral_protos = self.get_spectral_protos()
        band_importances = self.get_band_importances()

        return {
            "means": self.get_means(),
            "bases": self.get_bases(),
            "eigvals": self.get_eigvals(),
            "res_vars": self.get_res_vars(),
            "resvars": self.get_res_vars(),
            "variances": self.get_variances(),
            "active_ranks": self.get_active_ranks(),
            "reliability": self.get_reliability(),
            "sample_counts": self.sample_counts,
            "geometry_volumes": self.geometry_volumes,
            "class_dispersions": self.class_dispersions,
            "class_risk": self.class_risk,
            "spectral_protos": spectral_protos,
            "band_importances": band_importances,
            # aliases for downstream compatibility
            "spectral_prototypes": spectral_protos,
            "band_importance": band_importances,
            "inter_center_dist": self.inter_center_dist,
            "inter_subspace_overlap": self.inter_subspace_overlap,
            "inter_spectral_dist": self.inter_spectral_dist,
            "inter_adjacency": self.inter_adjacency,
        }

    # ============================================================
    # Snapshot IO
    # ============================================================
    @torch.no_grad()
    def export_snapshot(self):
        bank = self.get_bank()
        snap = {}
        for k, v in bank.items():
            snap[k] = v.detach().clone() if torch.is_tensor(v) else v
        snap["spectral_dim"] = int(self._spectral_dim.item())
        snap["variance_floor"] = float(self.variance_floor)
        return snap

    @torch.no_grad()
    def load_snapshot(self, snapshot: Dict[str, torch.Tensor], strict: bool = True):
        if snapshot is None:
            if strict:
                raise ValueError("snapshot is None")
            return

        means = snapshot.get("means", None)
        bases = snapshot.get("bases", None)
        eigvals = snapshot.get("eigvals", None)
        res_vars = snapshot.get("res_vars", snapshot.get("resvars", None))
        variances = snapshot.get("variances", None)

        if variances is not None and (eigvals is None or res_vars is None):
            eigvals = variances[:, :-1]
            res_vars = variances[:, -1]

        if any(v is None for v in [means, bases, eigvals, res_vars]):
            if strict:
                raise ValueError("snapshot missing required geometry keys")
            return

        means = torch.as_tensor(means, device=self.device, dtype=self._dtype())
        bases = torch.as_tensor(bases, device=self.device, dtype=means.dtype)
        eigvals = torch.as_tensor(eigvals, device=self.device, dtype=means.dtype)
        res_vars = torch.as_tensor(res_vars, device=self.device, dtype=means.dtype)

        if means.dim() != 2 or means.size(1) != self.d_model:
            raise ValueError(f"snapshot means shape invalid: {tuple(means.shape)}")
        if bases.dim() != 3 or bases.size(1) != self.d_model:
            raise ValueError(f"snapshot bases shape invalid: {tuple(bases.shape)}")

        C = means.size(0)
        self.means = torch.nan_to_num(means, nan=0.0, posinf=0.0, neginf=0.0)

        fixed_bases = []
        for c in range(C):
            fixed_bases.append(self._sanitize_basis(bases[c]))
        self.bases = torch.stack(fixed_bases, dim=0) if C > 0 else torch.empty((0, self.d_model, self.rank), device=self.device, dtype=means.dtype)

        fixed_eig = []
        for c in range(C):
            rv = res_vars[c] if res_vars.numel() > c else torch.tensor(self.variance_floor, device=self.device, dtype=means.dtype)
            fixed_eig.append(self._prepare_eigvals(eigvals[c], fallback_resvar=rv))
        self.eigvals = torch.stack(fixed_eig, dim=0) if C > 0 else torch.empty((0, self.rank), device=self.device, dtype=means.dtype)
        self.res_vars = torch.nan_to_num(res_vars.flatten(), nan=self.variance_floor, posinf=self.variance_floor, neginf=self.variance_floor).clamp_min(self.variance_floor)

        active_ranks = snapshot.get("active_ranks", None)
        if active_ranks is None:
            active_ranks = torch.full((C,), self.rank, device=self.device, dtype=torch.long)
        else:
            active_ranks = torch.as_tensor(active_ranks, device=self.device, dtype=torch.long).flatten()
            if active_ranks.numel() < C:
                pad = torch.full((C - active_ranks.numel(),), self.rank, device=self.device, dtype=torch.long)
                active_ranks = torch.cat([active_ranks, pad], dim=0)
            active_ranks = active_ranks[:C]
        self.active_ranks = active_ranks.clamp(min=0, max=self.rank)

        reliability = snapshot.get("reliability", None)
        if reliability is None:
            reliability = torch.ones((C,), device=self.device, dtype=means.dtype)
        else:
            reliability = torch.as_tensor(reliability, device=self.device, dtype=means.dtype).flatten()
            if reliability.numel() < C:
                pad = torch.ones((C - reliability.numel(),), device=self.device, dtype=means.dtype)
                reliability = torch.cat([reliability, pad], dim=0)
            reliability = reliability[:C]
        self.reliability = reliability.clamp(self.min_reliability, 1.0)

        def _load_vector(name, default_value=0.0):
            value = snapshot.get(name, None)
            if value is None:
                return torch.full((C,), float(default_value), device=self.device, dtype=means.dtype)
            value = torch.as_tensor(value, device=self.device, dtype=means.dtype).flatten()
            if value.numel() < C:
                pad = torch.full((C - value.numel(),), float(default_value), device=self.device, dtype=means.dtype)
                value = torch.cat([value, pad], dim=0)
            return value[:C]

        self.sample_counts = _load_vector("sample_counts", 0.0)
        self.geometry_volumes = _load_vector("geometry_volumes", 0.0)
        self.class_dispersions = _load_vector("class_dispersions", 0.0)
        self.class_risk = _load_vector("class_risk", 0.0)

        spectral_protos = snapshot.get("spectral_prototypes", snapshot.get("spectral_protos", None))
        band_importances = snapshot.get("band_importance", snapshot.get("band_importances", None))

        if spectral_protos is not None and band_importances is not None:
            spectral_protos = torch.as_tensor(spectral_protos, device=self.device, dtype=means.dtype)
            band_importances = torch.as_tensor(band_importances, device=self.device, dtype=means.dtype)

            if spectral_protos.dim() == 2 and band_importances.dim() == 2:
                self.spectral_protos = spectral_protos
                self.band_importances = band_importances
                self._spectral_dim = torch.tensor(
                    spectral_protos.size(1),
                    device=self.device,
                    dtype=torch.long,
                )
            elif strict:
                raise ValueError("snapshot spectral metadata has invalid shapes")
        else:
            self.spectral_protos = torch.empty((C, 0), device=self.device, dtype=means.dtype)
            self.band_importances = torch.empty((C, 0), device=self.device, dtype=means.dtype)
            self._spectral_dim = torch.tensor(0, device=self.device, dtype=torch.long)

        self._resize_inter_class_buffers(C, means.dtype)
        self.refresh_inter_class_geometry()

    # ============================================================
    # Add / update class
    # ============================================================
    @torch.no_grad()
    def add_class(
        self,
        mean,
        basis,
        eigvals,
        res_var,
        spectral_proto=None,
        band_importance=None,
        active_rank=None,
        reliability=None,
        sample_count=None,
    ):
        mean = self._prepare_mean(mean)
        res_var = self._prepare_res_var(res_var)
        basis = self._sanitize_basis(basis)
        eigvals = self._prepare_eigvals(eigvals, fallback_resvar=res_var)
        spectral_proto, band_importance = self._prepare_spectral_pair(
            spectral_proto=spectral_proto,
            band_importance=band_importance,
            dtype=mean.dtype,
        )

        if active_rank is None:
            active_rank_t = torch.tensor(self.rank, device=self.device, dtype=torch.long)
        else:
            active_rank_t = torch.as_tensor(active_rank, device=self.device, dtype=torch.long).reshape(())
            active_rank_t = active_rank_t.clamp(min=0, max=self.rank)

        if reliability is None:
            reliability_t = torch.tensor(1.0, device=self.device, dtype=mean.dtype)
        else:
            reliability_t = torch.as_tensor(reliability, device=self.device, dtype=mean.dtype).reshape(()).clamp(self.min_reliability, 1.0)

        self.means = torch.cat([self.means, mean.unsqueeze(0)], dim=0)
        self.bases = torch.cat([self.bases, basis.unsqueeze(0)], dim=0)
        self.eigvals = torch.cat([self.eigvals, eigvals.unsqueeze(0)], dim=0)
        self.res_vars = torch.cat([self.res_vars, res_var.view(1)], dim=0)
        self.active_ranks = torch.cat([self.active_ranks, active_rank_t.view(1)], dim=0)
        self.reliability = torch.cat([self.reliability, reliability_t.view(1)], dim=0)

        sample_count_t = torch.as_tensor(
            0.0 if sample_count is None else sample_count,
            device=self.device,
            dtype=mean.dtype,
        ).reshape(())
        volume_t, dispersion_t = self._geometry_volume_and_dispersion(eigvals, res_var, active_rank_t)
        risk_t = (1.0 - reliability_t).clamp(0.0, 1.0)

        self.sample_counts = torch.cat([self.sample_counts, sample_count_t.view(1)], dim=0)
        self.geometry_volumes = torch.cat([self.geometry_volumes, volume_t.view(1)], dim=0)
        self.class_dispersions = torch.cat([self.class_dispersions, dispersion_t.view(1)], dim=0)
        self.class_risk = torch.cat([self.class_risk, risk_t.view(1)], dim=0)

        if int(self._spectral_dim.item()) > 0:
            self.spectral_protos = torch.cat([self.spectral_protos, spectral_proto.unsqueeze(0)], dim=0)
            self.band_importances = torch.cat([self.band_importances, band_importance.unsqueeze(0)], dim=0)
        else:
            self.spectral_protos = torch.empty((len(self), 0), device=self.device, dtype=mean.dtype)
            self.band_importances = torch.empty((len(self), 0), device=self.device, dtype=mean.dtype)

        self._resize_inter_class_buffers(len(self), mean.dtype)
        self.refresh_inter_class_geometry()

    @torch.no_grad()
    def update_class(
        self,
        cls_id,
        mean,
        basis,
        eigvals,
        res_var,
        spectral_proto=None,
        band_importance=None,
        active_rank=None,
        reliability=None,
        sample_count=None,
    ):
        cls_id = int(cls_id)
        if cls_id < 0 or cls_id >= len(self):
            raise IndexError(f"class index {cls_id} out of range for GeometryBank of size {len(self)}")

        mean = self._prepare_mean(mean)
        res_var = self._prepare_res_var(res_var)
        basis = self._sanitize_basis(basis)
        eigvals = self._prepare_eigvals(eigvals, fallback_resvar=res_var)
        spectral_proto, band_importance = self._prepare_spectral_pair(
            spectral_proto=spectral_proto,
            band_importance=band_importance,
            dtype=mean.dtype,
        )

        self.means[cls_id] = mean
        self.bases[cls_id] = basis
        self.eigvals[cls_id] = eigvals
        self.res_vars[cls_id] = res_var

        if active_rank is not None:
            self.active_ranks[cls_id] = torch.as_tensor(active_rank, device=self.device, dtype=torch.long).reshape(()).clamp(0, self.rank)

        if reliability is not None:
            self.reliability[cls_id] = torch.as_tensor(reliability, device=self.device, dtype=mean.dtype).reshape(()).clamp(self.min_reliability, 1.0)

        if sample_count is not None and self.sample_counts.numel() > cls_id:
            self.sample_counts[cls_id] = torch.as_tensor(sample_count, device=self.device, dtype=mean.dtype).reshape(())

        if self.geometry_volumes.numel() > cls_id and self.class_dispersions.numel() > cls_id:
            volume_t, dispersion_t = self._geometry_volume_and_dispersion(
                self.eigvals[cls_id],
                self.res_vars[cls_id],
                self.active_ranks[cls_id],
            )
            self.geometry_volumes[cls_id] = volume_t
            self.class_dispersions[cls_id] = dispersion_t

        if self.class_risk.numel() > cls_id and self.reliability.numel() > cls_id:
            self.class_risk[cls_id] = (1.0 - self.reliability[cls_id]).clamp(0.0, 1.0)

        if int(self._spectral_dim.item()) > 0:
            self.spectral_protos[cls_id] = spectral_proto
            self.band_importances[cls_id] = band_importance

        self.refresh_inter_class_geometry()

    @torch.no_grad()
    def update_class_geometry(
        self,
        class_id: int,
        mean: torch.Tensor,
        basis: torch.Tensor,
        eigvals: torch.Tensor,
        resvar: torch.Tensor,
        spectral_proto: Optional[torch.Tensor] = None,
        band_importance: Optional[torch.Tensor] = None,
        reliability: Optional[torch.Tensor] = None,
        active_rank: Optional[torch.Tensor] = None,
        sample_count: Optional[torch.Tensor] = None,
        ema: float = 0.0,
    ) -> None:
        """
        Compatibility method for newer trainer/model code.
        Supports optional EMA update of existing geometry.
        """
        class_id = int(class_id)
        self.ensure_class_count(class_id + 1)

        if float(ema) > 0.0 and class_id < len(self) and self.reliability[class_id] > 0:
            ema = float(max(0.0, min(ema, 0.999)))
            old = self.get_bank()
            mean = ema * old["means"][class_id] + (1.0 - ema) * mean.to(self.device)
            eigvals = ema * old["eigvals"][class_id] + (1.0 - ema) * eigvals.to(self.device)
            resvar = ema * old["res_vars"][class_id] + (1.0 - ema) * torch.as_tensor(resvar, device=self.device)
            # Basis is not EMA averaged directly because basis columns have sign/rotation ambiguity.
            # Use the fresh sanitized basis.

        self.update_class(
            cls_id=class_id,
            mean=mean,
            basis=basis,
            eigvals=eigvals,
            res_var=resvar,
            spectral_proto=spectral_proto,
            band_importance=band_importance,
            active_rank=active_rank,
            reliability=reliability,
            sample_count=sample_count,
        )

    @torch.no_grad()
    def ensure_class_count(self, count: int, spectral_dim: int = 0, dtype=torch.float32):
        count = int(count)
        if spectral_dim > 0:
            self._ensure_spectral_dim(int(spectral_dim), dtype=dtype)

        while len(self) < count:
            dummy_mean = torch.zeros(self.d_model, device=self.device, dtype=dtype)
            eye_basis = torch.eye(self.d_model, self.rank, device=self.device, dtype=dtype)
            dummy_eigvals = torch.full((self.rank,), self.variance_floor, device=self.device, dtype=dtype)
            dummy_res_var = torch.tensor(self.variance_floor, device=self.device, dtype=dtype)

            if int(self._spectral_dim.item()) > 0:
                sdim = int(self._spectral_dim.item())
                dummy_spec = torch.zeros((sdim,), device=self.device, dtype=dtype)
                dummy_bw = torch.full((sdim,), 1.0 / float(sdim), device=self.device, dtype=dtype)
            else:
                dummy_spec = torch.empty((0,), device=self.device, dtype=dtype)
                dummy_bw = torch.empty((0,), device=self.device, dtype=dtype)

            self.add_class(
                dummy_mean,
                eye_basis,
                dummy_eigvals,
                dummy_res_var,
                spectral_proto=dummy_spec,
                band_importance=dummy_bw,
                active_rank=torch.tensor(0, device=self.device, dtype=torch.long),
                reliability=torch.tensor(self.min_reliability, device=self.device, dtype=dtype),
                sample_count=torch.tensor(0.0, device=self.device, dtype=dtype),
            )


    @torch.no_grad()
    def ensure_num_classes(self, count: int):
        """Legacy alias used by older NECILModel versions."""
        self.ensure_class_count(count)

    # ============================================================
    # Inter-class geometry
    # ============================================================
    @torch.no_grad()
    def refresh_inter_class_geometry(self):
        C = len(self)
        if C == 0:
            return

        dtype = self.means.dtype
        self._resize_inter_class_buffers(C, dtype)

        means = self.means
        diff = means.unsqueeze(1) - means.unsqueeze(0)
        self.inter_center_dist = diff.pow(2).sum(dim=-1).sqrt()

        if self.bases.numel() > 0:
            # Frobenius norm of U_i^T U_j, normalized by rank.
            overlap = torch.einsum("idr,jds->ijrs", self.bases, self.bases).pow(2).sum(dim=(-1, -2))
            self.inter_subspace_overlap = overlap / max(float(self.rank), 1.0)
        else:
            self.inter_subspace_overlap = torch.zeros(C, C, device=self.device, dtype=dtype)

        if self.spectral_protos.numel() > 0 and self.spectral_protos.size(0) == C and self.spectral_protos.size(1) > 0:
            sd = self.spectral_protos.unsqueeze(1) - self.spectral_protos.unsqueeze(0)
            self.inter_spectral_dist = sd.pow(2).sum(dim=-1).sqrt()
        else:
            self.inter_spectral_dist = torch.zeros(C, C, device=self.device, dtype=dtype)

        # High adjacency = likely confusable. Use center closeness + subspace overlap.
        center_scale = self.inter_center_dist.detach()
        nonzero = center_scale[center_scale > 0]
        if nonzero.numel() > 0:
            scale = nonzero.median().clamp_min(1e-6)
        else:
            scale = torch.tensor(1.0, device=self.device, dtype=dtype)

        center_aff = torch.exp(-self.inter_center_dist / (scale * max(self.adjacency_temperature, 1e-6)))
        sub_aff = self.inter_subspace_overlap.clamp(0.0, 1.0)

        adjacency = 0.5 * center_aff + 0.5 * sub_aff
        eye = torch.eye(C, device=self.device, dtype=torch.bool)
        adjacency = adjacency.masked_fill(eye, 0.0)
        self.inter_adjacency = adjacency

    # ============================================================
    # Retrieval helpers
    # ============================================================
    def _similarity_weights(self, feat: torch.Tensor):
        means = self.get_means()
        if means.numel() == 0:
            return None, None

        feat_n = F.normalize(feat, dim=-1, eps=1e-6)
        means_n = F.normalize(means.to(feat.device, feat.dtype), dim=-1, eps=1e-6)
        sim = torch.matmul(feat_n, means_n.t())

        reliability = self.get_reliability()
        if reliability is not None and reliability.numel() == sim.size(1):
            rel = reliability.to(feat.device, feat.dtype).clamp(0.05, 1.0)
            sim = sim + torch.log(rel.unsqueeze(0))

        weights = torch.softmax(sim, dim=-1)
        return means, weights

    def retrieve_geometry_ref(self, feat: torch.Tensor) -> torch.Tensor:
        means, weights = self._similarity_weights(feat)
        if means is None:
            return torch.zeros(feat.size(0), self.d_model, device=feat.device, dtype=feat.dtype)
        return torch.matmul(weights, means.to(feat.device, feat.dtype))

    def retrieve_spectral_ref(self, feat: torch.Tensor) -> torch.Tensor:
        means, weights = self._similarity_weights(feat)
        spectral_protos = self.get_spectral_protos()

        if means is None or spectral_protos.numel() == 0:
            return torch.empty((feat.size(0), 0), device=feat.device, dtype=feat.dtype)

        spectral_protos = spectral_protos.to(feat.device, feat.dtype)
        return torch.matmul(weights, spectral_protos)

    def retrieve_band_importance_ref(self, feat: torch.Tensor) -> torch.Tensor:
        means, weights = self._similarity_weights(feat)
        band_importances = self.get_band_importances()

        if means is None or band_importances.numel() == 0:
            return torch.empty((feat.size(0), 0), device=feat.device, dtype=feat.dtype)

        band_importances = band_importances.to(feat.device, feat.dtype)
        return torch.matmul(weights, band_importances)




# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Dict, Optional, Tuple


# # ============================================================
# # Standalone helpers
# # ============================================================
# def _orthonormalize_columns(basis: torch.Tensor) -> torch.Tensor:
#     """
#     Orthonormalize basis columns.
#     Accepts [D,R] or [C,D,R]; returns same shape with orthonormal columns.
#     """
#     if basis is None or basis.numel() == 0:
#         return basis

#     if basis.dim() == 2:
#         q, _ = torch.linalg.qr(basis, mode="reduced")
#         if q.size(1) < basis.size(1):
#             pad = torch.zeros(
#                 q.size(0),
#                 basis.size(1) - q.size(1),
#                 device=q.device,
#                 dtype=q.dtype,
#             )
#             q = torch.cat([q, pad], dim=1)
#         return q[:, : basis.size(1)]

#     if basis.dim() == 3:
#         return torch.stack([_orthonormalize_columns(b) for b in basis], dim=0)

#     raise ValueError(f"basis must be [D,R] or [C,D,R], got {tuple(basis.shape)}")


# def _complete_orthonormal_basis(active_basis: torch.Tensor, rank: int) -> torch.Tensor:
#     """
#     Return [D, rank] orthonormal-ish basis. If active_basis has fewer columns,
#     fill the remaining columns with deterministic orthogonal complement vectors.

#     This avoids zero basis columns, which silently create degenerate geometry.
#     """
#     if active_basis.dim() != 2:
#         raise ValueError(f"active_basis must be [D,Q], got {tuple(active_basis.shape)}")

#     d = int(active_basis.size(0))
#     q = int(active_basis.size(1))
#     device, dtype = active_basis.device, active_basis.dtype
#     rank = min(int(rank), d)

#     if q > 0:
#         basis = _orthonormalize_columns(active_basis[:, :q])
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

#     # Fallback should almost never run, but avoids hard failure under numerical degeneracy.
#     while len(cols) < rank:
#         v = torch.randn(d, device=device, dtype=dtype)
#         for u in cols:
#             v = v - torch.dot(v, u) * u
#         n = v.norm().clamp_min(1e-6)
#         cols.append(v / n)

#     return torch.stack(cols[:rank], dim=1)


# class GeometryBank(nn.Module):
#     """
#     Reliability-aware Spectral-Guided Low-Rank Geometry Bank for NECIL-HSI.

#     Per class c, stores:
#         mean:              [D]
#         basis:             [D, R]
#         eigvals:           [R]
#         res_var:           scalar residual variance
#         active_rank:       reliable PCA rank, <= n - 1
#         reliability:       scalar memory reliability in [0, 1]
#         spectral_proto:    [S] optional spectral prototype
#         band_importance:   [S] optional band-importance distribution

#     Why this version is safer:
#         - does not invent fake PCA rank for low-sample classes
#         - avoids zero basis columns by completing an orthonormal complement
#         - applies variance shrinkage and robust floors
#         - stores active rank and reliability for safe replay/token preservation
#         - preserves the existing public API used by NECILModel and TrainerHelper
#     """

#     def __init__(
#         self,
#         d_model: int,
#         rank: int,
#         device: str = "cpu",
#         variance_floor: float = 1e-4,
#         variance_shrinkage: float = 0.10,
#         max_variance_ratio: float = 50.0,
#         adjacency_temperature: float = 1.0,
#     ):
#         super().__init__()
#         self.d_model = int(d_model)
#         self.rank = min(int(rank), self.d_model)

#         self.variance_floor = float(variance_floor)
#         self.variance_shrinkage = float(variance_shrinkage)
#         self.max_variance_ratio = float(max_variance_ratio)
#         self.adjacency_temperature = float(adjacency_temperature)

#         dev = torch.device(device)

#         self.register_buffer("means", torch.empty((0, self.d_model), device=dev))
#         self.register_buffer("bases", torch.empty((0, self.d_model, self.rank), device=dev))
#         self.register_buffer("eigvals", torch.empty((0, self.rank), device=dev))
#         self.register_buffer("res_vars", torch.empty((0,), device=dev))

#         # Reliability-aware geometry metadata.
#         self.register_buffer("active_ranks", torch.empty((0,), dtype=torch.long, device=dev))
#         self.register_buffer("reliability", torch.empty((0,), device=dev))

#         # Spectral metadata may be unknown initially; allow dynamic expansion.
#         self.register_buffer("spectral_protos", torch.empty((0, 0), device=dev))
#         self.register_buffer("band_importances", torch.empty((0, 0), device=dev))
#         self.register_buffer("_spectral_dim", torch.tensor(0, dtype=torch.long, device=dev))

#         # Inter-class geometry diagnostics / optional regularization support.
#         self.register_buffer("inter_center_dist", torch.empty((0, 0), device=dev))
#         self.register_buffer("inter_subspace_overlap", torch.empty((0, 0), device=dev))
#         self.register_buffer("inter_spectral_dist", torch.empty((0, 0), device=dev))
#         self.register_buffer("inter_adjacency", torch.empty((0, 0), device=dev))

#     # ============================================================
#     # Compatibility properties
#     # ============================================================
#     @property
#     def device(self) -> torch.device:
#         return self.means.device

#     def __len__(self) -> int:
#         return int(self.means.size(0))

#     @property
#     def resvars(self) -> torch.Tensor:
#         return self.res_vars

#     @resvars.setter
#     def resvars(self, value: torch.Tensor) -> None:
#         self.res_vars = value

#     # ============================================================
#     # Internal helpers
#     # ============================================================
#     def _dtype(self) -> torch.dtype:
#         return self.means.dtype if self.means.numel() > 0 else torch.float32

#     def _default_spectral_proto(self, spectral_dim: int, dtype: torch.dtype):
#         if spectral_dim <= 0:
#             return torch.empty((0,), device=self.device, dtype=dtype)
#         return torch.zeros((spectral_dim,), device=self.device, dtype=dtype)

#     def _default_band_importance(self, spectral_dim: int, dtype: torch.dtype):
#         if spectral_dim <= 0:
#             return torch.empty((0,), device=self.device, dtype=dtype)
#         return torch.full(
#             (spectral_dim,),
#             1.0 / float(spectral_dim),
#             device=self.device,
#             dtype=dtype,
#         )

#     def _robust_variance_floor(self, total_var: torch.Tensor, d: int) -> torch.Tensor:
#         data_floor = (total_var / max(d, 1)) * 1e-3
#         return torch.maximum(
#             torch.tensor(self.variance_floor, device=total_var.device, dtype=total_var.dtype),
#             data_floor,
#         )

#     def _shrink_eigvals(self, eig: torch.Tensor, total_var: torch.Tensor, d: int) -> torch.Tensor:
#         if eig.numel() == 0:
#             return eig

#         avg_var = (total_var / max(d, 1)).clamp_min(self.variance_floor)
#         shrink = float(max(0.0, min(self.variance_shrinkage, 1.0)))
#         eig = (1.0 - shrink) * eig + shrink * avg_var

#         floor = self._robust_variance_floor(total_var, d)
#         ceil = floor * self.max_variance_ratio
#         return eig.clamp(min=float(floor.item()), max=float(ceil.item()))

#     def _infer_spectral_dim(
#         self,
#         spectral_proto: Optional[torch.Tensor],
#         band_importance: Optional[torch.Tensor],
#     ) -> int:
#         dims = []
#         current = int(self._spectral_dim.item())
#         if current > 0:
#             dims.append(current)

#         if spectral_proto is not None:
#             spectral_proto = torch.as_tensor(spectral_proto)
#             if spectral_proto.numel() > 0:
#                 dims.append(int(spectral_proto.numel()))

#         if band_importance is not None:
#             band_importance = torch.as_tensor(band_importance)
#             if band_importance.numel() > 0:
#                 dims.append(int(band_importance.numel()))

#         if len(dims) == 0:
#             return current

#         if len(set(dims)) != 1:
#             raise ValueError(f"Inconsistent spectral dimensions detected: {dims}")

#         return dims[0]

#     def _ensure_spectral_dim(self, spectral_dim: int, dtype: torch.dtype):
#         spectral_dim = int(spectral_dim)
#         current = int(self._spectral_dim.item())

#         if spectral_dim == current:
#             return
#         if current > 0 and spectral_dim == 0:
#             return
#         if current > 0 and spectral_dim > 0 and current != spectral_dim:
#             raise ValueError(
#                 f"GeometryBank spectral dim mismatch: existing {current}, requested {spectral_dim}"
#             )

#         if current == 0 and spectral_dim > 0:
#             n = len(self)
#             if n == 0:
#                 self.spectral_protos = torch.empty((0, spectral_dim), device=self.device, dtype=dtype)
#                 self.band_importances = torch.empty((0, spectral_dim), device=self.device, dtype=dtype)
#             else:
#                 self.spectral_protos = torch.zeros((n, spectral_dim), device=self.device, dtype=dtype)
#                 self.band_importances = torch.full(
#                     (n, spectral_dim),
#                     1.0 / float(spectral_dim),
#                     device=self.device,
#                     dtype=dtype,
#                 )
#             self._spectral_dim = torch.tensor(spectral_dim, device=self.device, dtype=torch.long)

#     def _prepare_mean(self, mean) -> torch.Tensor:
#         mean = torch.as_tensor(mean, device=self.device, dtype=self._dtype()).flatten()
#         if mean.numel() != self.d_model:
#             raise ValueError(f"mean must have {self.d_model} values, got {mean.numel()}")
#         return torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)

#     def _sanitize_basis(self, basis: torch.Tensor) -> torch.Tensor:
#         """
#         Enforce shape [D, rank] and return non-degenerate orthonormal columns.
#         """
#         basis = torch.as_tensor(basis, device=self.device, dtype=self._dtype())

#         if basis.dim() != 2:
#             raise ValueError(f"basis must be 2D, got {tuple(basis.shape)}")

#         # Allow accidental transpose.
#         if basis.shape[0] == self.rank and basis.shape[1] == self.d_model:
#             basis = basis.transpose(0, 1)

#         if basis.shape[0] != self.d_model:
#             raise ValueError(
#                 f"basis first dim must equal d_model={self.d_model}, got {tuple(basis.shape)}"
#             )

#         if basis.shape[1] > self.rank:
#             basis = basis[:, : self.rank]

#         col_norms = basis.norm(dim=0) if basis.numel() > 0 else torch.empty(0, device=self.device)
#         active = col_norms > 1e-8
#         active_basis = basis[:, active] if active.numel() > 0 and active.any() else torch.zeros(
#             self.d_model, 0, device=self.device, dtype=basis.dtype
#         )

#         return _complete_orthonormal_basis(active_basis, self.rank)

#     def _prepare_eigvals(self, eigvals, fallback_resvar: Optional[torch.Tensor] = None) -> torch.Tensor:
#         eigvals = torch.as_tensor(eigvals, device=self.device, dtype=self._dtype()).flatten()
#         floor = float(self.variance_floor)

#         if eigvals.numel() > self.rank:
#             eigvals = eigvals[: self.rank]
#         elif eigvals.numel() < self.rank:
#             fill = floor
#             if fallback_resvar is not None:
#                 fill = float(torch.as_tensor(fallback_resvar).detach().clamp_min(floor).item())
#             pad = torch.full(
#                 (self.rank - eigvals.numel(),),
#                 fill,
#                 device=self.device,
#                 dtype=eigvals.dtype,
#             )
#             eigvals = torch.cat([eigvals, pad], dim=0)

#         return torch.nan_to_num(eigvals, nan=floor, posinf=floor, neginf=floor).clamp_min(floor)

#     def _prepare_res_var(self, res_var) -> torch.Tensor:
#         res_var = torch.as_tensor(res_var, device=self.device, dtype=self._dtype()).reshape(())
#         return torch.nan_to_num(res_var, nan=self.variance_floor, posinf=self.variance_floor, neginf=self.variance_floor).clamp_min(self.variance_floor)

#     def _prepare_spectral_pair(
#         self,
#         spectral_proto=None,
#         band_importance=None,
#         dtype=torch.float32,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         spectral_dim = self._infer_spectral_dim(spectral_proto, band_importance)
#         self._ensure_spectral_dim(spectral_dim, dtype=dtype)

#         if spectral_dim <= 0:
#             return (
#                 torch.empty((0,), device=self.device, dtype=dtype),
#                 torch.empty((0,), device=self.device, dtype=dtype),
#             )

#         if spectral_proto is None or torch.as_tensor(spectral_proto).numel() == 0:
#             spectral_proto = self._default_spectral_proto(spectral_dim, dtype=dtype)
#         else:
#             spectral_proto = torch.as_tensor(spectral_proto, device=self.device, dtype=dtype).flatten()
#             if spectral_proto.numel() != spectral_dim:
#                 raise ValueError(
#                     f"spectral_proto dim mismatch: expected {spectral_dim}, got {spectral_proto.numel()}"
#                 )
#             spectral_proto = torch.nan_to_num(spectral_proto, nan=0.0, posinf=0.0, neginf=0.0)

#         if band_importance is None or torch.as_tensor(band_importance).numel() == 0:
#             band_importance = self._default_band_importance(spectral_dim, dtype=dtype)
#         else:
#             band_importance = torch.as_tensor(band_importance, device=self.device, dtype=dtype).flatten()
#             if band_importance.numel() != spectral_dim:
#                 raise ValueError(
#                     f"band_importance dim mismatch: expected {spectral_dim}, got {band_importance.numel()}"
#                 )
#             band_importance = torch.nan_to_num(band_importance, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
#             s = band_importance.sum()
#             if not torch.isfinite(s) or s.abs() <= 0:
#                 band_importance = self._default_band_importance(spectral_dim, dtype=dtype)
#             else:
#                 band_importance = band_importance / s.clamp_min(1e-8)

#         return spectral_proto, band_importance

#     def _resize_inter_class_buffers(self, n: int, dtype: torch.dtype):
#         z = torch.zeros(n, n, device=self.device, dtype=dtype)

#         def copy_old(old_tensor: torch.Tensor) -> torch.Tensor:
#             if old_tensor is not None and old_tensor.numel() > 0:
#                 old = min(old_tensor.size(0), n)
#                 z_new = z.clone()
#                 z_new[:old, :old] = old_tensor[:old, :old].to(device=self.device, dtype=dtype)
#                 return z_new
#             return z.clone()

#         self.inter_center_dist = copy_old(self.inter_center_dist)
#         self.inter_subspace_overlap = copy_old(self.inter_subspace_overlap)
#         self.inter_spectral_dist = copy_old(self.inter_spectral_dist)
#         self.inter_adjacency = copy_old(self.inter_adjacency)

#     # ============================================================
#     # Geometry extraction from features
#     # ============================================================
#     @torch.no_grad()
#     def extract_geometry(
#         self,
#         feat: torch.Tensor,
#         labels: torch.Tensor,
#         spectral_summary: Optional[torch.Tensor] = None,
#         band_weights: Optional[torch.Tensor] = None,
#     ) -> Dict[int, Dict[str, torch.Tensor]]:
#         """
#         Extract robust low-rank class geometry from a feature batch.
#         This is optional but useful if trainer code wants the bank to compute geometry directly.
#         """
#         geo: Dict[int, Dict[str, torch.Tensor]] = {}
#         if feat is None or labels is None or feat.numel() == 0 or labels.numel() == 0:
#             return geo

#         feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
#         device, dtype = feat.device, feat.dtype

#         for cid_t in torch.unique(labels):
#             cid = int(cid_t.item())
#             mask = labels == cid_t
#             z = feat[mask]
#             if z.size(0) == 0:
#                 continue

#             n, d = z.shape
#             mean = z.mean(dim=0)
#             centered = z - mean.unsqueeze(0)
#             total_var = centered.pow(2).sum(dim=1).mean().clamp_min(self.variance_floor)

#             # Correct covariance rank: cannot exceed n - 1.
#             q = min(self.rank, max(n - 1, 0), d)

#             if q <= 0:
#                 active_basis = torch.zeros(d, 0, device=device, dtype=dtype)
#                 active_eig = torch.empty(0, device=device, dtype=dtype)
#                 res_var = (total_var / max(d, 1)).clamp_min(self.variance_floor)
#             else:
#                 try:
#                     _, s, vh = torch.linalg.svd(centered, full_matrices=False)
#                     active_basis = vh[:q].t().contiguous()
#                     active_eig = (s[:q] ** 2) / max(n - 1, 1)
#                 except RuntimeError:
#                     cov = centered.t().mm(centered) / max(n - 1, 1)
#                     evals, evecs = torch.linalg.eigh(cov)
#                     idx = torch.argsort(evals, descending=True)[:q]
#                     active_basis = evecs[:, idx]
#                     active_eig = evals[idx]

#                 active_basis = _orthonormalize_columns(active_basis)
#                 active_eig = self._shrink_eigvals(active_eig, total_var, d)

#                 proj = centered.mm(active_basis)
#                 recon = proj.mm(active_basis.t())
#                 residual = centered - recon
#                 residual_energy = residual.pow(2).sum(dim=1).mean()
#                 res_var = (residual_energy / max(d - q, 1)).clamp_min(
#                     self._robust_variance_floor(total_var, d)
#                 )

#             basis = _complete_orthonormal_basis(active_basis, self.rank)
#             eigvals = torch.full((self.rank,), self.variance_floor, device=device, dtype=dtype)
#             if q > 0:
#                 eigvals[:q] = active_eig[:q]
#             if q < self.rank:
#                 eigvals[q:] = res_var

#             spectral_proto = None
#             if spectral_summary is not None and spectral_summary.numel() > 0:
#                 spectral_proto = spectral_summary[mask].mean(dim=0)

#             band_importance = None
#             if band_weights is not None and band_weights.numel() > 0:
#                 band_importance = band_weights[mask].mean(dim=0)
#                 band_importance = band_importance / band_importance.sum().clamp_min(1e-8)

#             sample_rel = n / float(n + 5)
#             rank_rel = q / float(max(self.rank, 1))
#             reliability = torch.tensor(
#                 min(1.0, max(0.05, 0.7 * sample_rel + 0.3 * rank_rel)),
#                 device=device,
#                 dtype=dtype,
#             )

#             geo[cid] = {
#                 "mean": mean.detach(),
#                 "basis": basis.detach(),
#                 "eigvals": eigvals.detach(),
#                 "res_var": res_var.detach(),
#                 "resvar": res_var.detach(),
#                 "spectral_proto": spectral_proto.detach() if spectral_proto is not None else None,
#                 "band_importance": band_importance.detach() if band_importance is not None else None,
#                 "active_rank": torch.tensor(q, device=device, dtype=torch.long),
#                 "reliability": reliability.detach(),
#             }

#         return geo

#     # ============================================================
#     # Getters
#     # ============================================================
#     def get_means(self):
#         return self.means

#     def get_bases(self):
#         return self.bases

#     def get_eigvals(self):
#         return self.eigvals

#     def get_res_vars(self):
#         return self.res_vars

#     def get_active_ranks(self):
#         if self.active_ranks.numel() == 0:
#             return torch.empty((0,), device=self.device, dtype=torch.long)
#         return self.active_ranks

#     def get_reliability(self):
#         if self.reliability.numel() == 0:
#             return torch.empty((0,), device=self.device, dtype=self.means.dtype)
#         return self.reliability

#     def get_variances(self):
#         if self.eigvals.numel() == 0:
#             return torch.empty((0, self.rank + 1), device=self.device, dtype=self.means.dtype)
#         return torch.cat([self.eigvals, self.res_vars.unsqueeze(-1)], dim=-1)

#     def get_projectors(self):
#         if self.bases.numel() == 0:
#             return torch.empty((0, self.d_model, self.d_model), device=self.device, dtype=self.means.dtype)
#         return torch.einsum("cdr,cer->cde", self.bases, self.bases)

#     def get_spectral_protos(self):
#         return self.spectral_protos

#     def get_band_importances(self):
#         return self.band_importances

#     def get_bank(self) -> Dict[str, torch.Tensor]:
#         spectral_protos = self.get_spectral_protos()
#         band_importances = self.get_band_importances()

#         return {
#             "means": self.get_means(),
#             "bases": self.get_bases(),
#             "eigvals": self.get_eigvals(),
#             "res_vars": self.get_res_vars(),
#             "resvars": self.get_res_vars(),
#             "variances": self.get_variances(),
#             "active_ranks": self.get_active_ranks(),
#             "reliability": self.get_reliability(),
#             "spectral_protos": spectral_protos,
#             "band_importances": band_importances,
#             # aliases for downstream compatibility
#             "spectral_prototypes": spectral_protos,
#             "band_importance": band_importances,
#             "inter_center_dist": self.inter_center_dist,
#             "inter_subspace_overlap": self.inter_subspace_overlap,
#             "inter_spectral_dist": self.inter_spectral_dist,
#             "inter_adjacency": self.inter_adjacency,
#         }

#     # ============================================================
#     # Snapshot IO
#     # ============================================================
#     @torch.no_grad()
#     def export_snapshot(self):
#         bank = self.get_bank()
#         snap = {}
#         for k, v in bank.items():
#             snap[k] = v.detach().clone() if torch.is_tensor(v) else v
#         snap["spectral_dim"] = int(self._spectral_dim.item())
#         snap["variance_floor"] = float(self.variance_floor)
#         return snap

#     @torch.no_grad()
#     def load_snapshot(self, snapshot: Dict[str, torch.Tensor], strict: bool = True):
#         if snapshot is None:
#             if strict:
#                 raise ValueError("snapshot is None")
#             return

#         means = snapshot.get("means", None)
#         bases = snapshot.get("bases", None)
#         eigvals = snapshot.get("eigvals", None)
#         res_vars = snapshot.get("res_vars", snapshot.get("resvars", None))
#         variances = snapshot.get("variances", None)

#         if variances is not None and (eigvals is None or res_vars is None):
#             eigvals = variances[:, :-1]
#             res_vars = variances[:, -1]

#         if any(v is None for v in [means, bases, eigvals, res_vars]):
#             if strict:
#                 raise ValueError("snapshot missing required geometry keys")
#             return

#         means = torch.as_tensor(means, device=self.device, dtype=self._dtype())
#         bases = torch.as_tensor(bases, device=self.device, dtype=means.dtype)
#         eigvals = torch.as_tensor(eigvals, device=self.device, dtype=means.dtype)
#         res_vars = torch.as_tensor(res_vars, device=self.device, dtype=means.dtype)

#         if means.dim() != 2 or means.size(1) != self.d_model:
#             raise ValueError(f"snapshot means shape invalid: {tuple(means.shape)}")
#         if bases.dim() != 3 or bases.size(1) != self.d_model:
#             raise ValueError(f"snapshot bases shape invalid: {tuple(bases.shape)}")

#         C = means.size(0)
#         self.means = torch.nan_to_num(means, nan=0.0, posinf=0.0, neginf=0.0)

#         fixed_bases = []
#         for c in range(C):
#             fixed_bases.append(self._sanitize_basis(bases[c]))
#         self.bases = torch.stack(fixed_bases, dim=0) if C > 0 else torch.empty((0, self.d_model, self.rank), device=self.device, dtype=means.dtype)

#         fixed_eig = []
#         for c in range(C):
#             rv = res_vars[c] if res_vars.numel() > c else torch.tensor(self.variance_floor, device=self.device, dtype=means.dtype)
#             fixed_eig.append(self._prepare_eigvals(eigvals[c], fallback_resvar=rv))
#         self.eigvals = torch.stack(fixed_eig, dim=0) if C > 0 else torch.empty((0, self.rank), device=self.device, dtype=means.dtype)
#         self.res_vars = torch.nan_to_num(res_vars.flatten(), nan=self.variance_floor, posinf=self.variance_floor, neginf=self.variance_floor).clamp_min(self.variance_floor)

#         active_ranks = snapshot.get("active_ranks", None)
#         if active_ranks is None:
#             active_ranks = torch.full((C,), self.rank, device=self.device, dtype=torch.long)
#         else:
#             active_ranks = torch.as_tensor(active_ranks, device=self.device, dtype=torch.long).flatten()
#             if active_ranks.numel() < C:
#                 pad = torch.full((C - active_ranks.numel(),), self.rank, device=self.device, dtype=torch.long)
#                 active_ranks = torch.cat([active_ranks, pad], dim=0)
#             active_ranks = active_ranks[:C]
#         self.active_ranks = active_ranks.clamp(min=0, max=self.rank)

#         reliability = snapshot.get("reliability", None)
#         if reliability is None:
#             reliability = torch.ones((C,), device=self.device, dtype=means.dtype)
#         else:
#             reliability = torch.as_tensor(reliability, device=self.device, dtype=means.dtype).flatten()
#             if reliability.numel() < C:
#                 pad = torch.ones((C - reliability.numel(),), device=self.device, dtype=means.dtype)
#                 reliability = torch.cat([reliability, pad], dim=0)
#             reliability = reliability[:C]
#         self.reliability = reliability.clamp(0.0, 1.0)

#         spectral_protos = snapshot.get("spectral_prototypes", snapshot.get("spectral_protos", None))
#         band_importances = snapshot.get("band_importance", snapshot.get("band_importances", None))

#         if spectral_protos is not None and band_importances is not None:
#             spectral_protos = torch.as_tensor(spectral_protos, device=self.device, dtype=means.dtype)
#             band_importances = torch.as_tensor(band_importances, device=self.device, dtype=means.dtype)

#             if spectral_protos.dim() == 2 and band_importances.dim() == 2:
#                 self.spectral_protos = spectral_protos
#                 self.band_importances = band_importances
#                 self._spectral_dim = torch.tensor(
#                     spectral_protos.size(1),
#                     device=self.device,
#                     dtype=torch.long,
#                 )
#             elif strict:
#                 raise ValueError("snapshot spectral metadata has invalid shapes")
#         else:
#             self.spectral_protos = torch.empty((C, 0), device=self.device, dtype=means.dtype)
#             self.band_importances = torch.empty((C, 0), device=self.device, dtype=means.dtype)
#             self._spectral_dim = torch.tensor(0, device=self.device, dtype=torch.long)

#         self._resize_inter_class_buffers(C, means.dtype)
#         self.refresh_inter_class_geometry()

#     # ============================================================
#     # Add / update class
#     # ============================================================
#     @torch.no_grad()
#     def add_class(
#         self,
#         mean,
#         basis,
#         eigvals,
#         res_var,
#         spectral_proto=None,
#         band_importance=None,
#         active_rank=None,
#         reliability=None,
#     ):
#         mean = self._prepare_mean(mean)
#         res_var = self._prepare_res_var(res_var)
#         basis = self._sanitize_basis(basis)
#         eigvals = self._prepare_eigvals(eigvals, fallback_resvar=res_var)
#         spectral_proto, band_importance = self._prepare_spectral_pair(
#             spectral_proto=spectral_proto,
#             band_importance=band_importance,
#             dtype=mean.dtype,
#         )

#         if active_rank is None:
#             active_rank_t = torch.tensor(self.rank, device=self.device, dtype=torch.long)
#         else:
#             active_rank_t = torch.as_tensor(active_rank, device=self.device, dtype=torch.long).reshape(())
#             active_rank_t = active_rank_t.clamp(min=0, max=self.rank)

#         if reliability is None:
#             reliability_t = torch.tensor(1.0, device=self.device, dtype=mean.dtype)
#         else:
#             reliability_t = torch.as_tensor(reliability, device=self.device, dtype=mean.dtype).reshape(()).clamp(0.0, 1.0)

#         self.means = torch.cat([self.means, mean.unsqueeze(0)], dim=0)
#         self.bases = torch.cat([self.bases, basis.unsqueeze(0)], dim=0)
#         self.eigvals = torch.cat([self.eigvals, eigvals.unsqueeze(0)], dim=0)
#         self.res_vars = torch.cat([self.res_vars, res_var.view(1)], dim=0)
#         self.active_ranks = torch.cat([self.active_ranks, active_rank_t.view(1)], dim=0)
#         self.reliability = torch.cat([self.reliability, reliability_t.view(1)], dim=0)

#         if int(self._spectral_dim.item()) > 0:
#             self.spectral_protos = torch.cat([self.spectral_protos, spectral_proto.unsqueeze(0)], dim=0)
#             self.band_importances = torch.cat([self.band_importances, band_importance.unsqueeze(0)], dim=0)
#         else:
#             self.spectral_protos = torch.empty((len(self), 0), device=self.device, dtype=mean.dtype)
#             self.band_importances = torch.empty((len(self), 0), device=self.device, dtype=mean.dtype)

#         self._resize_inter_class_buffers(len(self), mean.dtype)
#         self.refresh_inter_class_geometry()

#     @torch.no_grad()
#     def update_class(
#         self,
#         cls_id,
#         mean,
#         basis,
#         eigvals,
#         res_var,
#         spectral_proto=None,
#         band_importance=None,
#         active_rank=None,
#         reliability=None,
#     ):
#         cls_id = int(cls_id)
#         if cls_id < 0 or cls_id >= len(self):
#             raise IndexError(f"class index {cls_id} out of range for GeometryBank of size {len(self)}")

#         mean = self._prepare_mean(mean)
#         res_var = self._prepare_res_var(res_var)
#         basis = self._sanitize_basis(basis)
#         eigvals = self._prepare_eigvals(eigvals, fallback_resvar=res_var)
#         spectral_proto, band_importance = self._prepare_spectral_pair(
#             spectral_proto=spectral_proto,
#             band_importance=band_importance,
#             dtype=mean.dtype,
#         )

#         self.means[cls_id] = mean
#         self.bases[cls_id] = basis
#         self.eigvals[cls_id] = eigvals
#         self.res_vars[cls_id] = res_var

#         if active_rank is not None:
#             self.active_ranks[cls_id] = torch.as_tensor(active_rank, device=self.device, dtype=torch.long).reshape(()).clamp(0, self.rank)

#         if reliability is not None:
#             self.reliability[cls_id] = torch.as_tensor(reliability, device=self.device, dtype=mean.dtype).reshape(()).clamp(0.0, 1.0)

#         if int(self._spectral_dim.item()) > 0:
#             self.spectral_protos[cls_id] = spectral_proto
#             self.band_importances[cls_id] = band_importance

#         self.refresh_inter_class_geometry()

#     @torch.no_grad()
#     def update_class_geometry(
#         self,
#         class_id: int,
#         mean: torch.Tensor,
#         basis: torch.Tensor,
#         eigvals: torch.Tensor,
#         resvar: torch.Tensor,
#         spectral_proto: Optional[torch.Tensor] = None,
#         band_importance: Optional[torch.Tensor] = None,
#         reliability: Optional[torch.Tensor] = None,
#         active_rank: Optional[torch.Tensor] = None,
#         ema: float = 0.0,
#     ) -> None:
#         """
#         Compatibility method for newer trainer/model code.
#         Supports optional EMA update of existing geometry.
#         """
#         class_id = int(class_id)
#         self.ensure_class_count(class_id + 1)

#         if float(ema) > 0.0 and class_id < len(self) and self.reliability[class_id] > 0:
#             ema = float(max(0.0, min(ema, 0.999)))
#             old = self.get_bank()
#             mean = ema * old["means"][class_id] + (1.0 - ema) * mean.to(self.device)
#             eigvals = ema * old["eigvals"][class_id] + (1.0 - ema) * eigvals.to(self.device)
#             resvar = ema * old["res_vars"][class_id] + (1.0 - ema) * torch.as_tensor(resvar, device=self.device)
#             # Basis is not EMA averaged directly because basis columns have sign/rotation ambiguity.
#             # Use the fresh sanitized basis.

#         self.update_class(
#             cls_id=class_id,
#             mean=mean,
#             basis=basis,
#             eigvals=eigvals,
#             res_var=resvar,
#             spectral_proto=spectral_proto,
#             band_importance=band_importance,
#             active_rank=active_rank,
#             reliability=reliability,
#         )

#     @torch.no_grad()
#     def ensure_class_count(self, count: int, spectral_dim: int = 0, dtype=torch.float32):
#         count = int(count)
#         if spectral_dim > 0:
#             self._ensure_spectral_dim(int(spectral_dim), dtype=dtype)

#         while len(self) < count:
#             dummy_mean = torch.zeros(self.d_model, device=self.device, dtype=dtype)
#             eye_basis = torch.eye(self.d_model, self.rank, device=self.device, dtype=dtype)
#             dummy_eigvals = torch.full((self.rank,), self.variance_floor, device=self.device, dtype=dtype)
#             dummy_res_var = torch.tensor(self.variance_floor, device=self.device, dtype=dtype)

#             if int(self._spectral_dim.item()) > 0:
#                 sdim = int(self._spectral_dim.item())
#                 dummy_spec = torch.zeros((sdim,), device=self.device, dtype=dtype)
#                 dummy_bw = torch.full((sdim,), 1.0 / float(sdim), device=self.device, dtype=dtype)
#             else:
#                 dummy_spec = torch.empty((0,), device=self.device, dtype=dtype)
#                 dummy_bw = torch.empty((0,), device=self.device, dtype=dtype)

#             self.add_class(
#                 dummy_mean,
#                 eye_basis,
#                 dummy_eigvals,
#                 dummy_res_var,
#                 spectral_proto=dummy_spec,
#                 band_importance=dummy_bw,
#                 active_rank=torch.tensor(0, device=self.device, dtype=torch.long),
#                 reliability=torch.tensor(0.0, device=self.device, dtype=dtype),
#             )

#     # ============================================================
#     # Inter-class geometry
#     # ============================================================
#     @torch.no_grad()
#     def refresh_inter_class_geometry(self):
#         C = len(self)
#         if C == 0:
#             return

#         dtype = self.means.dtype
#         self._resize_inter_class_buffers(C, dtype)

#         means = self.means
#         diff = means.unsqueeze(1) - means.unsqueeze(0)
#         self.inter_center_dist = diff.pow(2).sum(dim=-1).sqrt()

#         if self.bases.numel() > 0:
#             # Frobenius norm of U_i^T U_j, normalized by rank.
#             overlap = torch.einsum("idr,jds->ijrs", self.bases, self.bases).pow(2).sum(dim=(-1, -2))
#             self.inter_subspace_overlap = overlap / max(float(self.rank), 1.0)
#         else:
#             self.inter_subspace_overlap = torch.zeros(C, C, device=self.device, dtype=dtype)

#         if self.spectral_protos.numel() > 0 and self.spectral_protos.size(0) == C and self.spectral_protos.size(1) > 0:
#             sd = self.spectral_protos.unsqueeze(1) - self.spectral_protos.unsqueeze(0)
#             self.inter_spectral_dist = sd.pow(2).sum(dim=-1).sqrt()
#         else:
#             self.inter_spectral_dist = torch.zeros(C, C, device=self.device, dtype=dtype)

#         # High adjacency = likely confusable. Use center closeness + subspace overlap.
#         center_scale = self.inter_center_dist.detach()
#         nonzero = center_scale[center_scale > 0]
#         if nonzero.numel() > 0:
#             scale = nonzero.median().clamp_min(1e-6)
#         else:
#             scale = torch.tensor(1.0, device=self.device, dtype=dtype)

#         center_aff = torch.exp(-self.inter_center_dist / (scale * max(self.adjacency_temperature, 1e-6)))
#         sub_aff = self.inter_subspace_overlap.clamp(0.0, 1.0)

#         adjacency = 0.5 * center_aff + 0.5 * sub_aff
#         eye = torch.eye(C, device=self.device, dtype=torch.bool)
#         adjacency = adjacency.masked_fill(eye, 0.0)
#         self.inter_adjacency = adjacency

#     # ============================================================
#     # Retrieval helpers
#     # ============================================================
#     def _similarity_weights(self, feat: torch.Tensor):
#         means = self.get_means()
#         if means.numel() == 0:
#             return None, None

#         feat_n = F.normalize(feat, dim=-1, eps=1e-6)
#         means_n = F.normalize(means.to(feat.device, feat.dtype), dim=-1, eps=1e-6)
#         sim = torch.matmul(feat_n, means_n.t())

#         reliability = self.get_reliability()
#         if reliability is not None and reliability.numel() == sim.size(1):
#             rel = reliability.to(feat.device, feat.dtype).clamp(0.05, 1.0)
#             sim = sim + torch.log(rel.unsqueeze(0))

#         weights = torch.softmax(sim, dim=-1)
#         return means, weights

#     def retrieve_geometry_ref(self, feat: torch.Tensor) -> torch.Tensor:
#         means, weights = self._similarity_weights(feat)
#         if means is None:
#             return torch.zeros(feat.size(0), self.d_model, device=feat.device, dtype=feat.dtype)
#         return torch.matmul(weights, means.to(feat.device, feat.dtype))

#     def retrieve_spectral_ref(self, feat: torch.Tensor) -> torch.Tensor:
#         means, weights = self._similarity_weights(feat)
#         spectral_protos = self.get_spectral_protos()

#         if means is None or spectral_protos.numel() == 0:
#             return torch.empty((feat.size(0), 0), device=feat.device, dtype=feat.dtype)

#         spectral_protos = spectral_protos.to(feat.device, feat.dtype)
#         return torch.matmul(weights, spectral_protos)

#     def retrieve_band_importance_ref(self, feat: torch.Tensor) -> torch.Tensor:
#         means, weights = self._similarity_weights(feat)
#         band_importances = self.get_band_importances()

#         if means is None or band_importances.numel() == 0:
#             return torch.empty((feat.size(0), 0), device=feat.device, dtype=feat.dtype)

#         band_importances = band_importances.to(feat.device, feat.dtype)
#         return torch.matmul(weights, band_importances)
