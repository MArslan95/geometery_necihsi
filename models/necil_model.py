import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any

from models.backbone import SSMBackbone
from models.token import HSISemanticConceptEncoder
from models.classifier import SemanticClassifier
from models.geometry_bank import GeometryBank



def _filter_supported_kwargs(cls_or_fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only keyword arguments supported by the target __init__/call signature.

    This prevents version-mismatch crashes when main.py/necil_model.py parse newer
    research knobs but the active module implementation does not consume them yet.
    It does NOT change the behavior of supported arguments.
    """
    try:
        sig = inspect.signature(cls_or_fn)
    except (TypeError, ValueError):
        return kwargs

    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs

    allowed = set(sig.parameters.keys())
    allowed.discard("self")
    return {k: v for k, v in kwargs.items() if k in allowed}


def _zero_scalar(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(0.0, device=device, dtype=dtype)


def _projector_from_basis(basis: torch.Tensor) -> torch.Tensor:
    if basis.dim() == 2:
        return basis @ basis.t()
    return torch.matmul(basis, basis.transpose(-1, -2))


def _orthonormalize_batched_basis(basis: torch.Tensor, rank: int) -> torch.Tensor:
    """
    Orthonormalize basis columns.

    Args:
        basis: [C, D, R]
        rank: target rank

    Returns:
        [C, D, R] with orthonormal columns.
    """
    if basis is None or basis.numel() == 0:
        return basis

    outs = []
    for b in basis:
        q, _ = torch.linalg.qr(b, mode="reduced")
        if q.size(1) < rank:
            pad = torch.zeros(
                q.size(0),
                rank - q.size(1),
                device=q.device,
                dtype=q.dtype,
            )
            q = torch.cat([q, pad], dim=1)
        outs.append(q[:, :rank])
    return torch.stack(outs, dim=0)


class GeometryCalibrator(nn.Module):
    """
    Conservative geometry transport calibration for old classes.

    The calibrator is intentionally small. It corrects old geometry drift without
    letting the old class memory become a free classifier.

    Calibrates:
        mean       : mu -> mu + delta_mu
        variances  : log(v) -> log(v) + delta_logv
        basis      : optional projector-stable basis correction

    By default, basis calibration is disabled because it is easier to destabilize
    than mean/variance calibration. Enable it only after the geometry-only system
    is stable.
    """

    def __init__(
        self,
        d_model: int,
        rank: int,
        hidden_dim: Optional[int] = None,
        var_floor: float = 1e-4,
        dropout: float = 0.1,
        calibrate_basis: bool = False,
        max_mean_scale: float = 0.10,
        max_var_scale: float = 0.10,
        max_basis_scale: float = 0.03,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.rank = int(rank)
        self.var_floor = float(var_floor)
        self.calibrate_basis = bool(calibrate_basis)

        hidden_dim = int(hidden_dim or max(d_model, 128))

        # Context is the old class mean. Keep this lightweight; do not overfit.
        self.mean_calibrator = nn.Sequential(
            nn.Linear(self.d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.d_model),
        )

        self.var_calibrator = nn.Sequential(
            nn.Linear(self.d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.rank + 1),
        )

        if self.calibrate_basis:
            self.basis_calibrator = nn.Sequential(
                nn.Linear(self.d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.d_model * self.rank),
            )
        else:
            self.basis_calibrator = None

        # Raw parameters are squashed. This prevents calibration from exploding.
        self.mean_scale_raw = nn.Parameter(torch.tensor(-3.0))
        self.var_scale_raw = nn.Parameter(torch.tensor(-3.0))
        self.basis_scale_raw = nn.Parameter(torch.tensor(-4.0))

        self.max_mean_scale = float(max_mean_scale)
        self.max_var_scale = float(max_var_scale)
        self.max_basis_scale = float(max_basis_scale)

    def _mean_scale(self) -> torch.Tensor:
        return self.max_mean_scale * torch.sigmoid(self.mean_scale_raw)

    def _var_scale(self) -> torch.Tensor:
        return self.max_var_scale * torch.sigmoid(self.var_scale_raw)

    def _basis_scale(self) -> torch.Tensor:
        return self.max_basis_scale * torch.sigmoid(self.basis_scale_raw)

    def forward(
        self,
        means: torch.Tensor,
        bases: torch.Tensor,
        variances: torch.Tensor,
        reliability: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if means is None or means.numel() == 0:
            return {
                "means": means,
                "bases": bases,
                "variances": variances,
                "mean_delta": means,
                "var_delta": variances,
                "basis_delta": bases,
            }

        if reliability is None or not torch.is_tensor(reliability) or reliability.numel() == 0:
            rel = torch.ones(means.size(0), 1, device=means.device, dtype=means.dtype)
        else:
            rel = reliability.to(device=means.device, dtype=means.dtype).view(-1, 1).clamp(0.05, 1.0)

        # Less reliable geometry gets smaller learned transport.
        mean_delta = self._mean_scale() * rel * self.mean_calibrator(means)
        calibrated_means = means + mean_delta

        logv = torch.log(variances.clamp_min(self.var_floor))
        var_delta = self._var_scale() * rel * self.var_calibrator(means)
        calibrated_logv = logv + var_delta
        calibrated_variances = torch.exp(calibrated_logv).clamp_min(self.var_floor)

        if self.calibrate_basis and self.basis_calibrator is not None and bases is not None and bases.numel() > 0:
            raw_delta = self.basis_calibrator(means).view(-1, self.d_model, self.rank)
            basis_delta = self._basis_scale() * rel.view(-1, 1, 1) * raw_delta
            calibrated_bases = _orthonormalize_batched_basis(bases + basis_delta, self.rank)
        else:
            basis_delta = torch.zeros_like(bases)
            calibrated_bases = bases

        return {
            "means": calibrated_means,
            "bases": calibrated_bases,
            "variances": calibrated_variances,
            "mean_delta": mean_delta,
            "var_delta": var_delta,
            "basis_delta": basis_delta,
        }

    def regularization_loss(
        self,
        raw_means: Optional[torch.Tensor],
        raw_bases: Optional[torch.Tensor],
        raw_variances: Optional[torch.Tensor],
        calibrated_means: Optional[torch.Tensor],
        calibrated_bases: Optional[torch.Tensor],
        calibrated_variances: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if (
            raw_means is None
            or raw_variances is None
            or calibrated_means is None
            or calibrated_variances is None
            or raw_means.numel() == 0
        ):
            device = raw_means.device if torch.is_tensor(raw_means) else torch.device("cpu")
            z = _zero_scalar(device)
            return {"total": z, "mean": z, "basis": z, "var": z}

        mean_reg = F.mse_loss(calibrated_means, raw_means)

        raw_logv = torch.log(raw_variances.clamp_min(self.var_floor))
        cal_logv = torch.log(calibrated_variances.clamp_min(self.var_floor))
        var_reg = F.mse_loss(cal_logv, raw_logv)

        if (
            raw_bases is not None
            and calibrated_bases is not None
            and torch.is_tensor(raw_bases)
            and torch.is_tensor(calibrated_bases)
            and raw_bases.numel() > 0
            and calibrated_bases.numel() > 0
        ):
            basis_reg = F.mse_loss(
                _projector_from_basis(calibrated_bases),
                _projector_from_basis(raw_bases),
            )
        else:
            basis_reg = _zero_scalar(raw_means.device, raw_means.dtype)

        basis_weight = 0.2 if self.calibrate_basis else 0.0
        total = mean_reg + var_reg + basis_weight * basis_reg
        return {"total": total, "mean": mean_reg, "basis": basis_reg, "var": var_reg}


class NECILModel(nn.Module):
    """
    Geometry-centric NECIL-HSI model.

    Core policy:
        - Backbone produces spectral-spatial features/tokens.
        - Projection preserves Euclidean feature scale.
        - GeometryBank is the real non-exemplar memory.
        - Classifier scores by geometry energy.
        - Semantic concepts are auxiliary and should not rewrite geometry during
          incremental phases unless explicitly enabled.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device(args.device)

        self.d_model = int(args.d_model)
        self.subspace_rank = int(getattr(args, "subspace_rank", 5))
        self.num_concepts_per_class = int(getattr(args, "num_concepts_per_class", 4))

        self.default_base_classifier_mode = str(
            getattr(args, "base_classifier_mode", "geometry_only")
        ).lower()

        self.default_incremental_classifier_mode = str(
            getattr(args, "incremental_classifier_mode", "calibrated_geometry")
        ).lower()

        self.current_num_classes = 0
        self.old_class_count = 0
        self.current_phase = 0

        # Auxiliary semantic memory. This is NOT the main classifier memory.
        self._anchors_cpu = []
        self._anchor_deltas = nn.ParameterList()
        self._concepts_cpu = []
        self._concept_deltas = nn.ParameterList()

        self.backbone = SSMBackbone(args)

        self.semantic_encoder = HSISemanticConceptEncoder(
            feature_dim=self.d_model,
            concept_dim=self.d_model,
            dropout=float(getattr(args, "semantic_dropout", 0.1)),
            token_temperature=float(getattr(args, "token_temperature", 0.07)),
        )

        self.projection = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(float(getattr(args, "projection_dropout", 0.1))),
            nn.Linear(self.d_model, self.d_model),
        )
        self.norm = nn.LayerNorm(self.d_model)

        self.concept_projector = nn.Linear(int(args.num_bands), self.d_model, bias=True)

        geometry_bank_kwargs = _filter_supported_kwargs(
            GeometryBank.__init__,
            {
                "device": args.device,
                "variance_floor": float(getattr(args, "geom_var_floor", 1e-4)),
                "variance_shrinkage": float(getattr(args, "geometry_variance_shrinkage", 0.10)),
                "max_variance_ratio": float(getattr(args, "geometry_max_variance_ratio", 50.0)),
                "min_reliability": float(getattr(args, "geometry_min_reliability", 0.05)),
                "adjacency_temperature": float(getattr(args, "geometry_adjacency_temperature", 1.0)),
            },
        )

        self.geometry_bank = GeometryBank(
            self.d_model,
            self.subspace_rank,
            **geometry_bank_kwargs,
        )

        self.geometry_calibrator = GeometryCalibrator(
            d_model=self.d_model,
            rank=self.subspace_rank,
            hidden_dim=int(getattr(args, "geometry_calibration_hidden_dim", self.d_model)),
            var_floor=float(getattr(args, "geom_var_floor", 1e-4)),
            dropout=float(getattr(args, "geometry_calibration_dropout", 0.1)),
            calibrate_basis=bool(getattr(args, "geometry_calibrate_basis", False)),
            max_mean_scale=float(getattr(args, "geometry_max_mean_scale", 0.10)),
            max_var_scale=float(getattr(args, "geometry_max_var_scale", 0.10)),
            max_basis_scale=float(getattr(args, "geometry_max_basis_scale", 0.03)),
        )

        classifier_kwargs = _filter_supported_kwargs(
            SemanticClassifier.__init__,
            {
                "initial_classes": 0,
                "d_model": self.d_model,
                "logit_scale": float(getattr(args, "loss_scale", 8.0)),
                "use_bias": bool(getattr(args, "classifier_use_bias", True)),
                "variance_floor": float(getattr(args, "geom_var_floor", 1e-4)),
                "use_geom_temperature": bool(getattr(args, "use_geom_temperature", True)),
                "concept_agg_temperature": float(getattr(args, "cls_temperature", 0.07)),
                "init_alpha_old": float(getattr(args, "init_alpha_old", -0.5)),
                "init_alpha_new": float(getattr(args, "init_alpha_new", -0.2)),
                # Keep legacy fusion off by default. Geometry should carry the method.
                "use_adaptive_fusion": bool(getattr(args, "use_adaptive_fusion", False)),
                "min_temperature": float(getattr(args, "min_temperature", 0.25)),
                "max_temperature": float(getattr(args, "max_temperature", 4.0)),
                "energy_normalize_by_dim": bool(getattr(args, "energy_normalize_by_dim", True)),
                "debias_strength": float(getattr(args, "debias_strength", 0.10)),
                "reliability_energy_weight": float(getattr(args, "reliability_energy_weight", 0.05)),
                "volume_energy_weight": float(getattr(args, "volume_energy_weight", 0.0)),
                "max_bias_abs": float(getattr(args, "max_classifier_bias_abs", 0.50)),
                "max_debias_abs": float(getattr(args, "max_classifier_debias_abs", 0.25)),
                "max_classifier_bias_abs": float(getattr(args, "max_classifier_bias_abs", 0.50)),
                "max_classifier_debias_abs": float(getattr(args, "max_classifier_debias_abs", 0.25)),
                "enable_old_new_temp_offsets": bool(getattr(args, "enable_old_new_temp_offsets", True)),
                "geometry_logit_clip": float(getattr(args, "classifier_geometry_logit_clip", 0.0)),
            },
        )

        self.classifier = SemanticClassifier(**classifier_kwargs)

    # =========================================================
    # Basic helpers
    # =========================================================
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1, eps=1e-6)

    def _project(self, feat: torch.Tensor) -> torch.Tensor:
        # Preserve Euclidean geometry. No final L2 normalization.
        return self.norm(self.projection(feat) + feat)

    def _spectral_summary(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(-1, -2))

    def _resolve_classifier_mode(self, classifier_mode: Optional[str]) -> str:
        if classifier_mode is None:
            return (
                self.default_base_classifier_mode
                if int(self.current_phase) == 0
                else self.default_incremental_classifier_mode
            )
        return str(classifier_mode).lower()

    def _resolve_semantic_mode(self, semantic_mode: Optional[str]) -> str:
        """
        Critical design choice:
        default semantic mode is identity for all phases.

        Do not use semantic_mode='all' in base and 'identity' in incremental,
        because that builds geometry memory in one feature manifold and trains
        incremental scoring in another.
        """
        if semantic_mode is None or str(semantic_mode).lower() == "auto":
            return "identity"
        return str(semantic_mode).lower()

    def set_phase(self, phase: int):
        self.current_phase = int(phase)

    def set_old_class_count(self, old_class_count: int):
        self.old_class_count = int(old_class_count)

    # =========================================================
    # Capacity / load preparation
    # =========================================================
    @torch.no_grad()
    def ensure_class_capacity(
        self,
        class_count: int,
        spectral_dim: int = 0,
        dtype: Optional[torch.dtype] = None,
    ):
        class_count = int(class_count)
        dtype = dtype or self.projection[0].weight.dtype

        while len(self._anchors_cpu) < class_count:
            zero_anchor = torch.zeros(self.d_model, dtype=dtype)
            zero_concepts = torch.zeros(self.num_concepts_per_class, self.d_model, dtype=dtype)

            self._anchors_cpu.append(zero_anchor.clone())
            self._anchor_deltas.append(nn.Parameter(torch.zeros_like(zero_anchor, device=self.device)))

            self._concepts_cpu.append(zero_concepts.clone())
            self._concept_deltas.append(nn.Parameter(torch.zeros_like(zero_concepts, device=self.device)))

        while self.classifier.num_classes < class_count:
            self.classifier.expand(1, self.current_phase)

        # Updated bank uses ensure_class_count(count=..., spectral_dim=...).
        # Older variants may use ensure_num_classes.
        if hasattr(self.geometry_bank, "ensure_class_count"):
            self.geometry_bank.ensure_class_count(
                count=class_count,
                spectral_dim=int(spectral_dim),
                dtype=dtype,
            )
        elif hasattr(self.geometry_bank, "ensure_num_classes"):
            self.geometry_bank.ensure_num_classes(class_count)

        self.current_num_classes = max(self.current_num_classes, class_count)

    # =========================================================
    # Auxiliary semantic banks
    # =========================================================
    def get_anchor_bank(self) -> torch.Tensor:
        if len(self._anchors_cpu) == 0:
            return torch.empty((0, self.d_model), device=self.device)

        return torch.stack(
            [
                self._normalize(base.to(self.device) + self._anchor_deltas[i])
                for i, base in enumerate(self._anchors_cpu)
            ],
            dim=0,
        )

    def get_concept_bank(self) -> torch.Tensor:
        if len(self._concepts_cpu) == 0:
            return torch.empty((0, self.num_concepts_per_class, self.d_model), device=self.device)

        return torch.stack(
            [
                self._normalize(base.to(self.device) + self._concept_deltas[i])
                for i, base in enumerate(self._concepts_cpu)
            ],
            dim=0,
        )

    # =========================================================
    # Geometry bank access
    # =========================================================
    def get_subspace_bank(self) -> Dict[str, torch.Tensor]:
        bank = self.geometry_bank.get_bank()

        # Defensive aliases for old/new bank implementations.
        if "variances" not in bank:
            if "eigvals" in bank and ("res_vars" in bank or "resvars" in bank):
                res = bank.get("res_vars", bank.get("resvars"))
                bank["variances"] = torch.cat([bank["eigvals"], res.unsqueeze(-1)], dim=-1)

        if "resvars" not in bank and "res_vars" in bank:
            bank["resvars"] = bank["res_vars"]
        if "res_vars" not in bank and "resvars" in bank:
            bank["res_vars"] = bank["resvars"]

        C = bank["means"].size(0) if "means" in bank and torch.is_tensor(bank["means"]) else 0
        if "reliability" not in bank:
            bank["reliability"] = torch.ones(C, device=self.device)
        if "active_ranks" not in bank:
            bank["active_ranks"] = torch.full((C,), self.subspace_rank, device=self.device, dtype=torch.long)
        if "sample_counts" not in bank:
            bank["sample_counts"] = torch.zeros(C, device=self.device)
        if "geometry_volumes" not in bank:
            bank["geometry_volumes"] = torch.zeros(C, device=self.device)
        if "class_dispersions" not in bank:
            bank["class_dispersions"] = torch.zeros(C, device=self.device)
        if "class_risk" not in bank:
            bank["class_risk"] = torch.zeros(C, device=self.device)

        return bank

    def get_old_subspace_bank(self, old_class_count: Optional[int] = None) -> Dict[str, torch.Tensor]:
        old_class_count = int(self.old_class_count if old_class_count is None else old_class_count)
        bank = self.get_subspace_bank()

        if old_class_count <= 0:
            return {
                "means": torch.empty((0, self.d_model), device=self.device),
                "bases": torch.empty((0, self.d_model, self.subspace_rank), device=self.device),
                "variances": torch.empty((0, self.subspace_rank + 1), device=self.device),
                "reliability": torch.empty((0,), device=self.device),
                "active_ranks": torch.empty((0,), device=self.device, dtype=torch.long),
            }

        return {
            "means": bank["means"][:old_class_count],
            "bases": bank["bases"][:old_class_count],
            "variances": bank["variances"][:old_class_count],
            "reliability": bank.get("reliability", torch.ones(bank["means"].size(0), device=bank["means"].device))[:old_class_count],
            "active_ranks": bank.get("active_ranks", torch.full((bank["means"].size(0),), self.subspace_rank, device=bank["means"].device, dtype=torch.long))[:old_class_count],
        }

    def get_new_subspace_bank(self, old_class_count: Optional[int] = None) -> Dict[str, torch.Tensor]:
        old_class_count = int(self.old_class_count if old_class_count is None else old_class_count)
        bank = self.get_subspace_bank()

        return {
            "means": bank["means"][old_class_count:],
            "bases": bank["bases"][old_class_count:],
            "variances": bank["variances"][old_class_count:],
            "reliability": bank.get("reliability", torch.ones(bank["means"].size(0), device=bank["means"].device))[old_class_count:],
            "active_ranks": bank.get("active_ranks", torch.full((bank["means"].size(0),), self.subspace_rank, device=bank["means"].device, dtype=torch.long))[old_class_count:],
        }

    def get_calibrated_old_subspace_bank(
        self,
        old_class_count: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        old_class_count = int(self.old_class_count if old_class_count is None else old_class_count)
        raw_old = self.get_old_subspace_bank(old_class_count)

        if old_class_count <= 0 or raw_old["means"].numel() == 0:
            return raw_old

        calibrated = self.geometry_calibrator(
            raw_old["means"],
            raw_old["bases"],
            raw_old["variances"],
            reliability=raw_old.get("reliability", None),
        )

        return {
            "means": calibrated["means"],
            "bases": calibrated["bases"],
            "variances": calibrated["variances"],
            "mean_delta": calibrated["mean_delta"],
            "var_delta": calibrated["var_delta"],
            "basis_delta": calibrated["basis_delta"],
            "raw_means": raw_old["means"],
            "raw_bases": raw_old["bases"],
            "raw_variances": raw_old["variances"],
            "reliability": raw_old.get("reliability", None),
            "active_ranks": raw_old.get("active_ranks", None),
        }

    def calibration_regularization_loss(
        self,
        old_class_count: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        old_class_count = int(self.old_class_count if old_class_count is None else old_class_count)
        raw_old = self.get_old_subspace_bank(old_class_count)

        if old_class_count <= 0 or raw_old["means"].numel() == 0:
            z = _zero_scalar(self.device)
            return {"total": z, "mean": z, "basis": z, "var": z}

        calibrated = self.get_calibrated_old_subspace_bank(old_class_count)

        return self.geometry_calibrator.regularization_loss(
            raw_means=raw_old["means"],
            raw_bases=raw_old["bases"],
            raw_variances=raw_old["variances"],
            calibrated_means=calibrated["means"],
            calibrated_bases=calibrated["bases"],
            calibrated_variances=calibrated["variances"],
        )

    # =========================================================
    # Snapshot helpers
    # =========================================================
    @torch.no_grad()
    def export_memory_snapshot(self) -> Dict[str, Any]:
        geometry_snap = self.geometry_bank.export_snapshot()

        if len(self._anchors_cpu) > 0:
            anchor_base = torch.stack([x.detach().cpu() for x in self._anchors_cpu], dim=0)
            anchor_deltas = torch.stack([p.detach().cpu() for p in self._anchor_deltas], dim=0)
        else:
            anchor_base = torch.empty((0, self.d_model))
            anchor_deltas = torch.empty((0, self.d_model))

        if len(self._concepts_cpu) > 0:
            concept_base = torch.stack([x.detach().cpu() for x in self._concepts_cpu], dim=0)
            concept_deltas = torch.stack([p.detach().cpu() for p in self._concept_deltas], dim=0)
        else:
            concept_base = torch.empty((0, self.num_concepts_per_class, self.d_model))
            concept_deltas = torch.empty((0, self.num_concepts_per_class, self.d_model))

        snap = {
            "current_num_classes": int(self.current_num_classes),
            "old_class_count": int(self.old_class_count),
            "current_phase": int(self.current_phase),
            "anchor_base": anchor_base,
            "anchor_deltas": anchor_deltas,
            "concept_base": concept_base,
            "concept_deltas": concept_deltas,
        }
        snap.update(geometry_snap)
        return snap

    @torch.no_grad()
    def load_memory_snapshot(self, snapshot: Dict[str, Any], strict: bool = True):
        if snapshot is None:
            if strict:
                raise ValueError("snapshot is None")
            return

        means = snapshot.get("means", None)
        class_count = int(means.size(0)) if torch.is_tensor(means) else int(snapshot.get("current_num_classes", 0))
        spectral_dim = int(snapshot.get("spectral_dim", 0))

        self.ensure_class_capacity(class_count=class_count, spectral_dim=spectral_dim)
        self.geometry_bank.load_snapshot(snapshot, strict=strict)

        self._anchors_cpu = []
        self._anchor_deltas = nn.ParameterList()
        self._concepts_cpu = []
        self._concept_deltas = nn.ParameterList()

        anchor_base = snapshot.get("anchor_base", None)
        anchor_deltas = snapshot.get("anchor_deltas", None)
        concept_base = snapshot.get("concept_base", None)
        concept_deltas = snapshot.get("concept_deltas", None)

        for cls in range(class_count):
            a_base = anchor_base[cls].detach().cpu().clone() if torch.is_tensor(anchor_base) and anchor_base.size(0) > cls else torch.zeros(self.d_model)
            a_delta = anchor_deltas[cls].detach().to(self.device).clone() if torch.is_tensor(anchor_deltas) and anchor_deltas.size(0) > cls else torch.zeros(self.d_model, device=self.device)
            c_base = concept_base[cls].detach().cpu().clone() if torch.is_tensor(concept_base) and concept_base.size(0) > cls else torch.zeros(self.num_concepts_per_class, self.d_model)
            c_delta = concept_deltas[cls].detach().to(self.device).clone() if torch.is_tensor(concept_deltas) and concept_deltas.size(0) > cls else torch.zeros(self.num_concepts_per_class, self.d_model, device=self.device)

            self._anchors_cpu.append(a_base)
            self._anchor_deltas.append(nn.Parameter(a_delta))
            self._concepts_cpu.append(c_base)
            self._concept_deltas.append(nn.Parameter(c_delta))

        self.current_num_classes = class_count
        self.old_class_count = int(snapshot.get("old_class_count", self.old_class_count))
        self.current_phase = int(snapshot.get("current_phase", self.current_phase))

    # =========================================================
    # Backbone feature extraction
    # =========================================================
    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.backbone(x)

        if not isinstance(out, dict):
            feat = out
            bands = x.shape[1]
            band_weights = torch.full((feat.size(0), bands), 1.0 / bands, device=feat.device, dtype=feat.dtype)
            return {
                "features": feat,
                "backbone_features": feat,
                "band_weights": band_weights,
                "spectral_tokens": None,
                "spatial_tokens": None,
                "fused_tokens": None,
                "spectral_features": feat,
                "spatial_features": feat,
                "spatial_patterns": {},
            }

        feat = out["features"]
        band_weights = out.get("band_weights")
        if band_weights is None:
            bands = x.shape[1]
            band_weights = torch.full((feat.size(0), bands), 1.0 / bands, device=feat.device, dtype=feat.dtype)

        return {
            "features": feat,
            "backbone_features": feat,
            "band_weights": band_weights,
            "spectral_tokens": out.get("spectral_tokens"),
            "spatial_tokens": out.get("spatial_tokens"),
            "fused_tokens": out.get("fused_tokens"),
            "spectral_features": out.get("spectral_features", feat),
            "spatial_features": out.get("spatial_features", feat),
            "spatial_patterns": out.get("spatial_patterns", {}),
        }

    @torch.no_grad()
    def extract_backbone_outputs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.extract_features(x)
        raw_feat = out["features"]
        projected_feat = self._project(raw_feat)

        return {
            "features": projected_feat,
            "projected_features": projected_feat,
            "preproject_features": raw_feat,
            "backbone_features": raw_feat,
            "band_weights": out["band_weights"],
            "spectral_summary": self._spectral_summary(x),
            "spectral_tokens": out["spectral_tokens"],
            "spatial_tokens": out["spatial_tokens"],
            "fused_tokens": out["fused_tokens"],
            "spectral_features": out["spectral_features"],
            "spatial_features": out["spatial_features"],
            "spatial_patterns": out["spatial_patterns"],
        }

    def build_geometry_ref(self, feat: torch.Tensor):
        if hasattr(self.geometry_bank, "retrieve_geometry_ref"):
            return self.geometry_bank.retrieve_geometry_ref(feat)
        return torch.zeros(feat.size(0), self.d_model, device=feat.device, dtype=feat.dtype)

    # =========================================================
    # Concept / anchor handling
    # =========================================================
    def _embed_concepts(self, concepts: torch.Tensor) -> torch.Tensor:
        concepts = concepts.float().to(self.device)

        if concepts.dim() != 2:
            raise ValueError(f"concepts must be 2D, got {tuple(concepts.shape)}")

        if concepts.size(1) == self.d_model:
            embedded = concepts
        elif concepts.size(1) == self.concept_projector.in_features:
            embedded = self.concept_projector(concepts)
        else:
            raise ValueError(
                f"concept dim mismatch: expected {self.d_model} or {self.concept_projector.in_features}, got {concepts.size(1)}"
            )

        embedded = self._normalize(embedded)

        k = embedded.size(0)
        if k != self.num_concepts_per_class:
            if k > self.num_concepts_per_class:
                embedded = embedded[: self.num_concepts_per_class]
            else:
                pad = embedded[-1:].repeat(self.num_concepts_per_class - k, 1)
                embedded = torch.cat([embedded, pad], dim=0)

        return embedded

    @torch.no_grad()
    def add_new_class_concepts(self, concepts: torch.Tensor):
        cls = int(self.current_num_classes)
        self.ensure_class_capacity(cls + 1)
        self.refresh_class_concepts(cls, concepts, reset_delta=True)

        anchor = self.get_anchor_bank()[cls].detach()
        basis = torch.eye(self.d_model, self.subspace_rank, device=self.device, dtype=anchor.dtype)
        eigvals = torch.full((self.subspace_rank,), 1e-4, device=self.device, dtype=anchor.dtype)
        res_var = torch.tensor(1e-4, device=self.device, dtype=anchor.dtype)

        self.refresh_class_subspace(
            cls=cls,
            mean=anchor,
            basis=basis,
            eigvals=eigvals,
            res_var=res_var,
            spectral_proto=None,
            band_importance=None,
            active_rank=torch.tensor(0, device=self.device, dtype=torch.long),
            reliability=torch.tensor(0.05, device=self.device, dtype=anchor.dtype),
        )

        self.current_num_classes = cls + 1

    @torch.no_grad()
    def refresh_class_concepts(
        self,
        cls: int,
        concepts: torch.Tensor,
        reset_delta: bool = True,
    ):
        cls = int(cls)
        self.ensure_class_capacity(cls + 1)

        embedded = self._embed_concepts(concepts)
        anchor = self._normalize(embedded.mean(dim=0))

        self._anchors_cpu[cls] = anchor.detach().cpu()
        self._concepts_cpu[cls] = embedded.detach().cpu()

        if reset_delta:
            self._anchor_deltas[cls].data.zero_()
            self._concept_deltas[cls].data.zero_()

        self.current_num_classes = max(self.current_num_classes, cls + 1)

    @torch.no_grad()
    def refresh_class_subspace(
        self,
        cls: int,
        mean: torch.Tensor,
        basis: torch.Tensor,
        eigvals: torch.Tensor,
        res_var,
        spectral_proto=None,
        band_importance=None,
        active_rank=None,
        reliability=None,
    ):
        cls = int(cls)

        spectral_dim = 0
        if spectral_proto is not None and torch.as_tensor(spectral_proto).numel() > 0:
            spectral_dim = int(torch.as_tensor(spectral_proto).numel())
        elif band_importance is not None and torch.as_tensor(band_importance).numel() > 0:
            spectral_dim = int(torch.as_tensor(band_importance).numel())

        self.ensure_class_capacity(cls + 1, spectral_dim=spectral_dim)

        if hasattr(self.geometry_bank, "update_class_geometry"):
            self.geometry_bank.update_class_geometry(
                class_id=cls,
                mean=mean.float().to(self.device),
                basis=basis.float().to(self.device),
                eigvals=eigvals.float().to(self.device),
                resvar=torch.as_tensor(res_var, device=self.device, dtype=mean.dtype),
                spectral_proto=spectral_proto,
                band_importance=band_importance,
                active_rank=active_rank,
                reliability=reliability,
            )
        else:
            self.geometry_bank.update_class(
                cls_id=cls,
                mean=mean.float().to(self.device),
                basis=basis.float().to(self.device),
                eigvals=eigvals.float().to(self.device),
                res_var=torch.as_tensor(res_var, device=self.device, dtype=mean.dtype),
                spectral_proto=spectral_proto,
                band_importance=band_importance,
            )

            if active_rank is not None and hasattr(self.geometry_bank, "active_ranks"):
                self.geometry_bank.active_ranks[cls] = active_rank.to(self.device)
            if reliability is not None and hasattr(self.geometry_bank, "reliability"):
                self.geometry_bank.reliability[cls] = reliability.to(self.device)

        self.current_num_classes = max(self.current_num_classes, cls + 1)

    @torch.no_grad()
    def refresh_inter_class_geometry(self):
        if hasattr(self.geometry_bank, "refresh_inter_class_geometry"):
            self.geometry_bank.refresh_inter_class_geometry()

    # =========================================================
    # Logit computation
    # =========================================================
    def compute_logits_from_features(
        self,
        features: torch.Tensor,
        classifier_mode: str = "geometry_only",
    ):
        classifier_mode = str(classifier_mode).lower()
        subspace_bank = self.get_subspace_bank()
        anchors = self.get_anchor_bank()
        concepts = self.get_concept_bank()
        calibrated_old = self.get_calibrated_old_subspace_bank(self.old_class_count)

        return self.classifier(
            features,
            anchors=anchors if anchors.numel() > 0 else None,
            concept_bank=concepts if concepts.numel() > 0 else None,
            subspace_means=subspace_bank["means"] if subspace_bank["means"].numel() > 0 else None,
            subspace_bases=subspace_bank["bases"] if subspace_bank["bases"].numel() > 0 else None,
            subspace_variances=subspace_bank["variances"] if subspace_bank["variances"].numel() > 0 else None,
            subspace_reliability=subspace_bank.get("reliability", None),
            subspace_active_ranks=subspace_bank.get("active_ranks", None),
            calibrated_old_means=calibrated_old["means"] if calibrated_old.get("means", None) is not None and calibrated_old["means"].numel() > 0 else None,
            calibrated_old_bases=calibrated_old["bases"] if calibrated_old.get("bases", None) is not None and calibrated_old["bases"].numel() > 0 else None,
            calibrated_old_variances=calibrated_old["variances"] if calibrated_old.get("variances", None) is not None and calibrated_old["variances"].numel() > 0 else None,
            calibrated_old_reliability=calibrated_old.get("reliability", None),
            calibrated_old_active_ranks=calibrated_old.get("active_ranks", None),
            mode=classifier_mode,
            old_class_count=int(self.old_class_count),
        )

    # =========================================================
    # Trainability controls
    # =========================================================
    def freeze_backbone_only(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def freeze_semantic_encoder(self):
        for p in self.semantic_encoder.parameters():
            p.requires_grad = False

    def unfreeze_semantic_encoder(self):
        for p in self.semantic_encoder.parameters():
            p.requires_grad = True

    def freeze_projection_head(self):
        for p in self.projection.parameters():
            p.requires_grad = False
        for p in self.norm.parameters():
            p.requires_grad = False
        for p in self.concept_projector.parameters():
            p.requires_grad = False

    def unfreeze_projection_head(self):
        for p in self.projection.parameters():
            p.requires_grad = True
        for p in self.norm.parameters():
            p.requires_grad = True
        for p in self.concept_projector.parameters():
            p.requires_grad = True

    def freeze_old_anchor_deltas(self, old_class_count: int):
        for i, p in enumerate(self._anchor_deltas):
            if i < int(old_class_count):
                p.requires_grad = False

    def unfreeze_new_anchor_deltas(self, old_class_count: int):
        for i, p in enumerate(self._anchor_deltas):
            p.requires_grad = i >= int(old_class_count)

    def freeze_old_concept_deltas(self, old_class_count: int):
        for i, p in enumerate(self._concept_deltas):
            if i < int(old_class_count):
                p.requires_grad = False

    def unfreeze_new_concept_deltas(self, old_class_count: int):
        for i, p in enumerate(self._concept_deltas):
            p.requires_grad = i >= int(old_class_count)

    def freeze_classifier_adaptation(self):
        if hasattr(self.classifier, "freeze_all_adaptation"):
            self.classifier.freeze_all_adaptation()

    def freeze_old_classifier_adaptation(self, old_class_count: int):
        if hasattr(self.classifier, "freeze_old_adaptation"):
            self.classifier.freeze_old_adaptation(old_class_count)

    def unfreeze_classifier_adaptation(self):
        if hasattr(self.classifier, "unfreeze_all_adaptation"):
            self.classifier.unfreeze_all_adaptation()

    def freeze_fusion_module(self):
        if hasattr(self.classifier, "freeze_fusion_module"):
            self.classifier.freeze_fusion_module()

    def unfreeze_fusion_module(self):
        if hasattr(self.classifier, "unfreeze_fusion_module"):
            self.classifier.unfreeze_fusion_module()

    def freeze_geometry_calibrator(self):
        for p in self.geometry_calibrator.parameters():
            p.requires_grad = False

    def unfreeze_geometry_calibrator(self):
        for p in self.geometry_calibrator.parameters():
            p.requires_grad = True

    # =========================================================
    # Semantic refinement helper
    # =========================================================
    def _semantic_refine(
        self,
        feat: torch.Tensor,
        anchors: torch.Tensor,
        concepts: torch.Tensor,
        spectral_tokens,
        spatial_tokens,
        semantic_mode: str,
        return_token_relations: bool,
    ):
        bypass_semantic = semantic_mode in {"off", "none", "bypass", "identity", "raw"}
        if anchors.numel() == 0 or bypass_semantic:
            return feat, None

        semantic_bank = concepts if concepts.numel() > 0 else anchors.unsqueeze(1)

        semantic_out = self.semantic_encoder(
            feat,
            concept_bank=semantic_bank,
            phase=int(self.current_phase),
            disable=False,
            spectral_tokens=spectral_tokens,
            spatial_tokens=spatial_tokens,
            return_token_relations=return_token_relations,
        )

        if isinstance(semantic_out, dict):
            feat_refined = semantic_out["features"]
            token_relations = semantic_out.get("token_relations", None)
        else:
            feat_refined = semantic_out
            token_relations = None

        return feat_refined, token_relations

    # =========================================================
    # Forward
    # =========================================================
    def forward(self, x: torch.Tensor, **kwargs):
        semantic_mode = self._resolve_semantic_mode(kwargs.get("semantic_mode", "auto"))
        classifier_mode = self._resolve_classifier_mode(kwargs.get("classifier_mode", None))
        return_token_relations = bool(kwargs.get("return_token_relations", False))

        backbone_out = self.extract_features(x)

        feat = backbone_out["features"]
        band_weights = backbone_out["band_weights"]
        spectral_tokens = backbone_out["spectral_tokens"]
        spatial_tokens = backbone_out["spatial_tokens"]
        fused_tokens = backbone_out["fused_tokens"]
        spectral_summary = self._spectral_summary(x)

        anchors = self.get_anchor_bank()
        concepts = self.get_concept_bank()

        feat_pre, token_relations = self._semantic_refine(
            feat=feat,
            anchors=anchors,
            concepts=concepts,
            spectral_tokens=spectral_tokens,
            spatial_tokens=spatial_tokens,
            semantic_mode=semantic_mode,
            return_token_relations=return_token_relations,
        )

        if (
            return_token_relations
            and token_relations is None
            and spectral_tokens is not None
            and spatial_tokens is not None
        ):
            token_relations = self.semantic_encoder.build_token_relations(
                spectral_tokens,
                spatial_tokens,
            )

        projected = self._project(feat_pre)

        subspace = self.get_subspace_bank()
        calibrated_old = self.get_calibrated_old_subspace_bank(self.old_class_count)

        geometry_ref = self.build_geometry_ref(projected)

        if hasattr(self.geometry_bank, "retrieve_spectral_ref"):
            spectral_ref = self.geometry_bank.retrieve_spectral_ref(projected)
        else:
            spectral_ref = torch.empty(projected.size(0), 0, device=projected.device, dtype=projected.dtype)

        if hasattr(self.geometry_bank, "retrieve_band_importance_ref"):
            band_importance_ref = self.geometry_bank.retrieve_band_importance_ref(projected)
        else:
            band_importance_ref = torch.empty(projected.size(0), 0, device=projected.device, dtype=projected.dtype)

        logits = self.compute_logits_from_features(
            projected,
            classifier_mode=classifier_mode,
        )

        calibration_reg = self.calibration_regularization_loss(self.old_class_count)

        return {
            "logits": logits,
            "features": projected,
            "preproject_features": feat_pre,
            "backbone_features": feat,
            "band_weights": band_weights,
            "spectral_summary": spectral_summary,
            "spectral_ref": spectral_ref,
            "band_importance_ref": band_importance_ref,
            "geometry_ref": geometry_ref,
            "anchors": anchors,
            "concept_bank": concepts,
            "subspace_means": subspace["means"],
            "subspace_bases": subspace["bases"],
            "subspace_variances": subspace["variances"],
            "subspace_reliability": subspace.get("reliability", None),
            "subspace_active_ranks": subspace.get("active_ranks", None),
            "subspace_sample_counts": subspace.get("sample_counts", None),
            "subspace_geometry_volumes": subspace.get("geometry_volumes", None),
            "subspace_class_dispersions": subspace.get("class_dispersions", None),
            "subspace_class_risk": subspace.get("class_risk", None),
            "calibrated_old_means": calibrated_old.get("means", None),
            "calibrated_old_bases": calibrated_old.get("bases", None),
            "calibrated_old_variances": calibrated_old.get("variances", None),
            "calibrated_old_reliability": calibrated_old.get("reliability", None),
            "calibration_reg": calibration_reg,
            "spectral_tokens": spectral_tokens,
            "spatial_tokens": spatial_tokens,
            "fused_tokens": fused_tokens,
            "token_relations": token_relations,
            "spectral_features": backbone_out["spectral_features"],
            "spatial_features": backbone_out["spatial_features"],
            "spatial_patterns": backbone_out["spatial_patterns"],
        }






# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Dict, Optional, Any

# from models.backbone import SSMBackbone
# from models.token import HSISemanticConceptEncoder
# from models.classifier import SemanticClassifier
# from models.geometry_bank import GeometryBank


# def _zero_scalar(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
#     return torch.tensor(0.0, device=device, dtype=dtype)


# def _projector_from_basis(basis: torch.Tensor) -> torch.Tensor:
#     if basis.dim() == 2:
#         return basis @ basis.t()
#     return torch.matmul(basis, basis.transpose(-1, -2))


# def _orthonormalize_batched_basis(basis: torch.Tensor, rank: int) -> torch.Tensor:
#     """
#     Orthonormalize basis columns.

#     Args:
#         basis: [C, D, R]
#         rank: target rank

#     Returns:
#         [C, D, R] with orthonormal columns.
#     """
#     if basis is None or basis.numel() == 0:
#         return basis

#     outs = []
#     for b in basis:
#         q, _ = torch.linalg.qr(b, mode="reduced")
#         if q.size(1) < rank:
#             pad = torch.zeros(
#                 q.size(0),
#                 rank - q.size(1),
#                 device=q.device,
#                 dtype=q.dtype,
#             )
#             q = torch.cat([q, pad], dim=1)
#         outs.append(q[:, :rank])
#     return torch.stack(outs, dim=0)


# class GeometryCalibrator(nn.Module):
#     """
#     Conservative geometry transport calibration for old classes.

#     The calibrator is intentionally small. It corrects old geometry drift without
#     letting the old class memory become a free classifier.

#     Calibrates:
#         mean       : mu -> mu + delta_mu
#         variances  : log(v) -> log(v) + delta_logv
#         basis      : optional projector-stable basis correction

#     By default, basis calibration is disabled because it is easier to destabilize
#     than mean/variance calibration. Enable it only after the geometry-only system
#     is stable.
#     """

#     def __init__(
#         self,
#         d_model: int,
#         rank: int,
#         hidden_dim: Optional[int] = None,
#         var_floor: float = 1e-4,
#         dropout: float = 0.1,
#         calibrate_basis: bool = False,
#         max_mean_scale: float = 0.10,
#         max_var_scale: float = 0.10,
#         max_basis_scale: float = 0.03,
#     ):
#         super().__init__()
#         self.d_model = int(d_model)
#         self.rank = int(rank)
#         self.var_floor = float(var_floor)
#         self.calibrate_basis = bool(calibrate_basis)

#         hidden_dim = int(hidden_dim or max(d_model, 128))

#         # Context is the old class mean. Keep this lightweight; do not overfit.
#         self.mean_calibrator = nn.Sequential(
#             nn.Linear(self.d_model, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, self.d_model),
#         )

#         self.var_calibrator = nn.Sequential(
#             nn.Linear(self.d_model, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, self.rank + 1),
#         )

#         if self.calibrate_basis:
#             self.basis_calibrator = nn.Sequential(
#                 nn.Linear(self.d_model, hidden_dim),
#                 nn.GELU(),
#                 nn.Dropout(dropout),
#                 nn.Linear(hidden_dim, self.d_model * self.rank),
#             )
#         else:
#             self.basis_calibrator = None

#         # Raw parameters are squashed. This prevents calibration from exploding.
#         self.mean_scale_raw = nn.Parameter(torch.tensor(-3.0))
#         self.var_scale_raw = nn.Parameter(torch.tensor(-3.0))
#         self.basis_scale_raw = nn.Parameter(torch.tensor(-4.0))

#         self.max_mean_scale = float(max_mean_scale)
#         self.max_var_scale = float(max_var_scale)
#         self.max_basis_scale = float(max_basis_scale)

#     def _mean_scale(self) -> torch.Tensor:
#         return self.max_mean_scale * torch.sigmoid(self.mean_scale_raw)

#     def _var_scale(self) -> torch.Tensor:
#         return self.max_var_scale * torch.sigmoid(self.var_scale_raw)

#     def _basis_scale(self) -> torch.Tensor:
#         return self.max_basis_scale * torch.sigmoid(self.basis_scale_raw)

#     def forward(
#         self,
#         means: torch.Tensor,
#         bases: torch.Tensor,
#         variances: torch.Tensor,
#         reliability: Optional[torch.Tensor] = None,
#     ) -> Dict[str, torch.Tensor]:
#         if means is None or means.numel() == 0:
#             return {
#                 "means": means,
#                 "bases": bases,
#                 "variances": variances,
#                 "mean_delta": means,
#                 "var_delta": variances,
#                 "basis_delta": bases,
#             }

#         if reliability is None or not torch.is_tensor(reliability) or reliability.numel() == 0:
#             rel = torch.ones(means.size(0), 1, device=means.device, dtype=means.dtype)
#         else:
#             rel = reliability.to(device=means.device, dtype=means.dtype).view(-1, 1).clamp(0.05, 1.0)

#         # Less reliable geometry gets smaller learned transport.
#         mean_delta = self._mean_scale() * rel * self.mean_calibrator(means)
#         calibrated_means = means + mean_delta

#         logv = torch.log(variances.clamp_min(self.var_floor))
#         var_delta = self._var_scale() * rel * self.var_calibrator(means)
#         calibrated_logv = logv + var_delta
#         calibrated_variances = torch.exp(calibrated_logv).clamp_min(self.var_floor)

#         if self.calibrate_basis and self.basis_calibrator is not None and bases is not None and bases.numel() > 0:
#             raw_delta = self.basis_calibrator(means).view(-1, self.d_model, self.rank)
#             basis_delta = self._basis_scale() * rel.view(-1, 1, 1) * raw_delta
#             calibrated_bases = _orthonormalize_batched_basis(bases + basis_delta, self.rank)
#         else:
#             basis_delta = torch.zeros_like(bases)
#             calibrated_bases = bases

#         return {
#             "means": calibrated_means,
#             "bases": calibrated_bases,
#             "variances": calibrated_variances,
#             "mean_delta": mean_delta,
#             "var_delta": var_delta,
#             "basis_delta": basis_delta,
#         }

#     def regularization_loss(
#         self,
#         raw_means: Optional[torch.Tensor],
#         raw_bases: Optional[torch.Tensor],
#         raw_variances: Optional[torch.Tensor],
#         calibrated_means: Optional[torch.Tensor],
#         calibrated_bases: Optional[torch.Tensor],
#         calibrated_variances: Optional[torch.Tensor],
#     ) -> Dict[str, torch.Tensor]:
#         if (
#             raw_means is None
#             or raw_variances is None
#             or calibrated_means is None
#             or calibrated_variances is None
#             or raw_means.numel() == 0
#         ):
#             device = raw_means.device if torch.is_tensor(raw_means) else torch.device("cpu")
#             z = _zero_scalar(device)
#             return {"total": z, "mean": z, "basis": z, "var": z}

#         mean_reg = F.mse_loss(calibrated_means, raw_means)

#         raw_logv = torch.log(raw_variances.clamp_min(self.var_floor))
#         cal_logv = torch.log(calibrated_variances.clamp_min(self.var_floor))
#         var_reg = F.mse_loss(cal_logv, raw_logv)

#         if (
#             raw_bases is not None
#             and calibrated_bases is not None
#             and torch.is_tensor(raw_bases)
#             and torch.is_tensor(calibrated_bases)
#             and raw_bases.numel() > 0
#             and calibrated_bases.numel() > 0
#         ):
#             basis_reg = F.mse_loss(
#                 _projector_from_basis(calibrated_bases),
#                 _projector_from_basis(raw_bases),
#             )
#         else:
#             basis_reg = _zero_scalar(raw_means.device, raw_means.dtype)

#         basis_weight = 0.2 if self.calibrate_basis else 0.0
#         total = mean_reg + var_reg + basis_weight * basis_reg
#         return {"total": total, "mean": mean_reg, "basis": basis_reg, "var": var_reg}


# class NECILModel(nn.Module):
#     """
#     Geometry-centric NECIL-HSI model.

#     Core policy:
#         - Backbone produces spectral-spatial features/tokens.
#         - Projection preserves Euclidean feature scale.
#         - GeometryBank is the real non-exemplar memory.
#         - Classifier scores by geometry energy.
#         - Semantic concepts are auxiliary and should not rewrite geometry during
#           incremental phases unless explicitly enabled.
#     """

#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         self.device = torch.device(args.device)

#         self.d_model = int(args.d_model)
#         self.subspace_rank = int(getattr(args, "subspace_rank", 5))
#         self.num_concepts_per_class = int(getattr(args, "num_concepts_per_class", 4))

#         self.default_base_classifier_mode = str(
#             getattr(args, "base_classifier_mode", "geometry_only")
#         ).lower()

#         self.default_incremental_classifier_mode = str(
#             getattr(args, "incremental_classifier_mode", "calibrated_geometry")
#         ).lower()

#         self.current_num_classes = 0
#         self.old_class_count = 0
#         self.current_phase = 0

#         # Auxiliary semantic memory. This is NOT the main classifier memory.
#         self._anchors_cpu = []
#         self._anchor_deltas = nn.ParameterList()
#         self._concepts_cpu = []
#         self._concept_deltas = nn.ParameterList()

#         self.backbone = SSMBackbone(args)

#         self.semantic_encoder = HSISemanticConceptEncoder(
#             feature_dim=self.d_model,
#             concept_dim=self.d_model,
#             dropout=float(getattr(args, "semantic_dropout", 0.1)),
#             token_temperature=float(getattr(args, "token_temperature", 0.07)),
#         )

#         self.projection = nn.Sequential(
#             nn.Linear(self.d_model, self.d_model),
#             nn.GELU(),
#             nn.Dropout(float(getattr(args, "projection_dropout", 0.1))),
#             nn.Linear(self.d_model, self.d_model),
#         )
#         self.norm = nn.LayerNorm(self.d_model)

#         self.concept_projector = nn.Linear(int(args.num_bands), self.d_model, bias=True)

#         self.geometry_bank = GeometryBank(
#             self.d_model,
#             self.subspace_rank,
#             device=args.device,
#             variance_floor=float(getattr(args, "geom_var_floor", 1e-4)),
#             variance_shrinkage=float(getattr(args, "geometry_variance_shrinkage", 0.10)),
#             max_variance_ratio=float(getattr(args, "geometry_max_variance_ratio", 50.0)),
#             min_reliability=float(getattr(args, "geometry_min_reliability", 0.05)),
#         )

#         self.geometry_calibrator = GeometryCalibrator(
#             d_model=self.d_model,
#             rank=self.subspace_rank,
#             hidden_dim=int(getattr(args, "geometry_calibration_hidden_dim", self.d_model)),
#             var_floor=float(getattr(args, "geom_var_floor", 1e-4)),
#             dropout=float(getattr(args, "geometry_calibration_dropout", 0.1)),
#             calibrate_basis=bool(getattr(args, "geometry_calibrate_basis", False)),
#             max_mean_scale=float(getattr(args, "geometry_max_mean_scale", 0.10)),
#             max_var_scale=float(getattr(args, "geometry_max_var_scale", 0.10)),
#             max_basis_scale=float(getattr(args, "geometry_max_basis_scale", 0.03)),
#         )

#         self.classifier = SemanticClassifier(
#             initial_classes=0,
#             d_model=self.d_model,
#             logit_scale=float(getattr(args, "loss_scale", 8.0)),
#             use_bias=bool(getattr(args, "classifier_use_bias", True)),
#             variance_floor=float(getattr(args, "geom_var_floor", 1e-4)),
#             use_geom_temperature=bool(getattr(args, "use_geom_temperature", True)),
#             concept_agg_temperature=float(getattr(args, "cls_temperature", 0.07)),
#             init_alpha_old=float(getattr(args, "init_alpha_old", -0.5)),
#             init_alpha_new=float(getattr(args, "init_alpha_new", -0.2)),
#             # Keep legacy fusion off by default. Geometry should carry the method.
#             use_adaptive_fusion=bool(getattr(args, "use_adaptive_fusion", False)),
#             min_temperature=float(getattr(args, "min_temperature", 0.25)),
#             max_temperature=float(getattr(args, "max_temperature", 4.0)),
#             energy_normalize_by_dim=bool(getattr(args, "energy_normalize_by_dim", True)),
#             debias_strength=float(getattr(args, "debias_strength", 0.10)),
#             reliability_energy_weight=float(getattr(args, "reliability_energy_weight", 0.05)),
#             max_bias_abs=float(getattr(args, "max_classifier_bias_abs", 0.50)),
#             max_debias_abs=float(getattr(args, "max_classifier_debias_abs", 0.25)),
#         )

#     # =========================================================
#     # Basic helpers
#     # =========================================================
#     def _normalize(self, x: torch.Tensor) -> torch.Tensor:
#         return F.normalize(x, dim=-1, eps=1e-6)

#     def _project(self, feat: torch.Tensor) -> torch.Tensor:
#         # Preserve Euclidean geometry. No final L2 normalization.
#         return self.norm(self.projection(feat) + feat)

#     def _spectral_summary(self, x: torch.Tensor) -> torch.Tensor:
#         return x.mean(dim=(-1, -2))

#     def _resolve_classifier_mode(self, classifier_mode: Optional[str]) -> str:
#         if classifier_mode is None:
#             return (
#                 self.default_base_classifier_mode
#                 if int(self.current_phase) == 0
#                 else self.default_incremental_classifier_mode
#             )
#         return str(classifier_mode).lower()

#     def _resolve_semantic_mode(self, semantic_mode: Optional[str]) -> str:
#         """
#         Critical design choice:
#         default semantic mode is identity for all phases.

#         Do not use semantic_mode='all' in base and 'identity' in incremental,
#         because that builds geometry memory in one feature manifold and trains
#         incremental scoring in another.
#         """
#         if semantic_mode is None or str(semantic_mode).lower() == "auto":
#             return "identity"
#         return str(semantic_mode).lower()

#     def set_phase(self, phase: int):
#         self.current_phase = int(phase)

#     def set_old_class_count(self, old_class_count: int):
#         self.old_class_count = int(old_class_count)

#     # =========================================================
#     # Capacity / load preparation
#     # =========================================================
#     @torch.no_grad()
#     def ensure_class_capacity(
#         self,
#         class_count: int,
#         spectral_dim: int = 0,
#         dtype: Optional[torch.dtype] = None,
#     ):
#         class_count = int(class_count)
#         dtype = dtype or self.projection[0].weight.dtype

#         while len(self._anchors_cpu) < class_count:
#             zero_anchor = torch.zeros(self.d_model, dtype=dtype)
#             zero_concepts = torch.zeros(self.num_concepts_per_class, self.d_model, dtype=dtype)

#             self._anchors_cpu.append(zero_anchor.clone())
#             self._anchor_deltas.append(nn.Parameter(torch.zeros_like(zero_anchor, device=self.device)))

#             self._concepts_cpu.append(zero_concepts.clone())
#             self._concept_deltas.append(nn.Parameter(torch.zeros_like(zero_concepts, device=self.device)))

#         while self.classifier.num_classes < class_count:
#             self.classifier.expand(1, self.current_phase)

#         # Updated bank uses ensure_class_count(count=..., spectral_dim=...).
#         # Older variants may use ensure_num_classes.
#         if hasattr(self.geometry_bank, "ensure_class_count"):
#             self.geometry_bank.ensure_class_count(
#                 count=class_count,
#                 spectral_dim=int(spectral_dim),
#                 dtype=dtype,
#             )
#         elif hasattr(self.geometry_bank, "ensure_num_classes"):
#             self.geometry_bank.ensure_num_classes(class_count)

#         self.current_num_classes = max(self.current_num_classes, class_count)

#     # =========================================================
#     # Auxiliary semantic banks
#     # =========================================================
#     def get_anchor_bank(self) -> torch.Tensor:
#         if len(self._anchors_cpu) == 0:
#             return torch.empty((0, self.d_model), device=self.device)

#         return torch.stack(
#             [
#                 self._normalize(base.to(self.device) + self._anchor_deltas[i])
#                 for i, base in enumerate(self._anchors_cpu)
#             ],
#             dim=0,
#         )

#     def get_concept_bank(self) -> torch.Tensor:
#         if len(self._concepts_cpu) == 0:
#             return torch.empty((0, self.num_concepts_per_class, self.d_model), device=self.device)

#         return torch.stack(
#             [
#                 self._normalize(base.to(self.device) + self._concept_deltas[i])
#                 for i, base in enumerate(self._concepts_cpu)
#             ],
#             dim=0,
#         )

#     # =========================================================
#     # Geometry bank access
#     # =========================================================
#     def get_subspace_bank(self) -> Dict[str, torch.Tensor]:
#         bank = self.geometry_bank.get_bank()

#         # Defensive aliases for old/new bank implementations.
#         if "variances" not in bank:
#             if "eigvals" in bank and ("res_vars" in bank or "resvars" in bank):
#                 res = bank.get("res_vars", bank.get("resvars"))
#                 bank["variances"] = torch.cat([bank["eigvals"], res.unsqueeze(-1)], dim=-1)

#         if "resvars" not in bank and "res_vars" in bank:
#             bank["resvars"] = bank["res_vars"]
#         if "res_vars" not in bank and "resvars" in bank:
#             bank["res_vars"] = bank["resvars"]

#         C = bank["means"].size(0) if "means" in bank and torch.is_tensor(bank["means"]) else 0
#         if "reliability" not in bank:
#             bank["reliability"] = torch.ones(C, device=self.device)
#         if "active_ranks" not in bank:
#             bank["active_ranks"] = torch.full((C,), self.subspace_rank, device=self.device, dtype=torch.long)
#         if "sample_counts" not in bank:
#             bank["sample_counts"] = torch.zeros(C, device=self.device)
#         if "geometry_volumes" not in bank:
#             bank["geometry_volumes"] = torch.zeros(C, device=self.device)
#         if "class_dispersions" not in bank:
#             bank["class_dispersions"] = torch.zeros(C, device=self.device)
#         if "class_risk" not in bank:
#             bank["class_risk"] = torch.zeros(C, device=self.device)

#         return bank

#     def get_old_subspace_bank(self, old_class_count: Optional[int] = None) -> Dict[str, torch.Tensor]:
#         old_class_count = int(self.old_class_count if old_class_count is None else old_class_count)
#         bank = self.get_subspace_bank()

#         if old_class_count <= 0:
#             return {
#                 "means": torch.empty((0, self.d_model), device=self.device),
#                 "bases": torch.empty((0, self.d_model, self.subspace_rank), device=self.device),
#                 "variances": torch.empty((0, self.subspace_rank + 1), device=self.device),
#                 "reliability": torch.empty((0,), device=self.device),
#                 "active_ranks": torch.empty((0,), device=self.device, dtype=torch.long),
#             }

#         return {
#             "means": bank["means"][:old_class_count],
#             "bases": bank["bases"][:old_class_count],
#             "variances": bank["variances"][:old_class_count],
#             "reliability": bank.get("reliability", torch.ones(bank["means"].size(0), device=bank["means"].device))[:old_class_count],
#             "active_ranks": bank.get("active_ranks", torch.full((bank["means"].size(0),), self.subspace_rank, device=bank["means"].device, dtype=torch.long))[:old_class_count],
#         }

#     def get_new_subspace_bank(self, old_class_count: Optional[int] = None) -> Dict[str, torch.Tensor]:
#         old_class_count = int(self.old_class_count if old_class_count is None else old_class_count)
#         bank = self.get_subspace_bank()

#         return {
#             "means": bank["means"][old_class_count:],
#             "bases": bank["bases"][old_class_count:],
#             "variances": bank["variances"][old_class_count:],
#             "reliability": bank.get("reliability", torch.ones(bank["means"].size(0), device=bank["means"].device))[old_class_count:],
#             "active_ranks": bank.get("active_ranks", torch.full((bank["means"].size(0),), self.subspace_rank, device=bank["means"].device, dtype=torch.long))[old_class_count:],
#         }

#     def get_calibrated_old_subspace_bank(
#         self,
#         old_class_count: Optional[int] = None,
#     ) -> Dict[str, torch.Tensor]:
#         old_class_count = int(self.old_class_count if old_class_count is None else old_class_count)
#         raw_old = self.get_old_subspace_bank(old_class_count)

#         if old_class_count <= 0 or raw_old["means"].numel() == 0:
#             return raw_old

#         calibrated = self.geometry_calibrator(
#             raw_old["means"],
#             raw_old["bases"],
#             raw_old["variances"],
#             reliability=raw_old.get("reliability", None),
#         )

#         return {
#             "means": calibrated["means"],
#             "bases": calibrated["bases"],
#             "variances": calibrated["variances"],
#             "mean_delta": calibrated["mean_delta"],
#             "var_delta": calibrated["var_delta"],
#             "basis_delta": calibrated["basis_delta"],
#             "raw_means": raw_old["means"],
#             "raw_bases": raw_old["bases"],
#             "raw_variances": raw_old["variances"],
#             "reliability": raw_old.get("reliability", None),
#             "active_ranks": raw_old.get("active_ranks", None),
#         }

#     def calibration_regularization_loss(
#         self,
#         old_class_count: Optional[int] = None,
#     ) -> Dict[str, torch.Tensor]:
#         old_class_count = int(self.old_class_count if old_class_count is None else old_class_count)
#         raw_old = self.get_old_subspace_bank(old_class_count)

#         if old_class_count <= 0 or raw_old["means"].numel() == 0:
#             z = _zero_scalar(self.device)
#             return {"total": z, "mean": z, "basis": z, "var": z}

#         calibrated = self.get_calibrated_old_subspace_bank(old_class_count)

#         return self.geometry_calibrator.regularization_loss(
#             raw_means=raw_old["means"],
#             raw_bases=raw_old["bases"],
#             raw_variances=raw_old["variances"],
#             calibrated_means=calibrated["means"],
#             calibrated_bases=calibrated["bases"],
#             calibrated_variances=calibrated["variances"],
#         )

#     # =========================================================
#     # Snapshot helpers
#     # =========================================================
#     @torch.no_grad()
#     def export_memory_snapshot(self) -> Dict[str, Any]:
#         geometry_snap = self.geometry_bank.export_snapshot()

#         if len(self._anchors_cpu) > 0:
#             anchor_base = torch.stack([x.detach().cpu() for x in self._anchors_cpu], dim=0)
#             anchor_deltas = torch.stack([p.detach().cpu() for p in self._anchor_deltas], dim=0)
#         else:
#             anchor_base = torch.empty((0, self.d_model))
#             anchor_deltas = torch.empty((0, self.d_model))

#         if len(self._concepts_cpu) > 0:
#             concept_base = torch.stack([x.detach().cpu() for x in self._concepts_cpu], dim=0)
#             concept_deltas = torch.stack([p.detach().cpu() for p in self._concept_deltas], dim=0)
#         else:
#             concept_base = torch.empty((0, self.num_concepts_per_class, self.d_model))
#             concept_deltas = torch.empty((0, self.num_concepts_per_class, self.d_model))

#         snap = {
#             "current_num_classes": int(self.current_num_classes),
#             "old_class_count": int(self.old_class_count),
#             "current_phase": int(self.current_phase),
#             "anchor_base": anchor_base,
#             "anchor_deltas": anchor_deltas,
#             "concept_base": concept_base,
#             "concept_deltas": concept_deltas,
#         }
#         snap.update(geometry_snap)
#         return snap

#     @torch.no_grad()
#     def load_memory_snapshot(self, snapshot: Dict[str, Any], strict: bool = True):
#         if snapshot is None:
#             if strict:
#                 raise ValueError("snapshot is None")
#             return

#         means = snapshot.get("means", None)
#         class_count = int(means.size(0)) if torch.is_tensor(means) else int(snapshot.get("current_num_classes", 0))
#         spectral_dim = int(snapshot.get("spectral_dim", 0))

#         self.ensure_class_capacity(class_count=class_count, spectral_dim=spectral_dim)
#         self.geometry_bank.load_snapshot(snapshot, strict=strict)

#         self._anchors_cpu = []
#         self._anchor_deltas = nn.ParameterList()
#         self._concepts_cpu = []
#         self._concept_deltas = nn.ParameterList()

#         anchor_base = snapshot.get("anchor_base", None)
#         anchor_deltas = snapshot.get("anchor_deltas", None)
#         concept_base = snapshot.get("concept_base", None)
#         concept_deltas = snapshot.get("concept_deltas", None)

#         for cls in range(class_count):
#             a_base = anchor_base[cls].detach().cpu().clone() if torch.is_tensor(anchor_base) and anchor_base.size(0) > cls else torch.zeros(self.d_model)
#             a_delta = anchor_deltas[cls].detach().to(self.device).clone() if torch.is_tensor(anchor_deltas) and anchor_deltas.size(0) > cls else torch.zeros(self.d_model, device=self.device)
#             c_base = concept_base[cls].detach().cpu().clone() if torch.is_tensor(concept_base) and concept_base.size(0) > cls else torch.zeros(self.num_concepts_per_class, self.d_model)
#             c_delta = concept_deltas[cls].detach().to(self.device).clone() if torch.is_tensor(concept_deltas) and concept_deltas.size(0) > cls else torch.zeros(self.num_concepts_per_class, self.d_model, device=self.device)

#             self._anchors_cpu.append(a_base)
#             self._anchor_deltas.append(nn.Parameter(a_delta))
#             self._concepts_cpu.append(c_base)
#             self._concept_deltas.append(nn.Parameter(c_delta))

#         self.current_num_classes = class_count
#         self.old_class_count = int(snapshot.get("old_class_count", self.old_class_count))
#         self.current_phase = int(snapshot.get("current_phase", self.current_phase))

#     # =========================================================
#     # Backbone feature extraction
#     # =========================================================
#     def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
#         out = self.backbone(x)

#         if not isinstance(out, dict):
#             feat = out
#             bands = x.shape[1]
#             band_weights = torch.full((feat.size(0), bands), 1.0 / bands, device=feat.device, dtype=feat.dtype)
#             return {
#                 "features": feat,
#                 "backbone_features": feat,
#                 "band_weights": band_weights,
#                 "spectral_tokens": None,
#                 "spatial_tokens": None,
#                 "fused_tokens": None,
#                 "spectral_features": feat,
#                 "spatial_features": feat,
#                 "spatial_patterns": {},
#             }

#         feat = out["features"]
#         band_weights = out.get("band_weights")
#         if band_weights is None:
#             bands = x.shape[1]
#             band_weights = torch.full((feat.size(0), bands), 1.0 / bands, device=feat.device, dtype=feat.dtype)

#         return {
#             "features": feat,
#             "backbone_features": feat,
#             "band_weights": band_weights,
#             "spectral_tokens": out.get("spectral_tokens"),
#             "spatial_tokens": out.get("spatial_tokens"),
#             "fused_tokens": out.get("fused_tokens"),
#             "spectral_features": out.get("spectral_features", feat),
#             "spatial_features": out.get("spatial_features", feat),
#             "spatial_patterns": out.get("spatial_patterns", {}),
#         }

#     @torch.no_grad()
#     def extract_backbone_outputs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
#         out = self.extract_features(x)
#         raw_feat = out["features"]
#         projected_feat = self._project(raw_feat)

#         return {
#             "features": projected_feat,
#             "projected_features": projected_feat,
#             "preproject_features": raw_feat,
#             "backbone_features": raw_feat,
#             "band_weights": out["band_weights"],
#             "spectral_summary": self._spectral_summary(x),
#             "spectral_tokens": out["spectral_tokens"],
#             "spatial_tokens": out["spatial_tokens"],
#             "fused_tokens": out["fused_tokens"],
#             "spectral_features": out["spectral_features"],
#             "spatial_features": out["spatial_features"],
#             "spatial_patterns": out["spatial_patterns"],
#         }

#     def build_geometry_ref(self, feat: torch.Tensor):
#         if hasattr(self.geometry_bank, "retrieve_geometry_ref"):
#             return self.geometry_bank.retrieve_geometry_ref(feat)
#         return torch.zeros(feat.size(0), self.d_model, device=feat.device, dtype=feat.dtype)

#     # =========================================================
#     # Concept / anchor handling
#     # =========================================================
#     def _embed_concepts(self, concepts: torch.Tensor) -> torch.Tensor:
#         concepts = concepts.float().to(self.device)

#         if concepts.dim() != 2:
#             raise ValueError(f"concepts must be 2D, got {tuple(concepts.shape)}")

#         if concepts.size(1) == self.d_model:
#             embedded = concepts
#         elif concepts.size(1) == self.concept_projector.in_features:
#             embedded = self.concept_projector(concepts)
#         else:
#             raise ValueError(
#                 f"concept dim mismatch: expected {self.d_model} or {self.concept_projector.in_features}, got {concepts.size(1)}"
#             )

#         embedded = self._normalize(embedded)

#         k = embedded.size(0)
#         if k != self.num_concepts_per_class:
#             if k > self.num_concepts_per_class:
#                 embedded = embedded[: self.num_concepts_per_class]
#             else:
#                 pad = embedded[-1:].repeat(self.num_concepts_per_class - k, 1)
#                 embedded = torch.cat([embedded, pad], dim=0)

#         return embedded

#     @torch.no_grad()
#     def add_new_class_concepts(self, concepts: torch.Tensor):
#         cls = int(self.current_num_classes)
#         self.ensure_class_capacity(cls + 1)
#         self.refresh_class_concepts(cls, concepts, reset_delta=True)

#         anchor = self.get_anchor_bank()[cls].detach()
#         basis = torch.eye(self.d_model, self.subspace_rank, device=self.device, dtype=anchor.dtype)
#         eigvals = torch.full((self.subspace_rank,), 1e-4, device=self.device, dtype=anchor.dtype)
#         res_var = torch.tensor(1e-4, device=self.device, dtype=anchor.dtype)

#         self.refresh_class_subspace(
#             cls=cls,
#             mean=anchor,
#             basis=basis,
#             eigvals=eigvals,
#             res_var=res_var,
#             spectral_proto=None,
#             band_importance=None,
#             active_rank=torch.tensor(0, device=self.device, dtype=torch.long),
#             reliability=torch.tensor(0.05, device=self.device, dtype=anchor.dtype),
#         )

#         self.current_num_classes = cls + 1

#     @torch.no_grad()
#     def refresh_class_concepts(
#         self,
#         cls: int,
#         concepts: torch.Tensor,
#         reset_delta: bool = True,
#     ):
#         cls = int(cls)
#         self.ensure_class_capacity(cls + 1)

#         embedded = self._embed_concepts(concepts)
#         anchor = self._normalize(embedded.mean(dim=0))

#         self._anchors_cpu[cls] = anchor.detach().cpu()
#         self._concepts_cpu[cls] = embedded.detach().cpu()

#         if reset_delta:
#             self._anchor_deltas[cls].data.zero_()
#             self._concept_deltas[cls].data.zero_()

#         self.current_num_classes = max(self.current_num_classes, cls + 1)

#     @torch.no_grad()
#     def refresh_class_subspace(
#         self,
#         cls: int,
#         mean: torch.Tensor,
#         basis: torch.Tensor,
#         eigvals: torch.Tensor,
#         res_var,
#         spectral_proto=None,
#         band_importance=None,
#         active_rank=None,
#         reliability=None,
#     ):
#         cls = int(cls)

#         spectral_dim = 0
#         if spectral_proto is not None and torch.as_tensor(spectral_proto).numel() > 0:
#             spectral_dim = int(torch.as_tensor(spectral_proto).numel())
#         elif band_importance is not None and torch.as_tensor(band_importance).numel() > 0:
#             spectral_dim = int(torch.as_tensor(band_importance).numel())

#         self.ensure_class_capacity(cls + 1, spectral_dim=spectral_dim)

#         if hasattr(self.geometry_bank, "update_class_geometry"):
#             self.geometry_bank.update_class_geometry(
#                 class_id=cls,
#                 mean=mean.float().to(self.device),
#                 basis=basis.float().to(self.device),
#                 eigvals=eigvals.float().to(self.device),
#                 resvar=torch.as_tensor(res_var, device=self.device, dtype=mean.dtype),
#                 spectral_proto=spectral_proto,
#                 band_importance=band_importance,
#                 active_rank=active_rank,
#                 reliability=reliability,
#             )
#         else:
#             self.geometry_bank.update_class(
#                 cls_id=cls,
#                 mean=mean.float().to(self.device),
#                 basis=basis.float().to(self.device),
#                 eigvals=eigvals.float().to(self.device),
#                 res_var=torch.as_tensor(res_var, device=self.device, dtype=mean.dtype),
#                 spectral_proto=spectral_proto,
#                 band_importance=band_importance,
#             )

#             if active_rank is not None and hasattr(self.geometry_bank, "active_ranks"):
#                 self.geometry_bank.active_ranks[cls] = active_rank.to(self.device)
#             if reliability is not None and hasattr(self.geometry_bank, "reliability"):
#                 self.geometry_bank.reliability[cls] = reliability.to(self.device)

#         self.current_num_classes = max(self.current_num_classes, cls + 1)

#     @torch.no_grad()
#     def refresh_inter_class_geometry(self):
#         if hasattr(self.geometry_bank, "refresh_inter_class_geometry"):
#             self.geometry_bank.refresh_inter_class_geometry()

#     # =========================================================
#     # Logit computation
#     # =========================================================
#     def compute_logits_from_features(
#         self,
#         features: torch.Tensor,
#         classifier_mode: str = "geometry_only",
#     ):
#         classifier_mode = str(classifier_mode).lower()
#         subspace_bank = self.get_subspace_bank()
#         anchors = self.get_anchor_bank()
#         concepts = self.get_concept_bank()
#         calibrated_old = self.get_calibrated_old_subspace_bank(self.old_class_count)

#         return self.classifier(
#             features,
#             anchors=anchors if anchors.numel() > 0 else None,
#             concept_bank=concepts if concepts.numel() > 0 else None,
#             subspace_means=subspace_bank["means"] if subspace_bank["means"].numel() > 0 else None,
#             subspace_bases=subspace_bank["bases"] if subspace_bank["bases"].numel() > 0 else None,
#             subspace_variances=subspace_bank["variances"] if subspace_bank["variances"].numel() > 0 else None,
#             subspace_reliability=subspace_bank.get("reliability", None),
#             subspace_active_ranks=subspace_bank.get("active_ranks", None),
#             calibrated_old_means=calibrated_old["means"] if calibrated_old.get("means", None) is not None and calibrated_old["means"].numel() > 0 else None,
#             calibrated_old_bases=calibrated_old["bases"] if calibrated_old.get("bases", None) is not None and calibrated_old["bases"].numel() > 0 else None,
#             calibrated_old_variances=calibrated_old["variances"] if calibrated_old.get("variances", None) is not None and calibrated_old["variances"].numel() > 0 else None,
#             calibrated_old_reliability=calibrated_old.get("reliability", None),
#             calibrated_old_active_ranks=calibrated_old.get("active_ranks", None),
#             mode=classifier_mode,
#             old_class_count=int(self.old_class_count),
#         )

#     # =========================================================
#     # Trainability controls
#     # =========================================================
#     def freeze_backbone_only(self):
#         for p in self.backbone.parameters():
#             p.requires_grad = False

#     def freeze_semantic_encoder(self):
#         for p in self.semantic_encoder.parameters():
#             p.requires_grad = False

#     def unfreeze_semantic_encoder(self):
#         for p in self.semantic_encoder.parameters():
#             p.requires_grad = True

#     def freeze_projection_head(self):
#         for p in self.projection.parameters():
#             p.requires_grad = False
#         for p in self.norm.parameters():
#             p.requires_grad = False
#         for p in self.concept_projector.parameters():
#             p.requires_grad = False

#     def unfreeze_projection_head(self):
#         for p in self.projection.parameters():
#             p.requires_grad = True
#         for p in self.norm.parameters():
#             p.requires_grad = True
#         for p in self.concept_projector.parameters():
#             p.requires_grad = True

#     def freeze_old_anchor_deltas(self, old_class_count: int):
#         for i, p in enumerate(self._anchor_deltas):
#             if i < int(old_class_count):
#                 p.requires_grad = False

#     def unfreeze_new_anchor_deltas(self, old_class_count: int):
#         for i, p in enumerate(self._anchor_deltas):
#             p.requires_grad = i >= int(old_class_count)

#     def freeze_old_concept_deltas(self, old_class_count: int):
#         for i, p in enumerate(self._concept_deltas):
#             if i < int(old_class_count):
#                 p.requires_grad = False

#     def unfreeze_new_concept_deltas(self, old_class_count: int):
#         for i, p in enumerate(self._concept_deltas):
#             p.requires_grad = i >= int(old_class_count)

#     def freeze_classifier_adaptation(self):
#         if hasattr(self.classifier, "freeze_all_adaptation"):
#             self.classifier.freeze_all_adaptation()

#     def freeze_old_classifier_adaptation(self, old_class_count: int):
#         if hasattr(self.classifier, "freeze_old_adaptation"):
#             self.classifier.freeze_old_adaptation(old_class_count)

#     def unfreeze_classifier_adaptation(self):
#         if hasattr(self.classifier, "unfreeze_all_adaptation"):
#             self.classifier.unfreeze_all_adaptation()

#     def freeze_fusion_module(self):
#         if hasattr(self.classifier, "freeze_fusion_module"):
#             self.classifier.freeze_fusion_module()

#     def unfreeze_fusion_module(self):
#         if hasattr(self.classifier, "unfreeze_fusion_module"):
#             self.classifier.unfreeze_fusion_module()

#     def freeze_geometry_calibrator(self):
#         for p in self.geometry_calibrator.parameters():
#             p.requires_grad = False

#     def unfreeze_geometry_calibrator(self):
#         for p in self.geometry_calibrator.parameters():
#             p.requires_grad = True

#     # =========================================================
#     # Semantic refinement helper
#     # =========================================================
#     def _semantic_refine(
#         self,
#         feat: torch.Tensor,
#         anchors: torch.Tensor,
#         concepts: torch.Tensor,
#         spectral_tokens,
#         spatial_tokens,
#         semantic_mode: str,
#         return_token_relations: bool,
#     ):
#         bypass_semantic = semantic_mode in {"off", "none", "bypass", "identity", "raw"}
#         if anchors.numel() == 0 or bypass_semantic:
#             return feat, None

#         semantic_bank = concepts if concepts.numel() > 0 else anchors.unsqueeze(1)

#         semantic_out = self.semantic_encoder(
#             feat,
#             concept_bank=semantic_bank,
#             phase=int(self.current_phase),
#             disable=False,
#             spectral_tokens=spectral_tokens,
#             spatial_tokens=spatial_tokens,
#             return_token_relations=return_token_relations,
#         )

#         if isinstance(semantic_out, dict):
#             feat_refined = semantic_out["features"]
#             token_relations = semantic_out.get("token_relations", None)
#         else:
#             feat_refined = semantic_out
#             token_relations = None

#         return feat_refined, token_relations

#     # =========================================================
#     # Forward
#     # =========================================================
#     def forward(self, x: torch.Tensor, **kwargs):
#         semantic_mode = self._resolve_semantic_mode(kwargs.get("semantic_mode", "auto"))
#         classifier_mode = self._resolve_classifier_mode(kwargs.get("classifier_mode", None))
#         return_token_relations = bool(kwargs.get("return_token_relations", False))

#         backbone_out = self.extract_features(x)

#         feat = backbone_out["features"]
#         band_weights = backbone_out["band_weights"]
#         spectral_tokens = backbone_out["spectral_tokens"]
#         spatial_tokens = backbone_out["spatial_tokens"]
#         fused_tokens = backbone_out["fused_tokens"]
#         spectral_summary = self._spectral_summary(x)

#         anchors = self.get_anchor_bank()
#         concepts = self.get_concept_bank()

#         feat_pre, token_relations = self._semantic_refine(
#             feat=feat,
#             anchors=anchors,
#             concepts=concepts,
#             spectral_tokens=spectral_tokens,
#             spatial_tokens=spatial_tokens,
#             semantic_mode=semantic_mode,
#             return_token_relations=return_token_relations,
#         )

#         if (
#             return_token_relations
#             and token_relations is None
#             and spectral_tokens is not None
#             and spatial_tokens is not None
#         ):
#             token_relations = self.semantic_encoder.build_token_relations(
#                 spectral_tokens,
#                 spatial_tokens,
#             )

#         projected = self._project(feat_pre)

#         subspace = self.get_subspace_bank()
#         calibrated_old = self.get_calibrated_old_subspace_bank(self.old_class_count)

#         geometry_ref = self.build_geometry_ref(projected)

#         if hasattr(self.geometry_bank, "retrieve_spectral_ref"):
#             spectral_ref = self.geometry_bank.retrieve_spectral_ref(projected)
#         else:
#             spectral_ref = torch.empty(projected.size(0), 0, device=projected.device, dtype=projected.dtype)

#         if hasattr(self.geometry_bank, "retrieve_band_importance_ref"):
#             band_importance_ref = self.geometry_bank.retrieve_band_importance_ref(projected)
#         else:
#             band_importance_ref = torch.empty(projected.size(0), 0, device=projected.device, dtype=projected.dtype)

#         logits = self.compute_logits_from_features(
#             projected,
#             classifier_mode=classifier_mode,
#         )

#         calibration_reg = self.calibration_regularization_loss(self.old_class_count)

#         return {
#             "logits": logits,
#             "features": projected,
#             "preproject_features": feat_pre,
#             "backbone_features": feat,
#             "band_weights": band_weights,
#             "spectral_summary": spectral_summary,
#             "spectral_ref": spectral_ref,
#             "band_importance_ref": band_importance_ref,
#             "geometry_ref": geometry_ref,
#             "anchors": anchors,
#             "concept_bank": concepts,
#             "subspace_means": subspace["means"],
#             "subspace_bases": subspace["bases"],
#             "subspace_variances": subspace["variances"],
#             "subspace_reliability": subspace.get("reliability", None),
#             "subspace_active_ranks": subspace.get("active_ranks", None),
#             "subspace_sample_counts": subspace.get("sample_counts", None),
#             "subspace_geometry_volumes": subspace.get("geometry_volumes", None),
#             "subspace_class_dispersions": subspace.get("class_dispersions", None),
#             "subspace_class_risk": subspace.get("class_risk", None),
#             "calibrated_old_means": calibrated_old.get("means", None),
#             "calibrated_old_bases": calibrated_old.get("bases", None),
#             "calibrated_old_variances": calibrated_old.get("variances", None),
#             "calibrated_old_reliability": calibrated_old.get("reliability", None),
#             "calibration_reg": calibration_reg,
#             "spectral_tokens": spectral_tokens,
#             "spatial_tokens": spatial_tokens,
#             "fused_tokens": fused_tokens,
#             "token_relations": token_relations,
#             "spectral_features": backbone_out["spectral_features"],
#             "spatial_features": backbone_out["spatial_features"],
#             "spatial_patterns": backbone_out["spatial_patterns"],
#         }






# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Dict, Optional, Any

# from models.backbone import SSMBackbone
# from models.token import HSISemanticConceptEncoder
# from models.classifier import SemanticClassifier
# from models.geometry_bank import GeometryBank


# def _zero_scalar(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
#     return torch.tensor(0.0, device=device, dtype=dtype)


# def _projector_from_basis(basis: torch.Tensor) -> torch.Tensor:
#     if basis.dim() == 2:
#         return basis @ basis.t()
#     return torch.matmul(basis, basis.transpose(-1, -2))


# def _orthonormalize_batched_basis(basis: torch.Tensor, rank: int) -> torch.Tensor:
#     """
#     Orthonormalize basis columns.

#     Args:
#         basis: [C, D, R]
#         rank: target rank

#     Returns:
#         [C, D, R] with orthonormal columns.
#     """
#     if basis is None or basis.numel() == 0:
#         return basis

#     outs = []
#     for b in basis:
#         q, _ = torch.linalg.qr(b, mode="reduced")
#         if q.size(1) < rank:
#             pad = torch.zeros(
#                 q.size(0),
#                 rank - q.size(1),
#                 device=q.device,
#                 dtype=q.dtype,
#             )
#             q = torch.cat([q, pad], dim=1)
#         outs.append(q[:, :rank])
#     return torch.stack(outs, dim=0)


# class GeometryCalibrator(nn.Module):
#     """
#     Conservative geometry transport calibration for old classes.

#     The calibrator is intentionally small. It corrects old geometry drift without
#     letting the old class memory become a free classifier.

#     Calibrates:
#         mean       : mu -> mu + delta_mu
#         variances  : log(v) -> log(v) + delta_logv
#         basis      : optional projector-stable basis correction

#     By default, basis calibration is disabled because it is easier to destabilize
#     than mean/variance calibration. Enable it only after the geometry-only system
#     is stable.
#     """

#     def __init__(
#         self,
#         d_model: int,
#         rank: int,
#         hidden_dim: Optional[int] = None,
#         var_floor: float = 1e-4,
#         dropout: float = 0.1,
#         calibrate_basis: bool = False,
#         max_mean_scale: float = 0.10,
#         max_var_scale: float = 0.10,
#         max_basis_scale: float = 0.03,
#     ):
#         super().__init__()
#         self.d_model = int(d_model)
#         self.rank = int(rank)
#         self.var_floor = float(var_floor)
#         self.calibrate_basis = bool(calibrate_basis)

#         hidden_dim = int(hidden_dim or max(d_model, 128))

#         # Context is the old class mean. Keep this lightweight; do not overfit.
#         self.mean_calibrator = nn.Sequential(
#             nn.Linear(self.d_model, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, self.d_model),
#         )

#         self.var_calibrator = nn.Sequential(
#             nn.Linear(self.d_model, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, self.rank + 1),
#         )

#         if self.calibrate_basis:
#             self.basis_calibrator = nn.Sequential(
#                 nn.Linear(self.d_model, hidden_dim),
#                 nn.GELU(),
#                 nn.Dropout(dropout),
#                 nn.Linear(hidden_dim, self.d_model * self.rank),
#             )
#         else:
#             self.basis_calibrator = None

#         # Raw parameters are squashed. This prevents calibration from exploding.
#         self.mean_scale_raw = nn.Parameter(torch.tensor(-3.0))
#         self.var_scale_raw = nn.Parameter(torch.tensor(-3.0))
#         self.basis_scale_raw = nn.Parameter(torch.tensor(-4.0))

#         self.max_mean_scale = float(max_mean_scale)
#         self.max_var_scale = float(max_var_scale)
#         self.max_basis_scale = float(max_basis_scale)

#     def _mean_scale(self) -> torch.Tensor:
#         return self.max_mean_scale * torch.sigmoid(self.mean_scale_raw)

#     def _var_scale(self) -> torch.Tensor:
#         return self.max_var_scale * torch.sigmoid(self.var_scale_raw)

#     def _basis_scale(self) -> torch.Tensor:
#         return self.max_basis_scale * torch.sigmoid(self.basis_scale_raw)

#     def forward(
#         self,
#         means: torch.Tensor,
#         bases: torch.Tensor,
#         variances: torch.Tensor,
#         reliability: Optional[torch.Tensor] = None,
#     ) -> Dict[str, torch.Tensor]:
#         if means is None or means.numel() == 0:
#             return {
#                 "means": means,
#                 "bases": bases,
#                 "variances": variances,
#                 "mean_delta": means,
#                 "var_delta": variances,
#                 "basis_delta": bases,
#             }

#         if reliability is None or not torch.is_tensor(reliability) or reliability.numel() == 0:
#             rel = torch.ones(means.size(0), 1, device=means.device, dtype=means.dtype)
#         else:
#             rel = reliability.to(device=means.device, dtype=means.dtype).view(-1, 1).clamp(0.05, 1.0)

#         # Less reliable geometry gets smaller learned transport.
#         mean_delta = self._mean_scale() * rel * self.mean_calibrator(means)
#         calibrated_means = means + mean_delta

#         logv = torch.log(variances.clamp_min(self.var_floor))
#         var_delta = self._var_scale() * rel * self.var_calibrator(means)
#         calibrated_logv = logv + var_delta
#         calibrated_variances = torch.exp(calibrated_logv).clamp_min(self.var_floor)

#         if self.calibrate_basis and self.basis_calibrator is not None and bases is not None and bases.numel() > 0:
#             raw_delta = self.basis_calibrator(means).view(-1, self.d_model, self.rank)
#             basis_delta = self._basis_scale() * rel.view(-1, 1, 1) * raw_delta
#             calibrated_bases = _orthonormalize_batched_basis(bases + basis_delta, self.rank)
#         else:
#             basis_delta = torch.zeros_like(bases)
#             calibrated_bases = bases

#         return {
#             "means": calibrated_means,
#             "bases": calibrated_bases,
#             "variances": calibrated_variances,
#             "mean_delta": mean_delta,
#             "var_delta": var_delta,
#             "basis_delta": basis_delta,
#         }

#     def regularization_loss(
#         self,
#         raw_means: Optional[torch.Tensor],
#         raw_bases: Optional[torch.Tensor],
#         raw_variances: Optional[torch.Tensor],
#         calibrated_means: Optional[torch.Tensor],
#         calibrated_bases: Optional[torch.Tensor],
#         calibrated_variances: Optional[torch.Tensor],
#     ) -> Dict[str, torch.Tensor]:
#         if (
#             raw_means is None
#             or raw_variances is None
#             or calibrated_means is None
#             or calibrated_variances is None
#             or raw_means.numel() == 0
#         ):
#             device = raw_means.device if torch.is_tensor(raw_means) else torch.device("cpu")
#             z = _zero_scalar(device)
#             return {"total": z, "mean": z, "basis": z, "var": z}

#         mean_reg = F.mse_loss(calibrated_means, raw_means)

#         raw_logv = torch.log(raw_variances.clamp_min(self.var_floor))
#         cal_logv = torch.log(calibrated_variances.clamp_min(self.var_floor))
#         var_reg = F.mse_loss(cal_logv, raw_logv)

#         if (
#             raw_bases is not None
#             and calibrated_bases is not None
#             and torch.is_tensor(raw_bases)
#             and torch.is_tensor(calibrated_bases)
#             and raw_bases.numel() > 0
#             and calibrated_bases.numel() > 0
#         ):
#             basis_reg = F.mse_loss(
#                 _projector_from_basis(calibrated_bases),
#                 _projector_from_basis(raw_bases),
#             )
#         else:
#             basis_reg = _zero_scalar(raw_means.device, raw_means.dtype)

#         basis_weight = 0.2 if self.calibrate_basis else 0.0
#         total = mean_reg + var_reg + basis_weight * basis_reg
#         return {"total": total, "mean": mean_reg, "basis": basis_reg, "var": var_reg}


# class NECILModel(nn.Module):
#     """
#     Geometry-centric NECIL-HSI model.

#     Core policy:
#         - Backbone produces spectral-spatial features/tokens.
#         - Projection preserves Euclidean feature scale.
#         - GeometryBank is the real non-exemplar memory.
#         - Classifier scores by geometry energy.
#         - Semantic concepts are auxiliary and should not rewrite geometry during
#           incremental phases unless explicitly enabled.
#     """

#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         self.device = torch.device(args.device)

#         self.d_model = int(args.d_model)
#         self.subspace_rank = int(getattr(args, "subspace_rank", 5))
#         self.num_concepts_per_class = int(getattr(args, "num_concepts_per_class", 4))

#         self.default_base_classifier_mode = str(
#             getattr(args, "base_classifier_mode", "geometry_only")
#         ).lower()

#         self.default_incremental_classifier_mode = str(
#             getattr(args, "incremental_classifier_mode", "calibrated_geometry")
#         ).lower()

#         self.current_num_classes = 0
#         self.old_class_count = 0
#         self.current_phase = 0

#         # Auxiliary semantic memory. This is NOT the main classifier memory.
#         self._anchors_cpu = []
#         self._anchor_deltas = nn.ParameterList()
#         self._concepts_cpu = []
#         self._concept_deltas = nn.ParameterList()

#         self.backbone = SSMBackbone(args)

#         self.semantic_encoder = HSISemanticConceptEncoder(
#             feature_dim=self.d_model,
#             concept_dim=self.d_model,
#             dropout=float(getattr(args, "semantic_dropout", 0.1)),
#             token_temperature=float(getattr(args, "token_temperature", 0.07)),
#         )

#         self.projection = nn.Sequential(
#             nn.Linear(self.d_model, self.d_model),
#             nn.GELU(),
#             nn.Dropout(float(getattr(args, "projection_dropout", 0.1))),
#             nn.Linear(self.d_model, self.d_model),
#         )
#         self.norm = nn.LayerNorm(self.d_model)

#         self.concept_projector = nn.Linear(int(args.num_bands), self.d_model, bias=True)

#         self.geometry_bank = GeometryBank(
#             self.d_model,
#             self.subspace_rank,
#             device=args.device,
#         )

#         self.geometry_calibrator = GeometryCalibrator(
#             d_model=self.d_model,
#             rank=self.subspace_rank,
#             hidden_dim=int(getattr(args, "geometry_calibration_hidden_dim", self.d_model)),
#             var_floor=float(getattr(args, "geom_var_floor", 1e-4)),
#             dropout=float(getattr(args, "geometry_calibration_dropout", 0.1)),
#             calibrate_basis=bool(getattr(args, "geometry_calibrate_basis", False)),
#             max_mean_scale=float(getattr(args, "geometry_max_mean_scale", 0.10)),
#             max_var_scale=float(getattr(args, "geometry_max_var_scale", 0.10)),
#             max_basis_scale=float(getattr(args, "geometry_max_basis_scale", 0.03)),
#         )

#         self.classifier = SemanticClassifier(
#             initial_classes=0,
#             d_model=self.d_model,
#             logit_scale=float(getattr(args, "loss_scale", 8.0)),
#             use_bias=bool(getattr(args, "classifier_use_bias", True)),
#             variance_floor=float(getattr(args, "geom_var_floor", 1e-4)),
#             use_geom_temperature=bool(getattr(args, "use_geom_temperature", True)),
#             concept_agg_temperature=float(getattr(args, "cls_temperature", 0.07)),
#             init_alpha_old=float(getattr(args, "init_alpha_old", -0.5)),
#             init_alpha_new=float(getattr(args, "init_alpha_new", -0.2)),
#             # Keep legacy fusion off by default. Geometry should carry the method.
#             use_adaptive_fusion=bool(getattr(args, "use_adaptive_fusion", False)),
#             min_temperature=float(getattr(args, "min_temperature", 0.25)),
#             max_temperature=float(getattr(args, "max_temperature", 4.0)),
#             energy_normalize_by_dim=bool(getattr(args, "energy_normalize_by_dim", True)),
#             debias_strength=float(getattr(args, "debias_strength", 0.10)),
#             reliability_energy_weight=float(getattr(args, "reliability_energy_weight", 0.05)),
#             volume_energy_weight=float(getattr(args, "volume_energy_weight", 0.0)),
#             max_classifier_bias_abs=float(getattr(args, "max_classifier_bias_abs", 0.0)),
#             max_classifier_debias_abs=float(getattr(args, "max_classifier_debias_abs", 0.0)),
#             enable_old_new_temp_offsets=bool(getattr(args, "enable_old_new_temp_offsets", True)),
#             geometry_logit_clip=float(getattr(args, "classifier_geometry_logit_clip", 0.0)),
#         )

#     # =========================================================
#     # Basic helpers
#     # =========================================================
#     def _normalize(self, x: torch.Tensor) -> torch.Tensor:
#         return F.normalize(x, dim=-1, eps=1e-6)

#     def _project(self, feat: torch.Tensor) -> torch.Tensor:
#         # Preserve Euclidean geometry. No final L2 normalization.
#         return self.norm(self.projection(feat) + feat)

#     def _spectral_summary(self, x: torch.Tensor) -> torch.Tensor:
#         return x.mean(dim=(-1, -2))

#     def _resolve_classifier_mode(self, classifier_mode: Optional[str]) -> str:
#         if classifier_mode is None:
#             return (
#                 self.default_base_classifier_mode
#                 if int(self.current_phase) == 0
#                 else self.default_incremental_classifier_mode
#             )
#         return str(classifier_mode).lower()

#     def _resolve_semantic_mode(self, semantic_mode: Optional[str]) -> str:
#         """
#         Critical design choice:
#         default semantic mode is identity for all phases.

#         Do not use semantic_mode='all' in base and 'identity' in incremental,
#         because that builds geometry memory in one feature manifold and trains
#         incremental scoring in another.
#         """
#         if semantic_mode is None or str(semantic_mode).lower() == "auto":
#             return "identity"
#         return str(semantic_mode).lower()

#     def set_phase(self, phase: int):
#         self.current_phase = int(phase)

#     def set_old_class_count(self, old_class_count: int):
#         self.old_class_count = int(old_class_count)

#     # =========================================================
#     # Capacity / load preparation
#     # =========================================================
#     @torch.no_grad()
#     def ensure_class_capacity(
#         self,
#         class_count: int,
#         spectral_dim: int = 0,
#         dtype: Optional[torch.dtype] = None,
#     ):
#         class_count = int(class_count)
#         dtype = dtype or self.projection[0].weight.dtype

#         while len(self._anchors_cpu) < class_count:
#             zero_anchor = torch.zeros(self.d_model, dtype=dtype)
#             zero_concepts = torch.zeros(self.num_concepts_per_class, self.d_model, dtype=dtype)

#             self._anchors_cpu.append(zero_anchor.clone())
#             self._anchor_deltas.append(nn.Parameter(torch.zeros_like(zero_anchor, device=self.device)))

#             self._concepts_cpu.append(zero_concepts.clone())
#             self._concept_deltas.append(nn.Parameter(torch.zeros_like(zero_concepts, device=self.device)))

#         while self.classifier.num_classes < class_count:
#             self.classifier.expand(1, self.current_phase)

#         # Updated bank uses ensure_class_count(count=..., spectral_dim=...).
#         # Older variants may use ensure_num_classes.
#         if hasattr(self.geometry_bank, "ensure_class_count"):
#             self.geometry_bank.ensure_class_count(
#                 count=class_count,
#                 spectral_dim=int(spectral_dim),
#                 dtype=dtype,
#             )
#         elif hasattr(self.geometry_bank, "ensure_num_classes"):
#             self.geometry_bank.ensure_num_classes(class_count)

#         self.current_num_classes = max(self.current_num_classes, class_count)

#     # =========================================================
#     # Auxiliary semantic banks
#     # =========================================================
#     def get_anchor_bank(self) -> torch.Tensor:
#         if len(self._anchors_cpu) == 0:
#             return torch.empty((0, self.d_model), device=self.device)

#         return torch.stack(
#             [
#                 self._normalize(base.to(self.device) + self._anchor_deltas[i])
#                 for i, base in enumerate(self._anchors_cpu)
#             ],
#             dim=0,
#         )

#     def get_concept_bank(self) -> torch.Tensor:
#         if len(self._concepts_cpu) == 0:
#             return torch.empty((0, self.num_concepts_per_class, self.d_model), device=self.device)

#         return torch.stack(
#             [
#                 self._normalize(base.to(self.device) + self._concept_deltas[i])
#                 for i, base in enumerate(self._concepts_cpu)
#             ],
#             dim=0,
#         )

#     # =========================================================
#     # Geometry bank access
#     # =========================================================
#     def get_subspace_bank(self) -> Dict[str, torch.Tensor]:
#         bank = self.geometry_bank.get_bank()

#         # Defensive aliases for old/new bank implementations.
#         if "variances" not in bank:
#             if "eigvals" in bank and ("res_vars" in bank or "resvars" in bank):
#                 res = bank.get("res_vars", bank.get("resvars"))
#                 bank["variances"] = torch.cat([bank["eigvals"], res.unsqueeze(-1)], dim=-1)

#         if "resvars" not in bank and "res_vars" in bank:
#             bank["resvars"] = bank["res_vars"]
#         if "res_vars" not in bank and "resvars" in bank:
#             bank["res_vars"] = bank["resvars"]

#         C = bank["means"].size(0) if "means" in bank and torch.is_tensor(bank["means"]) else 0
#         if "reliability" not in bank:
#             bank["reliability"] = torch.ones(C, device=self.device)
#         if "active_ranks" not in bank:
#             bank["active_ranks"] = torch.full((C,), self.subspace_rank, device=self.device, dtype=torch.long)

#         return bank

#     def get_old_subspace_bank(self, old_class_count: Optional[int] = None) -> Dict[str, torch.Tensor]:
#         old_class_count = int(self.old_class_count if old_class_count is None else old_class_count)
#         bank = self.get_subspace_bank()

#         if old_class_count <= 0:
#             return {
#                 "means": torch.empty((0, self.d_model), device=self.device),
#                 "bases": torch.empty((0, self.d_model, self.subspace_rank), device=self.device),
#                 "variances": torch.empty((0, self.subspace_rank + 1), device=self.device),
#                 "reliability": torch.empty((0,), device=self.device),
#                 "active_ranks": torch.empty((0,), device=self.device, dtype=torch.long),
#             }

#         return {
#             "means": bank["means"][:old_class_count],
#             "bases": bank["bases"][:old_class_count],
#             "variances": bank["variances"][:old_class_count],
#             "reliability": bank.get("reliability", torch.ones(bank["means"].size(0), device=bank["means"].device))[:old_class_count],
#             "active_ranks": bank.get("active_ranks", torch.full((bank["means"].size(0),), self.subspace_rank, device=bank["means"].device, dtype=torch.long))[:old_class_count],
#         }

#     def get_new_subspace_bank(self, old_class_count: Optional[int] = None) -> Dict[str, torch.Tensor]:
#         old_class_count = int(self.old_class_count if old_class_count is None else old_class_count)
#         bank = self.get_subspace_bank()

#         return {
#             "means": bank["means"][old_class_count:],
#             "bases": bank["bases"][old_class_count:],
#             "variances": bank["variances"][old_class_count:],
#             "reliability": bank.get("reliability", torch.ones(bank["means"].size(0), device=bank["means"].device))[old_class_count:],
#             "active_ranks": bank.get("active_ranks", torch.full((bank["means"].size(0),), self.subspace_rank, device=bank["means"].device, dtype=torch.long))[old_class_count:],
#         }

#     def get_calibrated_old_subspace_bank(
#         self,
#         old_class_count: Optional[int] = None,
#     ) -> Dict[str, torch.Tensor]:
#         old_class_count = int(self.old_class_count if old_class_count is None else old_class_count)
#         raw_old = self.get_old_subspace_bank(old_class_count)

#         if old_class_count <= 0 or raw_old["means"].numel() == 0:
#             return raw_old

#         calibrated = self.geometry_calibrator(
#             raw_old["means"],
#             raw_old["bases"],
#             raw_old["variances"],
#             reliability=raw_old.get("reliability", None),
#         )

#         return {
#             "means": calibrated["means"],
#             "bases": calibrated["bases"],
#             "variances": calibrated["variances"],
#             "mean_delta": calibrated["mean_delta"],
#             "var_delta": calibrated["var_delta"],
#             "basis_delta": calibrated["basis_delta"],
#             "raw_means": raw_old["means"],
#             "raw_bases": raw_old["bases"],
#             "raw_variances": raw_old["variances"],
#             "reliability": raw_old.get("reliability", None),
#             "active_ranks": raw_old.get("active_ranks", None),
#         }

#     def calibration_regularization_loss(
#         self,
#         old_class_count: Optional[int] = None,
#     ) -> Dict[str, torch.Tensor]:
#         old_class_count = int(self.old_class_count if old_class_count is None else old_class_count)
#         raw_old = self.get_old_subspace_bank(old_class_count)

#         if old_class_count <= 0 or raw_old["means"].numel() == 0:
#             z = _zero_scalar(self.device)
#             return {"total": z, "mean": z, "basis": z, "var": z}

#         calibrated = self.get_calibrated_old_subspace_bank(old_class_count)

#         return self.geometry_calibrator.regularization_loss(
#             raw_means=raw_old["means"],
#             raw_bases=raw_old["bases"],
#             raw_variances=raw_old["variances"],
#             calibrated_means=calibrated["means"],
#             calibrated_bases=calibrated["bases"],
#             calibrated_variances=calibrated["variances"],
#         )

#     # =========================================================
#     # Snapshot helpers
#     # =========================================================
#     @torch.no_grad()
#     def export_memory_snapshot(self) -> Dict[str, Any]:
#         geometry_snap = self.geometry_bank.export_snapshot()

#         if len(self._anchors_cpu) > 0:
#             anchor_base = torch.stack([x.detach().cpu() for x in self._anchors_cpu], dim=0)
#             anchor_deltas = torch.stack([p.detach().cpu() for p in self._anchor_deltas], dim=0)
#         else:
#             anchor_base = torch.empty((0, self.d_model))
#             anchor_deltas = torch.empty((0, self.d_model))

#         if len(self._concepts_cpu) > 0:
#             concept_base = torch.stack([x.detach().cpu() for x in self._concepts_cpu], dim=0)
#             concept_deltas = torch.stack([p.detach().cpu() for p in self._concept_deltas], dim=0)
#         else:
#             concept_base = torch.empty((0, self.num_concepts_per_class, self.d_model))
#             concept_deltas = torch.empty((0, self.num_concepts_per_class, self.d_model))

#         snap = {
#             "current_num_classes": int(self.current_num_classes),
#             "old_class_count": int(self.old_class_count),
#             "current_phase": int(self.current_phase),
#             "anchor_base": anchor_base,
#             "anchor_deltas": anchor_deltas,
#             "concept_base": concept_base,
#             "concept_deltas": concept_deltas,
#         }
#         snap.update(geometry_snap)
#         return snap

#     @torch.no_grad()
#     def load_memory_snapshot(self, snapshot: Dict[str, Any], strict: bool = True):
#         if snapshot is None:
#             if strict:
#                 raise ValueError("snapshot is None")
#             return

#         means = snapshot.get("means", None)
#         class_count = int(means.size(0)) if torch.is_tensor(means) else int(snapshot.get("current_num_classes", 0))
#         spectral_dim = int(snapshot.get("spectral_dim", 0))

#         self.ensure_class_capacity(class_count=class_count, spectral_dim=spectral_dim)
#         self.geometry_bank.load_snapshot(snapshot, strict=strict)

#         self._anchors_cpu = []
#         self._anchor_deltas = nn.ParameterList()
#         self._concepts_cpu = []
#         self._concept_deltas = nn.ParameterList()

#         anchor_base = snapshot.get("anchor_base", None)
#         anchor_deltas = snapshot.get("anchor_deltas", None)
#         concept_base = snapshot.get("concept_base", None)
#         concept_deltas = snapshot.get("concept_deltas", None)

#         for cls in range(class_count):
#             a_base = anchor_base[cls].detach().cpu().clone() if torch.is_tensor(anchor_base) and anchor_base.size(0) > cls else torch.zeros(self.d_model)
#             a_delta = anchor_deltas[cls].detach().to(self.device).clone() if torch.is_tensor(anchor_deltas) and anchor_deltas.size(0) > cls else torch.zeros(self.d_model, device=self.device)
#             c_base = concept_base[cls].detach().cpu().clone() if torch.is_tensor(concept_base) and concept_base.size(0) > cls else torch.zeros(self.num_concepts_per_class, self.d_model)
#             c_delta = concept_deltas[cls].detach().to(self.device).clone() if torch.is_tensor(concept_deltas) and concept_deltas.size(0) > cls else torch.zeros(self.num_concepts_per_class, self.d_model, device=self.device)

#             self._anchors_cpu.append(a_base)
#             self._anchor_deltas.append(nn.Parameter(a_delta))
#             self._concepts_cpu.append(c_base)
#             self._concept_deltas.append(nn.Parameter(c_delta))

#         self.current_num_classes = class_count
#         self.old_class_count = int(snapshot.get("old_class_count", self.old_class_count))
#         self.current_phase = int(snapshot.get("current_phase", self.current_phase))

#     # =========================================================
#     # Backbone feature extraction
#     # =========================================================
#     def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
#         out = self.backbone(x)

#         if not isinstance(out, dict):
#             feat = out
#             bands = x.shape[1]
#             band_weights = torch.full((feat.size(0), bands), 1.0 / bands, device=feat.device, dtype=feat.dtype)
#             return {
#                 "features": feat,
#                 "backbone_features": feat,
#                 "band_weights": band_weights,
#                 "spectral_tokens": None,
#                 "spatial_tokens": None,
#                 "fused_tokens": None,
#                 "spectral_features": feat,
#                 "spatial_features": feat,
#                 "spatial_patterns": {},
#             }

#         feat = out["features"]
#         band_weights = out.get("band_weights")
#         if band_weights is None:
#             bands = x.shape[1]
#             band_weights = torch.full((feat.size(0), bands), 1.0 / bands, device=feat.device, dtype=feat.dtype)

#         return {
#             "features": feat,
#             "backbone_features": feat,
#             "band_weights": band_weights,
#             "spectral_tokens": out.get("spectral_tokens"),
#             "spatial_tokens": out.get("spatial_tokens"),
#             "fused_tokens": out.get("fused_tokens"),
#             "spectral_features": out.get("spectral_features", feat),
#             "spatial_features": out.get("spatial_features", feat),
#             "spatial_patterns": out.get("spatial_patterns", {}),
#         }

#     @torch.no_grad()
#     def extract_backbone_outputs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
#         out = self.extract_features(x)
#         raw_feat = out["features"]
#         projected_feat = self._project(raw_feat)

#         return {
#             "features": projected_feat,
#             "projected_features": projected_feat,
#             "preproject_features": raw_feat,
#             "backbone_features": raw_feat,
#             "band_weights": out["band_weights"],
#             "spectral_summary": self._spectral_summary(x),
#             "spectral_tokens": out["spectral_tokens"],
#             "spatial_tokens": out["spatial_tokens"],
#             "fused_tokens": out["fused_tokens"],
#             "spectral_features": out["spectral_features"],
#             "spatial_features": out["spatial_features"],
#             "spatial_patterns": out["spatial_patterns"],
#         }

#     def build_geometry_ref(self, feat: torch.Tensor):
#         if hasattr(self.geometry_bank, "retrieve_geometry_ref"):
#             return self.geometry_bank.retrieve_geometry_ref(feat)
#         return torch.zeros(feat.size(0), self.d_model, device=feat.device, dtype=feat.dtype)

#     # =========================================================
#     # Concept / anchor handling
#     # =========================================================
#     def _embed_concepts(self, concepts: torch.Tensor) -> torch.Tensor:
#         concepts = concepts.float().to(self.device)

#         if concepts.dim() != 2:
#             raise ValueError(f"concepts must be 2D, got {tuple(concepts.shape)}")

#         if concepts.size(1) == self.d_model:
#             embedded = concepts
#         elif concepts.size(1) == self.concept_projector.in_features:
#             embedded = self.concept_projector(concepts)
#         else:
#             raise ValueError(
#                 f"concept dim mismatch: expected {self.d_model} or {self.concept_projector.in_features}, got {concepts.size(1)}"
#             )

#         embedded = self._normalize(embedded)

#         k = embedded.size(0)
#         if k != self.num_concepts_per_class:
#             if k > self.num_concepts_per_class:
#                 embedded = embedded[: self.num_concepts_per_class]
#             else:
#                 pad = embedded[-1:].repeat(self.num_concepts_per_class - k, 1)
#                 embedded = torch.cat([embedded, pad], dim=0)

#         return embedded

#     @torch.no_grad()
#     def add_new_class_concepts(self, concepts: torch.Tensor):
#         cls = int(self.current_num_classes)
#         self.ensure_class_capacity(cls + 1)
#         self.refresh_class_concepts(cls, concepts, reset_delta=True)

#         anchor = self.get_anchor_bank()[cls].detach()
#         basis = torch.eye(self.d_model, self.subspace_rank, device=self.device, dtype=anchor.dtype)
#         eigvals = torch.full((self.subspace_rank,), 1e-4, device=self.device, dtype=anchor.dtype)
#         res_var = torch.tensor(1e-4, device=self.device, dtype=anchor.dtype)

#         self.refresh_class_subspace(
#             cls=cls,
#             mean=anchor,
#             basis=basis,
#             eigvals=eigvals,
#             res_var=res_var,
#             spectral_proto=None,
#             band_importance=None,
#             active_rank=torch.tensor(0, device=self.device, dtype=torch.long),
#             reliability=torch.tensor(0.05, device=self.device, dtype=anchor.dtype),
#         )

#         self.current_num_classes = cls + 1

#     @torch.no_grad()
#     def refresh_class_concepts(
#         self,
#         cls: int,
#         concepts: torch.Tensor,
#         reset_delta: bool = True,
#     ):
#         cls = int(cls)
#         self.ensure_class_capacity(cls + 1)

#         embedded = self._embed_concepts(concepts)
#         anchor = self._normalize(embedded.mean(dim=0))

#         self._anchors_cpu[cls] = anchor.detach().cpu()
#         self._concepts_cpu[cls] = embedded.detach().cpu()

#         if reset_delta:
#             self._anchor_deltas[cls].data.zero_()
#             self._concept_deltas[cls].data.zero_()

#         self.current_num_classes = max(self.current_num_classes, cls + 1)

#     @torch.no_grad()
#     def refresh_class_subspace(
#         self,
#         cls: int,
#         mean: torch.Tensor,
#         basis: torch.Tensor,
#         eigvals: torch.Tensor,
#         res_var,
#         spectral_proto=None,
#         band_importance=None,
#         active_rank=None,
#         reliability=None,
#     ):
#         cls = int(cls)

#         spectral_dim = 0
#         if spectral_proto is not None and torch.as_tensor(spectral_proto).numel() > 0:
#             spectral_dim = int(torch.as_tensor(spectral_proto).numel())
#         elif band_importance is not None and torch.as_tensor(band_importance).numel() > 0:
#             spectral_dim = int(torch.as_tensor(band_importance).numel())

#         self.ensure_class_capacity(cls + 1, spectral_dim=spectral_dim)

#         if hasattr(self.geometry_bank, "update_class_geometry"):
#             self.geometry_bank.update_class_geometry(
#                 class_id=cls,
#                 mean=mean.float().to(self.device),
#                 basis=basis.float().to(self.device),
#                 eigvals=eigvals.float().to(self.device),
#                 resvar=torch.as_tensor(res_var, device=self.device, dtype=mean.dtype),
#                 spectral_proto=spectral_proto,
#                 band_importance=band_importance,
#                 active_rank=active_rank,
#                 reliability=reliability,
#             )
#         else:
#             self.geometry_bank.update_class(
#                 cls_id=cls,
#                 mean=mean.float().to(self.device),
#                 basis=basis.float().to(self.device),
#                 eigvals=eigvals.float().to(self.device),
#                 res_var=torch.as_tensor(res_var, device=self.device, dtype=mean.dtype),
#                 spectral_proto=spectral_proto,
#                 band_importance=band_importance,
#             )

#             if active_rank is not None and hasattr(self.geometry_bank, "active_ranks"):
#                 self.geometry_bank.active_ranks[cls] = active_rank.to(self.device)
#             if reliability is not None and hasattr(self.geometry_bank, "reliability"):
#                 self.geometry_bank.reliability[cls] = reliability.to(self.device)

#         self.current_num_classes = max(self.current_num_classes, cls + 1)

#     @torch.no_grad()
#     def refresh_inter_class_geometry(self):
#         if hasattr(self.geometry_bank, "refresh_inter_class_geometry"):
#             self.geometry_bank.refresh_inter_class_geometry()

#     # =========================================================
#     # Logit computation
#     # =========================================================
#     def compute_logits_from_features(
#         self,
#         features: torch.Tensor,
#         classifier_mode: str = "geometry_only",
#     ):
#         classifier_mode = str(classifier_mode).lower()
#         subspace_bank = self.get_subspace_bank()
#         anchors = self.get_anchor_bank()
#         concepts = self.get_concept_bank()
#         calibrated_old = self.get_calibrated_old_subspace_bank(self.old_class_count)

#         return self.classifier(
#             features,
#             anchors=anchors if anchors.numel() > 0 else None,
#             concept_bank=concepts if concepts.numel() > 0 else None,
#             subspace_means=subspace_bank["means"] if subspace_bank["means"].numel() > 0 else None,
#             subspace_bases=subspace_bank["bases"] if subspace_bank["bases"].numel() > 0 else None,
#             subspace_variances=subspace_bank["variances"] if subspace_bank["variances"].numel() > 0 else None,
#             subspace_reliability=subspace_bank.get("reliability", None),
#             subspace_active_ranks=subspace_bank.get("active_ranks", None),
#             calibrated_old_means=calibrated_old["means"] if calibrated_old.get("means", None) is not None and calibrated_old["means"].numel() > 0 else None,
#             calibrated_old_bases=calibrated_old["bases"] if calibrated_old.get("bases", None) is not None and calibrated_old["bases"].numel() > 0 else None,
#             calibrated_old_variances=calibrated_old["variances"] if calibrated_old.get("variances", None) is not None and calibrated_old["variances"].numel() > 0 else None,
#             calibrated_old_reliability=calibrated_old.get("reliability", None),
#             calibrated_old_active_ranks=calibrated_old.get("active_ranks", None),
#             mode=classifier_mode,
#             old_class_count=int(self.old_class_count),
#         )

#     # =========================================================
#     # Trainability controls
#     # =========================================================
#     def freeze_backbone_only(self):
#         for p in self.backbone.parameters():
#             p.requires_grad = False

#     def freeze_semantic_encoder(self):
#         for p in self.semantic_encoder.parameters():
#             p.requires_grad = False

#     def unfreeze_semantic_encoder(self):
#         for p in self.semantic_encoder.parameters():
#             p.requires_grad = True

#     def freeze_projection_head(self):
#         for p in self.projection.parameters():
#             p.requires_grad = False
#         for p in self.norm.parameters():
#             p.requires_grad = False
#         for p in self.concept_projector.parameters():
#             p.requires_grad = False

#     def unfreeze_projection_head(self):
#         for p in self.projection.parameters():
#             p.requires_grad = True
#         for p in self.norm.parameters():
#             p.requires_grad = True
#         for p in self.concept_projector.parameters():
#             p.requires_grad = True

#     def freeze_old_anchor_deltas(self, old_class_count: int):
#         for i, p in enumerate(self._anchor_deltas):
#             if i < int(old_class_count):
#                 p.requires_grad = False

#     def unfreeze_new_anchor_deltas(self, old_class_count: int):
#         for i, p in enumerate(self._anchor_deltas):
#             p.requires_grad = i >= int(old_class_count)

#     def freeze_old_concept_deltas(self, old_class_count: int):
#         for i, p in enumerate(self._concept_deltas):
#             if i < int(old_class_count):
#                 p.requires_grad = False

#     def freeze_classifier_adaptation(self):
#         if hasattr(self.classifier, "freeze_all_adaptation"):
#             self.classifier.freeze_all_adaptation()

#     def freeze_old_classifier_adaptation(self, old_class_count: int):
#         if hasattr(self.classifier, "freeze_old_adaptation"):
#             self.classifier.freeze_old_adaptation(old_class_count)

#     def unfreeze_classifier_adaptation(self):
#         if hasattr(self.classifier, "unfreeze_all_adaptation"):
#             self.classifier.unfreeze_all_adaptation()

#     def freeze_fusion_module(self):
#         if hasattr(self.classifier, "freeze_fusion_module"):
#             self.classifier.freeze_fusion_module()

#     def unfreeze_fusion_module(self):
#         if hasattr(self.classifier, "unfreeze_fusion_module"):
#             self.classifier.unfreeze_fusion_module()

#     def freeze_geometry_calibrator(self):
#         for p in self.geometry_calibrator.parameters():
#             p.requires_grad = False

#     def unfreeze_geometry_calibrator(self):
#         for p in self.geometry_calibrator.parameters():
#             p.requires_grad = True

#     # =========================================================
#     # Semantic refinement helper
#     # =========================================================
#     def _semantic_refine(
#         self,
#         feat: torch.Tensor,
#         anchors: torch.Tensor,
#         concepts: torch.Tensor,
#         spectral_tokens,
#         spatial_tokens,
#         semantic_mode: str,
#         return_token_relations: bool,
#     ):
#         bypass_semantic = semantic_mode in {"off", "none", "bypass", "identity", "raw"}
#         if anchors.numel() == 0 or bypass_semantic:
#             return feat, None

#         semantic_bank = concepts if concepts.numel() > 0 else anchors.unsqueeze(1)

#         semantic_out = self.semantic_encoder(
#             feat,
#             concept_bank=semantic_bank,
#             phase=int(self.current_phase),
#             disable=False,
#             spectral_tokens=spectral_tokens,
#             spatial_tokens=spatial_tokens,
#             return_token_relations=return_token_relations,
#         )

#         if isinstance(semantic_out, dict):
#             feat_refined = semantic_out["features"]
#             token_relations = semantic_out.get("token_relations", None)
#         else:
#             feat_refined = semantic_out
#             token_relations = None

#         return feat_refined, token_relations

#     # =========================================================
#     # Forward
#     # =========================================================
#     def forward(self, x: torch.Tensor, **kwargs):
#         semantic_mode = self._resolve_semantic_mode(kwargs.get("semantic_mode", "auto"))
#         classifier_mode = self._resolve_classifier_mode(kwargs.get("classifier_mode", None))
#         return_token_relations = bool(kwargs.get("return_token_relations", False))

#         backbone_out = self.extract_features(x)

#         feat = backbone_out["features"]
#         band_weights = backbone_out["band_weights"]
#         spectral_tokens = backbone_out["spectral_tokens"]
#         spatial_tokens = backbone_out["spatial_tokens"]
#         fused_tokens = backbone_out["fused_tokens"]
#         spectral_summary = self._spectral_summary(x)

#         anchors = self.get_anchor_bank()
#         concepts = self.get_concept_bank()

#         feat_pre, token_relations = self._semantic_refine(
#             feat=feat,
#             anchors=anchors,
#             concepts=concepts,
#             spectral_tokens=spectral_tokens,
#             spatial_tokens=spatial_tokens,
#             semantic_mode=semantic_mode,
#             return_token_relations=return_token_relations,
#         )

#         if (
#             return_token_relations
#             and token_relations is None
#             and spectral_tokens is not None
#             and spatial_tokens is not None
#         ):
#             token_relations = self.semantic_encoder.build_token_relations(
#                 spectral_tokens,
#                 spatial_tokens,
#             )

#         projected = self._project(feat_pre)

#         subspace = self.get_subspace_bank()
#         calibrated_old = self.get_calibrated_old_subspace_bank(self.old_class_count)

#         geometry_ref = self.build_geometry_ref(projected)

#         if hasattr(self.geometry_bank, "retrieve_spectral_ref"):
#             spectral_ref = self.geometry_bank.retrieve_spectral_ref(projected)
#         else:
#             spectral_ref = torch.empty(projected.size(0), 0, device=projected.device, dtype=projected.dtype)

#         if hasattr(self.geometry_bank, "retrieve_band_importance_ref"):
#             band_importance_ref = self.geometry_bank.retrieve_band_importance_ref(projected)
#         else:
#             band_importance_ref = torch.empty(projected.size(0), 0, device=projected.device, dtype=projected.dtype)

#         logits = self.compute_logits_from_features(
#             projected,
#             classifier_mode=classifier_mode,
#         )

#         calibration_reg = self.calibration_regularization_loss(self.old_class_count)

#         return {
#             "logits": logits,
#             "features": projected,
#             "preproject_features": feat_pre,
#             "backbone_features": feat,
#             "band_weights": band_weights,
#             "spectral_summary": spectral_summary,
#             "spectral_ref": spectral_ref,
#             "band_importance_ref": band_importance_ref,
#             "geometry_ref": geometry_ref,
#             "anchors": anchors,
#             "concept_bank": concepts,
#             "subspace_means": subspace["means"],
#             "subspace_bases": subspace["bases"],
#             "subspace_variances": subspace["variances"],
#             "subspace_reliability": subspace.get("reliability", None),
#             "subspace_active_ranks": subspace.get("active_ranks", None),
#             "calibrated_old_means": calibrated_old.get("means", None),
#             "calibrated_old_bases": calibrated_old.get("bases", None),
#             "calibrated_old_variances": calibrated_old.get("variances", None),
#             "calibrated_old_reliability": calibrated_old.get("reliability", None),
#             "calibration_reg": calibration_reg,
#             "spectral_tokens": spectral_tokens,
#             "spatial_tokens": spatial_tokens,
#             "fused_tokens": fused_tokens,
#             "token_relations": token_relations,
#             "spectral_features": backbone_out["spectral_features"],
#             "spatial_features": backbone_out["spatial_features"],
#             "spatial_patterns": backbone_out["spatial_patterns"],
#         }

















# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Dict, Optional, Any

# from models.backbone import SSMBackbone
# from models.token import HSISemanticConceptEncoder
# from models.classifier import SemanticClassifier
# from models.geometry_bank import GeometryBank


# def _zero_scalar(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
#     return torch.tensor(0.0, device=device, dtype=dtype)


# def _projector_from_basis(basis: torch.Tensor) -> torch.Tensor:
#     if basis.dim() == 2:
#         return basis @ basis.t()
#     return torch.matmul(basis, basis.transpose(-1, -2))


# def _orthonormalize_batched_basis(basis: torch.Tensor, rank: int) -> torch.Tensor:
#     """
#     Orthonormalize basis columns.

#     Args:
#         basis: [C, D, R]
#         rank: target rank

#     Returns:
#         [C, D, R] with orthonormal columns.
#     """
#     if basis is None or basis.numel() == 0:
#         return basis

#     outs = []
#     for b in basis:
#         q, _ = torch.linalg.qr(b, mode="reduced")
#         if q.size(1) < rank:
#             pad = torch.zeros(
#                 q.size(0),
#                 rank - q.size(1),
#                 device=q.device,
#                 dtype=q.dtype,
#             )
#             q = torch.cat([q, pad], dim=1)
#         outs.append(q[:, :rank])
#     return torch.stack(outs, dim=0)


# class GeometryCalibrator(nn.Module):
#     """
#     Conservative geometry transport calibration for old classes.

#     The calibrator is intentionally small. It corrects old geometry drift without
#     letting the old class memory become a free classifier.

#     Calibrates:
#         mean       : mu -> mu + delta_mu
#         variances  : log(v) -> log(v) + delta_logv
#         basis      : optional projector-stable basis correction

#     By default, basis calibration is disabled because it is easier to destabilize
#     than mean/variance calibration. Enable it only after the geometry-only system
#     is stable.
#     """

#     def __init__(
#         self,
#         d_model: int,
#         rank: int,
#         hidden_dim: Optional[int] = None,
#         var_floor: float = 1e-4,
#         dropout: float = 0.1,
#         calibrate_basis: bool = False,
#         max_mean_scale: float = 0.10,
#         max_var_scale: float = 0.10,
#         max_basis_scale: float = 0.03,
#     ):
#         super().__init__()
#         self.d_model = int(d_model)
#         self.rank = int(rank)
#         self.var_floor = float(var_floor)
#         self.calibrate_basis = bool(calibrate_basis)

#         hidden_dim = int(hidden_dim or max(d_model, 128))

#         # Context is the old class mean. Keep this lightweight; do not overfit.
#         self.mean_calibrator = nn.Sequential(
#             nn.Linear(self.d_model, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, self.d_model),
#         )

#         self.var_calibrator = nn.Sequential(
#             nn.Linear(self.d_model, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, self.rank + 1),
#         )

#         if self.calibrate_basis:
#             self.basis_calibrator = nn.Sequential(
#                 nn.Linear(self.d_model, hidden_dim),
#                 nn.GELU(),
#                 nn.Dropout(dropout),
#                 nn.Linear(hidden_dim, self.d_model * self.rank),
#             )
#         else:
#             self.basis_calibrator = None

#         # Raw parameters are squashed. This prevents calibration from exploding.
#         self.mean_scale_raw = nn.Parameter(torch.tensor(-3.0))
#         self.var_scale_raw = nn.Parameter(torch.tensor(-3.0))
#         self.basis_scale_raw = nn.Parameter(torch.tensor(-4.0))

#         self.max_mean_scale = float(max_mean_scale)
#         self.max_var_scale = float(max_var_scale)
#         self.max_basis_scale = float(max_basis_scale)

#     def _mean_scale(self) -> torch.Tensor:
#         return self.max_mean_scale * torch.sigmoid(self.mean_scale_raw)

#     def _var_scale(self) -> torch.Tensor:
#         return self.max_var_scale * torch.sigmoid(self.var_scale_raw)

#     def _basis_scale(self) -> torch.Tensor:
#         return self.max_basis_scale * torch.sigmoid(self.basis_scale_raw)

#     def forward(
#         self,
#         means: torch.Tensor,
#         bases: torch.Tensor,
#         variances: torch.Tensor,
#         reliability: Optional[torch.Tensor] = None,
#     ) -> Dict[str, torch.Tensor]:
#         if means is None or means.numel() == 0:
#             return {
#                 "means": means,
#                 "bases": bases,
#                 "variances": variances,
#                 "mean_delta": means,
#                 "var_delta": variances,
#                 "basis_delta": bases,
#             }

#         if reliability is None or not torch.is_tensor(reliability) or reliability.numel() == 0:
#             rel = torch.ones(means.size(0), 1, device=means.device, dtype=means.dtype)
#         else:
#             rel = reliability.to(device=means.device, dtype=means.dtype).view(-1, 1).clamp(0.05, 1.0)

#         # Less reliable geometry gets smaller learned transport.
#         mean_delta = self._mean_scale() * rel * self.mean_calibrator(means)
#         calibrated_means = means + mean_delta

#         logv = torch.log(variances.clamp_min(self.var_floor))
#         var_delta = self._var_scale() * rel * self.var_calibrator(means)
#         calibrated_logv = logv + var_delta
#         calibrated_variances = torch.exp(calibrated_logv).clamp_min(self.var_floor)

#         if self.calibrate_basis and self.basis_calibrator is not None and bases is not None and bases.numel() > 0:
#             raw_delta = self.basis_calibrator(means).view(-1, self.d_model, self.rank)
#             basis_delta = self._basis_scale() * rel.view(-1, 1, 1) * raw_delta
#             calibrated_bases = _orthonormalize_batched_basis(bases + basis_delta, self.rank)
#         else:
#             basis_delta = torch.zeros_like(bases)
#             calibrated_bases = bases

#         return {
#             "means": calibrated_means,
#             "bases": calibrated_bases,
#             "variances": calibrated_variances,
#             "mean_delta": mean_delta,
#             "var_delta": var_delta,
#             "basis_delta": basis_delta,
#         }

#     def regularization_loss(
#         self,
#         raw_means: Optional[torch.Tensor],
#         raw_bases: Optional[torch.Tensor],
#         raw_variances: Optional[torch.Tensor],
#         calibrated_means: Optional[torch.Tensor],
#         calibrated_bases: Optional[torch.Tensor],
#         calibrated_variances: Optional[torch.Tensor],
#     ) -> Dict[str, torch.Tensor]:
#         if (
#             raw_means is None
#             or raw_variances is None
#             or calibrated_means is None
#             or calibrated_variances is None
#             or raw_means.numel() == 0
#         ):
#             device = raw_means.device if torch.is_tensor(raw_means) else torch.device("cpu")
#             z = _zero_scalar(device)
#             return {"total": z, "mean": z, "basis": z, "var": z}

#         mean_reg = F.mse_loss(calibrated_means, raw_means)

#         raw_logv = torch.log(raw_variances.clamp_min(self.var_floor))
#         cal_logv = torch.log(calibrated_variances.clamp_min(self.var_floor))
#         var_reg = F.mse_loss(cal_logv, raw_logv)

#         if (
#             raw_bases is not None
#             and calibrated_bases is not None
#             and torch.is_tensor(raw_bases)
#             and torch.is_tensor(calibrated_bases)
#             and raw_bases.numel() > 0
#             and calibrated_bases.numel() > 0
#         ):
#             basis_reg = F.mse_loss(
#                 _projector_from_basis(calibrated_bases),
#                 _projector_from_basis(raw_bases),
#             )
#         else:
#             basis_reg = _zero_scalar(raw_means.device, raw_means.dtype)

#         basis_weight = 0.2 if self.calibrate_basis else 0.0
#         total = mean_reg + var_reg + basis_weight * basis_reg
#         return {"total": total, "mean": mean_reg, "basis": basis_reg, "var": var_reg}


# class NECILModel(nn.Module):
#     """
#     Geometry-centric NECIL-HSI model.

#     Core policy:
#         - Backbone produces spectral-spatial features/tokens.
#         - Projection preserves Euclidean feature scale.
#         - GeometryBank is the real non-exemplar memory.
#         - Classifier scores by geometry energy.
#         - Semantic concepts are auxiliary and should not rewrite geometry during
#           incremental phases unless explicitly enabled.
#     """

#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         self.device = torch.device(args.device)

#         self.d_model = int(args.d_model)
#         self.subspace_rank = int(getattr(args, "subspace_rank", 5))
#         self.num_concepts_per_class = int(getattr(args, "num_concepts_per_class", 4))

#         self.default_base_classifier_mode = str(
#             getattr(args, "base_classifier_mode", "geometry_only")
#         ).lower()

#         self.default_incremental_classifier_mode = str(
#             getattr(args, "incremental_classifier_mode", "calibrated_geometry")
#         ).lower()

#         self.current_num_classes = 0
#         self.old_class_count = 0
#         self.current_phase = 0

#         # Auxiliary semantic memory. This is NOT the main classifier memory.
#         self._anchors_cpu = []
#         self._anchor_deltas = nn.ParameterList()
#         self._concepts_cpu = []
#         self._concept_deltas = nn.ParameterList()

#         self.backbone = SSMBackbone(args)

#         self.semantic_encoder = HSISemanticConceptEncoder(
#             feature_dim=self.d_model,
#             concept_dim=self.d_model,
#             dropout=float(getattr(args, "semantic_dropout", 0.1)),
#             token_temperature=float(getattr(args, "token_temperature", 0.07)),
#         )

#         self.projection = nn.Sequential(
#             nn.Linear(self.d_model, self.d_model),
#             nn.GELU(),
#             nn.Dropout(float(getattr(args, "projection_dropout", 0.1))),
#             nn.Linear(self.d_model, self.d_model),
#         )
#         self.norm = nn.LayerNorm(self.d_model)

#         self.concept_projector = nn.Linear(int(args.num_bands), self.d_model, bias=True)

#         self.geometry_bank = GeometryBank(
#             self.d_model,
#             self.subspace_rank,
#             device=args.device,
#         )

#         self.geometry_calibrator = GeometryCalibrator(
#             d_model=self.d_model,
#             rank=self.subspace_rank,
#             hidden_dim=int(getattr(args, "geometry_calibration_hidden_dim", self.d_model)),
#             var_floor=float(getattr(args, "geom_var_floor", 1e-4)),
#             dropout=float(getattr(args, "geometry_calibration_dropout", 0.1)),
#             calibrate_basis=bool(getattr(args, "geometry_calibrate_basis", False)),
#             max_mean_scale=float(getattr(args, "geometry_max_mean_scale", 0.10)),
#             max_var_scale=float(getattr(args, "geometry_max_var_scale", 0.10)),
#             max_basis_scale=float(getattr(args, "geometry_max_basis_scale", 0.03)),
#         )

#         self.classifier = SemanticClassifier(
#             initial_classes=0,
#             d_model=self.d_model,
#             logit_scale=float(getattr(args, "loss_scale", 8.0)),
#             use_bias=bool(getattr(args, "classifier_use_bias", True)),
#             variance_floor=float(getattr(args, "geom_var_floor", 1e-4)),
#             use_geom_temperature=bool(getattr(args, "use_geom_temperature", True)),
#             concept_agg_temperature=float(getattr(args, "cls_temperature", 0.07)),
#             init_alpha_old=float(getattr(args, "init_alpha_old", -0.5)),
#             init_alpha_new=float(getattr(args, "init_alpha_new", -0.2)),
#             # Keep legacy fusion off by default. Geometry should carry the method.
#             use_adaptive_fusion=bool(getattr(args, "use_adaptive_fusion", False)),
#         )

#     # =========================================================
#     # Basic helpers
#     # =========================================================
#     def _normalize(self, x: torch.Tensor) -> torch.Tensor:
#         return F.normalize(x, dim=-1, eps=1e-6)

#     def _project(self, feat: torch.Tensor) -> torch.Tensor:
#         # Preserve Euclidean geometry. No final L2 normalization.
#         return self.norm(self.projection(feat) + feat)

#     def _spectral_summary(self, x: torch.Tensor) -> torch.Tensor:
#         return x.mean(dim=(-1, -2))

#     def _resolve_classifier_mode(self, classifier_mode: Optional[str]) -> str:
#         if classifier_mode is None:
#             return (
#                 self.default_base_classifier_mode
#                 if int(self.current_phase) == 0
#                 else self.default_incremental_classifier_mode
#             )
#         return str(classifier_mode).lower()

#     def _resolve_semantic_mode(self, semantic_mode: Optional[str]) -> str:
#         """
#         Critical design choice:
#         default semantic mode is identity for all phases.

#         Do not use semantic_mode='all' in base and 'identity' in incremental,
#         because that builds geometry memory in one feature manifold and trains
#         incremental scoring in another.
#         """
#         if semantic_mode is None or str(semantic_mode).lower() == "auto":
#             return "identity"
#         return str(semantic_mode).lower()

#     def set_phase(self, phase: int):
#         self.current_phase = int(phase)

#     def set_old_class_count(self, old_class_count: int):
#         self.old_class_count = int(old_class_count)

#     # =========================================================
#     # Capacity / load preparation
#     # =========================================================
#     @torch.no_grad()
#     def ensure_class_capacity(
#         self,
#         class_count: int,
#         spectral_dim: int = 0,
#         dtype: Optional[torch.dtype] = None,
#     ):
#         class_count = int(class_count)
#         dtype = dtype or self.projection[0].weight.dtype

#         while len(self._anchors_cpu) < class_count:
#             zero_anchor = torch.zeros(self.d_model, dtype=dtype)
#             zero_concepts = torch.zeros(self.num_concepts_per_class, self.d_model, dtype=dtype)

#             self._anchors_cpu.append(zero_anchor.clone())
#             self._anchor_deltas.append(nn.Parameter(torch.zeros_like(zero_anchor, device=self.device)))

#             self._concepts_cpu.append(zero_concepts.clone())
#             self._concept_deltas.append(nn.Parameter(torch.zeros_like(zero_concepts, device=self.device)))

#         while self.classifier.num_classes < class_count:
#             self.classifier.expand(1, self.current_phase)

#         # Updated bank uses ensure_class_count(count=..., spectral_dim=...).
#         # Older variants may use ensure_num_classes.
#         if hasattr(self.geometry_bank, "ensure_class_count"):
#             self.geometry_bank.ensure_class_count(
#                 count=class_count,
#                 spectral_dim=int(spectral_dim),
#                 dtype=dtype,
#             )
#         elif hasattr(self.geometry_bank, "ensure_num_classes"):
#             self.geometry_bank.ensure_num_classes(class_count)

#         self.current_num_classes = max(self.current_num_classes, class_count)

#     # =========================================================
#     # Auxiliary semantic banks
#     # =========================================================
#     def get_anchor_bank(self) -> torch.Tensor:
#         if len(self._anchors_cpu) == 0:
#             return torch.empty((0, self.d_model), device=self.device)

#         return torch.stack(
#             [
#                 self._normalize(base.to(self.device) + self._anchor_deltas[i])
#                 for i, base in enumerate(self._anchors_cpu)
#             ],
#             dim=0,
#         )

#     def get_concept_bank(self) -> torch.Tensor:
#         if len(self._concepts_cpu) == 0:
#             return torch.empty((0, self.num_concepts_per_class, self.d_model), device=self.device)

#         return torch.stack(
#             [
#                 self._normalize(base.to(self.device) + self._concept_deltas[i])
#                 for i, base in enumerate(self._concepts_cpu)
#             ],
#             dim=0,
#         )

#     # =========================================================
#     # Geometry bank access
#     # =========================================================
#     def get_subspace_bank(self) -> Dict[str, torch.Tensor]:
#         bank = self.geometry_bank.get_bank()

#         # Defensive aliases for old/new bank implementations.
#         if "variances" not in bank:
#             if "eigvals" in bank and ("res_vars" in bank or "resvars" in bank):
#                 res = bank.get("res_vars", bank.get("resvars"))
#                 bank["variances"] = torch.cat([bank["eigvals"], res.unsqueeze(-1)], dim=-1)

#         if "resvars" not in bank and "res_vars" in bank:
#             bank["resvars"] = bank["res_vars"]
#         if "res_vars" not in bank and "resvars" in bank:
#             bank["res_vars"] = bank["resvars"]

#         C = bank["means"].size(0) if "means" in bank and torch.is_tensor(bank["means"]) else 0
#         if "reliability" not in bank:
#             bank["reliability"] = torch.ones(C, device=self.device)
#         if "active_ranks" not in bank:
#             bank["active_ranks"] = torch.full((C,), self.subspace_rank, device=self.device, dtype=torch.long)

#         return bank

#     def get_old_subspace_bank(self, old_class_count: Optional[int] = None) -> Dict[str, torch.Tensor]:
#         old_class_count = int(self.old_class_count if old_class_count is None else old_class_count)
#         bank = self.get_subspace_bank()

#         if old_class_count <= 0:
#             return {
#                 "means": torch.empty((0, self.d_model), device=self.device),
#                 "bases": torch.empty((0, self.d_model, self.subspace_rank), device=self.device),
#                 "variances": torch.empty((0, self.subspace_rank + 1), device=self.device),
#                 "reliability": torch.empty((0,), device=self.device),
#                 "active_ranks": torch.empty((0,), device=self.device, dtype=torch.long),
#             }

#         return {
#             "means": bank["means"][:old_class_count],
#             "bases": bank["bases"][:old_class_count],
#             "variances": bank["variances"][:old_class_count],
#             "reliability": bank.get("reliability", torch.ones(bank["means"].size(0), device=bank["means"].device))[:old_class_count],
#             "active_ranks": bank.get("active_ranks", torch.full((bank["means"].size(0),), self.subspace_rank, device=bank["means"].device, dtype=torch.long))[:old_class_count],
#         }

#     def get_new_subspace_bank(self, old_class_count: Optional[int] = None) -> Dict[str, torch.Tensor]:
#         old_class_count = int(self.old_class_count if old_class_count is None else old_class_count)
#         bank = self.get_subspace_bank()

#         return {
#             "means": bank["means"][old_class_count:],
#             "bases": bank["bases"][old_class_count:],
#             "variances": bank["variances"][old_class_count:],
#             "reliability": bank.get("reliability", torch.ones(bank["means"].size(0), device=bank["means"].device))[old_class_count:],
#             "active_ranks": bank.get("active_ranks", torch.full((bank["means"].size(0),), self.subspace_rank, device=bank["means"].device, dtype=torch.long))[old_class_count:],
#         }

#     def get_calibrated_old_subspace_bank(
#         self,
#         old_class_count: Optional[int] = None,
#     ) -> Dict[str, torch.Tensor]:
#         old_class_count = int(self.old_class_count if old_class_count is None else old_class_count)
#         raw_old = self.get_old_subspace_bank(old_class_count)

#         if old_class_count <= 0 or raw_old["means"].numel() == 0:
#             return raw_old

#         calibrated = self.geometry_calibrator(
#             raw_old["means"],
#             raw_old["bases"],
#             raw_old["variances"],
#             reliability=raw_old.get("reliability", None),
#         )

#         return {
#             "means": calibrated["means"],
#             "bases": calibrated["bases"],
#             "variances": calibrated["variances"],
#             "mean_delta": calibrated["mean_delta"],
#             "var_delta": calibrated["var_delta"],
#             "basis_delta": calibrated["basis_delta"],
#             "raw_means": raw_old["means"],
#             "raw_bases": raw_old["bases"],
#             "raw_variances": raw_old["variances"],
#             "reliability": raw_old.get("reliability", None),
#             "active_ranks": raw_old.get("active_ranks", None),
#         }

#     def calibration_regularization_loss(
#         self,
#         old_class_count: Optional[int] = None,
#     ) -> Dict[str, torch.Tensor]:
#         old_class_count = int(self.old_class_count if old_class_count is None else old_class_count)
#         raw_old = self.get_old_subspace_bank(old_class_count)

#         if old_class_count <= 0 or raw_old["means"].numel() == 0:
#             z = _zero_scalar(self.device)
#             return {"total": z, "mean": z, "basis": z, "var": z}

#         calibrated = self.get_calibrated_old_subspace_bank(old_class_count)

#         return self.geometry_calibrator.regularization_loss(
#             raw_means=raw_old["means"],
#             raw_bases=raw_old["bases"],
#             raw_variances=raw_old["variances"],
#             calibrated_means=calibrated["means"],
#             calibrated_bases=calibrated["bases"],
#             calibrated_variances=calibrated["variances"],
#         )

#     # =========================================================
#     # Snapshot helpers
#     # =========================================================
#     @torch.no_grad()
#     def export_memory_snapshot(self) -> Dict[str, Any]:
#         geometry_snap = self.geometry_bank.export_snapshot()

#         if len(self._anchors_cpu) > 0:
#             anchor_base = torch.stack([x.detach().cpu() for x in self._anchors_cpu], dim=0)
#             anchor_deltas = torch.stack([p.detach().cpu() for p in self._anchor_deltas], dim=0)
#         else:
#             anchor_base = torch.empty((0, self.d_model))
#             anchor_deltas = torch.empty((0, self.d_model))

#         if len(self._concepts_cpu) > 0:
#             concept_base = torch.stack([x.detach().cpu() for x in self._concepts_cpu], dim=0)
#             concept_deltas = torch.stack([p.detach().cpu() for p in self._concept_deltas], dim=0)
#         else:
#             concept_base = torch.empty((0, self.num_concepts_per_class, self.d_model))
#             concept_deltas = torch.empty((0, self.num_concepts_per_class, self.d_model))

#         snap = {
#             "current_num_classes": int(self.current_num_classes),
#             "old_class_count": int(self.old_class_count),
#             "current_phase": int(self.current_phase),
#             "anchor_base": anchor_base,
#             "anchor_deltas": anchor_deltas,
#             "concept_base": concept_base,
#             "concept_deltas": concept_deltas,
#         }
#         snap.update(geometry_snap)
#         return snap

#     @torch.no_grad()
#     def load_memory_snapshot(self, snapshot: Dict[str, Any], strict: bool = True):
#         if snapshot is None:
#             if strict:
#                 raise ValueError("snapshot is None")
#             return

#         means = snapshot.get("means", None)
#         class_count = int(means.size(0)) if torch.is_tensor(means) else int(snapshot.get("current_num_classes", 0))
#         spectral_dim = int(snapshot.get("spectral_dim", 0))

#         self.ensure_class_capacity(class_count=class_count, spectral_dim=spectral_dim)
#         self.geometry_bank.load_snapshot(snapshot, strict=strict)

#         self._anchors_cpu = []
#         self._anchor_deltas = nn.ParameterList()
#         self._concepts_cpu = []
#         self._concept_deltas = nn.ParameterList()

#         anchor_base = snapshot.get("anchor_base", None)
#         anchor_deltas = snapshot.get("anchor_deltas", None)
#         concept_base = snapshot.get("concept_base", None)
#         concept_deltas = snapshot.get("concept_deltas", None)

#         for cls in range(class_count):
#             a_base = anchor_base[cls].detach().cpu().clone() if torch.is_tensor(anchor_base) and anchor_base.size(0) > cls else torch.zeros(self.d_model)
#             a_delta = anchor_deltas[cls].detach().to(self.device).clone() if torch.is_tensor(anchor_deltas) and anchor_deltas.size(0) > cls else torch.zeros(self.d_model, device=self.device)
#             c_base = concept_base[cls].detach().cpu().clone() if torch.is_tensor(concept_base) and concept_base.size(0) > cls else torch.zeros(self.num_concepts_per_class, self.d_model)
#             c_delta = concept_deltas[cls].detach().to(self.device).clone() if torch.is_tensor(concept_deltas) and concept_deltas.size(0) > cls else torch.zeros(self.num_concepts_per_class, self.d_model, device=self.device)

#             self._anchors_cpu.append(a_base)
#             self._anchor_deltas.append(nn.Parameter(a_delta))
#             self._concepts_cpu.append(c_base)
#             self._concept_deltas.append(nn.Parameter(c_delta))

#         self.current_num_classes = class_count
#         self.old_class_count = int(snapshot.get("old_class_count", self.old_class_count))
#         self.current_phase = int(snapshot.get("current_phase", self.current_phase))

#     # =========================================================
#     # Backbone feature extraction
#     # =========================================================
#     def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
#         out = self.backbone(x)

#         if not isinstance(out, dict):
#             feat = out
#             bands = x.shape[1]
#             band_weights = torch.full((feat.size(0), bands), 1.0 / bands, device=feat.device, dtype=feat.dtype)
#             return {
#                 "features": feat,
#                 "backbone_features": feat,
#                 "band_weights": band_weights,
#                 "spectral_tokens": None,
#                 "spatial_tokens": None,
#                 "fused_tokens": None,
#                 "spectral_features": feat,
#                 "spatial_features": feat,
#                 "spatial_patterns": {},
#             }

#         feat = out["features"]
#         band_weights = out.get("band_weights")
#         if band_weights is None:
#             bands = x.shape[1]
#             band_weights = torch.full((feat.size(0), bands), 1.0 / bands, device=feat.device, dtype=feat.dtype)

#         return {
#             "features": feat,
#             "backbone_features": feat,
#             "band_weights": band_weights,
#             "spectral_tokens": out.get("spectral_tokens"),
#             "spatial_tokens": out.get("spatial_tokens"),
#             "fused_tokens": out.get("fused_tokens"),
#             "spectral_features": out.get("spectral_features", feat),
#             "spatial_features": out.get("spatial_features", feat),
#             "spatial_patterns": out.get("spatial_patterns", {}),
#         }

#     @torch.no_grad()
#     def extract_backbone_outputs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
#         out = self.extract_features(x)
#         raw_feat = out["features"]
#         projected_feat = self._project(raw_feat)

#         return {
#             "features": projected_feat,
#             "projected_features": projected_feat,
#             "preproject_features": raw_feat,
#             "backbone_features": raw_feat,
#             "band_weights": out["band_weights"],
#             "spectral_summary": self._spectral_summary(x),
#             "spectral_tokens": out["spectral_tokens"],
#             "spatial_tokens": out["spatial_tokens"],
#             "fused_tokens": out["fused_tokens"],
#             "spectral_features": out["spectral_features"],
#             "spatial_features": out["spatial_features"],
#             "spatial_patterns": out["spatial_patterns"],
#         }

#     def build_geometry_ref(self, feat: torch.Tensor):
#         if hasattr(self.geometry_bank, "retrieve_geometry_ref"):
#             return self.geometry_bank.retrieve_geometry_ref(feat)
#         return torch.zeros(feat.size(0), self.d_model, device=feat.device, dtype=feat.dtype)

#     # =========================================================
#     # Concept / anchor handling
#     # =========================================================
#     def _embed_concepts(self, concepts: torch.Tensor) -> torch.Tensor:
#         concepts = concepts.float().to(self.device)

#         if concepts.dim() != 2:
#             raise ValueError(f"concepts must be 2D, got {tuple(concepts.shape)}")

#         if concepts.size(1) == self.d_model:
#             embedded = concepts
#         elif concepts.size(1) == self.concept_projector.in_features:
#             embedded = self.concept_projector(concepts)
#         else:
#             raise ValueError(
#                 f"concept dim mismatch: expected {self.d_model} or {self.concept_projector.in_features}, got {concepts.size(1)}"
#             )

#         embedded = self._normalize(embedded)

#         k = embedded.size(0)
#         if k != self.num_concepts_per_class:
#             if k > self.num_concepts_per_class:
#                 embedded = embedded[: self.num_concepts_per_class]
#             else:
#                 pad = embedded[-1:].repeat(self.num_concepts_per_class - k, 1)
#                 embedded = torch.cat([embedded, pad], dim=0)

#         return embedded

#     @torch.no_grad()
#     def add_new_class_concepts(self, concepts: torch.Tensor):
#         cls = int(self.current_num_classes)
#         self.ensure_class_capacity(cls + 1)
#         self.refresh_class_concepts(cls, concepts, reset_delta=True)

#         anchor = self.get_anchor_bank()[cls].detach()
#         basis = torch.eye(self.d_model, self.subspace_rank, device=self.device, dtype=anchor.dtype)
#         eigvals = torch.full((self.subspace_rank,), 1e-4, device=self.device, dtype=anchor.dtype)
#         res_var = torch.tensor(1e-4, device=self.device, dtype=anchor.dtype)

#         self.refresh_class_subspace(
#             cls=cls,
#             mean=anchor,
#             basis=basis,
#             eigvals=eigvals,
#             res_var=res_var,
#             spectral_proto=None,
#             band_importance=None,
#             active_rank=torch.tensor(0, device=self.device, dtype=torch.long),
#             reliability=torch.tensor(0.05, device=self.device, dtype=anchor.dtype),
#         )

#         self.current_num_classes = cls + 1

#     @torch.no_grad()
#     def refresh_class_concepts(
#         self,
#         cls: int,
#         concepts: torch.Tensor,
#         reset_delta: bool = True,
#     ):
#         cls = int(cls)
#         self.ensure_class_capacity(cls + 1)

#         embedded = self._embed_concepts(concepts)
#         anchor = self._normalize(embedded.mean(dim=0))

#         self._anchors_cpu[cls] = anchor.detach().cpu()
#         self._concepts_cpu[cls] = embedded.detach().cpu()

#         if reset_delta:
#             self._anchor_deltas[cls].data.zero_()
#             self._concept_deltas[cls].data.zero_()

#         self.current_num_classes = max(self.current_num_classes, cls + 1)

#     @torch.no_grad()
#     def refresh_class_subspace(
#         self,
#         cls: int,
#         mean: torch.Tensor,
#         basis: torch.Tensor,
#         eigvals: torch.Tensor,
#         res_var,
#         spectral_proto=None,
#         band_importance=None,
#         active_rank=None,
#         reliability=None,
#     ):
#         cls = int(cls)

#         spectral_dim = 0
#         if spectral_proto is not None and torch.as_tensor(spectral_proto).numel() > 0:
#             spectral_dim = int(torch.as_tensor(spectral_proto).numel())
#         elif band_importance is not None and torch.as_tensor(band_importance).numel() > 0:
#             spectral_dim = int(torch.as_tensor(band_importance).numel())

#         self.ensure_class_capacity(cls + 1, spectral_dim=spectral_dim)

#         if hasattr(self.geometry_bank, "update_class_geometry"):
#             self.geometry_bank.update_class_geometry(
#                 class_id=cls,
#                 mean=mean.float().to(self.device),
#                 basis=basis.float().to(self.device),
#                 eigvals=eigvals.float().to(self.device),
#                 resvar=torch.as_tensor(res_var, device=self.device, dtype=mean.dtype),
#                 spectral_proto=spectral_proto,
#                 band_importance=band_importance,
#                 active_rank=active_rank,
#                 reliability=reliability,
#             )
#         else:
#             self.geometry_bank.update_class(
#                 cls_id=cls,
#                 mean=mean.float().to(self.device),
#                 basis=basis.float().to(self.device),
#                 eigvals=eigvals.float().to(self.device),
#                 res_var=torch.as_tensor(res_var, device=self.device, dtype=mean.dtype),
#                 spectral_proto=spectral_proto,
#                 band_importance=band_importance,
#             )

#             if active_rank is not None and hasattr(self.geometry_bank, "active_ranks"):
#                 self.geometry_bank.active_ranks[cls] = active_rank.to(self.device)
#             if reliability is not None and hasattr(self.geometry_bank, "reliability"):
#                 self.geometry_bank.reliability[cls] = reliability.to(self.device)

#         self.current_num_classes = max(self.current_num_classes, cls + 1)

#     @torch.no_grad()
#     def refresh_inter_class_geometry(self):
#         if hasattr(self.geometry_bank, "refresh_inter_class_geometry"):
#             self.geometry_bank.refresh_inter_class_geometry()

#     # =========================================================
#     # Logit computation
#     # =========================================================
#     def compute_logits_from_features(
#         self,
#         features: torch.Tensor,
#         classifier_mode: str = "geometry_only",
#     ):
#         classifier_mode = str(classifier_mode).lower()
#         subspace_bank = self.get_subspace_bank()
#         anchors = self.get_anchor_bank()
#         concepts = self.get_concept_bank()
#         calibrated_old = self.get_calibrated_old_subspace_bank(self.old_class_count)

#         return self.classifier(
#             features,
#             anchors=anchors if anchors.numel() > 0 else None,
#             concept_bank=concepts if concepts.numel() > 0 else None,
#             subspace_means=subspace_bank["means"] if subspace_bank["means"].numel() > 0 else None,
#             subspace_bases=subspace_bank["bases"] if subspace_bank["bases"].numel() > 0 else None,
#             subspace_variances=subspace_bank["variances"] if subspace_bank["variances"].numel() > 0 else None,
#             calibrated_old_means=calibrated_old["means"] if calibrated_old.get("means", None) is not None and calibrated_old["means"].numel() > 0 else None,
#             calibrated_old_bases=calibrated_old["bases"] if calibrated_old.get("bases", None) is not None and calibrated_old["bases"].numel() > 0 else None,
#             calibrated_old_variances=calibrated_old["variances"] if calibrated_old.get("variances", None) is not None and calibrated_old["variances"].numel() > 0 else None,
#             mode=classifier_mode,
#             old_class_count=int(self.old_class_count),
#         )

#     # =========================================================
#     # Trainability controls
#     # =========================================================
#     def freeze_backbone_only(self):
#         for p in self.backbone.parameters():
#             p.requires_grad = False

#     def freeze_semantic_encoder(self):
#         for p in self.semantic_encoder.parameters():
#             p.requires_grad = False

#     def unfreeze_semantic_encoder(self):
#         for p in self.semantic_encoder.parameters():
#             p.requires_grad = True

#     def freeze_projection_head(self):
#         for p in self.projection.parameters():
#             p.requires_grad = False
#         for p in self.norm.parameters():
#             p.requires_grad = False
#         for p in self.concept_projector.parameters():
#             p.requires_grad = False

#     def unfreeze_projection_head(self):
#         for p in self.projection.parameters():
#             p.requires_grad = True
#         for p in self.norm.parameters():
#             p.requires_grad = True
#         for p in self.concept_projector.parameters():
#             p.requires_grad = True

#     def freeze_old_anchor_deltas(self, old_class_count: int):
#         for i, p in enumerate(self._anchor_deltas):
#             if i < int(old_class_count):
#                 p.requires_grad = False

#     def unfreeze_new_anchor_deltas(self, old_class_count: int):
#         for i, p in enumerate(self._anchor_deltas):
#             p.requires_grad = i >= int(old_class_count)

#     def freeze_old_concept_deltas(self, old_class_count: int):
#         for i, p in enumerate(self._concept_deltas):
#             if i < int(old_class_count):
#                 p.requires_grad = False

#     def freeze_classifier_adaptation(self):
#         if hasattr(self.classifier, "freeze_all_adaptation"):
#             self.classifier.freeze_all_adaptation()

#     def freeze_old_classifier_adaptation(self, old_class_count: int):
#         if hasattr(self.classifier, "freeze_old_adaptation"):
#             self.classifier.freeze_old_adaptation(old_class_count)

#     def unfreeze_classifier_adaptation(self):
#         if hasattr(self.classifier, "unfreeze_all_adaptation"):
#             self.classifier.unfreeze_all_adaptation()

#     def freeze_fusion_module(self):
#         if hasattr(self.classifier, "freeze_fusion_module"):
#             self.classifier.freeze_fusion_module()

#     def unfreeze_fusion_module(self):
#         if hasattr(self.classifier, "unfreeze_fusion_module"):
#             self.classifier.unfreeze_fusion_module()

#     def freeze_geometry_calibrator(self):
#         for p in self.geometry_calibrator.parameters():
#             p.requires_grad = False

#     def unfreeze_geometry_calibrator(self):
#         for p in self.geometry_calibrator.parameters():
#             p.requires_grad = True

#     # =========================================================
#     # Semantic refinement helper
#     # =========================================================
#     def _semantic_refine(
#         self,
#         feat: torch.Tensor,
#         anchors: torch.Tensor,
#         concepts: torch.Tensor,
#         spectral_tokens,
#         spatial_tokens,
#         semantic_mode: str,
#         return_token_relations: bool,
#     ):
#         bypass_semantic = semantic_mode in {"off", "none", "bypass", "identity", "raw"}
#         if anchors.numel() == 0 or bypass_semantic:
#             return feat, None

#         semantic_bank = concepts if concepts.numel() > 0 else anchors.unsqueeze(1)

#         semantic_out = self.semantic_encoder(
#             feat,
#             concept_bank=semantic_bank,
#             phase=int(self.current_phase),
#             disable=False,
#             spectral_tokens=spectral_tokens,
#             spatial_tokens=spatial_tokens,
#             return_token_relations=return_token_relations,
#         )

#         if isinstance(semantic_out, dict):
#             feat_refined = semantic_out["features"]
#             token_relations = semantic_out.get("token_relations", None)
#         else:
#             feat_refined = semantic_out
#             token_relations = None

#         return feat_refined, token_relations

#     # =========================================================
#     # Forward
#     # =========================================================
#     def forward(self, x: torch.Tensor, **kwargs):
#         semantic_mode = self._resolve_semantic_mode(kwargs.get("semantic_mode", "auto"))
#         classifier_mode = self._resolve_classifier_mode(kwargs.get("classifier_mode", None))
#         return_token_relations = bool(kwargs.get("return_token_relations", False))

#         backbone_out = self.extract_features(x)

#         feat = backbone_out["features"]
#         band_weights = backbone_out["band_weights"]
#         spectral_tokens = backbone_out["spectral_tokens"]
#         spatial_tokens = backbone_out["spatial_tokens"]
#         fused_tokens = backbone_out["fused_tokens"]
#         spectral_summary = self._spectral_summary(x)

#         anchors = self.get_anchor_bank()
#         concepts = self.get_concept_bank()

#         feat_pre, token_relations = self._semantic_refine(
#             feat=feat,
#             anchors=anchors,
#             concepts=concepts,
#             spectral_tokens=spectral_tokens,
#             spatial_tokens=spatial_tokens,
#             semantic_mode=semantic_mode,
#             return_token_relations=return_token_relations,
#         )

#         if (
#             return_token_relations
#             and token_relations is None
#             and spectral_tokens is not None
#             and spatial_tokens is not None
#         ):
#             token_relations = self.semantic_encoder.build_token_relations(
#                 spectral_tokens,
#                 spatial_tokens,
#             )

#         projected = self._project(feat_pre)

#         subspace = self.get_subspace_bank()
#         calibrated_old = self.get_calibrated_old_subspace_bank(self.old_class_count)

#         geometry_ref = self.build_geometry_ref(projected)

#         if hasattr(self.geometry_bank, "retrieve_spectral_ref"):
#             spectral_ref = self.geometry_bank.retrieve_spectral_ref(projected)
#         else:
#             spectral_ref = torch.empty(projected.size(0), 0, device=projected.device, dtype=projected.dtype)

#         if hasattr(self.geometry_bank, "retrieve_band_importance_ref"):
#             band_importance_ref = self.geometry_bank.retrieve_band_importance_ref(projected)
#         else:
#             band_importance_ref = torch.empty(projected.size(0), 0, device=projected.device, dtype=projected.dtype)

#         logits = self.compute_logits_from_features(
#             projected,
#             classifier_mode=classifier_mode,
#         )

#         calibration_reg = self.calibration_regularization_loss(self.old_class_count)

#         return {
#             "logits": logits,
#             "features": projected,
#             "preproject_features": feat_pre,
#             "backbone_features": feat,
#             "band_weights": band_weights,
#             "spectral_summary": spectral_summary,
#             "spectral_ref": spectral_ref,
#             "band_importance_ref": band_importance_ref,
#             "geometry_ref": geometry_ref,
#             "anchors": anchors,
#             "concept_bank": concepts,
#             "subspace_means": subspace["means"],
#             "subspace_bases": subspace["bases"],
#             "subspace_variances": subspace["variances"],
#             "subspace_reliability": subspace.get("reliability", None),
#             "subspace_active_ranks": subspace.get("active_ranks", None),
#             "calibrated_old_means": calibrated_old.get("means", None),
#             "calibrated_old_bases": calibrated_old.get("bases", None),
#             "calibrated_old_variances": calibrated_old.get("variances", None),
#             "calibrated_old_reliability": calibrated_old.get("reliability", None),
#             "calibration_reg": calibration_reg,
#             "spectral_tokens": spectral_tokens,
#             "spatial_tokens": spatial_tokens,
#             "fused_tokens": fused_tokens,
#             "token_relations": token_relations,
#             "spectral_features": backbone_out["spectral_features"],
#             "spatial_features": backbone_out["spatial_features"],
#             "spatial_patterns": backbone_out["spatial_patterns"],
#         }
