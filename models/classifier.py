import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class SemanticClassifier(nn.Module):
    """
    Geometry-native classifier for NECIL-HSI.

    Main modes
    ----------
    1) geometry_only
       Scores all classes directly from the current GeometryBank.

    2) calibrated_geometry
       Scores old classes using calibrated old geometry and new classes using
       current new geometry:
           [ calibrated old geometry | current new geometry ]

    Legacy modes
    ------------
    3) anchor_concept
    4) adaptive_hybrid

    These are retained only for ablation/backward compatibility. The proposed
    method should use geometry_only / calibrated_geometry.
    """

    def __init__(
        self,
        initial_classes: int = 0,
        d_model: int = 128,
        logit_scale: float = 8.0,
        use_bias: bool = True,
        variance_floor: float = 1e-4,
        use_geom_temperature: bool = True,
        concept_agg_temperature: float = 0.07,
        init_alpha_old: float = -0.5,
        init_alpha_new: float = -0.2,
        use_adaptive_fusion: bool = False,
        min_temperature: float = 0.25,
        max_temperature: float = 4.0,
        energy_normalize_by_dim: bool = True,
        debias_strength: float = 0.10,
        reliability_energy_weight: float = 0.05,
        max_bias_abs: float = 0.50,
        max_debias_abs: float = 0.25,
    ):
        super().__init__()

        self.num_classes = int(initial_classes)
        self.d_model = int(d_model)
        self.logit_scale = float(logit_scale)
        self.use_bias = bool(use_bias)
        self.variance_floor = float(variance_floor)
        self.use_geom_temperature = bool(use_geom_temperature)
        self.concept_agg_temperature = float(concept_agg_temperature)

        self.init_alpha_old = float(init_alpha_old)
        self.init_alpha_new = float(init_alpha_new)
        self.use_adaptive_fusion = bool(use_adaptive_fusion)

        self.min_temperature = float(min_temperature)
        self.max_temperature = float(max_temperature)
        self.energy_normalize_by_dim = bool(energy_normalize_by_dim)
        self.debias_strength = float(debias_strength)
        self.reliability_energy_weight = float(reliability_energy_weight)
        self.max_bias_abs = float(max_bias_abs)
        self.max_debias_abs = float(max_debias_abs)

        # ------------------------------------------------------------
        # Learnable calibration parameters
        # ------------------------------------------------------------
        self.bias = nn.Parameter(
            torch.zeros(self.num_classes),
            requires_grad=self.use_bias,
        )

        # Legacy fusion scalar per class. Kept only for compatibility.
        self.alpha = nn.Parameter(
            torch.full((self.num_classes,), self.init_alpha_old)
        )

        # Raw class temperature. Converted to bounded temperature in _safe_temperature.
        self.geom_temperature = nn.Parameter(torch.zeros(self.num_classes))

        # Old/new debias offsets. These are intentionally bounded at application.
        self.old_bias_offset = nn.Parameter(torch.zeros(self.num_classes))
        self.new_bias_offset = nn.Parameter(torch.zeros(self.num_classes))

        # Old/new temperature offsets. These are also bounded.
        self.old_temp_offset = nn.Parameter(torch.zeros(self.num_classes))
        self.new_temp_offset = nn.Parameter(torch.zeros(self.num_classes))

        if self.use_adaptive_fusion:
            self.fusion_mlp = nn.Sequential(
                nn.Linear(3, 16),
                nn.GELU(),
                nn.Linear(16, 3),
            )
        else:
            self.fusion_mlp = None

        # ------------------------------------------------------------
        # Gradient masks
        # ------------------------------------------------------------
        self.register_buffer("bias_grad_mask", torch.ones(self.num_classes))
        self.register_buffer("alpha_grad_mask", torch.ones(self.num_classes))
        self.register_buffer("temp_grad_mask", torch.ones(self.num_classes))
        self.register_buffer("old_bias_offset_mask", torch.ones(self.num_classes))
        self.register_buffer("new_bias_offset_mask", torch.ones(self.num_classes))
        self.register_buffer("old_temp_offset_mask", torch.ones(self.num_classes))
        self.register_buffer("new_temp_offset_mask", torch.ones(self.num_classes))

        self._hook_handles = []
        self._register_gradient_hooks()

    # ============================================================
    # Hook registration
    # ============================================================
    def _clear_hooks(self):
        for h in getattr(self, "_hook_handles", []):
            try:
                h.remove()
            except Exception:
                pass
        self._hook_handles = []

    def _register_gradient_hooks(self):
        """
        Important:
        expand() replaces Parameter objects. Re-register hooks after expansion,
        but first remove old handles to avoid accumulating stale hooks.
        """
        self._clear_hooks()

        if self.use_bias:
            self._hook_handles.append(
                self.bias.register_hook(lambda g: g * self.bias_grad_mask)
            )

        self._hook_handles.append(
            self.alpha.register_hook(lambda g: g * self.alpha_grad_mask)
        )

        if self.use_geom_temperature:
            self._hook_handles.append(
                self.geom_temperature.register_hook(lambda g: g * self.temp_grad_mask)
            )

        self._hook_handles.append(
            self.old_bias_offset.register_hook(lambda g: g * self.old_bias_offset_mask)
        )
        self._hook_handles.append(
            self.new_bias_offset.register_hook(lambda g: g * self.new_bias_offset_mask)
        )
        self._hook_handles.append(
            self.old_temp_offset.register_hook(lambda g: g * self.old_temp_offset_mask)
        )
        self._hook_handles.append(
            self.new_temp_offset.register_hook(lambda g: g * self.new_temp_offset_mask)
        )

    # ============================================================
    # Expansion
    # ============================================================
    def expand(self, num_new_classes: int, phase: int):
        del phase

        old = int(self.num_classes)
        num_new_classes = int(num_new_classes)
        if num_new_classes <= 0:
            return

        self.num_classes += num_new_classes

        # Use a stable device/dtype anchor.
        ref_param = self.alpha
        device = ref_param.device
        dtype = ref_param.dtype

        def _expand_param(old_param: nn.Parameter, fill_value: float) -> nn.Parameter:
            new_p = torch.full((self.num_classes,), fill_value, device=device, dtype=dtype)
            if old > 0:
                new_p[:old] = old_param.data
            return nn.Parameter(new_p, requires_grad=True)

        if self.use_bias:
            self.bias = _expand_param(self.bias, 0.0)

        self.alpha = _expand_param(self.alpha, self.init_alpha_new)
        self.geom_temperature = _expand_param(self.geom_temperature, 0.0)

        self.old_bias_offset = _expand_param(self.old_bias_offset, 0.0)
        self.new_bias_offset = _expand_param(self.new_bias_offset, 0.0)
        self.old_temp_offset = _expand_param(self.old_temp_offset, 0.0)
        self.new_temp_offset = _expand_param(self.new_temp_offset, 0.0)

        def _expand_mask(old_mask: Optional[torch.Tensor]) -> torch.Tensor:
            new_mask = torch.ones(self.num_classes, device=device, dtype=dtype)
            if old > 0 and old_mask is not None and old_mask.numel() > 0:
                new_mask[:old] = old_mask[:old].to(device=device, dtype=dtype)
            return new_mask

        old_bias_mask = self.bias_grad_mask.clone() if hasattr(self, "bias_grad_mask") else None
        old_alpha_mask = self.alpha_grad_mask.clone() if hasattr(self, "alpha_grad_mask") else None
        old_temp_mask = self.temp_grad_mask.clone() if hasattr(self, "temp_grad_mask") else None
        old_old_bias_mask = self.old_bias_offset_mask.clone() if hasattr(self, "old_bias_offset_mask") else None
        old_new_bias_mask = self.new_bias_offset_mask.clone() if hasattr(self, "new_bias_offset_mask") else None
        old_old_temp_mask = self.old_temp_offset_mask.clone() if hasattr(self, "old_temp_offset_mask") else None
        old_new_temp_mask = self.new_temp_offset_mask.clone() if hasattr(self, "new_temp_offset_mask") else None

        self.register_buffer("bias_grad_mask", _expand_mask(old_bias_mask))
        self.register_buffer("alpha_grad_mask", _expand_mask(old_alpha_mask))
        self.register_buffer("temp_grad_mask", _expand_mask(old_temp_mask))
        self.register_buffer("old_bias_offset_mask", _expand_mask(old_old_bias_mask))
        self.register_buffer("new_bias_offset_mask", _expand_mask(old_new_bias_mask))
        self.register_buffer("old_temp_offset_mask", _expand_mask(old_old_temp_mask))
        self.register_buffer("new_temp_offset_mask", _expand_mask(old_new_temp_mask))

        self._register_gradient_hooks()

    # ============================================================
    # Adaptation control
    # ============================================================
    def freeze_all_adaptation(self):
        if self.use_bias:
            self.bias_grad_mask.zero_()
            self.bias.requires_grad = True

        self.alpha_grad_mask.zero_()
        self.temp_grad_mask.zero_()
        self.old_bias_offset_mask.zero_()
        self.new_bias_offset_mask.zero_()
        self.old_temp_offset_mask.zero_()
        self.new_temp_offset_mask.zero_()

        self.alpha.requires_grad = True
        self.geom_temperature.requires_grad = True
        self.old_bias_offset.requires_grad = True
        self.new_bias_offset.requires_grad = True
        self.old_temp_offset.requires_grad = True
        self.new_temp_offset.requires_grad = True

    def unfreeze_all_adaptation(self):
        if self.use_bias:
            self.bias_grad_mask.fill_(1.0)
            self.bias.requires_grad = True

        self.alpha_grad_mask.fill_(1.0)
        self.temp_grad_mask.fill_(1.0)
        self.old_bias_offset_mask.fill_(1.0)
        self.new_bias_offset_mask.fill_(1.0)
        self.old_temp_offset_mask.fill_(1.0)
        self.new_temp_offset_mask.fill_(1.0)

        self.alpha.requires_grad = True
        self.geom_temperature.requires_grad = True
        self.old_bias_offset.requires_grad = True
        self.new_bias_offset.requires_grad = True
        self.old_temp_offset.requires_grad = True
        self.new_temp_offset.requires_grad = True

    def freeze_old_adaptation(self, old_class_count: int):
        old_class_count = int(old_class_count)

        if self.use_bias:
            self.bias_grad_mask.fill_(1.0)
            self.bias_grad_mask[:old_class_count] = 0.0

        self.alpha_grad_mask.fill_(1.0)
        self.alpha_grad_mask[:old_class_count] = 0.0

        self.temp_grad_mask.fill_(1.0)
        self.temp_grad_mask[:old_class_count] = 0.0

        self.old_bias_offset_mask.fill_(1.0)
        self.old_bias_offset_mask[:old_class_count] = 0.0

        self.new_bias_offset_mask.fill_(1.0)
        self.new_bias_offset_mask[:old_class_count] = 0.0

        self.old_temp_offset_mask.fill_(1.0)
        self.old_temp_offset_mask[:old_class_count] = 0.0

        self.new_temp_offset_mask.fill_(1.0)
        self.new_temp_offset_mask[:old_class_count] = 0.0

    def freeze_fusion_module(self):
        if self.fusion_mlp is None:
            return
        for p in self.fusion_mlp.parameters():
            p.requires_grad = False

    def unfreeze_fusion_module(self):
        if self.fusion_mlp is None:
            return
        for p in self.fusion_mlp.parameters():
            p.requires_grad = True

    # ============================================================
    # Helpers
    # ============================================================
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1, eps=1e-6)

    def _bounded_temperature_from_raw(self, raw: torch.Tensor) -> torch.Tensor:
        """
        Map raw parameter to [min_temperature, max_temperature].
        """
        return self.min_temperature + (self.max_temperature - self.min_temperature) * torch.sigmoid(raw)

    def _safe_temperature(self, num_classes: Optional[int] = None) -> torch.Tensor:
        if not self.use_geom_temperature:
            if num_classes is None:
                num_classes = self.num_classes
            return torch.ones(num_classes, device=self.geom_temperature.device, dtype=self.geom_temperature.dtype)

        t = self._bounded_temperature_from_raw(self.geom_temperature)

        if num_classes is None:
            return t
        if t.numel() == num_classes:
            return t
        if t.numel() > num_classes:
            return t[:num_classes]

        pad = torch.ones(num_classes - t.numel(), device=t.device, dtype=t.dtype)
        return torch.cat([t, pad], dim=0)

    def _safe_concept_temperature(self) -> float:
        return max(self.concept_agg_temperature, 1e-6)

    def _apply_bias(self, logits: torch.Tensor) -> torch.Tensor:
        if self.use_bias and self.bias.numel() > 0:
            b = self.bias[:logits.size(1)]
            if b.numel() < logits.size(1):
                pad = torch.zeros(logits.size(1) - b.numel(), device=logits.device, dtype=logits.dtype)
                b = torch.cat([b.to(logits.device, logits.dtype), pad], dim=0)
            else:
                b = b.to(logits.device, logits.dtype)
            b = self.max_bias_abs * torch.tanh(b / max(self.max_bias_abs, 1e-6))
            logits = logits + b.unsqueeze(0)
        return logits

    # ============================================================
    # Legacy normalized auxiliary logits
    # ============================================================
    def _anchor_logits(self, f: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        f_n = self._normalize(f)
        a_n = self._normalize(anchors)
        return self.logit_scale * F.linear(f_n, a_n)

    def _concept_logits(self, f: torch.Tensor, concept_bank: torch.Tensor) -> torch.Tensor:
        f_n = self._normalize(f)
        c_n = self._normalize(concept_bank)

        sim = torch.einsum("bd,ckd->bck", f_n, c_n)
        tau = self._safe_concept_temperature()
        concept_score = tau * torch.logsumexp(sim / tau, dim=-1)
        return self.logit_scale * concept_score

    # ============================================================
    # Geometry energy
    # ============================================================
    def _validate_geometry_inputs(
        self,
        f: torch.Tensor,
        means: torch.Tensor,
        bases: torch.Tensor,
        vars_: torch.Tensor,
    ):
        if f.dim() != 2:
            raise ValueError(f"Expected features [B,D], got {tuple(f.shape)}")
        if means is None or bases is None or vars_ is None:
            raise ValueError("Geometry tensors cannot be None.")
        if means.dim() != 2:
            raise ValueError(f"means must be [C,D], got {tuple(means.shape)}")
        if bases.dim() != 3:
            raise ValueError(f"bases must be [C,D,R], got {tuple(bases.shape)}")
        if vars_.dim() != 2:
            raise ValueError(f"vars must be [C,R+1], got {tuple(vars_.shape)}")
        if f.size(1) != means.size(1) or f.size(1) != bases.size(1):
            raise ValueError(
                f"Feature/geometry dim mismatch: f={f.size(1)}, means={means.size(1)}, bases={bases.size(1)}"
            )
        if means.size(0) != bases.size(0) or means.size(0) != vars_.size(0):
            raise ValueError(
                f"Class count mismatch: means={means.size(0)}, bases={bases.size(0)}, vars={vars_.size(0)}"
            )
        if bases.size(2) + 1 != vars_.size(1):
            raise ValueError(
                f"Rank/variance mismatch: bases rank={bases.size(2)}, vars dim={vars_.size(1)}"
            )

    def _geometry_energy(
        self,
        f: torch.Tensor,
        means: torch.Tensor,
        bases: torch.Tensor,
        vars_: torch.Tensor,
        reliability: Optional[torch.Tensor] = None,
        active_ranks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Low-rank Gaussian/Mahalanobis-style energy. Lower is better.

        f:      [B, D]
        means:  [C, D]
        bases:  [C, D, R]
        vars_:  [C, R+1]
        """
        self._validate_geometry_inputs(f, means, bases, vars_)

        means = means.to(device=f.device, dtype=f.dtype)
        bases = bases.to(device=f.device, dtype=f.dtype)
        vars_ = vars_.to(device=f.device, dtype=f.dtype)

        residual = f.unsqueeze(1) - means.unsqueeze(0)       # [B,C,D]
        coeff = torch.einsum("bcd,cdr->bcr", residual, bases)
        recon = torch.einsum("bcr,cdr->bcd", coeff, bases)
        orth = residual - recon

        eigvals = vars_[:, :-1].clamp_min(self.variance_floor)
        resvar = vars_[:, -1].clamp_min(self.variance_floor)

        if active_ranks is not None and torch.is_tensor(active_ranks) and active_ranks.numel() == means.size(0):
            ar = active_ranks.to(device=f.device).long().clamp(0, bases.size(2))
            rank_mask = torch.arange(bases.size(2), device=f.device).unsqueeze(0) < ar.unsqueeze(1)
            rank_mask = rank_mask.to(dtype=f.dtype)

            # Inactive dimensions are ignored in the parallel term.
            parallel_term = ((coeff ** 2) / eigvals.unsqueeze(0)) * rank_mask.unsqueeze(0)
            parallel_term = parallel_term.sum(dim=-1)
        else:
            parallel_term = ((coeff ** 2) / eigvals.unsqueeze(0)).sum(dim=-1)

        orth_dist2 = orth.pow(2).sum(dim=-1)
        orth_term = orth_dist2 / resvar.unsqueeze(0)

        energy = parallel_term + orth_term

        if self.energy_normalize_by_dim:
            energy = energy / max(f.size(1), 1)

        # Reliability should not dominate classification, but unreliable geometry
        # should be slightly penalized to reduce false confidence.
        if reliability is not None and torch.is_tensor(reliability) and reliability.numel() == means.size(0):
            rel = reliability.to(device=f.device, dtype=f.dtype).clamp(0.05, 1.0)
            reliability_penalty = -torch.log(rel).unsqueeze(0)
            energy = energy + self.reliability_energy_weight * reliability_penalty

        return torch.nan_to_num(energy, nan=1e6, posinf=1e6, neginf=1e6)

    def _geometry_logits(
        self,
        f: torch.Tensor,
        means: torch.Tensor,
        bases: torch.Tensor,
        vars_: torch.Tensor,
        reliability: Optional[torch.Tensor] = None,
        active_ranks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        energy = self._geometry_energy(
            f,
            means,
            bases,
            vars_,
            reliability=reliability,
            active_ranks=active_ranks,
        )
        temp = self._safe_temperature(num_classes=means.size(0)).to(device=f.device, dtype=f.dtype).unsqueeze(0)
        logits = -energy / temp
        return self.logit_scale * logits

    # ============================================================
    # Fusion / debias
    # ============================================================
    def _static_alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha).unsqueeze(0)

    def _adaptive_fusion_logits(
        self,
        anchor_logits: torch.Tensor,
        concept_logits: torch.Tensor,
        geom_logits: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_adaptive_fusion or self.fusion_mlp is None:
            alpha = self._static_alpha()
            alpha = alpha[:, :anchor_logits.size(1)]
            return alpha * anchor_logits + (1.0 - alpha) * geom_logits

        stacked = torch.stack([anchor_logits, concept_logits, geom_logits], dim=-1)
        gate_logits = self.fusion_mlp(stacked)
        gate = F.softmax(gate_logits, dim=-1)

        return (
            gate[..., 0] * anchor_logits
            + gate[..., 1] * concept_logits
            + gate[..., 2] * geom_logits
        )

    def _apply_old_new_debias(
        self,
        logits: torch.Tensor,
        old_class_count: int,
    ) -> torch.Tensor:
        if old_class_count <= 0 or old_class_count >= logits.size(1):
            return logits

        C = logits.size(1)
        device = logits.device
        dtype = logits.dtype

        old_mask = torch.zeros(C, device=device, dtype=dtype)
        new_mask = torch.zeros(C, device=device, dtype=dtype)
        old_mask[:old_class_count] = 1.0
        new_mask[old_class_count:] = 1.0

        # Bound offsets so debias cannot become a hidden classifier.
        debias_bound = min(float(self.debias_strength), float(self.max_debias_abs))
        old_bias = debias_bound * torch.tanh(self.old_bias_offset[:C]).to(device=device, dtype=dtype)
        new_bias = debias_bound * torch.tanh(self.new_bias_offset[:C]).to(device=device, dtype=dtype)

        logits = logits + old_mask.unsqueeze(0) * old_bias.unsqueeze(0)
        logits = logits - new_mask.unsqueeze(0) * new_bias.unsqueeze(0)

        old_temp_raw = self.old_temp_offset[:C].to(device=device, dtype=dtype)
        new_temp_raw = self.new_temp_offset[:C].to(device=device, dtype=dtype)

        old_temp = self._bounded_temperature_from_raw(old_temp_raw)
        new_temp = self._bounded_temperature_from_raw(new_temp_raw)

        temp = torch.ones(C, device=device, dtype=dtype)
        temp[:old_class_count] = old_temp[:old_class_count]
        temp[old_class_count:] = new_temp[old_class_count:]

        return logits / temp.unsqueeze(0)

    def adaptation_regularization_loss(self) -> Dict[str, torch.Tensor]:
        """
        Penalize classifier calibration from becoming the actual classifier.

        Add this to a trainer if you want stricter geometry-native behavior:
            reg = model.classifier.adaptation_regularization_loss()["total"]
        """
        if self.old_bias_offset.numel() == 0:
            z = torch.tensor(0.0, device=self.alpha.device, dtype=self.alpha.dtype)
            return {"total": z, "bias": z, "temp": z, "alpha": z}

        bias_reg = (
            self.old_bias_offset.pow(2).mean()
            + self.new_bias_offset.pow(2).mean()
        )

        temp_reg = (
            self.old_temp_offset.pow(2).mean()
            + self.new_temp_offset.pow(2).mean()
            + self.geom_temperature.pow(2).mean()
        )

        alpha_reg = self.alpha.pow(2).mean() if self.alpha.numel() > 0 else torch.tensor(
            0.0, device=self.old_bias_offset.device, dtype=self.old_bias_offset.dtype
        )

        total = bias_reg + 0.25 * temp_reg + 0.01 * alpha_reg
        return {
            "total": total,
            "bias": bias_reg,
            "temp": temp_reg,
            "alpha": alpha_reg,
        }

    # ============================================================
    # Bank resolution
    # ============================================================
    def _resolve_calibrated_geometry_bank(
        self,
        *,
        subspace_means: Optional[torch.Tensor],
        subspace_bases: Optional[torch.Tensor],
        subspace_variances: Optional[torch.Tensor],
        calibrated_old_means: Optional[torch.Tensor],
        calibrated_old_bases: Optional[torch.Tensor],
        calibrated_old_variances: Optional[torch.Tensor],
        old_class_count: int,
        subspace_reliability: Optional[torch.Tensor] = None,
        calibrated_old_reliability: Optional[torch.Tensor] = None,
        subspace_active_ranks: Optional[torch.Tensor] = None,
        calibrated_old_active_ranks: Optional[torch.Tensor] = None,
    ):
        if (
            subspace_means is None
            or subspace_bases is None
            or subspace_variances is None
            or subspace_means.numel() == 0
        ):
            raise ValueError("Geometry bank is required for geometry scoring.")

        if old_class_count <= 0:
            return subspace_means, subspace_bases, subspace_variances, subspace_reliability, subspace_active_ranks

        if (
            calibrated_old_means is None
            or calibrated_old_bases is None
            or calibrated_old_variances is None
            or calibrated_old_means.numel() == 0
        ):
            return subspace_means, subspace_bases, subspace_variances, subspace_reliability, subspace_active_ranks

        if old_class_count > subspace_means.size(0):
            raise ValueError(
                f"old_class_count={old_class_count} exceeds available classes={subspace_means.size(0)}."
            )

        new_means = subspace_means[old_class_count:]
        new_bases = subspace_bases[old_class_count:]
        new_vars = subspace_variances[old_class_count:]

        means = torch.cat([calibrated_old_means, new_means], dim=0)
        bases = torch.cat([calibrated_old_bases, new_bases], dim=0)
        vars_ = torch.cat([calibrated_old_variances, new_vars], dim=0)

        reliability = None
        if subspace_reliability is not None and torch.is_tensor(subspace_reliability) and subspace_reliability.numel() >= subspace_means.size(0):
            new_rel = subspace_reliability[old_class_count:]
            if calibrated_old_reliability is not None and torch.is_tensor(calibrated_old_reliability):
                old_rel = calibrated_old_reliability[:old_class_count]
            else:
                old_rel = subspace_reliability[:old_class_count]
            reliability = torch.cat([old_rel, new_rel], dim=0)

        active_ranks = None
        if subspace_active_ranks is not None and torch.is_tensor(subspace_active_ranks) and subspace_active_ranks.numel() >= subspace_means.size(0):
            new_ar = subspace_active_ranks[old_class_count:]
            if calibrated_old_active_ranks is not None and torch.is_tensor(calibrated_old_active_ranks):
                old_ar = calibrated_old_active_ranks[:old_class_count]
            else:
                old_ar = subspace_active_ranks[:old_class_count]
            active_ranks = torch.cat([old_ar, new_ar], dim=0)

        return means, bases, vars_, reliability, active_ranks

    # ============================================================
    # Forward
    # ============================================================
    def forward(
        self,
        features: torch.Tensor,
        *,
        anchors: Optional[torch.Tensor] = None,
        concept_bank: Optional[torch.Tensor] = None,
        subspace_means: Optional[torch.Tensor] = None,
        subspace_bases: Optional[torch.Tensor] = None,
        subspace_variances: Optional[torch.Tensor] = None,
        subspace_reliability: Optional[torch.Tensor] = None,
        subspace_active_ranks: Optional[torch.Tensor] = None,
        calibrated_old_means: Optional[torch.Tensor] = None,
        calibrated_old_bases: Optional[torch.Tensor] = None,
        calibrated_old_variances: Optional[torch.Tensor] = None,
        calibrated_old_reliability: Optional[torch.Tensor] = None,
        calibrated_old_active_ranks: Optional[torch.Tensor] = None,
        mode: str = "geometry_only",
        old_class_count: int = 0,
        return_energy: bool = False,
        **kwargs,
    ):
        del kwargs
        mode = str(mode).lower()

        if features.dim() != 2:
            raise ValueError(f"Expected features to be [B, D], got {tuple(features.shape)}")

        # --------------------------------------------------------
        # Geometry-only scoring
        # --------------------------------------------------------
        if mode == "geometry_only":
            if (
                subspace_means is None
                or subspace_bases is None
                or subspace_variances is None
                or subspace_means.numel() == 0
            ):
                raise ValueError("geometry_only mode requires subspace bank.")

            energy = self._geometry_energy(
                features,
                subspace_means,
                subspace_bases,
                subspace_variances,
                reliability=subspace_reliability,
                active_ranks=subspace_active_ranks,
            )
            logits = self._geometry_logits(
                features,
                subspace_means,
                subspace_bases,
                subspace_variances,
                reliability=subspace_reliability,
                active_ranks=subspace_active_ranks,
            )
            logits = self._apply_bias(logits)
            logits = self._apply_old_new_debias(logits, old_class_count)

            if return_energy:
                return {"logits": logits, "energy": energy}
            return logits

        # --------------------------------------------------------
        # Calibrated old geometry + current new geometry
        # --------------------------------------------------------
        if mode == "calibrated_geometry":
            means, bases, vars_, reliability, active_ranks = self._resolve_calibrated_geometry_bank(
                subspace_means=subspace_means,
                subspace_bases=subspace_bases,
                subspace_variances=subspace_variances,
                calibrated_old_means=calibrated_old_means,
                calibrated_old_bases=calibrated_old_bases,
                calibrated_old_variances=calibrated_old_variances,
                old_class_count=old_class_count,
                subspace_reliability=subspace_reliability,
                calibrated_old_reliability=calibrated_old_reliability,
                subspace_active_ranks=subspace_active_ranks,
                calibrated_old_active_ranks=calibrated_old_active_ranks,
            )

            energy = self._geometry_energy(
                features,
                means,
                bases,
                vars_,
                reliability=reliability,
                active_ranks=active_ranks,
            )
            logits = self._geometry_logits(
                features,
                means,
                bases,
                vars_,
                reliability=reliability,
                active_ranks=active_ranks,
            )
            logits = self._apply_bias(logits)
            logits = self._apply_old_new_debias(logits, old_class_count)

            if return_energy:
                return {"logits": logits, "energy": energy}
            return logits

        # --------------------------------------------------------
        # Legacy anchor/concept scoring
        # --------------------------------------------------------
        if mode == "anchor_concept":
            if anchors is None or anchors.numel() == 0:
                raise ValueError("anchor_concept mode requires anchors.")
            if concept_bank is None or concept_bank.numel() == 0:
                raise ValueError("anchor_concept mode requires concept_bank.")

            anchor_logits = self._anchor_logits(features, anchors)
            concept_logits = self._concept_logits(features, concept_bank)

            alpha = self._static_alpha()[:, :anchor_logits.size(1)]
            logits = alpha * anchor_logits + (1.0 - alpha) * concept_logits
            logits = self._apply_bias(logits)
            logits = self._apply_old_new_debias(logits, old_class_count)
            return logits

        # --------------------------------------------------------
        # Legacy hybrid scoring
        # --------------------------------------------------------
        if mode == "adaptive_hybrid":
            if anchors is None or anchors.numel() == 0:
                raise ValueError("adaptive_hybrid mode requires anchors.")
            if concept_bank is None or concept_bank.numel() == 0:
                raise ValueError("adaptive_hybrid mode requires concept_bank.")
            if (
                subspace_means is None
                or subspace_bases is None
                or subspace_variances is None
                or subspace_means.numel() == 0
            ):
                raise ValueError("adaptive_hybrid mode requires subspace bank.")

            anchor_logits = self._anchor_logits(features, anchors)
            concept_logits = self._concept_logits(features, concept_bank)
            geom_logits = self._geometry_logits(
                features,
                subspace_means,
                subspace_bases,
                subspace_variances,
                reliability=subspace_reliability,
                active_ranks=subspace_active_ranks,
            )

            logits = self._adaptive_fusion_logits(
                anchor_logits=anchor_logits,
                concept_logits=concept_logits,
                geom_logits=geom_logits,
            )
            logits = self._apply_bias(logits)
            logits = self._apply_old_new_debias(logits, old_class_count)
            return logits

        raise ValueError(f"Unsupported mode: {mode}")



# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Optional, Dict


# class SemanticClassifier(nn.Module):
#     """
#     Geometry-native classifier for NECIL-HSI.

#     Main modes
#     ----------
#     1) geometry_only
#        Scores all classes directly from the current GeometryBank.

#     2) calibrated_geometry
#        Scores old classes using calibrated old geometry and new classes using
#        current new geometry:
#            [ calibrated old geometry | current new geometry ]

#     Legacy modes
#     ------------
#     3) anchor_concept
#     4) adaptive_hybrid

#     These are retained only for ablation/backward compatibility. The proposed
#     method should use geometry_only / calibrated_geometry.
#     """

#     def __init__(
#         self,
#         initial_classes: int = 0,
#         d_model: int = 128,
#         logit_scale: float = 8.0,
#         use_bias: bool = True,
#         variance_floor: float = 1e-4,
#         use_geom_temperature: bool = True,
#         concept_agg_temperature: float = 0.07,
#         init_alpha_old: float = -0.5,
#         init_alpha_new: float = -0.2,
#         use_adaptive_fusion: bool = False,
#         min_temperature: float = 0.25,
#         max_temperature: float = 4.0,
#         energy_normalize_by_dim: bool = True,
#         debias_strength: float = 0.10,
#     ):
#         super().__init__()

#         self.num_classes = int(initial_classes)
#         self.d_model = int(d_model)
#         self.logit_scale = float(logit_scale)
#         self.use_bias = bool(use_bias)
#         self.variance_floor = float(variance_floor)
#         self.use_geom_temperature = bool(use_geom_temperature)
#         self.concept_agg_temperature = float(concept_agg_temperature)

#         self.init_alpha_old = float(init_alpha_old)
#         self.init_alpha_new = float(init_alpha_new)
#         self.use_adaptive_fusion = bool(use_adaptive_fusion)

#         self.min_temperature = float(min_temperature)
#         self.max_temperature = float(max_temperature)
#         self.energy_normalize_by_dim = bool(energy_normalize_by_dim)
#         self.debias_strength = float(debias_strength)

#         # ------------------------------------------------------------
#         # Learnable calibration parameters
#         # ------------------------------------------------------------
#         self.bias = nn.Parameter(
#             torch.zeros(self.num_classes),
#             requires_grad=self.use_bias,
#         )

#         # Legacy fusion scalar per class. Kept only for compatibility.
#         self.alpha = nn.Parameter(
#             torch.full((self.num_classes,), self.init_alpha_old)
#         )

#         # Raw class temperature. Converted to bounded temperature in _safe_temperature.
#         self.geom_temperature = nn.Parameter(torch.zeros(self.num_classes))

#         # Old/new debias offsets. These are intentionally bounded at application.
#         self.old_bias_offset = nn.Parameter(torch.zeros(self.num_classes))
#         self.new_bias_offset = nn.Parameter(torch.zeros(self.num_classes))

#         # Old/new temperature offsets. These are also bounded.
#         self.old_temp_offset = nn.Parameter(torch.zeros(self.num_classes))
#         self.new_temp_offset = nn.Parameter(torch.zeros(self.num_classes))

#         if self.use_adaptive_fusion:
#             self.fusion_mlp = nn.Sequential(
#                 nn.Linear(3, 16),
#                 nn.GELU(),
#                 nn.Linear(16, 3),
#             )
#         else:
#             self.fusion_mlp = None

#         # ------------------------------------------------------------
#         # Gradient masks
#         # ------------------------------------------------------------
#         self.register_buffer("bias_grad_mask", torch.ones(self.num_classes))
#         self.register_buffer("alpha_grad_mask", torch.ones(self.num_classes))
#         self.register_buffer("temp_grad_mask", torch.ones(self.num_classes))
#         self.register_buffer("old_bias_offset_mask", torch.ones(self.num_classes))
#         self.register_buffer("new_bias_offset_mask", torch.ones(self.num_classes))
#         self.register_buffer("old_temp_offset_mask", torch.ones(self.num_classes))
#         self.register_buffer("new_temp_offset_mask", torch.ones(self.num_classes))

#         self._hook_handles = []
#         self._register_gradient_hooks()

#     # ============================================================
#     # Hook registration
#     # ============================================================
#     def _clear_hooks(self):
#         for h in getattr(self, "_hook_handles", []):
#             try:
#                 h.remove()
#             except Exception:
#                 pass
#         self._hook_handles = []

#     def _register_gradient_hooks(self):
#         """
#         Important:
#         expand() replaces Parameter objects. Re-register hooks after expansion,
#         but first remove old handles to avoid accumulating stale hooks.
#         """
#         self._clear_hooks()

#         if self.use_bias:
#             self._hook_handles.append(
#                 self.bias.register_hook(lambda g: g * self.bias_grad_mask)
#             )

#         self._hook_handles.append(
#             self.alpha.register_hook(lambda g: g * self.alpha_grad_mask)
#         )

#         if self.use_geom_temperature:
#             self._hook_handles.append(
#                 self.geom_temperature.register_hook(lambda g: g * self.temp_grad_mask)
#             )

#         self._hook_handles.append(
#             self.old_bias_offset.register_hook(lambda g: g * self.old_bias_offset_mask)
#         )
#         self._hook_handles.append(
#             self.new_bias_offset.register_hook(lambda g: g * self.new_bias_offset_mask)
#         )
#         self._hook_handles.append(
#             self.old_temp_offset.register_hook(lambda g: g * self.old_temp_offset_mask)
#         )
#         self._hook_handles.append(
#             self.new_temp_offset.register_hook(lambda g: g * self.new_temp_offset_mask)
#         )

#     # ============================================================
#     # Expansion
#     # ============================================================
#     def expand(self, num_new_classes: int, phase: int):
#         del phase

#         old = int(self.num_classes)
#         num_new_classes = int(num_new_classes)
#         if num_new_classes <= 0:
#             return

#         self.num_classes += num_new_classes

#         # Use a stable device/dtype anchor.
#         ref_param = self.alpha
#         device = ref_param.device
#         dtype = ref_param.dtype

#         def _expand_param(old_param: nn.Parameter, fill_value: float) -> nn.Parameter:
#             new_p = torch.full((self.num_classes,), fill_value, device=device, dtype=dtype)
#             if old > 0:
#                 new_p[:old] = old_param.data
#             return nn.Parameter(new_p, requires_grad=True)

#         if self.use_bias:
#             self.bias = _expand_param(self.bias, 0.0)

#         self.alpha = _expand_param(self.alpha, self.init_alpha_new)
#         self.geom_temperature = _expand_param(self.geom_temperature, 0.0)

#         self.old_bias_offset = _expand_param(self.old_bias_offset, 0.0)
#         self.new_bias_offset = _expand_param(self.new_bias_offset, 0.0)
#         self.old_temp_offset = _expand_param(self.old_temp_offset, 0.0)
#         self.new_temp_offset = _expand_param(self.new_temp_offset, 0.0)

#         def _expand_mask(old_mask: Optional[torch.Tensor]) -> torch.Tensor:
#             new_mask = torch.ones(self.num_classes, device=device, dtype=dtype)
#             if old > 0 and old_mask is not None and old_mask.numel() > 0:
#                 new_mask[:old] = old_mask[:old].to(device=device, dtype=dtype)
#             return new_mask

#         old_bias_mask = self.bias_grad_mask.clone() if hasattr(self, "bias_grad_mask") else None
#         old_alpha_mask = self.alpha_grad_mask.clone() if hasattr(self, "alpha_grad_mask") else None
#         old_temp_mask = self.temp_grad_mask.clone() if hasattr(self, "temp_grad_mask") else None
#         old_old_bias_mask = self.old_bias_offset_mask.clone() if hasattr(self, "old_bias_offset_mask") else None
#         old_new_bias_mask = self.new_bias_offset_mask.clone() if hasattr(self, "new_bias_offset_mask") else None
#         old_old_temp_mask = self.old_temp_offset_mask.clone() if hasattr(self, "old_temp_offset_mask") else None
#         old_new_temp_mask = self.new_temp_offset_mask.clone() if hasattr(self, "new_temp_offset_mask") else None

#         self.register_buffer("bias_grad_mask", _expand_mask(old_bias_mask))
#         self.register_buffer("alpha_grad_mask", _expand_mask(old_alpha_mask))
#         self.register_buffer("temp_grad_mask", _expand_mask(old_temp_mask))
#         self.register_buffer("old_bias_offset_mask", _expand_mask(old_old_bias_mask))
#         self.register_buffer("new_bias_offset_mask", _expand_mask(old_new_bias_mask))
#         self.register_buffer("old_temp_offset_mask", _expand_mask(old_old_temp_mask))
#         self.register_buffer("new_temp_offset_mask", _expand_mask(old_new_temp_mask))

#         self._register_gradient_hooks()

#     # ============================================================
#     # Adaptation control
#     # ============================================================
#     def freeze_all_adaptation(self):
#         if self.use_bias:
#             self.bias_grad_mask.zero_()
#             self.bias.requires_grad = True

#         self.alpha_grad_mask.zero_()
#         self.temp_grad_mask.zero_()
#         self.old_bias_offset_mask.zero_()
#         self.new_bias_offset_mask.zero_()
#         self.old_temp_offset_mask.zero_()
#         self.new_temp_offset_mask.zero_()

#         self.alpha.requires_grad = True
#         self.geom_temperature.requires_grad = True
#         self.old_bias_offset.requires_grad = True
#         self.new_bias_offset.requires_grad = True
#         self.old_temp_offset.requires_grad = True
#         self.new_temp_offset.requires_grad = True

#     def unfreeze_all_adaptation(self):
#         if self.use_bias:
#             self.bias_grad_mask.fill_(1.0)
#             self.bias.requires_grad = True

#         self.alpha_grad_mask.fill_(1.0)
#         self.temp_grad_mask.fill_(1.0)
#         self.old_bias_offset_mask.fill_(1.0)
#         self.new_bias_offset_mask.fill_(1.0)
#         self.old_temp_offset_mask.fill_(1.0)
#         self.new_temp_offset_mask.fill_(1.0)

#         self.alpha.requires_grad = True
#         self.geom_temperature.requires_grad = True
#         self.old_bias_offset.requires_grad = True
#         self.new_bias_offset.requires_grad = True
#         self.old_temp_offset.requires_grad = True
#         self.new_temp_offset.requires_grad = True

#     def freeze_old_adaptation(self, old_class_count: int):
#         old_class_count = int(old_class_count)

#         if self.use_bias:
#             self.bias_grad_mask.fill_(1.0)
#             self.bias_grad_mask[:old_class_count] = 0.0

#         self.alpha_grad_mask.fill_(1.0)
#         self.alpha_grad_mask[:old_class_count] = 0.0

#         self.temp_grad_mask.fill_(1.0)
#         self.temp_grad_mask[:old_class_count] = 0.0

#         self.old_bias_offset_mask.fill_(1.0)
#         self.old_bias_offset_mask[:old_class_count] = 0.0

#         self.new_bias_offset_mask.fill_(1.0)
#         self.new_bias_offset_mask[:old_class_count] = 0.0

#         self.old_temp_offset_mask.fill_(1.0)
#         self.old_temp_offset_mask[:old_class_count] = 0.0

#         self.new_temp_offset_mask.fill_(1.0)
#         self.new_temp_offset_mask[:old_class_count] = 0.0

#     def freeze_fusion_module(self):
#         if self.fusion_mlp is None:
#             return
#         for p in self.fusion_mlp.parameters():
#             p.requires_grad = False

#     def unfreeze_fusion_module(self):
#         if self.fusion_mlp is None:
#             return
#         for p in self.fusion_mlp.parameters():
#             p.requires_grad = True

#     # ============================================================
#     # Helpers
#     # ============================================================
#     def _normalize(self, x: torch.Tensor) -> torch.Tensor:
#         return F.normalize(x, dim=-1, eps=1e-6)

#     def _bounded_temperature_from_raw(self, raw: torch.Tensor) -> torch.Tensor:
#         """
#         Map raw parameter to [min_temperature, max_temperature].
#         """
#         return self.min_temperature + (self.max_temperature - self.min_temperature) * torch.sigmoid(raw)

#     def _safe_temperature(self, num_classes: Optional[int] = None) -> torch.Tensor:
#         if not self.use_geom_temperature:
#             if num_classes is None:
#                 num_classes = self.num_classes
#             return torch.ones(num_classes, device=self.geom_temperature.device, dtype=self.geom_temperature.dtype)

#         t = self._bounded_temperature_from_raw(self.geom_temperature)

#         if num_classes is None:
#             return t
#         if t.numel() == num_classes:
#             return t
#         if t.numel() > num_classes:
#             return t[:num_classes]

#         pad = torch.ones(num_classes - t.numel(), device=t.device, dtype=t.dtype)
#         return torch.cat([t, pad], dim=0)

#     def _safe_concept_temperature(self) -> float:
#         return max(self.concept_agg_temperature, 1e-6)

#     def _apply_bias(self, logits: torch.Tensor) -> torch.Tensor:
#         if self.use_bias and self.bias.numel() > 0:
#             b = self.bias[:logits.size(1)]
#             if b.numel() < logits.size(1):
#                 pad = torch.zeros(logits.size(1) - b.numel(), device=logits.device, dtype=logits.dtype)
#                 b = torch.cat([b.to(logits.device, logits.dtype), pad], dim=0)
#             else:
#                 b = b.to(logits.device, logits.dtype)
#             logits = logits + b.unsqueeze(0)
#         return logits

#     # ============================================================
#     # Legacy normalized auxiliary logits
#     # ============================================================
#     def _anchor_logits(self, f: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
#         f_n = self._normalize(f)
#         a_n = self._normalize(anchors)
#         return self.logit_scale * F.linear(f_n, a_n)

#     def _concept_logits(self, f: torch.Tensor, concept_bank: torch.Tensor) -> torch.Tensor:
#         f_n = self._normalize(f)
#         c_n = self._normalize(concept_bank)

#         sim = torch.einsum("bd,ckd->bck", f_n, c_n)
#         tau = self._safe_concept_temperature()
#         concept_score = tau * torch.logsumexp(sim / tau, dim=-1)
#         return self.logit_scale * concept_score

#     # ============================================================
#     # Geometry energy
#     # ============================================================
#     def _validate_geometry_inputs(
#         self,
#         f: torch.Tensor,
#         means: torch.Tensor,
#         bases: torch.Tensor,
#         vars_: torch.Tensor,
#     ):
#         if f.dim() != 2:
#             raise ValueError(f"Expected features [B,D], got {tuple(f.shape)}")
#         if means is None or bases is None or vars_ is None:
#             raise ValueError("Geometry tensors cannot be None.")
#         if means.dim() != 2:
#             raise ValueError(f"means must be [C,D], got {tuple(means.shape)}")
#         if bases.dim() != 3:
#             raise ValueError(f"bases must be [C,D,R], got {tuple(bases.shape)}")
#         if vars_.dim() != 2:
#             raise ValueError(f"vars must be [C,R+1], got {tuple(vars_.shape)}")
#         if f.size(1) != means.size(1) or f.size(1) != bases.size(1):
#             raise ValueError(
#                 f"Feature/geometry dim mismatch: f={f.size(1)}, means={means.size(1)}, bases={bases.size(1)}"
#             )
#         if means.size(0) != bases.size(0) or means.size(0) != vars_.size(0):
#             raise ValueError(
#                 f"Class count mismatch: means={means.size(0)}, bases={bases.size(0)}, vars={vars_.size(0)}"
#             )
#         if bases.size(2) + 1 != vars_.size(1):
#             raise ValueError(
#                 f"Rank/variance mismatch: bases rank={bases.size(2)}, vars dim={vars_.size(1)}"
#             )

#     def _geometry_energy(
#         self,
#         f: torch.Tensor,
#         means: torch.Tensor,
#         bases: torch.Tensor,
#         vars_: torch.Tensor,
#         reliability: Optional[torch.Tensor] = None,
#         active_ranks: Optional[torch.Tensor] = None,
#     ) -> torch.Tensor:
#         """
#         Low-rank Gaussian/Mahalanobis-style energy. Lower is better.

#         f:      [B, D]
#         means:  [C, D]
#         bases:  [C, D, R]
#         vars_:  [C, R+1]
#         """
#         self._validate_geometry_inputs(f, means, bases, vars_)

#         means = means.to(device=f.device, dtype=f.dtype)
#         bases = bases.to(device=f.device, dtype=f.dtype)
#         vars_ = vars_.to(device=f.device, dtype=f.dtype)

#         residual = f.unsqueeze(1) - means.unsqueeze(0)       # [B,C,D]
#         coeff = torch.einsum("bcd,cdr->bcr", residual, bases)
#         recon = torch.einsum("bcr,cdr->bcd", coeff, bases)
#         orth = residual - recon

#         eigvals = vars_[:, :-1].clamp_min(self.variance_floor)
#         resvar = vars_[:, -1].clamp_min(self.variance_floor)

#         if active_ranks is not None and torch.is_tensor(active_ranks) and active_ranks.numel() == means.size(0):
#             ar = active_ranks.to(device=f.device).long().clamp(0, bases.size(2))
#             rank_mask = torch.arange(bases.size(2), device=f.device).unsqueeze(0) < ar.unsqueeze(1)
#             rank_mask = rank_mask.to(dtype=f.dtype)

#             # Inactive dimensions are ignored in the parallel term.
#             parallel_term = ((coeff ** 2) / eigvals.unsqueeze(0)) * rank_mask.unsqueeze(0)
#             parallel_term = parallel_term.sum(dim=-1)
#         else:
#             parallel_term = ((coeff ** 2) / eigvals.unsqueeze(0)).sum(dim=-1)

#         orth_dist2 = orth.pow(2).sum(dim=-1)
#         orth_term = orth_dist2 / resvar.unsqueeze(0)

#         energy = parallel_term + orth_term

#         if self.energy_normalize_by_dim:
#             energy = energy / max(f.size(1), 1)

#         # Reliability should not dominate classification, but unreliable geometry
#         # should be slightly penalized to reduce false confidence.
#         if reliability is not None and torch.is_tensor(reliability) and reliability.numel() == means.size(0):
#             rel = reliability.to(device=f.device, dtype=f.dtype).clamp(0.05, 1.0)
#             reliability_penalty = -torch.log(rel).unsqueeze(0)
#             energy = energy + 0.05 * reliability_penalty

#         return torch.nan_to_num(energy, nan=1e6, posinf=1e6, neginf=1e6)

#     def _geometry_logits(
#         self,
#         f: torch.Tensor,
#         means: torch.Tensor,
#         bases: torch.Tensor,
#         vars_: torch.Tensor,
#         reliability: Optional[torch.Tensor] = None,
#         active_ranks: Optional[torch.Tensor] = None,
#     ) -> torch.Tensor:
#         energy = self._geometry_energy(
#             f,
#             means,
#             bases,
#             vars_,
#             reliability=reliability,
#             active_ranks=active_ranks,
#         )
#         temp = self._safe_temperature(num_classes=means.size(0)).to(device=f.device, dtype=f.dtype).unsqueeze(0)
#         logits = -energy / temp
#         return self.logit_scale * logits

#     # ============================================================
#     # Fusion / debias
#     # ============================================================
#     def _static_alpha(self) -> torch.Tensor:
#         return torch.sigmoid(self.alpha).unsqueeze(0)

#     def _adaptive_fusion_logits(
#         self,
#         anchor_logits: torch.Tensor,
#         concept_logits: torch.Tensor,
#         geom_logits: torch.Tensor,
#     ) -> torch.Tensor:
#         if not self.use_adaptive_fusion or self.fusion_mlp is None:
#             alpha = self._static_alpha()
#             alpha = alpha[:, :anchor_logits.size(1)]
#             return alpha * anchor_logits + (1.0 - alpha) * geom_logits

#         stacked = torch.stack([anchor_logits, concept_logits, geom_logits], dim=-1)
#         gate_logits = self.fusion_mlp(stacked)
#         gate = F.softmax(gate_logits, dim=-1)

#         return (
#             gate[..., 0] * anchor_logits
#             + gate[..., 1] * concept_logits
#             + gate[..., 2] * geom_logits
#         )

#     def _apply_old_new_debias(
#         self,
#         logits: torch.Tensor,
#         old_class_count: int,
#     ) -> torch.Tensor:
#         if old_class_count <= 0 or old_class_count >= logits.size(1):
#             return logits

#         C = logits.size(1)
#         device = logits.device
#         dtype = logits.dtype

#         old_mask = torch.zeros(C, device=device, dtype=dtype)
#         new_mask = torch.zeros(C, device=device, dtype=dtype)
#         old_mask[:old_class_count] = 1.0
#         new_mask[old_class_count:] = 1.0

#         # Bound offsets so debias cannot become a hidden classifier.
#         old_bias = self.debias_strength * torch.tanh(self.old_bias_offset[:C]).to(device=device, dtype=dtype)
#         new_bias = self.debias_strength * torch.tanh(self.new_bias_offset[:C]).to(device=device, dtype=dtype)

#         logits = logits + old_mask.unsqueeze(0) * old_bias.unsqueeze(0)
#         logits = logits - new_mask.unsqueeze(0) * new_bias.unsqueeze(0)

#         old_temp_raw = self.old_temp_offset[:C].to(device=device, dtype=dtype)
#         new_temp_raw = self.new_temp_offset[:C].to(device=device, dtype=dtype)

#         old_temp = self._bounded_temperature_from_raw(old_temp_raw)
#         new_temp = self._bounded_temperature_from_raw(new_temp_raw)

#         temp = torch.ones(C, device=device, dtype=dtype)
#         temp[:old_class_count] = old_temp[:old_class_count]
#         temp[old_class_count:] = new_temp[old_class_count:]

#         return logits / temp.unsqueeze(0)

#     def adaptation_regularization_loss(self) -> Dict[str, torch.Tensor]:
#         """
#         Penalize classifier calibration from becoming the actual classifier.

#         Add this to a trainer if you want stricter geometry-native behavior:
#             reg = model.classifier.adaptation_regularization_loss()["total"]
#         """
#         if self.old_bias_offset.numel() == 0:
#             z = torch.tensor(0.0, device=self.alpha.device, dtype=self.alpha.dtype)
#             return {"total": z, "bias": z, "temp": z, "alpha": z}

#         bias_reg = (
#             self.old_bias_offset.pow(2).mean()
#             + self.new_bias_offset.pow(2).mean()
#         )

#         temp_reg = (
#             self.old_temp_offset.pow(2).mean()
#             + self.new_temp_offset.pow(2).mean()
#             + self.geom_temperature.pow(2).mean()
#         )

#         alpha_reg = self.alpha.pow(2).mean() if self.alpha.numel() > 0 else torch.tensor(
#             0.0, device=self.old_bias_offset.device, dtype=self.old_bias_offset.dtype
#         )

#         total = bias_reg + 0.25 * temp_reg + 0.01 * alpha_reg
#         return {
#             "total": total,
#             "bias": bias_reg,
#             "temp": temp_reg,
#             "alpha": alpha_reg,
#         }

#     # ============================================================
#     # Bank resolution
#     # ============================================================
#     def _resolve_calibrated_geometry_bank(
#         self,
#         *,
#         subspace_means: Optional[torch.Tensor],
#         subspace_bases: Optional[torch.Tensor],
#         subspace_variances: Optional[torch.Tensor],
#         calibrated_old_means: Optional[torch.Tensor],
#         calibrated_old_bases: Optional[torch.Tensor],
#         calibrated_old_variances: Optional[torch.Tensor],
#         old_class_count: int,
#         subspace_reliability: Optional[torch.Tensor] = None,
#         calibrated_old_reliability: Optional[torch.Tensor] = None,
#         subspace_active_ranks: Optional[torch.Tensor] = None,
#         calibrated_old_active_ranks: Optional[torch.Tensor] = None,
#     ):
#         if (
#             subspace_means is None
#             or subspace_bases is None
#             or subspace_variances is None
#             or subspace_means.numel() == 0
#         ):
#             raise ValueError("Geometry bank is required for geometry scoring.")

#         if old_class_count <= 0:
#             return subspace_means, subspace_bases, subspace_variances, subspace_reliability, subspace_active_ranks

#         if (
#             calibrated_old_means is None
#             or calibrated_old_bases is None
#             or calibrated_old_variances is None
#             or calibrated_old_means.numel() == 0
#         ):
#             return subspace_means, subspace_bases, subspace_variances, subspace_reliability, subspace_active_ranks

#         if old_class_count > subspace_means.size(0):
#             raise ValueError(
#                 f"old_class_count={old_class_count} exceeds available classes={subspace_means.size(0)}."
#             )

#         new_means = subspace_means[old_class_count:]
#         new_bases = subspace_bases[old_class_count:]
#         new_vars = subspace_variances[old_class_count:]

#         means = torch.cat([calibrated_old_means, new_means], dim=0)
#         bases = torch.cat([calibrated_old_bases, new_bases], dim=0)
#         vars_ = torch.cat([calibrated_old_variances, new_vars], dim=0)

#         reliability = None
#         if subspace_reliability is not None and torch.is_tensor(subspace_reliability) and subspace_reliability.numel() >= subspace_means.size(0):
#             new_rel = subspace_reliability[old_class_count:]
#             if calibrated_old_reliability is not None and torch.is_tensor(calibrated_old_reliability):
#                 old_rel = calibrated_old_reliability[:old_class_count]
#             else:
#                 old_rel = subspace_reliability[:old_class_count]
#             reliability = torch.cat([old_rel, new_rel], dim=0)

#         active_ranks = None
#         if subspace_active_ranks is not None and torch.is_tensor(subspace_active_ranks) and subspace_active_ranks.numel() >= subspace_means.size(0):
#             new_ar = subspace_active_ranks[old_class_count:]
#             if calibrated_old_active_ranks is not None and torch.is_tensor(calibrated_old_active_ranks):
#                 old_ar = calibrated_old_active_ranks[:old_class_count]
#             else:
#                 old_ar = subspace_active_ranks[:old_class_count]
#             active_ranks = torch.cat([old_ar, new_ar], dim=0)

#         return means, bases, vars_, reliability, active_ranks

#     # ============================================================
#     # Forward
#     # ============================================================
#     def forward(
#         self,
#         features: torch.Tensor,
#         *,
#         anchors: Optional[torch.Tensor] = None,
#         concept_bank: Optional[torch.Tensor] = None,
#         subspace_means: Optional[torch.Tensor] = None,
#         subspace_bases: Optional[torch.Tensor] = None,
#         subspace_variances: Optional[torch.Tensor] = None,
#         subspace_reliability: Optional[torch.Tensor] = None,
#         subspace_active_ranks: Optional[torch.Tensor] = None,
#         calibrated_old_means: Optional[torch.Tensor] = None,
#         calibrated_old_bases: Optional[torch.Tensor] = None,
#         calibrated_old_variances: Optional[torch.Tensor] = None,
#         calibrated_old_reliability: Optional[torch.Tensor] = None,
#         calibrated_old_active_ranks: Optional[torch.Tensor] = None,
#         mode: str = "geometry_only",
#         old_class_count: int = 0,
#         return_energy: bool = False,
#         **kwargs,
#     ):
#         del kwargs
#         mode = str(mode).lower()

#         if features.dim() != 2:
#             raise ValueError(f"Expected features to be [B, D], got {tuple(features.shape)}")

#         # --------------------------------------------------------
#         # Geometry-only scoring
#         # --------------------------------------------------------
#         if mode == "geometry_only":
#             if (
#                 subspace_means is None
#                 or subspace_bases is None
#                 or subspace_variances is None
#                 or subspace_means.numel() == 0
#             ):
#                 raise ValueError("geometry_only mode requires subspace bank.")

#             energy = self._geometry_energy(
#                 features,
#                 subspace_means,
#                 subspace_bases,
#                 subspace_variances,
#                 reliability=subspace_reliability,
#                 active_ranks=subspace_active_ranks,
#             )
#             logits = self._geometry_logits(
#                 features,
#                 subspace_means,
#                 subspace_bases,
#                 subspace_variances,
#                 reliability=subspace_reliability,
#                 active_ranks=subspace_active_ranks,
#             )
#             logits = self._apply_bias(logits)
#             logits = self._apply_old_new_debias(logits, old_class_count)

#             if return_energy:
#                 return {"logits": logits, "energy": energy}
#             return logits

#         # --------------------------------------------------------
#         # Calibrated old geometry + current new geometry
#         # --------------------------------------------------------
#         if mode == "calibrated_geometry":
#             means, bases, vars_, reliability, active_ranks = self._resolve_calibrated_geometry_bank(
#                 subspace_means=subspace_means,
#                 subspace_bases=subspace_bases,
#                 subspace_variances=subspace_variances,
#                 calibrated_old_means=calibrated_old_means,
#                 calibrated_old_bases=calibrated_old_bases,
#                 calibrated_old_variances=calibrated_old_variances,
#                 old_class_count=old_class_count,
#                 subspace_reliability=subspace_reliability,
#                 calibrated_old_reliability=calibrated_old_reliability,
#                 subspace_active_ranks=subspace_active_ranks,
#                 calibrated_old_active_ranks=calibrated_old_active_ranks,
#             )

#             energy = self._geometry_energy(
#                 features,
#                 means,
#                 bases,
#                 vars_,
#                 reliability=reliability,
#                 active_ranks=active_ranks,
#             )
#             logits = self._geometry_logits(
#                 features,
#                 means,
#                 bases,
#                 vars_,
#                 reliability=reliability,
#                 active_ranks=active_ranks,
#             )
#             logits = self._apply_bias(logits)
#             logits = self._apply_old_new_debias(logits, old_class_count)

#             if return_energy:
#                 return {"logits": logits, "energy": energy}
#             return logits

#         # --------------------------------------------------------
#         # Legacy anchor/concept scoring
#         # --------------------------------------------------------
#         if mode == "anchor_concept":
#             if anchors is None or anchors.numel() == 0:
#                 raise ValueError("anchor_concept mode requires anchors.")
#             if concept_bank is None or concept_bank.numel() == 0:
#                 raise ValueError("anchor_concept mode requires concept_bank.")

#             anchor_logits = self._anchor_logits(features, anchors)
#             concept_logits = self._concept_logits(features, concept_bank)

#             alpha = self._static_alpha()[:, :anchor_logits.size(1)]
#             logits = alpha * anchor_logits + (1.0 - alpha) * concept_logits
#             logits = self._apply_bias(logits)
#             logits = self._apply_old_new_debias(logits, old_class_count)
#             return logits

#         # --------------------------------------------------------
#         # Legacy hybrid scoring
#         # --------------------------------------------------------
#         if mode == "adaptive_hybrid":
#             if anchors is None or anchors.numel() == 0:
#                 raise ValueError("adaptive_hybrid mode requires anchors.")
#             if concept_bank is None or concept_bank.numel() == 0:
#                 raise ValueError("adaptive_hybrid mode requires concept_bank.")
#             if (
#                 subspace_means is None
#                 or subspace_bases is None
#                 or subspace_variances is None
#                 or subspace_means.numel() == 0
#             ):
#                 raise ValueError("adaptive_hybrid mode requires subspace bank.")

#             anchor_logits = self._anchor_logits(features, anchors)
#             concept_logits = self._concept_logits(features, concept_bank)
#             geom_logits = self._geometry_logits(
#                 features,
#                 subspace_means,
#                 subspace_bases,
#                 subspace_variances,
#                 reliability=subspace_reliability,
#                 active_ranks=subspace_active_ranks,
#             )

#             logits = self._adaptive_fusion_logits(
#                 anchor_logits=anchor_logits,
#                 concept_logits=concept_logits,
#                 geom_logits=geom_logits,
#             )
#             logits = self._apply_bias(logits)
#             logits = self._apply_old_new_debias(logits, old_class_count)
#             return logits

#         raise ValueError(f"Unsupported mode: {mode}")


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Optional


# class SemanticClassifier(nn.Module):
#     """
#     Geometry-native classifier for NECIL-HSI.

#     Main design
#     -----------
#     1) geometry_only
#        - score all classes directly from the provided geometry bank

#     2) calibrated_geometry
#        - if calibrated old geometry is provided, use:
#             [ calibrated old classes | current new classes ]
#        - otherwise falls back to the provided current geometry bank

#     Legacy compatibility
#     --------------------
#     - anchor_concept
#     - adaptive_hybrid

#     These are retained only so the rest of the pipeline does not crash
#     while you migrate fully to geometry-native training.
#     """

#     def __init__(
#         self,
#         initial_classes: int = 0,
#         d_model: int = 128,
#         logit_scale: float = 8.0,
#         use_bias: bool = True,
#         variance_floor: float = 1e-4,
#         use_geom_temperature: bool = True,
#         concept_agg_temperature: float = 0.07,
#         init_alpha_old: float = -0.5,
#         init_alpha_new: float = -0.2,
#         use_adaptive_fusion: bool = True,
#     ):
#         super().__init__()

#         self.num_classes = int(initial_classes)
#         self.d_model = int(d_model)
#         self.logit_scale = float(logit_scale)
#         self.use_bias = bool(use_bias)
#         self.variance_floor = float(variance_floor)
#         self.use_geom_temperature = bool(use_geom_temperature)
#         self.concept_agg_temperature = float(concept_agg_temperature)

#         # Retained only for legacy modes
#         self.init_alpha_old = float(init_alpha_old)
#         self.init_alpha_new = float(init_alpha_new)
#         self.use_adaptive_fusion = bool(use_adaptive_fusion)

#         # ------------------------------------------------------------
#         # Learnable classwise calibration parameters
#         # ------------------------------------------------------------
#         self.bias = nn.Parameter(
#             torch.zeros(self.num_classes),
#             requires_grad=self.use_bias,
#         )

#         # Legacy fusion scalar per class
#         self.alpha = nn.Parameter(
#             torch.full((self.num_classes,), self.init_alpha_old)
#         )

#         # Geometry temperature per class
#         self.geom_temperature = nn.Parameter(torch.ones(self.num_classes))

#         # Explicit old/new debias offsets
#         self.old_bias_offset = nn.Parameter(torch.zeros(self.num_classes))
#         self.new_bias_offset = nn.Parameter(torch.zeros(self.num_classes))

#         # Explicit old/new temperature offsets
#         self.old_temp_offset = nn.Parameter(torch.ones(self.num_classes))
#         self.new_temp_offset = nn.Parameter(torch.ones(self.num_classes))

#         # Legacy fusion gate
#         if self.use_adaptive_fusion:
#             self.fusion_mlp = nn.Sequential(
#                 nn.Linear(3, 16),
#                 nn.GELU(),
#                 nn.Linear(16, 3),
#             )
#         else:
#             self.fusion_mlp = None

#         # ------------------------------------------------------------
#         # Gradient masks for selective adaptation
#         # ------------------------------------------------------------
#         self.register_buffer("bias_grad_mask", torch.ones(self.num_classes))
#         self.register_buffer("alpha_grad_mask", torch.ones(self.num_classes))
#         self.register_buffer("temp_grad_mask", torch.ones(self.num_classes))
#         self.register_buffer("old_bias_offset_mask", torch.ones(self.num_classes))
#         self.register_buffer("new_bias_offset_mask", torch.ones(self.num_classes))
#         self.register_buffer("old_temp_offset_mask", torch.ones(self.num_classes))
#         self.register_buffer("new_temp_offset_mask", torch.ones(self.num_classes))

#         self._register_gradient_hooks()

#     # ============================================================
#     # Hook registration
#     # ============================================================
#     def _register_gradient_hooks(self):
#         if self.use_bias:
#             self.bias.register_hook(lambda g: g * self.bias_grad_mask)

#         self.alpha.register_hook(lambda g: g * self.alpha_grad_mask)

#         if self.use_geom_temperature:
#             self.geom_temperature.register_hook(lambda g: g * self.temp_grad_mask)

#         self.old_bias_offset.register_hook(lambda g: g * self.old_bias_offset_mask)
#         self.new_bias_offset.register_hook(lambda g: g * self.new_bias_offset_mask)
#         self.old_temp_offset.register_hook(lambda g: g * self.old_temp_offset_mask)
#         self.new_temp_offset.register_hook(lambda g: g * self.new_temp_offset_mask)

#     # ============================================================
#     # Expansion
#     # ============================================================
#     def expand(self, num_new_classes: int, phase: int):
#         del phase

#         old = self.num_classes
#         num_new_classes = int(num_new_classes)
#         self.num_classes += num_new_classes

#         # Use a stable parameter/device anchor
#         ref_param = self.alpha
#         device = ref_param.device
#         dtype = ref_param.dtype

#         def _expand_param(old_param: nn.Parameter, fill_value: float) -> nn.Parameter:
#             new_p = torch.full((self.num_classes,), fill_value, device=device, dtype=dtype)
#             if old > 0:
#                 new_p[:old] = old_param.data
#             return nn.Parameter(new_p, requires_grad=True)

#         if self.use_bias:
#             new_bias = torch.zeros(self.num_classes, device=device, dtype=dtype)
#             if old > 0:
#                 new_bias[:old] = self.bias.data
#             self.bias = nn.Parameter(new_bias, requires_grad=True)

#         new_alpha = torch.full(
#             (self.num_classes,),
#             fill_value=self.init_alpha_new,
#             device=device,
#             dtype=dtype,
#         )
#         if old > 0:
#             new_alpha[:old] = self.alpha.data
#         self.alpha = nn.Parameter(new_alpha, requires_grad=True)

#         new_temp = torch.ones(self.num_classes, device=device, dtype=dtype)
#         if old > 0:
#             new_temp[:old] = self.geom_temperature.data
#         self.geom_temperature = nn.Parameter(new_temp, requires_grad=True)

#         self.old_bias_offset = _expand_param(self.old_bias_offset, 0.0)
#         self.new_bias_offset = _expand_param(self.new_bias_offset, 0.0)
#         self.old_temp_offset = _expand_param(self.old_temp_offset, 1.0)
#         self.new_temp_offset = _expand_param(self.new_temp_offset, 1.0)

#         def _expand_mask(old_mask: Optional[torch.Tensor]) -> torch.Tensor:
#             new_mask = torch.ones(self.num_classes, device=device, dtype=dtype)
#             if old > 0 and old_mask is not None:
#                 new_mask[:old] = old_mask[:old]
#             return new_mask

#         old_bias_mask = self.bias_grad_mask.clone() if hasattr(self, "bias_grad_mask") else None
#         old_alpha_mask = self.alpha_grad_mask.clone() if hasattr(self, "alpha_grad_mask") else None
#         old_temp_mask = self.temp_grad_mask.clone() if hasattr(self, "temp_grad_mask") else None
#         old_old_bias_mask = (
#             self.old_bias_offset_mask.clone() if hasattr(self, "old_bias_offset_mask") else None
#         )
#         old_new_bias_mask = (
#             self.new_bias_offset_mask.clone() if hasattr(self, "new_bias_offset_mask") else None
#         )
#         old_old_temp_mask = (
#             self.old_temp_offset_mask.clone() if hasattr(self, "old_temp_offset_mask") else None
#         )
#         old_new_temp_mask = (
#             self.new_temp_offset_mask.clone() if hasattr(self, "new_temp_offset_mask") else None
#         )

#         self.register_buffer("bias_grad_mask", _expand_mask(old_bias_mask))
#         self.register_buffer("alpha_grad_mask", _expand_mask(old_alpha_mask))
#         self.register_buffer("temp_grad_mask", _expand_mask(old_temp_mask))
#         self.register_buffer("old_bias_offset_mask", _expand_mask(old_old_bias_mask))
#         self.register_buffer("new_bias_offset_mask", _expand_mask(old_new_bias_mask))
#         self.register_buffer("old_temp_offset_mask", _expand_mask(old_old_temp_mask))
#         self.register_buffer("new_temp_offset_mask", _expand_mask(old_new_temp_mask))

#         self._register_gradient_hooks()

#     # ============================================================
#     # Adaptation control
#     # ============================================================
#     def freeze_all_adaptation(self):
#         if self.use_bias:
#             self.bias_grad_mask.zero_()
#             self.bias.requires_grad = True

#         self.alpha_grad_mask.zero_()
#         self.temp_grad_mask.zero_()
#         self.old_bias_offset_mask.zero_()
#         self.new_bias_offset_mask.zero_()
#         self.old_temp_offset_mask.zero_()
#         self.new_temp_offset_mask.zero_()

#         self.alpha.requires_grad = True
#         self.geom_temperature.requires_grad = True
#         self.old_bias_offset.requires_grad = True
#         self.new_bias_offset.requires_grad = True
#         self.old_temp_offset.requires_grad = True
#         self.new_temp_offset.requires_grad = True

#     def unfreeze_all_adaptation(self):
#         if self.use_bias:
#             self.bias_grad_mask.fill_(1.0)
#             self.bias.requires_grad = True

#         self.alpha_grad_mask.fill_(1.0)
#         self.temp_grad_mask.fill_(1.0)
#         self.old_bias_offset_mask.fill_(1.0)
#         self.new_bias_offset_mask.fill_(1.0)
#         self.old_temp_offset_mask.fill_(1.0)
#         self.new_temp_offset_mask.fill_(1.0)

#         self.alpha.requires_grad = True
#         self.geom_temperature.requires_grad = True
#         self.old_bias_offset.requires_grad = True
#         self.new_bias_offset.requires_grad = True
#         self.old_temp_offset.requires_grad = True
#         self.new_temp_offset.requires_grad = True

#     def freeze_old_adaptation(self, old_class_count: int):
#         old_class_count = int(old_class_count)

#         if self.use_bias:
#             self.bias_grad_mask.fill_(1.0)
#             self.bias_grad_mask[:old_class_count] = 0.0

#         self.alpha_grad_mask.fill_(1.0)
#         self.alpha_grad_mask[:old_class_count] = 0.0

#         self.temp_grad_mask.fill_(1.0)
#         self.temp_grad_mask[:old_class_count] = 0.0

#         self.old_bias_offset_mask.fill_(1.0)
#         self.old_bias_offset_mask[:old_class_count] = 0.0

#         self.new_bias_offset_mask.fill_(1.0)
#         self.new_bias_offset_mask[:old_class_count] = 0.0

#         self.old_temp_offset_mask.fill_(1.0)
#         self.old_temp_offset_mask[:old_class_count] = 0.0

#         self.new_temp_offset_mask.fill_(1.0)
#         self.new_temp_offset_mask[:old_class_count] = 0.0

#     def freeze_fusion_module(self):
#         if self.fusion_mlp is None:
#             return
#         for p in self.fusion_mlp.parameters():
#             p.requires_grad = False

#     def unfreeze_fusion_module(self):
#         if self.fusion_mlp is None:
#             return
#         for p in self.fusion_mlp.parameters():
#             p.requires_grad = True

#     # ============================================================
#     # Helpers
#     # ============================================================
#     def _normalize(self, x: torch.Tensor) -> torch.Tensor:
#         return F.normalize(x, dim=-1, eps=1e-6)

#     def _safe_temperature(self, num_classes: Optional[int] = None) -> torch.Tensor:
#         if not self.use_geom_temperature:
#             if num_classes is None:
#                 num_classes = self.num_classes
#             return torch.ones(num_classes, device=self.geom_temperature.device, dtype=self.geom_temperature.dtype)

#         t = F.softplus(self.geom_temperature) + 1e-4
#         if num_classes is None:
#             return t
#         if t.numel() == num_classes:
#             return t
#         if t.numel() > num_classes:
#             return t[:num_classes]
#         # defensive padding if classifier metadata lags behind bank size
#         pad = torch.ones(
#             num_classes - t.numel(),
#             device=t.device,
#             dtype=t.dtype,
#         )
#         return torch.cat([t, pad], dim=0)

#     def _safe_concept_temperature(self) -> float:
#         return max(self.concept_agg_temperature, 1e-6)

#     def _apply_bias(self, logits: torch.Tensor) -> torch.Tensor:
#         if self.use_bias and self.bias.numel() > 0:
#             if self.bias.numel() == logits.size(1):
#                 logits = logits + self.bias.unsqueeze(0)
#             else:
#                 logits = logits + self.bias[:logits.size(1)].unsqueeze(0)
#         return logits

#     def _anchor_logits(self, f: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
#         f_n = self._normalize(f)
#         a_n = self._normalize(anchors)
#         return self.logit_scale * F.linear(f_n, a_n)

#     def _concept_logits(self, f: torch.Tensor, concept_bank: torch.Tensor) -> torch.Tensor:
#         f_n = self._normalize(f)
#         c_n = self._normalize(concept_bank)

#         sim = torch.einsum("bd,ckd->bck", f_n, c_n)
#         tau = self._safe_concept_temperature()
#         concept_score = tau * torch.logsumexp(sim / tau, dim=-1)
#         return self.logit_scale * concept_score

#     def _geometry_energy(
#         self,
#         f: torch.Tensor,
#         means: torch.Tensor,
#         bases: torch.Tensor,
#         vars_: torch.Tensor,
#     ) -> torch.Tensor:
#         """
#         Returns geometry energy. Lower is better.
#         Shapes:
#             f:      [B, D]
#             means:  [C, D]
#             bases:  [C, D, R]
#             vars_:  [C, R+1]
#         """
#         residual = f.unsqueeze(1) - means.unsqueeze(0)  # [B, C, D]

#         coeff = torch.einsum("bcd,cdr->bcr", residual, bases)     # [B, C, R]
#         recon = torch.einsum("bcr,cdr->bcd", coeff, bases)        # [B, C, D]
#         orth = residual - recon                                   # [B, C, D]

#         orth_dist2 = (orth ** 2).sum(dim=-1)                      # [B, C]

#         eigvals = vars_[:, :-1].clamp_min(self.variance_floor)    # [C, R]
#         resvar = vars_[:, -1].clamp_min(self.variance_floor)      # [C]

#         parallel_term = ((coeff ** 2) / eigvals.unsqueeze(0)).sum(dim=-1)
#         orth_term = orth_dist2 / resvar.unsqueeze(0)

#         energy = parallel_term + orth_term
#         energy = energy / max(f.size(1), 1)
#         return energy

#     def _geometry_logits(
#         self,
#         f: torch.Tensor,
#         means: torch.Tensor,
#         bases: torch.Tensor,
#         vars_: torch.Tensor,
#     ) -> torch.Tensor:
#         energy = self._geometry_energy(f, means, bases, vars_)
#         temp = self._safe_temperature(num_classes=means.size(0)).unsqueeze(0)
#         score = -(energy / temp)
#         return self.logit_scale * score

#     def _static_alpha(self) -> torch.Tensor:
#         return torch.sigmoid(self.alpha).unsqueeze(0)

#     def _adaptive_fusion_logits(
#         self,
#         anchor_logits: torch.Tensor,
#         concept_logits: torch.Tensor,
#         geom_logits: torch.Tensor,
#     ) -> torch.Tensor:
#         if not self.use_adaptive_fusion or self.fusion_mlp is None:
#             alpha = self._static_alpha()
#             alpha = alpha[:, :anchor_logits.size(1)]
#             return alpha * anchor_logits + (1.0 - alpha) * geom_logits

#         stacked = torch.stack([anchor_logits, concept_logits, geom_logits], dim=-1)  # [B, C, 3]
#         gate_logits = self.fusion_mlp(stacked)
#         gate = F.softmax(gate_logits, dim=-1)

#         fused = (
#             gate[..., 0] * anchor_logits
#             + gate[..., 1] * concept_logits
#             + gate[..., 2] * geom_logits
#         )
#         return fused

#     def _apply_old_new_debias(
#         self,
#         logits: torch.Tensor,
#         old_class_count: int,
#     ) -> torch.Tensor:
#         if old_class_count <= 0 or old_class_count >= logits.size(1):
#             return logits

#         C = logits.size(1)
#         device = logits.device
#         dtype = logits.dtype

#         old_mask = torch.zeros(C, device=device, dtype=dtype)
#         new_mask = torch.zeros(C, device=device, dtype=dtype)
#         old_mask[:old_class_count] = 1.0
#         new_mask[old_class_count:] = 1.0

#         old_bias = self.old_bias_offset[:C]
#         new_bias = self.new_bias_offset[:C]
#         logits = logits + old_mask.unsqueeze(0) * old_bias.unsqueeze(0)
#         logits = logits - new_mask.unsqueeze(0) * new_bias.unsqueeze(0)

#         old_temp = F.softplus(self.old_temp_offset[:C]).clamp_min(1e-4)
#         new_temp = F.softplus(self.new_temp_offset[:C]).clamp_min(1e-4)

#         temp = torch.ones_like(old_temp)
#         temp[:old_class_count] = old_temp[:old_class_count]
#         temp[old_class_count:] = new_temp[old_class_count:]

#         logits = logits / temp.unsqueeze(0)
#         return logits

#     def _resolve_calibrated_geometry_bank(
#         self,
#         *,
#         subspace_means: Optional[torch.Tensor],
#         subspace_bases: Optional[torch.Tensor],
#         subspace_variances: Optional[torch.Tensor],
#         calibrated_old_means: Optional[torch.Tensor],
#         calibrated_old_bases: Optional[torch.Tensor],
#         calibrated_old_variances: Optional[torch.Tensor],
#         old_class_count: int,
#     ):
#         """
#         Build the geometry bank used for calibrated_geometry mode.

#         Priority:
#         1) If calibrated old bank is provided and old_class_count > 0:
#            use [ calibrated old bank | current new bank ]
#         2) Otherwise:
#            use the raw current bank directly
#         """
#         if (
#             subspace_means is None
#             or subspace_bases is None
#             or subspace_variances is None
#             or subspace_means.numel() == 0
#         ):
#             raise ValueError("Geometry bank is required for geometry scoring.")

#         if old_class_count <= 0:
#             return subspace_means, subspace_bases, subspace_variances

#         if (
#             calibrated_old_means is None
#             or calibrated_old_bases is None
#             or calibrated_old_variances is None
#             or calibrated_old_means.numel() == 0
#         ):
#             return subspace_means, subspace_bases, subspace_variances

#         if old_class_count > subspace_means.size(0):
#             raise ValueError(
#                 f"old_class_count={old_class_count} exceeds available classes={subspace_means.size(0)}."
#             )

#         new_means = subspace_means[old_class_count:]
#         new_bases = subspace_bases[old_class_count:]
#         new_vars = subspace_variances[old_class_count:]

#         means = torch.cat([calibrated_old_means, new_means], dim=0)
#         bases = torch.cat([calibrated_old_bases, new_bases], dim=0)
#         vars_ = torch.cat([calibrated_old_variances, new_vars], dim=0)

#         return means, bases, vars_

#     # ============================================================
#     # Forward
#     # ============================================================
#     def forward(
#         self,
#         features: torch.Tensor,
#         *,
#         anchors: Optional[torch.Tensor] = None,
#         concept_bank: Optional[torch.Tensor] = None,
#         subspace_means: Optional[torch.Tensor] = None,
#         subspace_bases: Optional[torch.Tensor] = None,
#         subspace_variances: Optional[torch.Tensor] = None,
#         calibrated_old_means: Optional[torch.Tensor] = None,
#         calibrated_old_bases: Optional[torch.Tensor] = None,
#         calibrated_old_variances: Optional[torch.Tensor] = None,
#         mode: str = "geometry_only",
#         old_class_count: int = 0,
#         return_energy: bool = False,
#         **kwargs,
#     ):
#         del kwargs
#         mode = str(mode).lower()

#         if features.dim() != 2:
#             raise ValueError(f"Expected features to be [B, D], got {tuple(features.shape)}")

#         # --------------------------------------------------------
#         # Pure geometry scoring
#         # --------------------------------------------------------
#         if mode == "geometry_only":
#             if (
#                 subspace_means is None
#                 or subspace_bases is None
#                 or subspace_variances is None
#                 or subspace_means.numel() == 0
#             ):
#                 raise ValueError("geometry_only mode requires subspace bank.")

#             energy = self._geometry_energy(
#                 features,
#                 subspace_means,
#                 subspace_bases,
#                 subspace_variances,
#             )
#             logits = self._geometry_logits(
#                 features,
#                 subspace_means,
#                 subspace_bases,
#                 subspace_variances,
#             )
#             logits = self._apply_bias(logits)
#             logits = self._apply_old_new_debias(logits, old_class_count)

#             if return_energy:
#                 return {"logits": logits, "energy": energy}
#             return logits

#         # --------------------------------------------------------
#         # Calibrated old geometry + current new geometry
#         # --------------------------------------------------------
#         elif mode == "calibrated_geometry":
#             means, bases, vars_ = self._resolve_calibrated_geometry_bank(
#                 subspace_means=subspace_means,
#                 subspace_bases=subspace_bases,
#                 subspace_variances=subspace_variances,
#                 calibrated_old_means=calibrated_old_means,
#                 calibrated_old_bases=calibrated_old_bases,
#                 calibrated_old_variances=calibrated_old_variances,
#                 old_class_count=old_class_count,
#             )

#             energy = self._geometry_energy(features, means, bases, vars_)
#             logits = self._geometry_logits(features, means, bases, vars_)
#             logits = self._apply_bias(logits)
#             logits = self._apply_old_new_debias(logits, old_class_count)

#             if return_energy:
#                 return {"logits": logits, "energy": energy}
#             return logits

#         # --------------------------------------------------------
#         # Legacy anchor-concept mode
#         # --------------------------------------------------------
#         elif mode == "anchor_concept":
#             if anchors is None or anchors.numel() == 0:
#                 raise ValueError("anchor_concept mode requires anchors.")
#             if concept_bank is None or concept_bank.numel() == 0:
#                 raise ValueError("anchor_concept mode requires concept_bank.")

#             anchor_logits = self._anchor_logits(features, anchors)
#             concept_logits = self._concept_logits(features, concept_bank)

#             alpha = self._static_alpha()[:, :anchor_logits.size(1)]
#             logits = alpha * anchor_logits + (1.0 - alpha) * concept_logits
#             logits = self._apply_bias(logits)
#             logits = self._apply_old_new_debias(logits, old_class_count)
#             return logits

#         # --------------------------------------------------------
#         # Legacy hybrid mode
#         # --------------------------------------------------------
#         elif mode == "adaptive_hybrid":
#             if anchors is None or anchors.numel() == 0:
#                 raise ValueError("adaptive_hybrid mode requires anchors.")
#             if concept_bank is None or concept_bank.numel() == 0:
#                 raise ValueError("adaptive_hybrid mode requires concept_bank.")
#             if (
#                 subspace_means is None
#                 or subspace_bases is None
#                 or subspace_variances is None
#                 or subspace_means.numel() == 0
#             ):
#                 raise ValueError("adaptive_hybrid mode requires subspace bank.")

#             anchor_logits = self._anchor_logits(features, anchors)
#             concept_logits = self._concept_logits(features, concept_bank)
#             geom_logits = self._geometry_logits(
#                 features,
#                 subspace_means,
#                 subspace_bases,
#                 subspace_variances,
#             )

#             logits = self._adaptive_fusion_logits(
#                 anchor_logits=anchor_logits,
#                 concept_logits=concept_logits,
#                 geom_logits=geom_logits,
#             )
#             logits = self._apply_bias(logits)
#             logits = self._apply_old_new_debias(logits, old_class_count)
#             return logits

#         else:
#             raise ValueError(f"Unsupported mode: {mode}")

