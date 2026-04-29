import os
from contextlib import nullcontext

import torch
import torch.nn.functional as F
import torch.optim as optim

from losses.necil_losses import GlobalLogitMargin, ConceptSeparation, FeatureConceptCompactness
from trainers.trainer_helpers import TrainerHelper


class Trainer(TrainerHelper):
    """
    Geometry-centric trainer for strict non-exemplar HSI CIL.

    Corrected stability policy
    --------------------------
    1. Validate only after current-phase geometry is refreshed.
    2. Select best incremental checkpoint by refreshed validation harmonic mean.
    3. Do not accidentally unfreeze old classifier adaptation.
    4. Keep old raw samples inaccessible under strict non-exemplar protocol.
    5. Refresh only current-phase geometry during incremental training.
    6. Use identity semantic path for all geometry-native training/eval.
    """

    def __init__(self, model, dataset, args):
        self.args = args
        self.device = torch.device(args.device)
        self.model = model.to(self.device)
        self.dataset = dataset
        self.save_dir = args.save_dir
        self.debug = bool(getattr(args, "debug_verbose", False)) or os.environ.get("NECIL_DEBUG", "0") == "1"

        self.subspace_rank = int(getattr(args, "subspace_rank", 5))
        self.geom_var_floor = float(getattr(args, "geom_var_floor", 1e-4))
        self.num_concepts_per_class = int(getattr(args, "num_concepts_per_class", 4))
        self.alignment_samples_per_class = int(getattr(args, "alignment_samples_per_class", 16))

        # ---------------- Base phase ----------------
        self.base_compact = float(getattr(args, "base_compact", 0.05))
        self.base_sep = float(getattr(args, "base_sep", 0.05))
        self.base_ortho = float(getattr(args, "base_ortho", 0.03))
        self.base_margin = float(getattr(args, "base_margin", 1.0))
        self.base_center_norm = float(getattr(args, "base_center_norm", 0.01))
        self.base_radius = float(getattr(args, "base_radius", 0.01))

        # Safe base geometry scheduling. These getattr defaults do not require
        # argparse changes unless you want CLI control.
        self.base_geometry_warmup_epochs = int(getattr(args, "base_geometry_warmup_epochs", 20))
        self.base_geometry_ramp_epochs = int(getattr(args, "base_geometry_ramp_epochs", 30))
        self.base_max_geo_loss = float(getattr(args, "base_max_geo_loss", 5.0))

        # Safe incremental objective controls. Defaults preserve your current
        # command unless you expose these args in main.py.
        self.incremental_new_ce_weight = float(getattr(args, "incremental_new_ce_weight", 1.0))
        self.incremental_replay_weight_mult = float(getattr(args, "incremental_replay_weight_mult", 1.0))
        self.incremental_max_aux_loss = float(getattr(args, "incremental_max_aux_loss", 50.0))

        # ---------------- Incremental phase ----------------
        self.replay_weight = float(getattr(args, "synthetic_replay_weight", 1.0))
        self.replay_per_class = int(getattr(args, "synthetic_replay_per_class", 32))
        self.align_mean_weight = float(getattr(args, "align_mean_weight", 0.10))
        self.align_basis_weight = float(getattr(args, "align_basis_weight", 0.05))
        self.align_var_weight = float(getattr(args, "align_var_weight", 0.02))
        self.insert_weight = float(getattr(args, "insert_weight", 0.01))
        self.insert_margin = float(getattr(args, "insert_margin", 5.0))
        self.new_volume_weight = float(getattr(args, "new_volume_weight", 0.005))
        self.new_volume_target = float(getattr(args, "new_volume_target", 1.5))
        self.incremental_warmup_epochs = int(getattr(args, "incremental_warmup_epochs", 5))

        # geometry-calibration regularization
        self.geometry_calibration_weight = float(
            getattr(args, "geometry_calibration_weight", 0.05)
        )

        # ---------------- Calibration ----------------
        self.calibration_epochs = int(getattr(args, "calibration_epochs", 5))
        self.calibration_lr = float(getattr(args, "calibration_lr", 5e-4))
        self.calibration_replay_weight = float(getattr(args, "calibration_replay_weight", 1.0))

        # ---------------- Validation / checkpoint policy ----------------
        self.refresh_before_validation = bool(getattr(args, "refresh_before_validation", True))
        self.validation_refresh_every = int(getattr(args, "validation_refresh_every", 1))
        self.best_state_metric = str(getattr(args, "best_state_metric", "hm")).lower()
        self.early_stop_patience = int(getattr(args, "early_stop_patience", 0))

        # ---------------- Auxiliary losses ----------------
        self.logit_margin_value = float(getattr(args, "logit_margin_value", 0.20))
        self.logit_margin_weight = float(getattr(args, "logit_margin_weight", 0.02))
        self.concept_sep_weight = float(getattr(args, "concept_sep_weight", 0.01))
        self.feature_concept_compact_weight = float(
            getattr(args, "feature_concept_compact_weight", 0.03)
        )
        self.inc_logit_margin_weight = float(getattr(args, "inc_logit_margin_weight", 0.005))
        self.inc_feature_concept_compact_weight = float(
            getattr(args, "inc_feature_concept_compact_weight", 0.005)
        )
        self.classifier_adaptation_weight = float(getattr(args, "classifier_adaptation_weight", 0.0))

        self.logit_margin = GlobalLogitMargin(margin=self.logit_margin_value)
        self.concept_sep = ConceptSeparation(
            max_cosine=float(getattr(args, "concept_sep_max_cosine", 0.25))
        )
        self.feature_concept_compact = FeatureConceptCompactness(
            temperature=float(getattr(args, "cls_temperature", 0.07))
        )

        self.token_memory = {}

    # ============================================================
    # Mode helpers
    # ============================================================
    def _inc_classifier_mode(self) -> str:
        return str(
            getattr(self.args, "incremental_classifier_mode", "calibrated_geometry")
        ).lower()

    def _eval_classifier_mode(self) -> str:
        return str(
            getattr(self.args, "eval_classifier_mode", "calibrated_geometry")
        ).lower()

    def _set_model_phase_and_old_count(self, phase: int, old_class_count: int):
        if hasattr(self.model, "set_phase"):
            self.model.set_phase(int(phase))
        else:
            self.model.current_phase = int(phase)

        if hasattr(self.model, "set_old_class_count"):
            self.model.set_old_class_count(int(old_class_count))
        else:
            self.model.old_class_count = int(old_class_count)

    # ============================================================
    # Geometry refresh helpers
    # ============================================================
    @torch.no_grad()
    def _refresh_classes_for_validation(self, phase: int, class_ids, split: str = "train"):
        """
        Refresh only currently accessible classes before validation.

        This fixes the stale-geometry validation bug:
        validation must evaluate the model with the same current-phase geometry
        that will be used after phase finalization.
        """
        if not self.refresh_before_validation:
            return

        phase = int(phase)
        old_training_state = self.model.training

        # current phase train classes are allowed in strict non-exemplar mode.
        ctx = (
            self.dataset.memory_build_context(phase)
            if hasattr(self.dataset, "memory_build_context")
            else nullcontext()
        )

        with ctx:
            for cls in class_ids:
                self._build_class_memory_from_current_phase(int(cls), split=split)

        gb = getattr(self.model, "geometry_bank", None)
        if gb is not None and hasattr(gb, "refresh_inter_class_geometry"):
            gb.refresh_inter_class_geometry()

        if old_training_state:
            self.model.train()
        else:
            self.model.eval()

    def _should_refresh_for_validation(self, epoch: int) -> bool:
        if not self.refresh_before_validation:
            return False
        if self.validation_refresh_every <= 0:
            return False
        return (int(epoch) + 1) % self.validation_refresh_every == 0

    def _select_score(self, val_stats: dict, phase: int) -> float:
        if phase == 0:
            return float(val_stats.get("acc", 0.0))

        metric = self.best_state_metric
        if metric in {"hm", "h", "harmonic"}:
            return float(val_stats.get("hm", 0.0))
        if metric in {"acc", "oa"}:
            return float(val_stats.get("acc", 0.0))
        if metric in {"old"}:
            return float(val_stats.get("old_acc", 0.0))
        if metric in {"new"}:
            return float(val_stats.get("new_acc", 0.0))

        # Default for incremental CIL: harmonic mean.
        return float(val_stats.get("hm", 0.0))

    def _capture_state(self):
        return {
            k: v.detach().cpu().clone()
            for k, v in self.model.state_dict().items()
        }

    def _soft_cap_loss(self, loss: torch.Tensor, max_value: float) -> torch.Tensor:
        """Scale an auxiliary loss when it explodes, without hard zeroing gradients.

        torch.clamp(loss, max=...) kills the gradient above the cap. This keeps
        the gradient direction but rescales its magnitude.
        """
        max_value = float(max_value)
        if max_value <= 0.0:
            return loss
        detached = loss.detach().abs()
        scale = torch.clamp(
            torch.as_tensor(max_value, device=loss.device, dtype=loss.dtype) / (detached + 1e-8),
            max=1.0,
        )
        return loss * scale

    def _base_geometry_ramp(self, epoch_idx: int) -> float:
        warm = int(getattr(self, "base_geometry_warmup_epochs", 20))
        ramp_epochs = max(int(getattr(self, "base_geometry_ramp_epochs", 30)), 1)
        epoch_idx = int(epoch_idx)
        if epoch_idx < warm:
            return 0.0
        return min(1.0, float(epoch_idx - warm + 1) / float(ramp_epochs))

    # ============================================================
    # Trainability
    # ============================================================
    def _set_incremental_trainable_params(self, old_class_count: int):
        for p in self.model.parameters():
            p.requires_grad = True

        if hasattr(self.model, "freeze_backbone_only"):
            self.model.freeze_backbone_only()

        bb = getattr(self.model, "backbone", None)
        if (
            bool(getattr(self.args, "unfreeze_last_backbone_during_incremental", False))
            and bb is not None
            and hasattr(bb, "unfreeze_last_blocks")
        ):
            bb.unfreeze_last_blocks()

        if (
            bool(getattr(self.args, "freeze_semantic_encoder_during_incremental", True))
            and hasattr(self.model, "freeze_semantic_encoder")
        ):
            self.model.freeze_semantic_encoder()
        elif hasattr(self.model, "unfreeze_semantic_encoder"):
            self.model.unfreeze_semantic_encoder()

        if bool(getattr(self.args, "freeze_projection_during_incremental", False)):
            if hasattr(self.model, "freeze_projection_head"):
                self.model.freeze_projection_head()
        else:
            if hasattr(self.model, "unfreeze_projection_head"):
                self.model.unfreeze_projection_head()

        if hasattr(self.model, "freeze_old_anchor_deltas"):
            self.model.freeze_old_anchor_deltas(old_class_count)
        if hasattr(self.model, "unfreeze_new_anchor_deltas"):
            self.model.unfreeze_new_anchor_deltas(old_class_count)
        if hasattr(self.model, "freeze_old_concept_deltas"):
            self.model.freeze_old_concept_deltas(old_class_count)

        freeze_classifier = bool(getattr(self.args, "freeze_classifier_during_incremental", False))
        if freeze_classifier:
            if hasattr(self.model, "freeze_classifier_adaptation"):
                self.model.freeze_classifier_adaptation()
        else:
            if hasattr(self.model, "unfreeze_classifier_adaptation"):
                self.model.unfreeze_classifier_adaptation()
            if hasattr(self.model, "freeze_old_classifier_adaptation"):
                self.model.freeze_old_classifier_adaptation(old_class_count)

        # Fusion is legacy/hybrid. Keep it frozen unless explicitly enabled.
        if bool(getattr(self.args, "use_adaptive_fusion", False)):
            if hasattr(self.model, "unfreeze_fusion_module"):
                self.model.unfreeze_fusion_module()
        else:
            if hasattr(self.model, "freeze_fusion_module"):
                self.model.freeze_fusion_module()

        # geometry calibrator remains trainable in incremental phases
        if hasattr(self.model, "unfreeze_geometry_calibrator"):
            self.model.unfreeze_geometry_calibrator()

    # ============================================================
    # Base auxiliary losses
    # ============================================================
    def _base_aux_losses(self, out, y):
        features = out["features"]
        logits = out["logits"]
        concept_bank = out.get("concept_bank", None)

        compact, sep, ortho, center_norm, radius = self._base_geometry_loss(features, y)

        margin = (
            self.logit_margin(logits, y)
            if self.logit_margin_weight > 0.0
            else self._zero(features)
        )

        concept_sep = (
            self.concept_sep(concept_bank)
            if self.concept_sep_weight > 0.0 and concept_bank is not None and concept_bank.numel() > 0
            else self._zero(features)
        )

        feature_concept = (
            self.feature_concept_compact(features, concept_bank, y)
            if self.feature_concept_compact_weight > 0.0 and concept_bank is not None and concept_bank.numel() > 0
            else self._zero(features)
        )

        return compact, sep, ortho, center_norm, radius, margin, concept_sep, feature_concept

    def _classifier_adaptation_reg(self, ref: torch.Tensor) -> torch.Tensor:
        if self.classifier_adaptation_weight <= 0.0:
            return self._zero(ref)

        classifier = getattr(self.model, "classifier", None)
        if classifier is None:
            return self._zero(ref)

        if hasattr(classifier, "adaptation_regularization_loss"):
            reg = classifier.adaptation_regularization_loss()
            if isinstance(reg, dict):
                return reg.get("total", self._zero(ref))
            return reg

        return self._zero(ref)

    # ============================================================
    # Train one epoch
    # ============================================================
    def _train_epoch_base(self, loader, optimizer, epoch_idx: int = 0):
        self.model.train()

        total_loss = 0.0
        total_ce = 0.0
        total_geo_raw = 0.0
        total_geo_used = 0.0
        total_correct = 0
        total = 0
        geo_ramp = self._base_geometry_ramp(epoch_idx)

        for x, y in loader:
            x = x.to(self.device).float()
            y = y.to(self.device).long()

            optimizer.zero_grad(set_to_none=True)

            out = self.model(
                x,
                semantic_mode="identity",
                classifier_mode=getattr(self.args, "base_classifier_mode", "geometry_only"),
            )

            ce = F.cross_entropy(
                out["logits"],
                y,
                label_smoothing=float(getattr(self.args, "label_smoothing", 0.0)),
            )
            compact, sep, ortho, center_norm, radius, margin, concept_sep, feature_concept = (
                self._base_aux_losses(out, y)
            )

            geo_raw = (
                self.base_compact * compact
                + self.base_sep * sep
                + self.base_ortho * ortho
                + self.base_center_norm * center_norm
                + self.base_radius * radius
                + self.logit_margin_weight * margin
                + self.concept_sep_weight * concept_sep
                + self.feature_concept_compact_weight * feature_concept
            )
            geo_used = self._soft_cap_loss(geo_raw, self.base_max_geo_loss)
            loss = ce + geo_ramp * geo_used

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                1.0,
            )
            optimizer.step()

            total_loss += float(loss.detach().item())
            total_ce += float(ce.detach().item())
            total_geo_raw += float(geo_raw.detach().item())
            total_geo_used += float((geo_ramp * geo_used).detach().item())
            total_correct += int((out["logits"].argmax(1) == y).sum().item())
            total += int(y.size(0))

        denom = max(len(loader), 1)
        return {
            "loss": total_loss / denom,
            "ce": total_ce / denom,
            "geo_raw": total_geo_raw / denom,
            "geo_used": total_geo_used / denom,
            "acc": 100.0 * total_correct / max(total, 1),
            "geo_ramp": float(geo_ramp),
        }

    def _train_epoch_incremental(
        self,
        loader,
        optimizer,
        old_class_count,
        new_class_ids,
        old_bank_snapshot,
        old_token_snapshot=None,
        epoch_idx: int = 0,
    ):
        self.model.train()
        self._set_model_phase_and_old_count(self.model.current_phase, old_class_count)

        total_loss = 0.0
        total_correct = 0
        total = 0

        classifier_mode = self._inc_classifier_mode()

        token_loss_weight = float(getattr(self.args, "token_loss_weight", 0.0))
        spectral_guidance_weight = float(getattr(self.args, "spectral_guidance_weight", 0.0))
        band_guidance_weight = float(getattr(self.args, "band_guidance_weight", 0.0))
        replay_geometry_weight = float(getattr(self.args, "replay_geometry_weight", 0.0))
        new_ce_weight = float(getattr(self, "incremental_new_ce_weight", 1.0))
        replay_weight_eff = self.replay_weight * float(getattr(self, "incremental_replay_weight_mult", 1.0))

        use_token_relations = token_loss_weight > 0.0

        if self.incremental_warmup_epochs > 0:
            post_warm_epochs = max(
                int(getattr(self.args, "epochs_inc", self.incremental_warmup_epochs)) - self.incremental_warmup_epochs,
                1,
            )
            if epoch_idx < self.incremental_warmup_epochs:
                structure_ramp = 0.0
            else:
                structure_ramp = min(
                    1.0,
                    float(epoch_idx - self.incremental_warmup_epochs + 1) / float(post_warm_epochs),
                )
        else:
            structure_ramp = 1.0

        for x, y in loader:
            x = x.to(self.device).float()
            y = y.to(self.device).long()

            optimizer.zero_grad(set_to_none=True)

            out = self.model(
                x,
                semantic_mode="identity",
                classifier_mode=classifier_mode,
                return_token_relations=use_token_relations,
            )
            logits = out["logits"]
            features = out["features"]
            concept_bank = out.get("concept_bank", None)

            # 1) New-class supervision
            ce_new = self._masked_weighted_ce_new(logits, y, new_class_ids)

            # 2) Old-class replay
            replay_x, replay_y = self._sample_replay_from_snapshot(
                old_bank_snapshot, old_class_count
            )

            if replay_x is not None:
                replay_logits = self.model.compute_logits_from_features(
                    replay_x,
                    classifier_mode=classifier_mode,
                )
                ce_replay = F.cross_entropy(replay_logits, replay_y)

                replay_geom_loss = self._replay_geometry_loss(
                    replay_x,
                    replay_y,
                    old_bank_snapshot=old_bank_snapshot,
                )
            else:
                ce_replay = self._zero(logits)
                replay_geom_loss = self._zero(logits)

            # 3) Weak bank-vs-snapshot consistency
            a_mean, a_basis, a_var, _ = self._geometry_alignment_losses(
                old_bank_snapshot,
                old_class_count,
            )

            # 4) Geometry-native separation and compactness
            if hasattr(self, "_symmetric_geometry_separation_loss"):
                sep_geo_loss = self._symmetric_geometry_separation_loss(
                    new_features=features,
                    new_labels=y,
                    replay_features=replay_x,
                    replay_labels=replay_y,
                    old_class_count=old_class_count,
                    new_class_ids=new_class_ids,
                    old_bank_snapshot=old_bank_snapshot,
                )
            else:
                sep_geo_loss = self._geometry_separation_loss(
                    features, y, old_class_count, new_class_ids
                )
            vol_loss = self._new_class_volume_loss(features, y, new_class_ids)

            # 5) Structural preservation
            token_loss, token_parts = self._token_manifold_loss(
                out, y, old_class_count, old_token_snapshot
            )
            del token_parts

            spec_loss, band_loss = self._spectral_guidance_losses(
                out, y, old_class_count
            )

            # 6) Calibration/adaptation regularization
            calibration_reg = out.get("calibration_reg", None)
            if calibration_reg is None:
                calibration_total = self._zero(logits)
            else:
                calibration_total = calibration_reg.get("total", self._zero(logits))

            cls_adapt_reg = self._classifier_adaptation_reg(logits)

            # 7) New-class auxiliaries
            margin_new = self._zero(logits)
            feature_concept_new = self._zero(logits)

            class_ids = torch.tensor(new_class_ids, device=logits.device, dtype=torch.long)

            if self._labels_are_local(y, new_class_ids):
                y_local = y
            else:
                y_local = torch.full_like(y, fill_value=-1)
                for local_idx, global_cls in enumerate(class_ids):
                    y_local[y == global_cls] = local_idx

            valid = y_local >= 0

            if self.inc_logit_margin_weight > 0.0 and valid.any():
                logits_new_only = logits.index_select(1, class_ids)
                margin_new = self.logit_margin(logits_new_only[valid], y_local[valid])

            if (
                self.inc_feature_concept_compact_weight > 0.0
                and concept_bank is not None
                and concept_bank.numel() > 0
                and valid.any()
            ):
                new_concept_bank = concept_bank.index_select(0, class_ids)
                feature_concept_new = self.feature_concept_compact(
                    features[valid],
                    new_concept_bank,
                    y_local[valid],
                )

            # 8) Loss assembly
            if epoch_idx < self.incremental_warmup_epochs:
                aux_loss = (
                    self.inc_logit_margin_weight * margin_new
                    + self.geometry_calibration_weight * calibration_total
                    + self.classifier_adaptation_weight * cls_adapt_reg
                )
            else:
                aux_loss = (
                    replay_geometry_weight * structure_ramp * replay_geom_loss
                    + self.align_mean_weight * structure_ramp * a_mean
                    + self.align_basis_weight * structure_ramp * a_basis
                    + self.align_var_weight * structure_ramp * a_var
                    + self.insert_weight * structure_ramp * sep_geo_loss
                    + self.new_volume_weight * structure_ramp * vol_loss
                    + self.geometry_calibration_weight * calibration_total
                    + self.classifier_adaptation_weight * cls_adapt_reg
                    + token_loss_weight * structure_ramp * token_loss
                    + spectral_guidance_weight * structure_ramp * spec_loss
                    + band_guidance_weight * structure_ramp * band_loss
                    + self.inc_logit_margin_weight * margin_new
                    + self.inc_feature_concept_compact_weight * feature_concept_new
                )

            aux_loss = self._soft_cap_loss(aux_loss, self.incremental_max_aux_loss)
            loss = new_ce_weight * ce_new + replay_weight_eff * ce_replay + aux_loss

            if not torch.isfinite(loss):
                if self.debug:
                    print(
                        "[WARN] Non-finite incremental loss encountered. "
                        f"ce_new={float(ce_new.detach().item()):.6f}, "
                        f"ce_replay={float(ce_replay.detach().item()):.6f}, "
                        f"replay_geom={float(replay_geom_loss.detach().item()):.6f}, "
                        f"a_mean={float(a_mean.detach().item()):.6f}, "
                        f"a_basis={float(a_basis.detach().item()):.6f}, "
                        f"a_var={float(a_var.detach().item()):.6f}, "
                        f"sep_geo={float(sep_geo_loss.detach().item()):.6f}, "
                        f"vol={float(vol_loss.detach().item()):.6f}, "
                        f"calib={float(calibration_total.detach().item()):.6f}, "
                        f"cls_adapt={float(cls_adapt_reg.detach().item()):.6f}, "
                        f"token={float(token_loss.detach().item()):.6f}, "
                        f"spec={float(spec_loss.detach().item()):.6f}, "
                        f"band={float(band_loss.detach().item()):.6f}, "
                        f"margin={float(margin_new.detach().item()):.6f}, "
                        f"feat_concept={float(feature_concept_new.detach().item()):.6f}"
                    )
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], 0.5
            )
            optimizer.step()

            total_loss += float(loss.item())

            correct_new, valid_new = self._incremental_accuracy_with_count(
                logits, y, new_class_ids
            )
            total_correct += correct_new
            total += valid_new

            if self.debug:
                print(
                    f"[IncDebug] loss={float(loss.item()):.4f} | "
                    f"ce_new={float(ce_new.detach().item()):.4f} | "
                    f"ce_replay={float(ce_replay.detach().item()):.4f} | "
                    f"replay_geom={float(replay_geom_loss.detach().item()):.4f} | "
                    f"a_mean={float(a_mean.detach().item()):.4f} | "
                    f"a_basis={float(a_basis.detach().item()):.4f} | "
                    f"a_var={float(a_var.detach().item()):.4f} | "
                    f"sep_geo={float(sep_geo_loss.detach().item()):.4f} | "
                    f"vol={float(vol_loss.detach().item()):.4f} | "
                    f"calib={float(calibration_total.detach().item()):.4f} | "
                    f"cls_adapt={float(cls_adapt_reg.detach().item()):.4f} | "
                    f"token={float(token_loss.detach().item()):.4f} | "
                    f"spec={float(spec_loss.detach().item()):.4f} | "
                    f"band={float(band_loss.detach().item()):.4f} | "
                    f"margin={float(margin_new.detach().item()):.4f} | "
                    f"feat_concept={float(feature_concept_new.detach().item()):.4f} | "
                    f"ramp={structure_ramp:.3f}"
                )

        train_acc = 100.0 * total_correct / max(total, 1)
        return total_loss / max(len(loader), 1), train_acc

    # ============================================================
    # Post-phase calibration
    # ============================================================
    def _set_calibration_trainable_params(self):
        for p in self.model.parameters():
            p.requires_grad = False

        if hasattr(self.model, "unfreeze_classifier_adaptation"):
            self.model.unfreeze_classifier_adaptation()

        # Freeze old classifier adaptation again if available; calibration should
        # tune global/allowed offsets conservatively, not rewrite old class params.
        old_class_count = int(getattr(self.model, "old_class_count", 0))
        if hasattr(self.model, "freeze_old_classifier_adaptation") and old_class_count > 0:
            self.model.freeze_old_classifier_adaptation(old_class_count)

        if hasattr(self.model, "unfreeze_geometry_calibrator"):
            self.model.unfreeze_geometry_calibrator()

    def _post_phase_calibration(
        self,
        phase: int,
        old_class_count: int,
        new_class_ids,
        old_bank_snapshot,
        batch_size: int,
    ):
        if self.calibration_epochs <= 0:
            return

        self._set_calibration_trainable_params()

        calib_params = [p for p in self.model.parameters() if p.requires_grad]
        if len(calib_params) == 0:
            return

        optimizer = optim.Adam(calib_params, lr=self.calibration_lr, weight_decay=0.0)

        train_loader = self.dataset.get_phase_dataloader(
            phase,
            split="train",
            batch_size=batch_size,
            shuffle=True,
        )

        classifier_mode = self._eval_classifier_mode()
        replay_geometry_weight = float(getattr(self.args, "replay_geometry_weight", 0.0))

        self.model.train()
        self._set_model_phase_and_old_count(phase, old_class_count)

        best_state = None
        best_loss = float("inf")

        for _ in range(self.calibration_epochs):
            epoch_loss = 0.0
            epoch_steps = 0

            for x, y in train_loader:
                x = x.to(self.device).float()
                y = y.to(self.device).long()

                optimizer.zero_grad(set_to_none=True)

                out = self.model(
                    x,
                    semantic_mode="identity",
                    classifier_mode=classifier_mode,
                )
                ce_new = self._masked_weighted_ce_new(out["logits"], y, new_class_ids)

                replay_x, replay_y = self._sample_replay_from_snapshot(
                    old_bank_snapshot,
                    old_class_count,
                )

                if replay_x is not None:
                    replay_logits = self.model.compute_logits_from_features(
                        replay_x,
                        classifier_mode=classifier_mode,
                    )
                    ce_old = F.cross_entropy(replay_logits, replay_y)

                    replay_geom_loss = self._replay_geometry_loss(
                        replay_x,
                        replay_y,
                        old_bank_snapshot=old_bank_snapshot,
                    )
                else:
                    ce_old = self._zero(out["logits"])
                    replay_geom_loss = self._zero(out["logits"])

                calibration_reg = out.get("calibration_reg", None)
                if calibration_reg is None:
                    calibration_total = self._zero(out["logits"])
                else:
                    calibration_total = calibration_reg.get("total", self._zero(out["logits"]))

                cls_adapt_reg = self._classifier_adaptation_reg(out["logits"])

                loss = (
                    ce_new
                    + self.calibration_replay_weight * ce_old
                    + self.geometry_calibration_weight * calibration_total
                    + self.classifier_adaptation_weight * cls_adapt_reg
                    + 0.5 * replay_geometry_weight * replay_geom_loss
                )

                if not torch.isfinite(loss):
                    if self.debug:
                        print(
                            "[WARN] Non-finite calibration loss encountered. "
                            f"ce_new={float(ce_new.detach().item()):.6f}, "
                            f"ce_old={float(ce_old.detach().item()):.6f}, "
                            f"calib={float(calibration_total.detach().item()):.6f}, "
                            f"cls_adapt={float(cls_adapt_reg.detach().item()):.6f}, "
                            f"replay_geom={float(replay_geom_loss.detach().item()):.6f}"
                        )
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(calib_params, 0.5)
                optimizer.step()

                epoch_loss += float(loss.item())
                epoch_steps += 1

            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_state = self._capture_state()

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self._set_model_phase_and_old_count(phase, old_class_count)

    # ============================================================
    # Validation
    # ============================================================
    @torch.no_grad()
    def _validate_split_metrics(self, loader, old_class_count: int):
        self.model.eval()
        self._set_model_phase_and_old_count(self.model.current_phase, old_class_count)

        total_loss = total_correct = total = 0
        old_correct = old_total = 0
        new_correct = new_total = 0

        if int(old_class_count) == 0:
            val_classifier_mode = getattr(self.args, "base_classifier_mode", "geometry_only")
            val_semantic_mode = "identity"
        else:
            val_classifier_mode = self._eval_classifier_mode()
            val_semantic_mode = getattr(self.args, "eval_semantic_mode", "identity")

        for x, y in loader:
            x = x.to(self.device).float()
            y = y.to(self.device).long()

            out = self.model(
                x,
                semantic_mode=val_semantic_mode,
                classifier_mode=val_classifier_mode,
            )

            logits = out["logits"]
            loss = F.cross_entropy(logits, y)
            preds = logits.argmax(dim=1)

            total_loss += float(loss.item())
            total_correct += int((preds == y).sum().item())
            total += int(y.size(0))

            old_mask = y < old_class_count
            new_mask = y >= old_class_count

            if old_mask.any():
                old_correct += int((preds[old_mask] == y[old_mask]).sum().item())
                old_total += int(old_mask.sum().item())

            if new_mask.any():
                new_correct += int((preds[new_mask] == y[new_mask]).sum().item())
                new_total += int(new_mask.sum().item())

        old_acc = 100.0 * old_correct / max(old_total, 1)
        new_acc = 100.0 * new_correct / max(new_total, 1)
        total_acc = 100.0 * total_correct / max(total, 1)
        hm = 0.0 if (old_acc + new_acc) == 0 else 2.0 * old_acc * new_acc / (old_acc + new_acc)

        return {
            "loss": total_loss / max(len(loader), 1),
            "acc": total_acc,
            "old_acc": old_acc,
            "new_acc": new_acc,
            "hm": hm,
        }

    # ============================================================
    # Phase training
    # ============================================================
    def train_phase(self, phase, epochs, batch_size=64, lr=1e-4):
        print(f"==== Training Phase {phase} ====")

        phase = int(phase)
        self.dataset.start_phase(phase)

        needed_classes = max(self.dataset.phase_to_classes[phase]) + 1
        if int(self.model.current_num_classes) < needed_classes:
            self._bootstrap_phase_classes(phase, split="train")

        old_class_count = 0 if phase == 0 else len(self.dataset.get_classes_up_to_phase(phase - 1))
        self._set_model_phase_and_old_count(phase, old_class_count)

        history = {
            "train_loss": [],
            # Train accuracy is the refreshed train-set diagnostic, not online/stale-bank accuracy.
            "train_acc": [],
            "train_old_acc": [],
            "train_new_acc": [],
            "train_hm": [],
            "val_loss": [],
            "val_acc": [],
            "val_old_acc": [],
            "val_new_acc": [],
            "val_hm": [],
        }

        val_loader = self.dataset.get_cumulative_dataloader(
            phase,
            split="val",
            batch_size=batch_size,
            shuffle=False,
        )

        # ---------------- Base phase ----------------
        if phase == 0:
            train_loader = self.dataset.get_phase_dataloader(
                phase,
                split="train",
                batch_size=batch_size,
                shuffle=True,
            )

            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

            best_state, best_score = None, -1.0
            no_improve = 0

            base_class_ids = [int(c) for c in self.dataset.phase_to_classes[phase]]

            for epoch in range(epochs):
                self._set_model_phase_and_old_count(phase, 0)

                tr_stats = self._train_epoch_base(train_loader, optimizer, epoch_idx=epoch)

                # Refresh base geometry before validation; otherwise phase-0
                # validation is evaluated against stale placeholder geometry.
                if self._should_refresh_for_validation(epoch):
                    self._refresh_classes_for_validation(phase, base_class_ids, split="train")

                # Evaluate train split after geometry refresh.
                # This refreshed train accuracy is the reported Train Acc.
                train_eval_stats = self._validate_split_metrics(train_loader, 0)
                val_stats = self._validate_split_metrics(val_loader, 0)
                scheduler.step()

                history["train_loss"].append(tr_stats["loss"])
                history.setdefault("train_ce_loss", []).append(tr_stats["ce"])
                history.setdefault("train_geo_raw", []).append(tr_stats["geo_raw"])
                history.setdefault("train_geo_used", []).append(tr_stats["geo_used"])
                history.setdefault("train_geo_ramp", []).append(tr_stats["geo_ramp"])
                history["train_acc"].append(train_eval_stats["acc"])
                history["train_old_acc"].append(train_eval_stats["old_acc"])
                history["train_new_acc"].append(train_eval_stats["new_acc"])
                history["train_hm"].append(train_eval_stats["hm"])
                history["val_loss"].append(val_stats["loss"])
                history["val_acc"].append(val_stats["acc"])
                history["val_old_acc"].append(val_stats["old_acc"])
                history["val_new_acc"].append(val_stats["new_acc"])
                history["val_hm"].append(val_stats["hm"])

                print(
                    f"Ep {epoch+1:03d}/{epochs} | "
                    f"Train Loss: {tr_stats['loss']:.4f} | "
                    f"CE: {tr_stats['ce']:.4f} | "
                    f"Geo(raw/used): {tr_stats['geo_raw']:.2f}/{tr_stats['geo_used']:.2f} | "
                    f"Ramp: {tr_stats['geo_ramp']:.2f} | "
                    f"Train Acc: {train_eval_stats['acc']:.2f}% | "
                    f"Val Acc: {val_stats['acc']:.2f}% | "
                    f"Val Loss: {val_stats['loss']:.4f} | "
                    f"Old: {val_stats['old_acc']:.2f}% | "
                    f"New: {val_stats['new_acc']:.2f}% | "
                    f"H: {val_stats['hm']:.2f}%"
                )

                score = self._select_score(val_stats, phase=0)
                if score > best_score:
                    best_score = score
                    best_state = self._capture_state()
                    no_improve = 0
                else:
                    no_improve += 1

                if self.early_stop_patience > 0 and no_improve >= self.early_stop_patience:
                    print(f"[EarlyStop] Phase {phase}: no improvement for {no_improve} epochs.")
                    break

            if best_state is not None:
                self.model.load_state_dict(best_state)
                self._set_model_phase_and_old_count(phase, 0)

            self._finalize_phase_memory(phase, split="train")
            gb = getattr(self.model, "geometry_bank", None)
            if gb is not None and hasattr(gb, "refresh_inter_class_geometry"):
                gb.refresh_inter_class_geometry()

            self.model.old_class_count = int(self.model.current_num_classes)
            # self.save_checkpoint(phase, history)
            return history

        # ---------------- Incremental phase ----------------
        old_bank_snapshot = self._snapshot_old_bank(old_class_count)
        old_token_snapshot = self._snapshot_old_token_memory(old_class_count)
        new_class_ids = [int(c) for c in self.dataset.phase_to_classes[phase]]

        self._set_incremental_trainable_params(old_class_count)

        train_loader = self.dataset.get_phase_dataloader(
            phase,
            split="train",
            batch_size=batch_size,
            shuffle=True,
        )

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise RuntimeError("No trainable parameters found for incremental phase.")

        optimizer = optim.Adam(
            trainable_params,
            lr=lr,
            weight_decay=1e-5,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        best_state, best_score = None, -1.0
        no_improve = 0

        for epoch in range(epochs):
            self._set_model_phase_and_old_count(phase, old_class_count)

            tr_loss, tr_acc = self._train_epoch_incremental(
                train_loader,
                optimizer,
                old_class_count,
                new_class_ids,
                old_bank_snapshot,
                old_token_snapshot,
                epoch_idx=epoch,
            )

            # --------------------------------------------------------
            # Correct order:
            #   1. train epoch
            #   2. refresh current-phase geometry
            #   3. validate
            #   4. save best state
            #
            # The old trainer validated before refresh, which made validation
            # look collapsed even when post-finalization evaluation recovered.
            # --------------------------------------------------------
            refresh_every = int(getattr(self.args, "bank_refresh_every", 0))
            do_periodic_refresh = refresh_every > 0 and (epoch + 1) % refresh_every == 0
            do_validation_refresh = self._should_refresh_for_validation(epoch)

            if do_periodic_refresh or do_validation_refresh:
                self._refresh_classes_for_validation(phase, new_class_ids, split="train")

            # Evaluate current-phase train split after geometry refresh.
            # This refreshed accuracy is the reported Train Acc.
            train_eval_stats = self._validate_split_metrics(train_loader, old_class_count)
            val_stats = self._validate_split_metrics(val_loader, old_class_count)
            scheduler.step()

            history["train_loss"].append(tr_loss)
            history["train_acc"].append(train_eval_stats["acc"])
            history["train_old_acc"].append(train_eval_stats["old_acc"])
            history["train_new_acc"].append(train_eval_stats["new_acc"])
            history["train_hm"].append(train_eval_stats["hm"])
            history["val_loss"].append(val_stats["loss"])
            history["val_acc"].append(val_stats["acc"])
            history["val_old_acc"].append(val_stats["old_acc"])
            history["val_new_acc"].append(val_stats["new_acc"])
            history["val_hm"].append(val_stats["hm"])

            print(
                f"Ep {epoch+1:03d}/{epochs} | "
                f"Train Loss: {tr_loss:.4f} | "
                f"Train Acc: {train_eval_stats['acc']:.2f}% | "
                f"Train O/N/H: {train_eval_stats['old_acc']:.2f}/{train_eval_stats['new_acc']:.2f}/{train_eval_stats['hm']:.2f}% | "
                f"Val Acc: {val_stats['acc']:.2f}% | "
                f"Val Loss: {val_stats['loss']:.4f} | "
                f"Val O/N/H: {val_stats['old_acc']:.2f}/{val_stats['new_acc']:.2f}/{val_stats['hm']:.2f}%"
            )

            score = self._select_score(val_stats, phase=phase)
            if score > best_score:
                best_score = score
                best_state = self._capture_state()
                no_improve = 0
            else:
                no_improve += 1

            if self.early_stop_patience > 0 and no_improve >= self.early_stop_patience:
                print(f"[EarlyStop] Phase {phase}: no improvement for {no_improve} epochs.")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self._set_model_phase_and_old_count(phase, old_class_count)

        self._post_phase_calibration(
            phase=phase,
            old_class_count=old_class_count,
            new_class_ids=new_class_ids,
            old_bank_snapshot=old_bank_snapshot,
            batch_size=batch_size,
        )

        self._finalize_phase_memory(phase, split="train")
        gb = getattr(self.model, "geometry_bank", None)
        if gb is not None and hasattr(gb, "refresh_inter_class_geometry"):
            gb.refresh_inter_class_geometry()

        self.model.old_class_count = int(self.model.current_num_classes)
        return history

    # ============================================================
    # Save
    # ============================================================
    def save_checkpoint(self, phase, history, evaluator_metrics=None):
        run_dir = getattr(self.args, "run_dir", None)
        if run_dir is not None and str(run_dir).strip():
            phase_dir = os.path.join(str(run_dir), f"phase_{phase}")
        else:
            phase_dir = os.path.join(self.save_dir, self.args.dataset, f"phase_{phase}")
        os.makedirs(phase_dir, exist_ok=True)

        ckpt = {
            "phase": int(phase),
            "model_state_dict": self.model.state_dict(),
            "memory_snapshot": self.model.export_memory_snapshot() if hasattr(self.model, "export_memory_snapshot") else None,
            "current_num_classes": int(self.model.current_num_classes),
            "old_class_count": int(getattr(self.model, "old_class_count", 0)),
            "history": history,
            "args": vars(self.args),
            "token_memory": {
                int(k): {kk: vv.detach().cpu() for kk, vv in v.items()}
                for k, v in getattr(self, "token_memory", {}).items()
            },
        }
        if evaluator_metrics is not None:
            ckpt["evaluator_metrics"] = evaluator_metrics

        path = os.path.join(phase_dir, "checkpoint.pth")
        torch.save(ckpt, path)
        print(f"[Saved] {path}")












# import os
# from contextlib import nullcontext

# import torch
# import torch.nn.functional as F
# import torch.optim as optim

# from losses.necil_losses import GlobalLogitMargin, ConceptSeparation, FeatureConceptCompactness
# from trainers.trainer_helpers import TrainerHelper


# class Trainer(TrainerHelper):
#     """
#     Geometry-centric trainer for strict non-exemplar HSI CIL.

#     Corrected stability policy
#     --------------------------
#     1. Validate only after current-phase geometry is refreshed.
#     2. Select best incremental checkpoint by refreshed validation harmonic mean.
#     3. Do not accidentally unfreeze old classifier adaptation.
#     4. Keep old raw samples inaccessible under strict non-exemplar protocol.
#     5. Refresh only current-phase geometry during incremental training.
#     6. Use identity semantic path for all geometry-native training/eval.
#     """

#     def __init__(self, model, dataset, args):
#         self.args = args
#         self.model = model.to(self.device)
#         self.dataset = dataset
#         self.save_dir = args.save_dir
#         self.debug = bool(getattr(args, "debug_verbose", False)) or os.environ.get("NECIL_DEBUG", "0") == "1"

#         self.subspace_rank = int(getattr(args, "subspace_rank", 5))
#         self.geom_var_floor = float(getattr(args, "geom_var_floor", 1e-4))
#         self.num_concepts_per_class = int(getattr(args, "num_concepts_per_class", 4))
#         self.alignment_samples_per_class = int(getattr(args, "alignment_samples_per_class", 16))

#         # ---------------- Base phase ----------------
#         self.base_compact = float(getattr(args, "base_compact", 0.05))
#         self.base_sep = float(getattr(args, "base_sep", 0.05))
#         self.base_ortho = float(getattr(args, "base_ortho", 0.03))
#         self.base_margin = float(getattr(args, "base_margin", 1.0))
#         self.base_center_norm = float(getattr(args, "base_center_norm", 0.01))
#         self.base_radius = float(getattr(args, "base_radius", 0.01))

#         # ---------------- Incremental phase ----------------
#         self.replay_weight = float(getattr(args, "synthetic_replay_weight", 1.0))
#         self.replay_per_class = int(getattr(args, "synthetic_replay_per_class", 32))
#         self.align_mean_weight = float(getattr(args, "align_mean_weight", 0.10))
#         self.align_basis_weight = float(getattr(args, "align_basis_weight", 0.05))
#         self.align_var_weight = float(getattr(args, "align_var_weight", 0.02))
#         self.insert_weight = float(getattr(args, "insert_weight", 0.01))
#         self.insert_margin = float(getattr(args, "insert_margin", 5.0))
#         self.new_volume_weight = float(getattr(args, "new_volume_weight", 0.005))
#         self.new_volume_target = float(getattr(args, "new_volume_target", 1.5))
#         self.incremental_warmup_epochs = int(getattr(args, "incremental_warmup_epochs", 5))

#         # geometry-calibration regularization
#         self.geometry_calibration_weight = float(
#             getattr(args, "geometry_calibration_weight", 0.05)
#         )

#         # ---------------- Calibration ----------------
#         self.calibration_epochs = int(getattr(args, "calibration_epochs", 5))
#         self.calibration_lr = float(getattr(args, "calibration_lr", 5e-4))
#         self.calibration_replay_weight = float(getattr(args, "calibration_replay_weight", 1.0))

#         # ---------------- Validation / checkpoint policy ----------------
#         self.refresh_before_validation = bool(getattr(args, "refresh_before_validation", True))
#         self.validation_refresh_every = int(getattr(args, "validation_refresh_every", 1))
#         self.best_state_metric = str(getattr(args, "best_state_metric", "hm")).lower()
#         self.early_stop_patience = int(getattr(args, "early_stop_patience", 0))

#         # ---------------- Auxiliary losses ----------------
#         self.logit_margin_value = float(getattr(args, "logit_margin_value", 0.20))
#         self.logit_margin_weight = float(getattr(args, "logit_margin_weight", 0.02))
#         self.concept_sep_weight = float(getattr(args, "concept_sep_weight", 0.01))
#         self.feature_concept_compact_weight = float(
#             getattr(args, "feature_concept_compact_weight", 0.03)
#         )
#         self.inc_logit_margin_weight = float(getattr(args, "inc_logit_margin_weight", 0.005))
#         self.inc_feature_concept_compact_weight = float(
#             getattr(args, "inc_feature_concept_compact_weight", 0.005)
#         )
#         self.classifier_adaptation_weight = float(getattr(args, "classifier_adaptation_weight", 0.0))

#         self.logit_margin = GlobalLogitMargin(margin=self.logit_margin_value)
#         self.concept_sep = ConceptSeparation(
#             max_cosine=float(getattr(args, "concept_sep_max_cosine", 0.25))
#         )
#         self.feature_concept_compact = FeatureConceptCompactness(
#             temperature=float(getattr(args, "cls_temperature", 0.07))
#         )

#         self.token_memory = {}

#     # ============================================================
#     # Mode helpers
#     # ============================================================
#     def _inc_classifier_mode(self) -> str:
#         return str(
#             getattr(self.args, "incremental_classifier_mode", "calibrated_geometry")
#         ).lower()

#     def _eval_classifier_mode(self) -> str:
#         return str(
#             getattr(self.args, "eval_classifier_mode", "calibrated_geometry")
#         ).lower()

#     def _set_model_phase_and_old_count(self, phase: int, old_class_count: int):
#         if hasattr(self.model, "set_phase"):
#             self.model.set_phase(int(phase))
#         else:
#             self.model.current_phase = int(phase)

#         if hasattr(self.model, "set_old_class_count"):
#             self.model.set_old_class_count(int(old_class_count))
#         else:
#             self.model.old_class_count = int(old_class_count)

#     # ============================================================
#     # Geometry refresh helpers
#     # ============================================================
#     @torch.no_grad()
#     def _refresh_classes_for_validation(self, phase: int, class_ids, split: str = "train"):
#         """
#         Refresh only currently accessible classes before validation.

#         This fixes the stale-geometry validation bug:
#         validation must evaluate the model with the same current-phase geometry
#         that will be used after phase finalization.
#         """
#         if not self.refresh_before_validation:
#             return

#         phase = int(phase)
#         old_training_state = self.model.training

#         # current phase train classes are allowed in strict non-exemplar mode.
#         ctx = (
#             self.dataset.memory_build_context(phase)
#             if hasattr(self.dataset, "memory_build_context")
#             else nullcontext()
#         )

#         with ctx:
#             for cls in class_ids:
#                 self._build_class_memory_from_current_phase(int(cls), split=split)

#         gb = getattr(self.model, "geometry_bank", None)
#         if gb is not None and hasattr(gb, "refresh_inter_class_geometry"):
#             gb.refresh_inter_class_geometry()

#         if old_training_state:
#             self.model.train()
#         else:
#             self.model.eval()

#     def _should_refresh_for_validation(self, epoch: int) -> bool:
#         if not self.refresh_before_validation:
#             return False
#         if self.validation_refresh_every <= 0:
#             return False
#         return (int(epoch) + 1) % self.validation_refresh_every == 0

#     def _select_score(self, val_stats: dict, phase: int) -> float:
#         if phase == 0:
#             return float(val_stats.get("acc", 0.0))

#         metric = self.best_state_metric
#         if metric in {"hm", "h", "harmonic"}:
#             return float(val_stats.get("hm", 0.0))
#         if metric in {"acc", "oa"}:
#             return float(val_stats.get("acc", 0.0))
#         if metric in {"old"}:
#             return float(val_stats.get("old_acc", 0.0))
#         if metric in {"new"}:
#             return float(val_stats.get("new_acc", 0.0))

#         # Default for incremental CIL: harmonic mean.
#         return float(val_stats.get("hm", 0.0))

#     def _capture_state(self):
#         return {
#             k: v.detach().cpu().clone()
#             for k, v in self.model.state_dict().items()
#         }

#     # ============================================================
#     # Trainability
#     # ============================================================
#     def _set_incremental_trainable_params(self, old_class_count: int):
#         for p in self.model.parameters():
#             p.requires_grad = True

#         if hasattr(self.model, "freeze_backbone_only"):
#             self.model.freeze_backbone_only()

#         bb = getattr(self.model, "backbone", None)
#         if (
#             bool(getattr(self.args, "unfreeze_last_backbone_during_incremental", False))
#             and bb is not None
#             and hasattr(bb, "unfreeze_last_blocks")
#         ):
#             bb.unfreeze_last_blocks()

#         if (
#             bool(getattr(self.args, "freeze_semantic_encoder_during_incremental", True))
#             and hasattr(self.model, "freeze_semantic_encoder")
#         ):
#             self.model.freeze_semantic_encoder()
#         elif hasattr(self.model, "unfreeze_semantic_encoder"):
#             self.model.unfreeze_semantic_encoder()

#         if bool(getattr(self.args, "freeze_projection_during_incremental", False)):
#             if hasattr(self.model, "freeze_projection_head"):
#                 self.model.freeze_projection_head()
#         else:
#             if hasattr(self.model, "unfreeze_projection_head"):
#                 self.model.unfreeze_projection_head()

#         if hasattr(self.model, "freeze_old_anchor_deltas"):
#             self.model.freeze_old_anchor_deltas(old_class_count)
#         if hasattr(self.model, "unfreeze_new_anchor_deltas"):
#             self.model.unfreeze_new_anchor_deltas(old_class_count)
#         if hasattr(self.model, "freeze_old_concept_deltas"):
#             self.model.freeze_old_concept_deltas(old_class_count)

#         freeze_classifier = bool(getattr(self.args, "freeze_classifier_during_incremental", False))
#         if freeze_classifier:
#             if hasattr(self.model, "freeze_classifier_adaptation"):
#                 self.model.freeze_classifier_adaptation()
#         else:
#             if hasattr(self.model, "unfreeze_classifier_adaptation"):
#                 self.model.unfreeze_classifier_adaptation()
#             if hasattr(self.model, "freeze_old_classifier_adaptation"):
#                 self.model.freeze_old_classifier_adaptation(old_class_count)

#         # Fusion is legacy/hybrid. Keep it frozen unless explicitly enabled.
#         if bool(getattr(self.args, "use_adaptive_fusion", False)):
#             if hasattr(self.model, "unfreeze_fusion_module"):
#                 self.model.unfreeze_fusion_module()
#         else:
#             if hasattr(self.model, "freeze_fusion_module"):
#                 self.model.freeze_fusion_module()

#         # geometry calibrator remains trainable in incremental phases
#         if hasattr(self.model, "unfreeze_geometry_calibrator"):
#             self.model.unfreeze_geometry_calibrator()

#     # ============================================================
#     # Base auxiliary losses
#     # ============================================================
#     def _base_aux_losses(self, out, y):
#         features = out["features"]
#         logits = out["logits"]
#         concept_bank = out.get("concept_bank", None)

#         compact, sep, ortho, center_norm, radius = self._base_geometry_loss(features, y)

#         margin = (
#             self.logit_margin(logits, y)
#             if self.logit_margin_weight > 0.0
#             else self._zero(features)
#         )

#         concept_sep = (
#             self.concept_sep(concept_bank)
#             if self.concept_sep_weight > 0.0 and concept_bank is not None and concept_bank.numel() > 0
#             else self._zero(features)
#         )

#         feature_concept = (
#             self.feature_concept_compact(features, concept_bank, y)
#             if self.feature_concept_compact_weight > 0.0 and concept_bank is not None and concept_bank.numel() > 0
#             else self._zero(features)
#         )

#         return compact, sep, ortho, center_norm, radius, margin, concept_sep, feature_concept

#     def _classifier_adaptation_reg(self, ref: torch.Tensor) -> torch.Tensor:
#         if self.classifier_adaptation_weight <= 0.0:
#             return self._zero(ref)

#         classifier = getattr(self.model, "classifier", None)
#         if classifier is None:
#             return self._zero(ref)

#         if hasattr(classifier, "adaptation_regularization_loss"):
#             reg = classifier.adaptation_regularization_loss()
#             if isinstance(reg, dict):
#                 return reg.get("total", self._zero(ref))
#             return reg

#         return self._zero(ref)

#     # ============================================================
#     # Train one epoch
#     # ============================================================
#     def _train_epoch_base(self, loader, optimizer):
#         self.model.train()

#         total_loss = total_correct = total = 0

#         for x, y in loader:
#             x = x.to(self.device).float()
#             y = y.to(self.device).long()

#             optimizer.zero_grad(set_to_none=True)

#             out = self.model(
#                 x,
#                 semantic_mode="identity",
#                 classifier_mode=getattr(self.args, "base_classifier_mode", "geometry_only"),
#             )

#             ce = F.cross_entropy(
#                 out["logits"],
#                 y,
#                 label_smoothing=float(getattr(self.args, "label_smoothing", 0.0)),
#             )
#             compact, sep, ortho, center_norm, radius, margin, concept_sep, feature_concept = (
#                 self._base_aux_losses(out, y)
#             )

#             loss = (
#                 ce
#                 + self.base_compact * compact
#                 + self.base_sep * sep
#                 + self.base_ortho * ortho
#                 + self.base_center_norm * center_norm
#                 + self.base_radius * radius
#                 + self.logit_margin_weight * margin
#                 + self.concept_sep_weight * concept_sep
#                 + self.feature_concept_compact_weight * feature_concept
#             )

#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(
#                 [p for p in self.model.parameters() if p.requires_grad],
#                 1.0,
#             )
#             optimizer.step()

#             total_loss += float(loss.item())
#             total_correct += int((out["logits"].argmax(1) == y).sum().item())
#             total += int(y.size(0))

#         return total_loss / max(len(loader), 1), 100.0 * total_correct / max(total, 1)

#     def _train_epoch_incremental(
#         self,
#         loader,
#         optimizer,
#         old_class_count,
#         new_class_ids,
#         old_bank_snapshot,
#         old_token_snapshot=None,
#         epoch_idx: int = 0,
#     ):
#         self.model.train()
#         self._set_model_phase_and_old_count(self.model.current_phase, old_class_count)

#         total_loss = 0.0
#         total_correct = 0
#         total = 0

#         classifier_mode = self._inc_classifier_mode()

#         token_loss_weight = float(getattr(self.args, "token_loss_weight", 0.0))
#         spectral_guidance_weight = float(getattr(self.args, "spectral_guidance_weight", 0.0))
#         band_guidance_weight = float(getattr(self.args, "band_guidance_weight", 0.0))
#         replay_geometry_weight = float(getattr(self.args, "replay_geometry_weight", 0.0))

#         use_token_relations = token_loss_weight > 0.0

#         if self.incremental_warmup_epochs > 0:
#             post_warm_epochs = max(
#                 int(getattr(self.args, "epochs_inc", self.incremental_warmup_epochs)) - self.incremental_warmup_epochs,
#                 1,
#             )
#             if epoch_idx < self.incremental_warmup_epochs:
#                 structure_ramp = 0.0
#             else:
#                 structure_ramp = min(
#                     1.0,
#                     float(epoch_idx - self.incremental_warmup_epochs + 1) / float(post_warm_epochs),
#                 )
#         else:
#             structure_ramp = 1.0

#         for x, y in loader:
#             x = x.to(self.device).float()
#             y = y.to(self.device).long()

#             optimizer.zero_grad(set_to_none=True)

#             out = self.model(
#                 x,
#                 semantic_mode="identity",
#                 classifier_mode=classifier_mode,
#                 return_token_relations=use_token_relations,
#             )
#             logits = out["logits"]
#             features = out["features"]
#             concept_bank = out.get("concept_bank", None)

#             # 1) New-class supervision
#             ce_new = self._masked_weighted_ce_new(logits, y, new_class_ids)

#             # 2) Old-class replay
#             replay_x, replay_y = self._sample_replay_from_snapshot(
#                 old_bank_snapshot, old_class_count
#             )

#             if replay_x is not None:
#                 replay_logits = self.model.compute_logits_from_features(
#                     replay_x,
#                     classifier_mode=classifier_mode,
#                 )
#                 ce_replay = F.cross_entropy(replay_logits, replay_y)

#                 replay_geom_loss = self._replay_geometry_loss(
#                     replay_x,
#                     replay_y,
#                     old_bank_snapshot=old_bank_snapshot,
#                 )
#             else:
#                 ce_replay = self._zero(logits)
#                 replay_geom_loss = self._zero(logits)

#             # 3) Weak bank-vs-snapshot consistency
#             a_mean, a_basis, a_var, _ = self._geometry_alignment_losses(
#                 old_bank_snapshot,
#                 old_class_count,
#             )

#             # 4) Geometry-native separation and compactness
#             if hasattr(self, "_symmetric_geometry_separation_loss"):
#                 sep_geo_loss = self._symmetric_geometry_separation_loss(
#                     new_features=features,
#                     new_labels=y,
#                     replay_features=replay_x,
#                     replay_labels=replay_y,
#                     old_class_count=old_class_count,
#                     new_class_ids=new_class_ids,
#                     old_bank_snapshot=old_bank_snapshot,
#                 )
#             else:
#                 sep_geo_loss = self._geometry_separation_loss(
#                     features, y, old_class_count, new_class_ids
#                 )
#             vol_loss = self._new_class_volume_loss(features, y, new_class_ids)

#             # 5) Structural preservation
#             token_loss, token_parts = self._token_manifold_loss(
#                 out, y, old_class_count, old_token_snapshot
#             )
#             del token_parts

#             spec_loss, band_loss = self._spectral_guidance_losses(
#                 out, y, old_class_count
#             )

#             # 6) Calibration/adaptation regularization
#             calibration_reg = out.get("calibration_reg", None)
#             if calibration_reg is None:
#                 calibration_total = self._zero(logits)
#             else:
#                 calibration_total = calibration_reg.get("total", self._zero(logits))

#             cls_adapt_reg = self._classifier_adaptation_reg(logits)

#             # 7) New-class auxiliaries
#             margin_new = self._zero(logits)
#             feature_concept_new = self._zero(logits)

#             class_ids = torch.tensor(new_class_ids, device=logits.device, dtype=torch.long)

#             if self._labels_are_local(y, new_class_ids):
#                 y_local = y
#             else:
#                 y_local = torch.full_like(y, fill_value=-1)
#                 for local_idx, global_cls in enumerate(class_ids):
#                     y_local[y == global_cls] = local_idx

#             valid = y_local >= 0

#             if self.inc_logit_margin_weight > 0.0 and valid.any():
#                 logits_new_only = logits.index_select(1, class_ids)
#                 margin_new = self.logit_margin(logits_new_only[valid], y_local[valid])

#             if (
#                 self.inc_feature_concept_compact_weight > 0.0
#                 and concept_bank is not None
#                 and concept_bank.numel() > 0
#                 and valid.any()
#             ):
#                 new_concept_bank = concept_bank.index_select(0, class_ids)
#                 feature_concept_new = self.feature_concept_compact(
#                     features[valid],
#                     new_concept_bank,
#                     y_local[valid],
#                 )

#             # 8) Loss assembly
#             if epoch_idx < self.incremental_warmup_epochs:
#                 loss = (
#                     ce_new
#                     + self.replay_weight * ce_replay
#                     + self.inc_logit_margin_weight * margin_new
#                     + self.geometry_calibration_weight * calibration_total
#                     + self.classifier_adaptation_weight * cls_adapt_reg
#                 )
#             else:
#                 loss = (
#                     ce_new
#                     + self.replay_weight * ce_replay
#                     + replay_geometry_weight * structure_ramp * replay_geom_loss
#                     + self.align_mean_weight * structure_ramp * a_mean
#                     + self.align_basis_weight * structure_ramp * a_basis
#                     + self.align_var_weight * structure_ramp * a_var
#                     + self.insert_weight * structure_ramp * sep_geo_loss
#                     + self.new_volume_weight * structure_ramp * vol_loss
#                     + self.geometry_calibration_weight * calibration_total
#                     + self.classifier_adaptation_weight * cls_adapt_reg
#                     + token_loss_weight * structure_ramp * token_loss
#                     + spectral_guidance_weight * structure_ramp * spec_loss
#                     + band_guidance_weight * structure_ramp * band_loss
#                     + self.inc_logit_margin_weight * margin_new
#                     + self.inc_feature_concept_compact_weight * feature_concept_new
#                 )

#             if not torch.isfinite(loss):
#                 if self.debug:
#                     print(
#                         "[WARN] Non-finite incremental loss encountered. "
#                         f"ce_new={float(ce_new.detach().item()):.6f}, "
#                         f"ce_replay={float(ce_replay.detach().item()):.6f}, "
#                         f"replay_geom={float(replay_geom_loss.detach().item()):.6f}, "
#                         f"a_mean={float(a_mean.detach().item()):.6f}, "
#                         f"a_basis={float(a_basis.detach().item()):.6f}, "
#                         f"a_var={float(a_var.detach().item()):.6f}, "
#                         f"sep_geo={float(sep_geo_loss.detach().item()):.6f}, "
#                         f"vol={float(vol_loss.detach().item()):.6f}, "
#                         f"calib={float(calibration_total.detach().item()):.6f}, "
#                         f"cls_adapt={float(cls_adapt_reg.detach().item()):.6f}, "
#                         f"token={float(token_loss.detach().item()):.6f}, "
#                         f"spec={float(spec_loss.detach().item()):.6f}, "
#                         f"band={float(band_loss.detach().item()):.6f}, "
#                         f"margin={float(margin_new.detach().item()):.6f}, "
#                         f"feat_concept={float(feature_concept_new.detach().item()):.6f}"
#                     )
#                 continue

#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(
#                 [p for p in self.model.parameters() if p.requires_grad], 0.5
#             )
#             optimizer.step()

#             total_loss += float(loss.item())

#             correct_new, valid_new = self._incremental_accuracy_with_count(
#                 logits, y, new_class_ids
#             )
#             total_correct += correct_new
#             total += valid_new

#             if self.debug:
#                 print(
#                     f"[IncDebug] loss={float(loss.item()):.4f} | "
#                     f"ce_new={float(ce_new.detach().item()):.4f} | "
#                     f"ce_replay={float(ce_replay.detach().item()):.4f} | "
#                     f"replay_geom={float(replay_geom_loss.detach().item()):.4f} | "
#                     f"a_mean={float(a_mean.detach().item()):.4f} | "
#                     f"a_basis={float(a_basis.detach().item()):.4f} | "
#                     f"a_var={float(a_var.detach().item()):.4f} | "
#                     f"sep_geo={float(sep_geo_loss.detach().item()):.4f} | "
#                     f"vol={float(vol_loss.detach().item()):.4f} | "
#                     f"calib={float(calibration_total.detach().item()):.4f} | "
#                     f"cls_adapt={float(cls_adapt_reg.detach().item()):.4f} | "
#                     f"token={float(token_loss.detach().item()):.4f} | "
#                     f"spec={float(spec_loss.detach().item()):.4f} | "
#                     f"band={float(band_loss.detach().item()):.4f} | "
#                     f"margin={float(margin_new.detach().item()):.4f} | "
#                     f"feat_concept={float(feature_concept_new.detach().item()):.4f} | "
#                     f"ramp={structure_ramp:.3f}"
#                 )

#         train_acc = 100.0 * total_correct / max(total, 1)
#         return total_loss / max(len(loader), 1), train_acc

#     # ============================================================
#     # Post-phase calibration
#     # ============================================================
#     def _set_calibration_trainable_params(self):
#         for p in self.model.parameters():
#             p.requires_grad = False

#         if hasattr(self.model, "unfreeze_classifier_adaptation"):
#             self.model.unfreeze_classifier_adaptation()

#         # Freeze old classifier adaptation again if available; calibration should
#         # tune global/allowed offsets conservatively, not rewrite old class params.
#         old_class_count = int(getattr(self.model, "old_class_count", 0))
#         if hasattr(self.model, "freeze_old_classifier_adaptation") and old_class_count > 0:
#             self.model.freeze_old_classifier_adaptation(old_class_count)

#         if hasattr(self.model, "unfreeze_geometry_calibrator"):
#             self.model.unfreeze_geometry_calibrator()

#     def _post_phase_calibration(
#         self,
#         phase: int,
#         old_class_count: int,
#         new_class_ids,
#         old_bank_snapshot,
#         batch_size: int,
#     ):
#         if self.calibration_epochs <= 0:
#             return

#         self._set_calibration_trainable_params()

#         calib_params = [p for p in self.model.parameters() if p.requires_grad]
#         if len(calib_params) == 0:
#             return

#         optimizer = optim.Adam(calib_params, lr=self.calibration_lr, weight_decay=0.0)

#         train_loader = self.dataset.get_phase_dataloader(
#             phase,
#             split="train",
#             batch_size=batch_size,
#             shuffle=True,
#         )

#         classifier_mode = self._eval_classifier_mode()
#         replay_geometry_weight = float(getattr(self.args, "replay_geometry_weight", 0.0))

#         self.model.train()
#         self._set_model_phase_and_old_count(phase, old_class_count)

#         best_state = None
#         best_loss = float("inf")

#         for _ in range(self.calibration_epochs):
#             epoch_loss = 0.0
#             epoch_steps = 0

#             for x, y in train_loader:
#                 x = x.to(self.device).float()
#                 y = y.to(self.device).long()

#                 optimizer.zero_grad(set_to_none=True)

#                 out = self.model(
#                     x,
#                     semantic_mode="identity",
#                     classifier_mode=classifier_mode,
#                 )
#                 ce_new = self._masked_weighted_ce_new(out["logits"], y, new_class_ids)

#                 replay_x, replay_y = self._sample_replay_from_snapshot(
#                     old_bank_snapshot,
#                     old_class_count,
#                 )

#                 if replay_x is not None:
#                     replay_logits = self.model.compute_logits_from_features(
#                         replay_x,
#                         classifier_mode=classifier_mode,
#                     )
#                     ce_old = F.cross_entropy(replay_logits, replay_y)

#                     replay_geom_loss = self._replay_geometry_loss(
#                         replay_x,
#                         replay_y,
#                         old_bank_snapshot=old_bank_snapshot,
#                     )
#                 else:
#                     ce_old = self._zero(out["logits"])
#                     replay_geom_loss = self._zero(out["logits"])

#                 calibration_reg = out.get("calibration_reg", None)
#                 if calibration_reg is None:
#                     calibration_total = self._zero(out["logits"])
#                 else:
#                     calibration_total = calibration_reg.get("total", self._zero(out["logits"]))

#                 cls_adapt_reg = self._classifier_adaptation_reg(out["logits"])

#                 loss = (
#                     ce_new
#                     + self.calibration_replay_weight * ce_old
#                     + self.geometry_calibration_weight * calibration_total
#                     + self.classifier_adaptation_weight * cls_adapt_reg
#                     + 0.5 * replay_geometry_weight * replay_geom_loss
#                 )

#                 if not torch.isfinite(loss):
#                     if self.debug:
#                         print(
#                             "[WARN] Non-finite calibration loss encountered. "
#                             f"ce_new={float(ce_new.detach().item()):.6f}, "
#                             f"ce_old={float(ce_old.detach().item()):.6f}, "
#                             f"calib={float(calibration_total.detach().item()):.6f}, "
#                             f"cls_adapt={float(cls_adapt_reg.detach().item()):.6f}, "
#                             f"replay_geom={float(replay_geom_loss.detach().item()):.6f}"
#                         )
#                     continue

#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(calib_params, 0.5)
#                 optimizer.step()

#                 epoch_loss += float(loss.item())
#                 epoch_steps += 1

#             avg_epoch_loss = epoch_loss / max(epoch_steps, 1)

#             if avg_epoch_loss < best_loss:
#                 best_loss = avg_epoch_loss
#                 best_state = self._capture_state()

#         if best_state is not None:
#             self.model.load_state_dict(best_state)
#             self._set_model_phase_and_old_count(phase, old_class_count)

#     # ============================================================
#     # Validation
#     # ============================================================
#     @torch.no_grad()
#     def _validate_split_metrics(self, loader, old_class_count: int):
#         self.model.eval()
#         self._set_model_phase_and_old_count(self.model.current_phase, old_class_count)

#         total_loss = total_correct = total = 0
#         old_correct = old_total = 0
#         new_correct = new_total = 0

#         if int(old_class_count) == 0:
#             val_classifier_mode = getattr(self.args, "base_classifier_mode", "geometry_only")
#             val_semantic_mode = "identity"
#         else:
#             val_classifier_mode = self._eval_classifier_mode()
#             val_semantic_mode = getattr(self.args, "eval_semantic_mode", "identity")

#         for x, y in loader:
#             x = x.to(self.device).float()
#             y = y.to(self.device).long()

#             out = self.model(
#                 x,
#                 semantic_mode=val_semantic_mode,
#                 classifier_mode=val_classifier_mode,
#             )

#             logits = out["logits"]
#             loss = F.cross_entropy(logits, y)
#             preds = logits.argmax(dim=1)

#             total_loss += float(loss.item())
#             total_correct += int((preds == y).sum().item())
#             total += int(y.size(0))

#             old_mask = y < old_class_count
#             new_mask = y >= old_class_count

#             if old_mask.any():
#                 old_correct += int((preds[old_mask] == y[old_mask]).sum().item())
#                 old_total += int(old_mask.sum().item())

#             if new_mask.any():
#                 new_correct += int((preds[new_mask] == y[new_mask]).sum().item())
#                 new_total += int(new_mask.sum().item())

#         old_acc = 100.0 * old_correct / max(old_total, 1)
#         new_acc = 100.0 * new_correct / max(new_total, 1)
#         total_acc = 100.0 * total_correct / max(total, 1)
#         hm = 0.0 if (old_acc + new_acc) == 0 else 2.0 * old_acc * new_acc / (old_acc + new_acc)

#         return {
#             "loss": total_loss / max(len(loader), 1),
#             "acc": total_acc,
#             "old_acc": old_acc,
#             "new_acc": new_acc,
#             "hm": hm,
#         }

#     # ============================================================
#     # Phase training
#     # ============================================================
#     def train_phase(self, phase, epochs, batch_size=64, lr=1e-4):
#         print(f"==== Training Phase {phase} ====")

#         phase = int(phase)
#         self.dataset.start_phase(phase)

#         needed_classes = max(self.dataset.phase_to_classes[phase]) + 1
#         if int(self.model.current_num_classes) < needed_classes:
#             self._bootstrap_phase_classes(phase, split="train")

#         old_class_count = 0 if phase == 0 else len(self.dataset.get_classes_up_to_phase(phase - 1))
#         self._set_model_phase_and_old_count(phase, old_class_count)

#         history = {
#             "train_loss": [],
#             # Train accuracy is the refreshed train-set diagnostic, not online/stale-bank accuracy.
#             "train_acc": [],
#             "val_loss": [],
#             "val_acc": [],
#             "val_old_acc": [],
#             "val_new_acc": [],
#             "val_hm": [],
#         }

#         val_loader = self.dataset.get_cumulative_dataloader(
#             phase,
#             split="val",
#             batch_size=batch_size,
#             shuffle=False,
#         )

#         # ---------------- Base phase ----------------
#         if phase == 0:
#             train_loader = self.dataset.get_phase_dataloader(
#                 phase,
#                 split="train",
#                 batch_size=batch_size,
#                 shuffle=True,
#             )

#             optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
#             scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

#             best_state, best_score = None, -1.0
#             no_improve = 0

#             base_class_ids = [int(c) for c in self.dataset.phase_to_classes[phase]]

#             for epoch in range(epochs):
#                 self._set_model_phase_and_old_count(phase, 0)

#                 tr_loss, tr_acc = self._train_epoch_base(train_loader, optimizer)

#                 # Refresh base geometry before validation; otherwise phase-0
#                 # validation is evaluated against stale placeholder geometry.
#                 if self._should_refresh_for_validation(epoch):
#                     self._refresh_classes_for_validation(phase, base_class_ids, split="train")

#                 # Evaluate train split after geometry refresh.
#                 # This refreshed train accuracy is the reported Train Acc.
#                 train_eval_stats = self._validate_split_metrics(train_loader, 0)
#                 val_stats = self._validate_split_metrics(val_loader, 0)
#                 scheduler.step()

#                 history["train_loss"].append(tr_loss)
#                 history["train_acc"].append(train_eval_stats["acc"])
#                 history["val_loss"].append(val_stats["loss"])
#                 history["val_acc"].append(val_stats["acc"])
#                 history["val_old_acc"].append(val_stats["old_acc"])
#                 history["val_new_acc"].append(val_stats["new_acc"])
#                 history["val_hm"].append(val_stats["hm"])

#                 print(
#                     f"Ep {epoch+1:03d}/{epochs} | "
#                     f"Train Loss: {tr_loss:.4f} | "
#                     f"Train Acc: {train_eval_stats['acc']:.2f}% | "
#                     f"Val Acc: {val_stats['acc']:.2f}% | "
#                     f"Val Loss: {val_stats['loss']:.4f} | "
#                     f"Old: {val_stats['old_acc']:.2f}% | "
#                     f"New: {val_stats['new_acc']:.2f}% | "
#                     f"H: {val_stats['hm']:.2f}%"
#                 )

#                 score = self._select_score(val_stats, phase=0)
#                 if score > best_score:
#                     best_score = score
#                     best_state = self._capture_state()
#                     no_improve = 0
#                 else:
#                     no_improve += 1

#                 if self.early_stop_patience > 0 and no_improve >= self.early_stop_patience:
#                     print(f"[EarlyStop] Phase {phase}: no improvement for {no_improve} epochs.")
#                     break

#             if best_state is not None:
#                 self.model.load_state_dict(best_state)
#                 self._set_model_phase_and_old_count(phase, 0)

#             self._finalize_phase_memory(phase, split="train")
#             gb = getattr(self.model, "geometry_bank", None)
#             if gb is not None and hasattr(gb, "refresh_inter_class_geometry"):
#                 gb.refresh_inter_class_geometry()

#             self.model.old_class_count = int(self.model.current_num_classes)
#             # self.save_checkpoint(phase, history)
#             return history

#         # ---------------- Incremental phase ----------------
#         old_bank_snapshot = self._snapshot_old_bank(old_class_count)
#         old_token_snapshot = self._snapshot_old_token_memory(old_class_count)
#         new_class_ids = [int(c) for c in self.dataset.phase_to_classes[phase]]

#         self._set_incremental_trainable_params(old_class_count)

#         train_loader = self.dataset.get_phase_dataloader(
#             phase,
#             split="train",
#             batch_size=batch_size,
#             shuffle=True,
#         )

#         trainable_params = [p for p in self.model.parameters() if p.requires_grad]
#         if len(trainable_params) == 0:
#             raise RuntimeError("No trainable parameters found for incremental phase.")

#         optimizer = optim.Adam(
#             trainable_params,
#             lr=lr,
#             weight_decay=1e-5,
#         )
#         scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

#         best_state, best_score = None, -1.0
#         no_improve = 0

#         for epoch in range(epochs):
#             self._set_model_phase_and_old_count(phase, old_class_count)

#             tr_loss, tr_acc = self._train_epoch_incremental(
#                 train_loader,
#                 optimizer,
#                 old_class_count,
#                 new_class_ids,
#                 old_bank_snapshot,
#                 old_token_snapshot,
#                 epoch_idx=epoch,
#             )

#             # --------------------------------------------------------
#             # Correct order:
#             #   1. train epoch
#             #   2. refresh current-phase geometry
#             #   3. validate
#             #   4. save best state
#             #
#             # The old trainer validated before refresh, which made validation
#             # look collapsed even when post-finalization evaluation recovered.
#             # --------------------------------------------------------
#             refresh_every = int(getattr(self.args, "bank_refresh_every", 0))
#             do_periodic_refresh = refresh_every > 0 and (epoch + 1) % refresh_every == 0
#             do_validation_refresh = self._should_refresh_for_validation(epoch)

#             if do_periodic_refresh or do_validation_refresh:
#                 self._refresh_classes_for_validation(phase, new_class_ids, split="train")

#             # Evaluate current-phase train split after geometry refresh.
#             # This refreshed accuracy is the reported Train Acc.
#             train_eval_stats = self._validate_split_metrics(train_loader, old_class_count)
#             val_stats = self._validate_split_metrics(val_loader, old_class_count)
#             scheduler.step()

#             history["train_loss"].append(tr_loss)
#             history["train_acc"].append(train_eval_stats["acc"])
#             history["val_loss"].append(val_stats["loss"])
#             history["val_acc"].append(val_stats["acc"])
#             history["val_old_acc"].append(val_stats["old_acc"])
#             history["val_new_acc"].append(val_stats["new_acc"])
#             history["val_hm"].append(val_stats["hm"])

#             print(
#                 f"Ep {epoch+1:03d}/{epochs} | "
#                 f"Train Loss: {tr_loss:.4f} | "
#                 f"Train Acc: {train_eval_stats['acc']:.2f}% | "
#                 f"Val Acc: {val_stats['acc']:.2f}% | "
#                 f"Val Loss: {val_stats['loss']:.4f}"
#             )

#             score = self._select_score(val_stats, phase=phase)
#             if score > best_score:
#                 best_score = score
#                 best_state = self._capture_state()
#                 no_improve = 0
#             else:
#                 no_improve += 1

#             if self.early_stop_patience > 0 and no_improve >= self.early_stop_patience:
#                 print(f"[EarlyStop] Phase {phase}: no improvement for {no_improve} epochs.")
#                 break

#         if best_state is not None:
#             self.model.load_state_dict(best_state)
#             self._set_model_phase_and_old_count(phase, old_class_count)

#         self._post_phase_calibration(
#             phase=phase,
#             old_class_count=old_class_count,
#             new_class_ids=new_class_ids,
#             old_bank_snapshot=old_bank_snapshot,
#             batch_size=batch_size,
#         )

#         self._finalize_phase_memory(phase, split="train")
#         gb = getattr(self.model, "geometry_bank", None)
#         if gb is not None and hasattr(gb, "refresh_inter_class_geometry"):
#             gb.refresh_inter_class_geometry()

#         self.model.old_class_count = int(self.model.current_num_classes)
#         return history

#     # ============================================================
#     # Save
#     # ============================================================
#     def save_checkpoint(self, phase, history, evaluator_metrics=None):
#         run_dir = getattr(self.args, "run_dir", None)
#         if run_dir is not None and str(run_dir).strip():
#             phase_dir = os.path.join(str(run_dir), f"phase_{phase}")
#         else:
#             phase_dir = os.path.join(self.save_dir, self.args.dataset, f"phase_{phase}")
#         os.makedirs(phase_dir, exist_ok=True)

#         ckpt = {
#             "phase": int(phase),
#             "model_state_dict": self.model.state_dict(),
#             "memory_snapshot": self.model.export_memory_snapshot() if hasattr(self.model, "export_memory_snapshot") else None,
#             "current_num_classes": int(self.model.current_num_classes),
#             "old_class_count": int(getattr(self.model, "old_class_count", 0)),
#             "history": history,
#             "args": vars(self.args),
#             "token_memory": {
#                 int(k): {kk: vv.detach().cpu() for kk, vv in v.items()}
#                 for k, v in getattr(self, "token_memory", {}).items()
#             },
#         }
#         if evaluator_metrics is not None:
#             ckpt["evaluator_metrics"] = evaluator_metrics

#         path = os.path.join(phase_dir, "checkpoint.pth")
#         torch.save(ckpt, path)
#         print(f"[Saved] {path}")















# import os
# from contextlib import nullcontext

# import torch
# import torch.nn.functional as F
# import numpy as np
# import torch.optim as optim

# from losses.necil_losses import GlobalLogitMargin, ConceptSeparation, FeatureConceptCompactness
# from trainers.trainer_helpers import TrainerHelper


# class Trainer(TrainerHelper):
#     """
#     Geometry-centric trainer for strict non-exemplar HSI CIL.

#     Corrected stability policy
#     --------------------------
#     1. Validate only after current-phase geometry is refreshed.
#     2. Select best incremental checkpoint by refreshed validation harmonic mean.
#     3. Do not accidentally unfreeze old classifier adaptation.
#     4. Keep old raw samples inaccessible under strict non-exemplar protocol.
#     5. Refresh only current-phase geometry during incremental training.
#     6. Use identity semantic path for all geometry-native training/eval.
#     """

#     def __init__(self, model, dataset, args):
#         self.args = args
#         self.device = torch.device(args.device)
#         self.model = model.to(self.device)
#         self.dataset = dataset
#         self.save_dir = args.save_dir
#         self.debug = bool(getattr(args, "debug_verbose", False)) or os.environ.get("NECIL_DEBUG", "0") == "1"

#         self.subspace_rank = int(getattr(args, "subspace_rank", 5))
#         self.geom_var_floor = float(getattr(args, "geom_var_floor", 1e-4))
#         self.num_concepts_per_class = int(getattr(args, "num_concepts_per_class", 4))
#         self.alignment_samples_per_class = int(getattr(args, "alignment_samples_per_class", 16))

#         # ---------------- Base phase ----------------
#         self.base_compact = float(getattr(args, "base_compact", 0.05))
#         self.base_sep = float(getattr(args, "base_sep", 0.05))
#         self.base_ortho = float(getattr(args, "base_ortho", 0.03))
#         self.base_margin = float(getattr(args, "base_margin", 1.0))
#         self.base_center_norm = float(getattr(args, "base_center_norm", 0.01))
#         self.base_radius = float(getattr(args, "base_radius", 0.01))

#         # ---------------- Incremental phase ----------------
#         self.replay_weight = float(getattr(args, "synthetic_replay_weight", 1.0))
#         self.replay_per_class = int(getattr(args, "synthetic_replay_per_class", 32))
#         self.align_mean_weight = float(getattr(args, "align_mean_weight", 0.10))
#         self.align_basis_weight = float(getattr(args, "align_basis_weight", 0.05))
#         self.align_var_weight = float(getattr(args, "align_var_weight", 0.02))
#         self.insert_weight = float(getattr(args, "insert_weight", 0.01))
#         self.insert_margin = float(getattr(args, "insert_margin", 5.0))
#         self.new_volume_weight = float(getattr(args, "new_volume_weight", 0.005))
#         self.new_volume_target = float(getattr(args, "new_volume_target", 1.5))
#         self.incremental_warmup_epochs = int(getattr(args, "incremental_warmup_epochs", 5))

#         # geometry-calibration regularization
#         self.geometry_calibration_weight = float(
#             getattr(args, "geometry_calibration_weight", 0.05)
#         )

#         # ---------------- Calibration ----------------
#         self.calibration_epochs = int(getattr(args, "calibration_epochs", 5))
#         self.calibration_lr = float(getattr(args, "calibration_lr", 5e-4))
#         self.calibration_replay_weight = float(getattr(args, "calibration_replay_weight", 1.0))

#         # ---------------- Validation / checkpoint policy ----------------
#         self.refresh_before_validation = bool(getattr(args, "refresh_before_validation", True))
#         self.validation_refresh_every = int(getattr(args, "validation_refresh_every", 1))
#         self.best_state_metric = str(getattr(args, "best_state_metric", "hm")).lower()
#         self.early_stop_patience = int(getattr(args, "early_stop_patience", 0))

#         # ---------------- Auxiliary losses ----------------
#         self.logit_margin_value = float(getattr(args, "logit_margin_value", 0.20))
#         self.logit_margin_weight = float(getattr(args, "logit_margin_weight", 0.02))
#         self.concept_sep_weight = float(getattr(args, "concept_sep_weight", 0.01))
#         self.feature_concept_compact_weight = float(
#             getattr(args, "feature_concept_compact_weight", 0.03)
#         )
#         self.inc_logit_margin_weight = float(getattr(args, "inc_logit_margin_weight", 0.005))
#         self.inc_feature_concept_compact_weight = float(
#             getattr(args, "inc_feature_concept_compact_weight", 0.005)
#         )
#         self.classifier_adaptation_weight = float(getattr(args, "classifier_adaptation_weight", 0.0))

#         self.logit_margin = GlobalLogitMargin(margin=self.logit_margin_value)
#         self.concept_sep = ConceptSeparation(
#             max_cosine=float(getattr(args, "concept_sep_max_cosine", 0.25))
#         )
#         self.feature_concept_compact = FeatureConceptCompactness(
#             temperature=float(getattr(args, "cls_temperature", 0.07))
#         )

#         self.token_memory = {}

#     # ============================================================
#     # Mode helpers
#     # ============================================================
#     def _inc_classifier_mode(self) -> str:
#         return str(
#             getattr(self.args, "incremental_classifier_mode", "calibrated_geometry")
#         ).lower()

#     def _eval_classifier_mode(self) -> str:
#         return str(
#             getattr(self.args, "eval_classifier_mode", "calibrated_geometry")
#         ).lower()

#     def _set_model_phase_and_old_count(self, phase: int, old_class_count: int):
#         if hasattr(self.model, "set_phase"):
#             self.model.set_phase(int(phase))
#         else:
#             self.model.current_phase = int(phase)

#         if hasattr(self.model, "set_old_class_count"):
#             self.model.set_old_class_count(int(old_class_count))
#         else:
#             self.model.old_class_count = int(old_class_count)

#     # ============================================================
#     # Geometry refresh helpers
#     # ============================================================
#     @torch.no_grad()
#     def _refresh_classes_for_validation(self, phase: int, class_ids, split: str = "train"):
#         """
#         Refresh only currently accessible classes before validation.

#         This fixes the stale-geometry validation bug:
#         validation must evaluate the model with the same current-phase geometry
#         that will be used after phase finalization.
#         """
#         if not self.refresh_before_validation:
#             return

#         phase = int(phase)
#         old_training_state = self.model.training

#         # current phase train classes are allowed in strict non-exemplar mode.
#         ctx = (
#             self.dataset.memory_build_context(phase)
#             if hasattr(self.dataset, "memory_build_context")
#             else nullcontext()
#         )

#         with ctx:
#             for cls in class_ids:
#                 self._build_class_memory_from_current_phase(int(cls), split=split)

#         gb = getattr(self.model, "geometry_bank", None)
#         if gb is not None and hasattr(gb, "refresh_inter_class_geometry"):
#             gb.refresh_inter_class_geometry()

#         if old_training_state:
#             self.model.train()
#         else:
#             self.model.eval()

#     def _should_refresh_for_validation(self, epoch: int) -> bool:
#         if not self.refresh_before_validation:
#             return False
#         if self.validation_refresh_every <= 0:
#             return False
#         return (int(epoch) + 1) % self.validation_refresh_every == 0

#     def _select_score(self, val_stats: dict, phase: int) -> float:
#         if phase == 0:
#             return float(val_stats.get("acc", 0.0))

#         metric = self.best_state_metric
#         if metric in {"hm", "h", "harmonic"}:
#             return float(val_stats.get("hm", 0.0))
#         if metric in {"acc", "oa"}:
#             return float(val_stats.get("acc", 0.0))
#         if metric in {"old"}:
#             return float(val_stats.get("old_acc", 0.0))
#         if metric in {"new"}:
#             return float(val_stats.get("new_acc", 0.0))

#         # Default for incremental CIL: harmonic mean.
#         return float(val_stats.get("hm", 0.0))

#     def _capture_state(self):
#         return {
#             k: v.detach().cpu().clone()
#             for k, v in self.model.state_dict().items()
#         }

#     # ============================================================
#     # Phase-risk / optimizer helpers
#     # ============================================================
#     def _print_trainable_summary(self, phase: int):
#         trainable = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
#         total = sum(p.numel() for _, p in trainable)
#         print(f"[Trainable] Phase {phase}: {len(trainable)} tensors | {total:,} params")
#         for n, p in trainable[:50]:
#             print(f"  - {n}: {tuple(p.shape)}")
#         if len(trainable) > 50:
#             print(f"  ... {len(trainable) - 50} more tensors")

#     def _build_incremental_optimizer(self, lr: float):
#         groups = {"projection": [], "calibrator": [], "classifier": [], "deltas": [], "other": []}
#         for name, p in self.model.named_parameters():
#             if not p.requires_grad:
#                 continue
#             lname = name.lower()
#             if "projection" in lname or "projector" in lname or ".proj" in lname or "concept_projector" in lname:
#                 groups["projection"].append(p)
#             elif "calibrator" in lname or "geometry_calibrator" in lname:
#                 groups["calibrator"].append(p)
#             elif "classifier" in lname:
#                 groups["classifier"].append(p)
#             elif "delta" in lname:
#                 groups["deltas"].append(p)
#             else:
#                 groups["other"].append(p)

#         param_groups = []
#         if groups["projection"]:
#             param_groups.append({"params": groups["projection"], "lr": lr * float(getattr(self.args, "projection_incremental_lr_scale", 0.05)), "weight_decay": 1e-5, "name": "projection"})
#         if groups["calibrator"]:
#             param_groups.append({"params": groups["calibrator"], "lr": lr, "weight_decay": 0.0, "name": "calibrator"})
#         if groups["classifier"]:
#             param_groups.append({"params": groups["classifier"], "lr": lr, "weight_decay": 0.0, "name": "classifier"})
#         if groups["deltas"]:
#             param_groups.append({"params": groups["deltas"], "lr": lr, "weight_decay": 1e-5, "name": "deltas"})
#         if groups["other"]:
#             param_groups.append({"params": groups["other"], "lr": lr * 0.2, "weight_decay": 1e-5, "name": "other"})
#         if len(param_groups) == 0:
#             raise RuntimeError("No trainable parameter groups for incremental optimizer.")
#         print("[Optimizer] Incremental parameter groups:")
#         for g in param_groups:
#             n_params = sum(p.numel() for p in g["params"])
#             print(f"  - {g['name']}: {n_params:,} params | lr={g['lr']:.3e} | wd={g['weight_decay']}")
#         return optim.Adam(param_groups)

#     def _load_captured_state(self, state):
#         self.model.load_state_dict(state)

#     def _build_phase_balance_policy(self, phase: int, old_class_count: int, new_class_ids):
#         new_class_ids = [int(c) for c in new_class_ids]
#         policy = {
#             "new_ce_weight": 1.0,
#             "replay_weight_mult": 1.0,
#             "replay_geometry_mult": 1.0,
#             "alignment_mult": 1.0,
#             "skip_post_calibration": False,
#             "enable_epoch_revert": True,
#             "reason": "normal_phase",
#         }
#         if int(phase) == 0 or int(old_class_count) <= 0 or len(new_class_ids) == 0:
#             return policy
#         labels = getattr(self.dataset, "remapped_labels", None)
#         if labels is None:
#             labels = getattr(self.dataset, "labels", None)
#         if labels is None:
#             return policy
#         labels_np = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
#         labels_np = labels_np.reshape(-1)
#         old_ids = list(range(int(old_class_count)))
#         old_counts = np.asarray([(labels_np == c).sum() for c in old_ids], dtype=np.float64)
#         new_counts = np.asarray([(labels_np == c).sum() for c in new_class_ids], dtype=np.float64)
#         old_counts = old_counts[old_counts > 0]
#         new_counts = new_counts[new_counts > 0]
#         if old_counts.size == 0 or new_counts.size == 0:
#             return policy
#         old_ref = float(np.median(old_counts))
#         new_ref = float(np.median(new_counts))
#         dominance = new_ref / max(old_ref, 1.0)
#         single_new_phase = len(new_class_ids) == 1
#         if dominance >= 2.0 or single_new_phase:
#             replay_mult = float(np.clip(np.sqrt(max(dominance, 1.0)), 1.5, 2.5))
#             new_ce_weight = float(np.clip(1.0 / np.sqrt(max(dominance, 1.0)), 0.35, 0.75))
#             policy.update({
#                 "new_ce_weight": new_ce_weight,
#                 "replay_weight_mult": replay_mult,
#                 "replay_geometry_mult": min(2.0, replay_mult),
#                 "alignment_mult": min(1.8, 1.0 + 0.25 * replay_mult),
#                 "skip_post_calibration": True,
#                 "reason": f"dominant_new_phase(dominance={dominance:.2f}, single={single_new_phase})",
#             })
#         return policy

#     def _is_harmful_incremental_update(self, prev_val, curr_val) -> bool:
#         if prev_val is None or curr_val is None:
#             return False
#         prev_old = float(prev_val.get("old_acc", prev_val.get("old_accuracy", 0.0)))
#         prev_new = float(prev_val.get("new_acc", prev_val.get("new_accuracy", 0.0)))
#         prev_hm = float(prev_val.get("hm", prev_val.get("harmonic_mean", 0.0)))
#         curr_old = float(curr_val.get("old_acc", curr_val.get("old_accuracy", 0.0)))
#         curr_new = float(curr_val.get("new_acc", curr_val.get("new_accuracy", 0.0)))
#         curr_hm = float(curr_val.get("hm", curr_val.get("harmonic_mean", 0.0)))
#         old_drop = prev_old - curr_old
#         hm_drop = prev_hm - curr_hm
#         new_saturated = curr_new >= 98.0 or (prev_new >= 98.0 and curr_new >= 97.5)
#         old_damaged = old_drop >= 2.0
#         hm_damaged = hm_drop >= 1.0
#         return bool(new_saturated and (old_damaged or hm_damaged))

    
#     # ============================================================
#     # Trainability
#     # ============================================================
#     def _set_incremental_trainable_params(self, old_class_count: int):
#         for p in self.model.parameters():
#             p.requires_grad = True

#         if hasattr(self.model, "freeze_backbone_only"):
#             self.model.freeze_backbone_only()

#         bb = getattr(self.model, "backbone", None)
#         if (
#             bool(getattr(self.args, "unfreeze_last_backbone_during_incremental", False))
#             and bb is not None
#             and hasattr(bb, "unfreeze_last_blocks")
#         ):
#             bb.unfreeze_last_blocks()

#         if (
#             bool(getattr(self.args, "freeze_semantic_encoder_during_incremental", True))
#             and hasattr(self.model, "freeze_semantic_encoder")
#         ):
#             self.model.freeze_semantic_encoder()
#         elif hasattr(self.model, "unfreeze_semantic_encoder"):
#             self.model.unfreeze_semantic_encoder()

#         if bool(getattr(self.args, "freeze_projection_during_incremental", False)):
#             if hasattr(self.model, "freeze_projection_head"):
#                 self.model.freeze_projection_head()
#         else:
#             if hasattr(self.model, "unfreeze_projection_head"):
#                 self.model.unfreeze_projection_head()

#         if hasattr(self.model, "freeze_old_anchor_deltas"):
#             self.model.freeze_old_anchor_deltas(old_class_count)
#         if hasattr(self.model, "unfreeze_new_anchor_deltas"):
#             self.model.unfreeze_new_anchor_deltas(old_class_count)
#         if hasattr(self.model, "freeze_old_concept_deltas"):
#             self.model.freeze_old_concept_deltas(old_class_count)

#         freeze_classifier = bool(getattr(self.args, "freeze_classifier_during_incremental", False))
#         if freeze_classifier:
#             if hasattr(self.model, "freeze_classifier_adaptation"):
#                 self.model.freeze_classifier_adaptation()
#         else:
#             if hasattr(self.model, "unfreeze_classifier_adaptation"):
#                 self.model.unfreeze_classifier_adaptation()
#             if hasattr(self.model, "freeze_old_classifier_adaptation"):
#                 self.model.freeze_old_classifier_adaptation(old_class_count)

#         # Fusion is legacy/hybrid. Keep it frozen unless explicitly enabled.
#         if bool(getattr(self.args, "use_adaptive_fusion", False)):
#             if hasattr(self.model, "unfreeze_fusion_module"):
#                 self.model.unfreeze_fusion_module()
#         else:
#             if hasattr(self.model, "freeze_fusion_module"):
#                 self.model.freeze_fusion_module()

#         # geometry calibrator remains trainable in incremental phases
#         if hasattr(self.model, "unfreeze_geometry_calibrator"):
#             self.model.unfreeze_geometry_calibrator()
#         self._print_trainable_summary(getattr(self.model, "current_phase", -1))

#     # ============================================================
#     # Base auxiliary losses
#     # ============================================================
#     def _base_aux_losses(self, out, y):
#         features = out["features"]
#         logits = out["logits"]
#         concept_bank = out.get("concept_bank", None)

#         compact, sep, ortho, center_norm, radius = self._base_geometry_loss(features, y)

#         margin = (
#             self.logit_margin(logits, y)
#             if self.logit_margin_weight > 0.0
#             else self._zero(features)
#         )

#         concept_sep = (
#             self.concept_sep(concept_bank)
#             if self.concept_sep_weight > 0.0 and concept_bank is not None and concept_bank.numel() > 0
#             else self._zero(features)
#         )

#         feature_concept = (
#             self.feature_concept_compact(features, concept_bank, y)
#             if self.feature_concept_compact_weight > 0.0 and concept_bank is not None and concept_bank.numel() > 0
#             else self._zero(features)
#         )

#         return compact, sep, ortho, center_norm, radius, margin, concept_sep, feature_concept

#     def _classifier_adaptation_reg(self, ref: torch.Tensor) -> torch.Tensor:
#         if self.classifier_adaptation_weight <= 0.0:
#             return self._zero(ref)

#         classifier = getattr(self.model, "classifier", None)
#         if classifier is None:
#             return self._zero(ref)

#         if hasattr(classifier, "adaptation_regularization_loss"):
#             reg = classifier.adaptation_regularization_loss()
#             if isinstance(reg, dict):
#                 return reg.get("total", self._zero(ref))
#             return reg

#         return self._zero(ref)

#     # ============================================================
#     # Train one epoch
#     # ============================================================
#     def _train_epoch_base(self, loader, optimizer):
#         self.model.train()

#         total_loss = total_correct = total = 0

#         for x, y in loader:
#             x = x.to(self.device).float()
#             y = y.to(self.device).long()

#             optimizer.zero_grad(set_to_none=True)

#             out = self.model(
#                 x,
#                 semantic_mode="identity",
#                 classifier_mode=getattr(self.args, "base_classifier_mode", "geometry_only"),
#             )

#             ce = F.cross_entropy(
#                 out["logits"],
#                 y,
#                 label_smoothing=float(getattr(self.args, "label_smoothing", 0.0)),
#             )
#             compact, sep, ortho, center_norm, radius, margin, concept_sep, feature_concept = (
#                 self._base_aux_losses(out, y)
#             )

#             loss = (
#                 ce
#                 + self.base_compact * compact
#                 + self.base_sep * sep
#                 + self.base_ortho * ortho
#                 + self.base_center_norm * center_norm
#                 + self.base_radius * radius
#                 + self.logit_margin_weight * margin
#                 + self.concept_sep_weight * concept_sep
#                 + self.feature_concept_compact_weight * feature_concept
#             )

#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(
#                 [p for p in self.model.parameters() if p.requires_grad],
#                 1.0,
#             )
#             optimizer.step()

#             total_loss += float(loss.item())
#             total_correct += int((out["logits"].argmax(1) == y).sum().item())
#             total += int(y.size(0))

#         return total_loss / max(len(loader), 1), 100.0 * total_correct / max(total, 1)

#     def _train_epoch_incremental(
#         self,
#         loader,
#         optimizer,
#         old_class_count,
#         new_class_ids,
#         old_bank_snapshot,
#         old_token_snapshot=None,
#         epoch_idx: int = 0,
#     ):
#         self.model.train()
#         self._set_model_phase_and_old_count(self.model.current_phase, old_class_count)

#         total_loss = 0.0
#         total_correct = 0
#         total = 0

#         classifier_mode = self._inc_classifier_mode()

#         token_loss_weight = float(getattr(self.args, "token_loss_weight", 0.0))
#         spectral_guidance_weight = float(getattr(self.args, "spectral_guidance_weight", 0.0))
#         band_guidance_weight = float(getattr(self.args, "band_guidance_weight", 0.0))
#         replay_geometry_weight = float(getattr(self.args, "replay_geometry_weight", 0.0))

#         use_token_relations = token_loss_weight > 0.0

#         if self.incremental_warmup_epochs > 0:
#             post_warm_epochs = max(
#                 int(getattr(self.args, "epochs_inc", self.incremental_warmup_epochs)) - self.incremental_warmup_epochs,
#                 1,
#             )
#             if epoch_idx < self.incremental_warmup_epochs:
#                 structure_ramp = 0.0
#             else:
#                 structure_ramp = min(
#                     1.0,
#                     float(epoch_idx - self.incremental_warmup_epochs + 1) / float(post_warm_epochs),
#                 )
#         else:
#             structure_ramp = 1.0

#         for x, y in loader:
#             x = x.to(self.device).float()
#             y = y.to(self.device).long()

#             optimizer.zero_grad(set_to_none=True)

#             out = self.model(
#                 x,
#                 semantic_mode="identity",
#                 classifier_mode=classifier_mode,
#                 return_token_relations=use_token_relations,
#             )
#             logits = out["logits"]
#             features = out["features"]
#             concept_bank = out.get("concept_bank", None)

#             # 1) New-class supervision
#             ce_new = self._masked_weighted_ce_new(logits, y, new_class_ids)

#             # 2) Old-class replay
#             replay_x, replay_y = self._sample_replay_from_snapshot(
#                 old_bank_snapshot, old_class_count
#             )

#             if replay_x is not None:
#                 replay_logits = self.model.compute_logits_from_features(
#                     replay_x,
#                     classifier_mode=classifier_mode,
#                 )
#                 ce_replay = F.cross_entropy(replay_logits, replay_y)

#                 replay_geom_loss = self._replay_geometry_loss(
#                     replay_x,
#                     replay_y,
#                     old_bank_snapshot=old_bank_snapshot,
#                 )
#             else:
#                 ce_replay = self._zero(logits)
#                 replay_geom_loss = self._zero(logits)

#             # 3) Weak bank-vs-snapshot consistency
#             a_mean, a_basis, a_var, _ = self._geometry_alignment_losses(
#                 old_bank_snapshot,
#                 old_class_count,
#             )

#             # 4) Geometry-native separation and compactness
#             if hasattr(self, "_symmetric_geometry_separation_loss"):
#                 sep_geo_loss = self._symmetric_geometry_separation_loss(
#                     new_features=features,
#                     new_labels=y,
#                     replay_features=replay_x,
#                     replay_labels=replay_y,
#                     old_class_count=old_class_count,
#                     new_class_ids=new_class_ids,
#                     old_bank_snapshot=old_bank_snapshot,
#                 )
#             else:
#                 sep_geo_loss = self._geometry_separation_loss(
#                     features, y, old_class_count, new_class_ids
#                 )
#             vol_loss = self._new_class_volume_loss(features, y, new_class_ids)

#             # 5) Structural preservation
#             token_loss, token_parts = self._token_manifold_loss(
#                 out, y, old_class_count, old_token_snapshot
#             )
#             del token_parts

#             spec_loss, band_loss = self._spectral_guidance_losses(
#                 out, y, old_class_count
#             )

#             # 6) Calibration/adaptation regularization
#             calibration_reg = out.get("calibration_reg", None)
#             if calibration_reg is None:
#                 calibration_total = self._zero(logits)
#             else:
#                 calibration_total = calibration_reg.get("total", self._zero(logits))

#             cls_adapt_reg = self._classifier_adaptation_reg(logits)

#             # 7) New-class auxiliaries
#             margin_new = self._zero(logits)
#             feature_concept_new = self._zero(logits)

#             class_ids = torch.tensor(new_class_ids, device=logits.device, dtype=torch.long)

#             if self._labels_are_local(y, new_class_ids):
#                 y_local = y
#             else:
#                 y_local = torch.full_like(y, fill_value=-1)
#                 for local_idx, global_cls in enumerate(class_ids):
#                     y_local[y == global_cls] = local_idx

#             valid = y_local >= 0

#             if self.inc_logit_margin_weight > 0.0 and valid.any():
#                 logits_new_only = logits.index_select(1, class_ids)
#                 margin_new = self.logit_margin(logits_new_only[valid], y_local[valid])

#             if (
#                 self.inc_feature_concept_compact_weight > 0.0
#                 and concept_bank is not None
#                 and concept_bank.numel() > 0
#                 and valid.any()
#             ):
#                 new_concept_bank = concept_bank.index_select(0, class_ids)
#                 feature_concept_new = self.feature_concept_compact(
#                     features[valid],
#                     new_concept_bank,
#                     y_local[valid],
#                 )

#             # 8) Loss assembly
#             if epoch_idx < self.incremental_warmup_epochs:
#                 loss = (
#                     ce_new
#                     + self.replay_weight * ce_replay
#                     + self.inc_logit_margin_weight * margin_new
#                     + self.geometry_calibration_weight * calibration_total
#                     + self.classifier_adaptation_weight * cls_adapt_reg
#                 )
#             else:
#                 loss = (
#                     ce_new
#                     + self.replay_weight * ce_replay
#                     + replay_geometry_weight * structure_ramp * replay_geom_loss
#                     + self.align_mean_weight * structure_ramp * a_mean
#                     + self.align_basis_weight * structure_ramp * a_basis
#                     + self.align_var_weight * structure_ramp * a_var
#                     + self.insert_weight * structure_ramp * sep_geo_loss
#                     + self.new_volume_weight * structure_ramp * vol_loss
#                     + self.geometry_calibration_weight * calibration_total
#                     + self.classifier_adaptation_weight * cls_adapt_reg
#                     + token_loss_weight * structure_ramp * token_loss
#                     + spectral_guidance_weight * structure_ramp * spec_loss
#                     + band_guidance_weight * structure_ramp * band_loss
#                     + self.inc_logit_margin_weight * margin_new
#                     + self.inc_feature_concept_compact_weight * feature_concept_new
#                 )

#             if not torch.isfinite(loss):
#                 if self.debug:
#                     print(
#                         "[WARN] Non-finite incremental loss encountered. "
#                         f"ce_new={float(ce_new.detach().item()):.6f}, "
#                         f"ce_replay={float(ce_replay.detach().item()):.6f}, "
#                         f"replay_geom={float(replay_geom_loss.detach().item()):.6f}, "
#                         f"a_mean={float(a_mean.detach().item()):.6f}, "
#                         f"a_basis={float(a_basis.detach().item()):.6f}, "
#                         f"a_var={float(a_var.detach().item()):.6f}, "
#                         f"sep_geo={float(sep_geo_loss.detach().item()):.6f}, "
#                         f"vol={float(vol_loss.detach().item()):.6f}, "
#                         f"calib={float(calibration_total.detach().item()):.6f}, "
#                         f"cls_adapt={float(cls_adapt_reg.detach().item()):.6f}, "
#                         f"token={float(token_loss.detach().item()):.6f}, "
#                         f"spec={float(spec_loss.detach().item()):.6f}, "
#                         f"band={float(band_loss.detach().item()):.6f}, "
#                         f"margin={float(margin_new.detach().item()):.6f}, "
#                         f"feat_concept={float(feature_concept_new.detach().item()):.6f}"
#                     )
#                 continue

#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(
#                 [p for p in self.model.parameters() if p.requires_grad], 0.5
#             )
#             optimizer.step()

#             total_loss += float(loss.item())

#             correct_new, valid_new = self._incremental_accuracy_with_count(
#                 logits, y, new_class_ids
#             )
#             total_correct += correct_new
#             total += valid_new

#             if self.debug:
#                 print(
#                     f"[IncDebug] loss={float(loss.item()):.4f} | "
#                     f"ce_new={float(ce_new.detach().item()):.4f} | "
#                     f"ce_replay={float(ce_replay.detach().item()):.4f} | "
#                     f"replay_geom={float(replay_geom_loss.detach().item()):.4f} | "
#                     f"a_mean={float(a_mean.detach().item()):.4f} | "
#                     f"a_basis={float(a_basis.detach().item()):.4f} | "
#                     f"a_var={float(a_var.detach().item()):.4f} | "
#                     f"sep_geo={float(sep_geo_loss.detach().item()):.4f} | "
#                     f"vol={float(vol_loss.detach().item()):.4f} | "
#                     f"calib={float(calibration_total.detach().item()):.4f} | "
#                     f"cls_adapt={float(cls_adapt_reg.detach().item()):.4f} | "
#                     f"token={float(token_loss.detach().item()):.4f} | "
#                     f"spec={float(spec_loss.detach().item()):.4f} | "
#                     f"band={float(band_loss.detach().item()):.4f} | "
#                     f"margin={float(margin_new.detach().item()):.4f} | "
#                     f"feat_concept={float(feature_concept_new.detach().item()):.4f} | "
#                     f"ramp={structure_ramp:.3f}"
#                 )

#         train_acc = 100.0 * total_correct / max(total, 1)
#         return total_loss / max(len(loader), 1), train_acc

#     # ============================================================
#     # Post-phase calibration
#     # ============================================================
#     def _set_calibration_trainable_params(self):
#         for p in self.model.parameters():
#             p.requires_grad = False

#         if hasattr(self.model, "unfreeze_classifier_adaptation"):
#             self.model.unfreeze_classifier_adaptation()

#         # Freeze old classifier adaptation again if available; calibration should
#         # tune global/allowed offsets conservatively, not rewrite old class params.
#         old_class_count = int(getattr(self.model, "old_class_count", 0))
#         if hasattr(self.model, "freeze_old_classifier_adaptation") and old_class_count > 0:
#             self.model.freeze_old_classifier_adaptation(old_class_count)

#         if hasattr(self.model, "unfreeze_geometry_calibrator"):
#             self.model.unfreeze_geometry_calibrator()

#     def _post_phase_calibration(
#         self,
#         phase: int,
#         old_class_count: int,
#         new_class_ids,
#         old_bank_snapshot,
#         batch_size: int,
#     ):
#         if self.calibration_epochs <= 0:
#             return

#         self._set_calibration_trainable_params()

#         calib_params = [p for p in self.model.parameters() if p.requires_grad]
#         if len(calib_params) == 0:
#             return

#         optimizer = optim.Adam(calib_params, lr=self.calibration_lr, weight_decay=0.0)

#         train_loader = self.dataset.get_phase_dataloader(
#             phase,
#             split="train",
#             batch_size=batch_size,
#             shuffle=True,
#         )

#         classifier_mode = self._eval_classifier_mode()
#         replay_geometry_weight = float(getattr(self.args, "replay_geometry_weight", 0.0))

#         self.model.train()
#         self._set_model_phase_and_old_count(phase, old_class_count)

#         best_state = None
#         best_loss = float("inf")

#         for _ in range(self.calibration_epochs):
#             epoch_loss = 0.0
#             epoch_steps = 0

#             for x, y in train_loader:
#                 x = x.to(self.device).float()
#                 y = y.to(self.device).long()

#                 optimizer.zero_grad(set_to_none=True)

#                 out = self.model(
#                     x,
#                     semantic_mode="identity",
#                     classifier_mode=classifier_mode,
#                 )
#                 ce_new = self._masked_weighted_ce_new(out["logits"], y, new_class_ids)

#                 replay_x, replay_y = self._sample_replay_from_snapshot(
#                     old_bank_snapshot,
#                     old_class_count,
#                 )

#                 if replay_x is not None:
#                     replay_logits = self.model.compute_logits_from_features(
#                         replay_x,
#                         classifier_mode=classifier_mode,
#                     )
#                     ce_old = F.cross_entropy(replay_logits, replay_y)

#                     replay_geom_loss = self._replay_geometry_loss(
#                         replay_x,
#                         replay_y,
#                         old_bank_snapshot=old_bank_snapshot,
#                     )
#                 else:
#                     ce_old = self._zero(out["logits"])
#                     replay_geom_loss = self._zero(out["logits"])

#                 calibration_reg = out.get("calibration_reg", None)
#                 if calibration_reg is None:
#                     calibration_total = self._zero(out["logits"])
#                 else:
#                     calibration_total = calibration_reg.get("total", self._zero(out["logits"]))

#                 cls_adapt_reg = self._classifier_adaptation_reg(out["logits"])

#                 loss = (
#                     ce_new
#                     + self.calibration_replay_weight * ce_old
#                     + self.geometry_calibration_weight * calibration_total
#                     + self.classifier_adaptation_weight * cls_adapt_reg
#                     + 0.5 * replay_geometry_weight * replay_geom_loss
#                 )

#                 if not torch.isfinite(loss):
#                     if self.debug:
#                         print(
#                             "[WARN] Non-finite calibration loss encountered. "
#                             f"ce_new={float(ce_new.detach().item()):.6f}, "
#                             f"ce_old={float(ce_old.detach().item()):.6f}, "
#                             f"calib={float(calibration_total.detach().item()):.6f}, "
#                             f"cls_adapt={float(cls_adapt_reg.detach().item()):.6f}, "
#                             f"replay_geom={float(replay_geom_loss.detach().item()):.6f}"
#                         )
#                     continue

#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(calib_params, 0.5)
#                 optimizer.step()

#                 epoch_loss += float(loss.item())
#                 epoch_steps += 1

#             avg_epoch_loss = epoch_loss / max(epoch_steps, 1)

#             if avg_epoch_loss < best_loss:
#                 best_loss = avg_epoch_loss
#                 best_state = self._capture_state()

#         if best_state is not None:
#             self.model.load_state_dict(best_state)
#             self._set_model_phase_and_old_count(phase, old_class_count)

#     # ============================================================
#     # Validation
#     # ============================================================
#     @torch.no_grad()
#     def _validate_split_metrics(self, loader, old_class_count: int):
#         self.model.eval()
#         self._set_model_phase_and_old_count(self.model.current_phase, old_class_count)

#         total_loss = total_correct = total = 0
#         old_correct = old_total = 0
#         new_correct = new_total = 0

#         if int(old_class_count) == 0:
#             val_classifier_mode = getattr(self.args, "base_classifier_mode", "geometry_only")
#             val_semantic_mode = "identity"
#         else:
#             val_classifier_mode = self._eval_classifier_mode()
#             val_semantic_mode = getattr(self.args, "eval_semantic_mode", "identity")

#         for x, y in loader:
#             x = x.to(self.device).float()
#             y = y.to(self.device).long()

#             out = self.model(
#                 x,
#                 semantic_mode=val_semantic_mode,
#                 classifier_mode=val_classifier_mode,
#             )

#             logits = out["logits"]
#             loss = F.cross_entropy(logits, y)
#             preds = logits.argmax(dim=1)

#             total_loss += float(loss.item())
#             total_correct += int((preds == y).sum().item())
#             total += int(y.size(0))

#             old_mask = y < old_class_count
#             new_mask = y >= old_class_count

#             if old_mask.any():
#                 old_correct += int((preds[old_mask] == y[old_mask]).sum().item())
#                 old_total += int(old_mask.sum().item())

#             if new_mask.any():
#                 new_correct += int((preds[new_mask] == y[new_mask]).sum().item())
#                 new_total += int(new_mask.sum().item())

#         old_acc = 100.0 * old_correct / max(old_total, 1)
#         new_acc = 100.0 * new_correct / max(new_total, 1)
#         total_acc = 100.0 * total_correct / max(total, 1)
#         hm = 0.0 if (old_acc + new_acc) == 0 else 2.0 * old_acc * new_acc / (old_acc + new_acc)

#         return {
#             "loss": total_loss / max(len(loader), 1),
#             "acc": total_acc,
#             "old_acc": old_acc,
#             "new_acc": new_acc,
#             "hm": hm,
#         }

#     # ============================================================
#     # Phase training
#     # ============================================================
#     def train_phase(self, phase, epochs, batch_size=64, lr=1e-4):
#         print(f"==== Training Phase {phase} ====")

#         phase = int(phase)
#         self.dataset.start_phase(phase)

#         needed_classes = max(self.dataset.phase_to_classes[phase]) + 1
#         if int(self.model.current_num_classes) < needed_classes:
#             self._bootstrap_phase_classes(phase, split="train")

#         old_class_count = 0 if phase == 0 else len(self.dataset.get_classes_up_to_phase(phase - 1))
#         self._set_model_phase_and_old_count(phase, old_class_count)

#         history = {
#             "train_loss": [],
#             # Train accuracy is the refreshed train-set diagnostic, not online/stale-bank accuracy.
#             "train_acc": [],
#             "val_loss": [],
#             "val_acc": [],
#             "val_old_acc": [],
#             "val_new_acc": [],
#             "val_hm": [],
#         }

#         val_loader = self.dataset.get_cumulative_dataloader(
#             phase,
#             split="val",
#             batch_size=batch_size,
#             shuffle=False,
#         )

#         # ---------------- Base phase ----------------
#         if phase == 0:
#             train_loader = self.dataset.get_phase_dataloader(
#                 phase,
#                 split="train",
#                 batch_size=batch_size,
#                 shuffle=True,
#             )

#             optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
#             scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

#             best_state, best_score = None, -1.0
#             no_improve = 0

#             base_class_ids = [int(c) for c in self.dataset.phase_to_classes[phase]]

#             for epoch in range(epochs):
#                 self._set_model_phase_and_old_count(phase, 0)

#                 tr_loss, tr_acc = self._train_epoch_base(train_loader, optimizer)

#                 # Refresh base geometry before validation; otherwise phase-0
#                 # validation is evaluated against stale placeholder geometry.
#                 if self._should_refresh_for_validation(epoch):
#                     self._refresh_classes_for_validation(phase, base_class_ids, split="train")

#                 # Evaluate train split after geometry refresh.
#                 # This refreshed train accuracy is the reported Train Acc.
#                 train_eval_stats = self._validate_split_metrics(train_loader, 0)
#                 val_stats = self._validate_split_metrics(val_loader, 0)
#                 scheduler.step()

#                 history["train_loss"].append(tr_loss)
#                 history["train_acc"].append(train_eval_stats["acc"])
#                 history["val_loss"].append(val_stats["loss"])
#                 history["val_acc"].append(val_stats["acc"])
#                 history["val_old_acc"].append(val_stats["old_acc"])
#                 history["val_new_acc"].append(val_stats["new_acc"])
#                 history["val_hm"].append(val_stats["hm"])

#                 print(
#                     f"Ep {epoch+1:03d}/{epochs} | "
#                     f"Train Loss: {tr_loss:.4f} | "
#                     f"Train Acc: {train_eval_stats['acc']:.2f}% | "
#                     f"Val Acc: {val_stats['acc']:.2f}% | "
#                     f"Val Loss: {val_stats['loss']:.4f}"
#                 )

#                 score = self._select_score(val_stats, phase=0)
#                 if score > best_score:
#                     best_score = score
#                     best_state = self._capture_state()
#                     no_improve = 0
#                 else:
#                     no_improve += 1

#                 if self.early_stop_patience > 0 and no_improve >= self.early_stop_patience:
#                     print(f"[EarlyStop] Phase {phase}: no improvement for {no_improve} epochs.")
#                     break

#             if best_state is not None:
#                 self.model.load_state_dict(best_state)
#                 self._set_model_phase_and_old_count(phase, 0)

#             self._finalize_phase_memory(phase, split="train")
#             gb = getattr(self.model, "geometry_bank", None)
#             if gb is not None and hasattr(gb, "refresh_inter_class_geometry"):
#                 gb.refresh_inter_class_geometry()

#             self.model.old_class_count = int(self.model.current_num_classes)
#             # self.save_checkpoint(phase, history)
#             return history

#         # ---------------- Incremental phase ----------------
#         old_bank_snapshot = self._snapshot_old_bank(old_class_count)
#         old_token_snapshot = self._snapshot_old_token_memory(old_class_count)
#         new_class_ids = [int(c) for c in self.dataset.phase_to_classes[phase]]

#         self._set_incremental_trainable_params(old_class_count)

#         train_loader = self.dataset.get_phase_dataloader(
#             phase,
#             split="train",
#             batch_size=batch_size,
#             shuffle=True,
#         )

#         trainable_params = [p for p in self.model.parameters() if p.requires_grad]
#         if len(trainable_params) == 0:
#             raise RuntimeError("No trainable parameters found for incremental phase.")

#         optimizer = optim.Adam(
#             trainable_params,
#             lr=lr,
#             weight_decay=1e-5,
#         )
#         scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

#         best_state, best_score = None, -1.0
#         no_improve = 0

#         for epoch in range(epochs):
#             self._set_model_phase_and_old_count(phase, old_class_count)

#             tr_loss, tr_acc = self._train_epoch_incremental(
#                 train_loader,
#                 optimizer,
#                 old_class_count,
#                 new_class_ids,
#                 old_bank_snapshot,
#                 old_token_snapshot,
#                 epoch_idx=epoch,
#             )

#             # --------------------------------------------------------
#             # Correct order:
#             #   1. train epoch
#             #   2. refresh current-phase geometry
#             #   3. validate
#             #   4. save best state
#             #
#             # The old trainer validated before refresh, which made validation
#             # look collapsed even when post-finalization evaluation recovered.
#             # --------------------------------------------------------
#             refresh_every = int(getattr(self.args, "bank_refresh_every", 0))
#             do_periodic_refresh = refresh_every > 0 and (epoch + 1) % refresh_every == 0
#             do_validation_refresh = self._should_refresh_for_validation(epoch)

#             if do_periodic_refresh or do_validation_refresh:
#                 self._refresh_classes_for_validation(phase, new_class_ids, split="train")

#             # Evaluate current-phase train split after geometry refresh.
#             # This refreshed accuracy is the reported Train Acc.
#             train_eval_stats = self._validate_split_metrics(train_loader, old_class_count)
#             val_stats = self._validate_split_metrics(val_loader, old_class_count)
#             scheduler.step()

#             history["train_loss"].append(tr_loss)
#             history["train_acc"].append(train_eval_stats["acc"])
#             history["val_loss"].append(val_stats["loss"])
#             history["val_acc"].append(val_stats["acc"])
#             history["val_old_acc"].append(val_stats["old_acc"])
#             history["val_new_acc"].append(val_stats["new_acc"])
#             history["val_hm"].append(val_stats["hm"])

#             print(
#                 f"Ep {epoch+1:03d}/{epochs} | "
#                 f"Train Loss: {tr_loss:.4f} | "
#                 f"Train Acc: {train_eval_stats['acc']:.2f}% | "
#                 f"Val Acc: {val_stats['acc']:.2f}% | "
#                 f"Val Loss: {val_stats['loss']:.4f}"
#             )

#             score = self._select_score(val_stats, phase=phase)
#             if score > best_score:
#                 best_score = score
#                 best_state = self._capture_state()
#                 no_improve = 0
#             else:
#                 no_improve += 1

#             if self.early_stop_patience > 0 and no_improve >= self.early_stop_patience:
#                 print(f"[EarlyStop] Phase {phase}: no improvement for {no_improve} epochs.")
#                 break

#         if best_state is not None:
#             self.model.load_state_dict(best_state)
#             self._set_model_phase_and_old_count(phase, old_class_count)

#         self._post_phase_calibration(
#             phase=phase,
#             old_class_count=old_class_count,
#             new_class_ids=new_class_ids,
#             old_bank_snapshot=old_bank_snapshot,
#             batch_size=batch_size,
#         )

#         self._finalize_phase_memory(phase, split="train")
#         gb = getattr(self.model, "geometry_bank", None)
#         if gb is not None and hasattr(gb, "refresh_inter_class_geometry"):
#             gb.refresh_inter_class_geometry()

#         self.model.old_class_count = int(self.model.current_num_classes)
#         self.save_checkpoint(phase, history)
#         return history

#     # ============================================================
#     # Save
#     # ============================================================
#     def save_checkpoint(self, phase, history, evaluator_metrics=None):
#         run_dir = getattr(self.args, "run_dir", None)
#         if run_dir is not None and str(run_dir).strip():
#             phase_dir = os.path.join(str(run_dir), f"phase_{phase}")
#         else:
#             phase_dir = os.path.join(self.save_dir, self.args.dataset, f"phase_{phase}")
#         os.makedirs(phase_dir, exist_ok=True)

#         ckpt = {
#             "phase": int(phase),
#             "model_state_dict": self.model.state_dict(),
#             "memory_snapshot": self.model.export_memory_snapshot() if hasattr(self.model, "export_memory_snapshot") else None,
#             "current_num_classes": int(self.model.current_num_classes),
#             "old_class_count": int(getattr(self.model, "old_class_count", 0)),
#             "history": history,
#             "args": vars(self.args),
#             "token_memory": {
#                 int(k): {kk: vv.detach().cpu() for kk, vv in v.items()}
#                 for k, v in getattr(self, "token_memory", {}).items()
#             },
#         }
#         if evaluator_metrics is not None:
#             ckpt["evaluator_metrics"] = evaluator_metrics

#         path = os.path.join(phase_dir, "checkpoint.pth")
#         torch.save(ckpt, path)
#         print(f"[Saved] {path}")
