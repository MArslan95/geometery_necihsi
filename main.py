"""
Main Entry Point for Geometry-Native NECIL-HSI
==============================================

Aligned with:
- strict non-exemplar IncrementalHSIDataset
- explicit HSI label policy where class 0 may be a real class
- reliability-aware low-rank GeometryBank
- geometry-centric Trainer / TrainerHelper
- NECILModel with conservative geometry calibration
- geometry-native classifier
- visualization/evaluation using semantic_mode='identity'
- multi-run evaluation with mean ± std
- per-phase and final classification reports

Critical policy
---------------
Do not evaluate phase 0 with semantic_mode='all' while incremental phases use
semantic_mode='identity'. Geometry memory must be built/evaluated in one feature
manifold. The default is identity for all phases.
"""

import argparse
import inspect
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import Optional, Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.hsi_dataloader_pytorch import LoadHSIData, ImageCubes
from data.incremental_dataset import IncrementalHSIDataset
from models.necil_model import NECILModel
from trainers.trainer import Trainer
from utils.eval import NECILEvaluator, save_classification_report, make_json_serializable
from utils.visualize import predict_phase_grid, plot_training_history


DATASET_INFO = {
    "IP": {"name": "Indian Pines", "bands": 200, "classes": 16},
    "SA": {"name": "Salinas", "bands": 204, "classes": 16},
    "PU": {"name": "Pavia University", "bands": 103, "classes": 9},
    "PC": {"name": "Pavia Centre", "bands": 102, "classes": 9},
    "BS": {"name": "Botswana", "bands": 145, "classes": 14},
    "LK": {"name": "LongKou", "bands": 270, "classes": 9},
    "HH": {"name": "HongHu", "bands": 270, "classes": 22},
    "HC": {"name": "HanChuan", "bands": 274, "classes": 16},
    "UH13": {"name": "Houston 2013", "bands": 144, "classes": 15},
    "QUH": {"name": "QUH-Qingyun", "bands": 270, "classes": 6},
    "PI": {"name": "QUH-Pingan", "bands": 270, "classes": 10},
    "TH": {"name": "QUH-Tangdaowan", "bands": 270, "classes": 18},
}

SUPPORTED_CLASSIFIER_MODES = {
    "geometry_only",
    "calibrated_geometry",
}

SUPPORTED_EVAL_SEMANTIC_MODES = {
    "identity",
    "off",
    "none",
    "bypass",
    "raw",
    # "all" is kept only for explicit ablation, not for main runs.
    "all",
    "auto",
}


# ============================================================
# Argument helpers
# ============================================================
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).lower() in ("true", "1", "yes", "y", "t")


def parse_seed_list(seed_list_str: str):
    if seed_list_str is None or str(seed_list_str).strip() == "":
        return None
    return [int(s.strip()) for s in str(seed_list_str).split(",") if s.strip() != ""]


def parse_args():
    parser = argparse.ArgumentParser(description="Geometry-Native NECIL-HSI")

    # ---------------- Dataset ----------------
    parser.add_argument("--dataset", type=str, default="IP", choices=DATASET_INFO.keys())
    parser.add_argument("--data_dir", type=str, default="./datasets")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--patch_size", type=int, default=11)
    parser.add_argument("--train_ratio", type=float, default=0.2)
    parser.add_argument("--val_ratio", type=float, default=0.1)

    # ---------------- Model ----------------
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--subspace_rank", type=int, default=5)
    parser.add_argument("--num_concepts_per_class", type=int, default=4)
    parser.add_argument("--d_state", type=int, default=16)
    parser.add_argument("--d_conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--num_spectral_layers", type=int, default=3)
    parser.add_argument("--num_layers", type=int, default=3)

    # ---------------- Backbone / projection ----------------
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--semantic_dropout", type=float, default=0.1)
    parser.add_argument("--projection_dropout", type=float, default=0.1)
    parser.add_argument("--backbone_norm", type=str, default="layer", choices=["layer", "rms"])
    parser.add_argument("--stem_norm_groups", type=int, default=8)
    parser.add_argument("--ssm_residual_scale_init", type=float, default=0.7)
    parser.add_argument("--fusion_residual_scale", type=float, default=0.3)
    parser.add_argument("--backbone_output_dropout", type=float, default=0.0)

    # ---------------- Token / semantic ----------------
    parser.add_argument("--token_temperature", type=float, default=0.07)
    parser.add_argument("--token_topk_ratio", type=float, default=1.0)
    parser.add_argument("--token_symmetric_affinity", type=str2bool, default=True)
    parser.add_argument("--disable_semantic_in_incremental", type=str2bool, default=True)

    # ---------------- Classifier ----------------
    parser.add_argument("--cls_temperature", type=float, default=0.07)
    parser.add_argument("--loss_scale", type=float, default=8.0)
    parser.add_argument("--geom_var_floor", type=float, default=1e-4)
    parser.add_argument("--classifier_use_bias", type=str2bool, default=True)
    parser.add_argument("--use_geom_temperature", type=str2bool, default=True)
    parser.add_argument("--min_temperature", type=float, default=0.25)
    parser.add_argument("--max_temperature", type=float, default=4.0)
    parser.add_argument("--debias_strength", type=float, default=0.10)
    parser.add_argument("--energy_normalize_by_dim", type=str2bool, default=True)
    # Accepted by strict classifier variants. Older classifier versions ignore these via getattr.
    parser.add_argument("--reliability_energy_weight", type=float, default=0.05)
    parser.add_argument("--volume_energy_weight", type=float, default=0.0)
    parser.add_argument("--max_classifier_bias_abs", type=float, default=0.50)
    parser.add_argument("--max_classifier_debias_abs", type=float, default=0.25)
    parser.add_argument("--allow_legacy_classifier_modes", type=str2bool, default=False)

    # Legacy compatibility knobs. Keep false for the main method.
    parser.add_argument("--use_adaptive_fusion", type=str2bool, default=False)
    parser.add_argument("--init_alpha_old", type=float, default=-0.5)
    parser.add_argument("--init_alpha_new", type=float, default=-0.2)

    # ---------------- Geometry bank ----------------
    parser.add_argument("--geometry_variance_shrinkage", type=float, default=0.10)
    parser.add_argument("--geometry_max_variance_ratio", type=float, default=50.0)
    parser.add_argument("--geometry_min_reliability", type=float, default=0.05)
    # Accepted by newer GeometryBank variants. Older banks ignore these through getattr/defaults.
    parser.add_argument("--geometry_adjacency_temperature", type=float, default=1.0)
    parser.add_argument("--geometry_energy_temperature", type=float, default=1.0)
    parser.add_argument("--geometry_volume_temperature", type=float, default=1.0)

    # ---------------- Adaptive geometry risk controller ----------------
    # These flags are parsed and logged for compatibility with adaptive trainers.
    # If your current trainer does not consume them, they are harmless no-op config fields.
    parser.add_argument("--use_geometry_risk_controller", type=str2bool, default=False)
    parser.add_argument("--adaptive_update_after_validation", type=str2bool, default=True)
    parser.add_argument("--risk_replay_alpha", type=float, default=0.8)
    parser.add_argument("--risk_sep_beta", type=float, default=1.2)
    parser.add_argument("--risk_align_gamma", type=float, default=0.8)
    parser.add_argument("--risk_margin_delta", type=float, default=0.6)
    parser.add_argument("--risk_max_replay_weight", type=float, default=2.5)
    parser.add_argument("--risk_max_insert_weight", type=float, default=0.04)
    parser.add_argument("--risk_max_margin", type=float, default=5.0)
    parser.add_argument("--adaptive_replay_min", type=int, default=32)
    parser.add_argument("--adaptive_replay_max", type=int, default=128)
    parser.add_argument("--adaptive_replay_power", type=float, default=1.0)
    parser.add_argument("--adaptive_margin_strength", type=float, default=1.0)

    # ---------------- Validation / checkpoint policy ----------------
    parser.add_argument("--refresh_before_validation", type=str2bool, default=True)
    parser.add_argument("--validation_refresh_every", type=int, default=1)
    parser.add_argument("--best_state_metric", type=str, default="hm", choices=["hm", "h", "harmonic", "acc", "oa", "old", "new"])
    parser.add_argument("--early_stop_metric", type=str, default="hm", choices=["hm", "h", "harmonic", "acc", "oa", "old", "new"])
    parser.add_argument("--early_stop_patience", type=int, default=0)

    # ---------------- Geometry calibration ----------------
    parser.add_argument("--geometry_calibration_hidden_dim", type=int, default=128)
    parser.add_argument("--geometry_calibration_dropout", type=float, default=0.1)
    parser.add_argument("--geometry_calibration_weight", type=float, default=0.03)
    parser.add_argument("--geometry_calibrate_basis", type=str2bool, default=False)
    parser.add_argument("--geometry_max_mean_scale", type=float, default=0.10)
    parser.add_argument("--geometry_max_var_scale", type=float, default=0.10)
    parser.add_argument("--geometry_max_basis_scale", type=float, default=0.03)

    # ---------------- Optimization ----------------
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs_base", type=int, default=100)
    parser.add_argument("--epochs_inc", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--label_smoothing", type=float, default=0.0)

    # ---------------- Base shaping ----------------
    parser.add_argument("--base_compact", type=float, default=0.05)
    parser.add_argument("--base_sep", type=float, default=0.05)
    parser.add_argument("--base_ortho", type=float, default=0.03)
    parser.add_argument("--base_margin", type=float, default=1.0)
    parser.add_argument("--base_center_norm", type=float, default=0.01)
    parser.add_argument("--base_radius", type=float, default=0.01)

    # ---------------- Incremental core ----------------
    parser.add_argument("--synthetic_replay_weight", type=float, default=1.0)
    parser.add_argument("--synthetic_replay_per_class", type=int, default=32)
    parser.add_argument("--replay_subspace_scale", type=float, default=0.8)
    parser.add_argument("--replay_residual_scale", type=float, default=0.25)
    parser.add_argument("--replay_min_reliability", type=float, default=0.05)
    parser.add_argument("--replay_geometry_weight", type=float, default=0.05)

    parser.add_argument("--align_mean_weight", type=float, default=0.05)
    parser.add_argument("--align_basis_weight", type=float, default=0.02)
    parser.add_argument("--align_var_weight", type=float, default=0.01)
    parser.add_argument("--align_spec_weight", type=float, default=0.0)

    parser.add_argument("--incremental_warmup_epochs", type=int, default=5)
    parser.add_argument("--bank_refresh_every", type=int, default=1)

    # ---------------- Geometry separation / compactness ----------------
    parser.add_argument("--insert_margin", type=float, default=5.0)
    parser.add_argument("--insert_weight", type=float, default=0.02)
    parser.add_argument("--old_new_energy_margin", type=float, default=5.0)
    parser.add_argument("--new_volume_weight", type=float, default=0.001)
    parser.add_argument("--new_volume_target", type=float, default=2.0)

    # ---------------- Token/spectral preservation ----------------
    parser.add_argument("--token_match_distance_threshold", type=float, default=1.5)
    parser.add_argument("--token_reliability_threshold", type=float, default=0.35)

    parser.add_argument("--token_loss_weight", type=float, default=0.0)
    parser.add_argument("--token_spectral_weight", type=float, default=0.25)
    parser.add_argument("--token_spatial_weight", type=float, default=0.25)
    parser.add_argument("--token_cross_weight", type=float, default=0.50)
    parser.add_argument("--token_fused_weight", type=float, default=0.0)

    parser.add_argument("--spectral_guidance_weight", type=float, default=0.01)
    parser.add_argument("--band_guidance_weight", type=float, default=0.005)
    parser.add_argument("--spectral_guidance_band_loss_type", type=str, default="kl")

    # ---------------- Auxiliary classification shaping ----------------
    parser.add_argument("--logit_margin_value", type=float, default=0.2)
    parser.add_argument("--logit_margin_weight", type=float, default=0.02)
    parser.add_argument("--inc_logit_margin_weight", type=float, default=0.005)

    parser.add_argument("--concept_sep_weight", type=float, default=0.01)
    parser.add_argument("--feature_concept_compact_weight", type=float, default=0.03)
    parser.add_argument("--inc_feature_concept_compact_weight", type=float, default=0.005)
    parser.add_argument("--concept_sep_max_cosine", type=float, default=0.25)

    parser.add_argument("--classifier_adaptation_weight", type=float, default=0.0)

    # ---------------- Incremental setup ----------------
    parser.add_argument("--base_classes", type=int, default=None)
    parser.add_argument("--increment", type=int, default=None)
    parser.add_argument("--min_train_per_class", type=int, default=20)
    parser.add_argument("--alignment_samples_per_class", type=int, default=8)

    # ---------------- Calibration ----------------
    parser.add_argument("--calibration_epochs", type=int, default=3)
    parser.add_argument("--calibration_lr", type=float, default=1e-4)
    parser.add_argument("--calibration_replay_weight", type=float, default=1.0)

    # ---------------- PCA / preprocessing ----------------
    parser.add_argument("--no_pca", action="store_true")
    parser.add_argument("--pca_components", type=int, default=30)
    parser.add_argument("--reduction_method", type=str, default="PCA")

    # ---------------- Modes ----------------
    parser.add_argument("--base_classifier_mode", type=str, default="geometry_only")
    parser.add_argument("--incremental_classifier_mode", type=str, default="calibrated_geometry")
    parser.add_argument("--eval_classifier_mode", type=str, default="calibrated_geometry")
    parser.add_argument("--eval_semantic_mode", type=str, default="identity")

    # ---------------- Freezing / protocol ----------------
    parser.add_argument("--freeze_classifier_during_incremental", type=str2bool, default=False)
    parser.add_argument("--freeze_semantic_encoder_during_incremental", type=str2bool, default=True)
    parser.add_argument("--unfreeze_last_backbone_during_incremental", type=str2bool, default=False)
    parser.add_argument("--freeze_projection_during_incremental", type=str2bool, default=True)
    parser.add_argument("--strict_non_exemplar", type=str2bool, default=True)

    # ---------------- Reproducibility / multi-run ----------------
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--seed_list", type=str, default="")
    parser.add_argument("--deterministic", type=str2bool, default=False)

    # ---------------- Visualization / reporting ----------------
    parser.add_argument("--skip_tsne", type=str2bool, default=False)
    parser.add_argument("--tsne_max_samples", type=int, default=1000)
    parser.add_argument("--skip_phase_maps", type=str2bool, default=False)
    parser.add_argument("--save_classification_report", type=str2bool, default=True)
    parser.add_argument("--save_final_classification_report", type=str2bool, default=True)

    # ---------------- Phase-map visualization ----------------
    parser.add_argument("--viz_class_cmap", type=str, default="turbo")
    parser.add_argument("--viz_confidence_cmap", type=str, default="magma")
    parser.add_argument("--viz_background_color", type=str, default="#20252B")
    parser.add_argument("--viz_save_error_map", type=str2bool, default=True)
    parser.add_argument("--viz_save_numpy", type=str2bool, default=True)

    # ---------------- System ----------------
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--subspace_extract_batch_size", type=int, default=256)
    parser.add_argument("--debug_verbose", type=str2bool, default=False)

    return parser.parse_args()


def validate_args(args):
    args.base_classifier_mode = str(args.base_classifier_mode).lower().strip()
    args.incremental_classifier_mode = str(args.incremental_classifier_mode).lower().strip()
    args.eval_classifier_mode = str(args.eval_classifier_mode).lower().strip()
    args.eval_semantic_mode = str(args.eval_semantic_mode).lower().strip()

    for mode_name, mode_value in [
        ("base_classifier_mode", args.base_classifier_mode),
        ("incremental_classifier_mode", args.incremental_classifier_mode),
        ("eval_classifier_mode", args.eval_classifier_mode),
    ]:
        if mode_value not in SUPPORTED_CLASSIFIER_MODES:
            raise ValueError(
                f"Unsupported {mode_name}='{mode_value}'. "
                f"Supported: {sorted(SUPPORTED_CLASSIFIER_MODES)}"
            )

    if args.eval_semantic_mode not in SUPPORTED_EVAL_SEMANTIC_MODES:
        raise ValueError(
            f"Unsupported eval_semantic_mode='{args.eval_semantic_mode}'. "
            f"Supported: {sorted(SUPPORTED_EVAL_SEMANTIC_MODES)}"
        )

    if args.eval_semantic_mode == "all":
        print(
            "[WARN] eval_semantic_mode='all' evaluates a different feature path. "
            "Use only for ablation. Main geometry runs should use identity."
        )

    if args.base_classes is not None and args.base_classes <= 0:
        raise ValueError("--base_classes must be positive.")
    if args.increment is not None and args.increment <= 0:
        raise ValueError("--increment must be positive.")
    if args.pca_components <= 0 and not args.no_pca:
        raise ValueError("--pca_components must be positive when PCA is enabled.")
    if args.subspace_rank <= 0:
        raise ValueError("--subspace_rank must be positive.")
    if args.num_concepts_per_class <= 0:
        raise ValueError("--num_concepts_per_class must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.epochs_base <= 0 or args.epochs_inc <= 0:
        raise ValueError("--epochs_base and --epochs_inc must be positive.")
    if args.synthetic_replay_weight < 0.0:
        raise ValueError("--synthetic_replay_weight must be >= 0.")
    if args.synthetic_replay_per_class < 0:
        raise ValueError("--synthetic_replay_per_class must be >= 0.")
    if args.calibration_epochs < 0:
        raise ValueError("--calibration_epochs must be >= 0.")
    if args.calibration_replay_weight < 0.0:
        raise ValueError("--calibration_replay_weight must be >= 0.")
    if args.geometry_calibration_weight < 0.0:
        raise ValueError("--geometry_calibration_weight must be >= 0.")
    if args.num_runs <= 0:
        raise ValueError("--num_runs must be >= 1.")
    if not (0.0 < args.token_topk_ratio <= 1.0):
        raise ValueError("--token_topk_ratio must be in (0, 1].")
    if args.bank_refresh_every < 0:
        raise ValueError("--bank_refresh_every must be >= 0.")
    if args.validation_refresh_every < 0:
        raise ValueError("--validation_refresh_every must be >= 0.")
    if args.early_stop_patience < 0:
        raise ValueError("--early_stop_patience must be >= 0.")
    if args.adaptive_replay_min < 0 or args.adaptive_replay_max < 0:
        raise ValueError("--adaptive_replay_min/max must be >= 0.")
    if args.adaptive_replay_min > args.adaptive_replay_max:
        raise ValueError("--adaptive_replay_min must be <= --adaptive_replay_max.")
    if args.risk_max_replay_weight <= 0.0:
        raise ValueError("--risk_max_replay_weight must be positive.")
    if args.risk_max_insert_weight < 0.0:
        raise ValueError("--risk_max_insert_weight must be >= 0.")
    if args.risk_max_margin <= 0.0:
        raise ValueError("--risk_max_margin must be positive.")
    if args.reliability_energy_weight < 0.0 or args.volume_energy_weight < 0.0:
        raise ValueError("--reliability_energy_weight and --volume_energy_weight must be >= 0.")
    if args.max_classifier_bias_abs <= 0.0 or args.max_classifier_debias_abs <= 0.0:
        raise ValueError("--max_classifier_bias_abs and --max_classifier_debias_abs must be positive.")

    if not (0.0 <= args.geometry_min_reliability <= 1.0):
        raise ValueError("--geometry_min_reliability must be in [0, 1].")
    if args.min_temperature <= 0.0 or args.max_temperature <= 0.0:
        raise ValueError("--min_temperature and --max_temperature must be positive.")
    if args.min_temperature > args.max_temperature:
        raise ValueError("--min_temperature must be <= --max_temperature.")
    if not (0.0 < args.train_ratio < 1.0) or not (0.0 <= args.val_ratio < 1.0):
        raise ValueError("--train_ratio must be in (0,1) and --val_ratio must be in [0,1).")
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("--train_ratio + --val_ratio must be < 1.0.")

    seed_list = parse_seed_list(args.seed_list)
    if seed_list is not None and len(seed_list) > 0 and len(seed_list) != args.num_runs:
        raise ValueError(
            f"--seed_list has {len(seed_list)} seeds but --num_runs={args.num_runs}. "
            f"These must match."
        )

    return args


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = True


# ============================================================
# Model / eval helpers
# ============================================================
def _set_model_phase_and_old_count(model, dataset, phase: int):
    phase = int(phase)
    old_class_count = 0 if phase == 0 else len(dataset.get_classes_up_to_phase(phase - 1))

    if hasattr(model, "set_phase"):
        model.set_phase(phase)
    else:
        model.current_phase = phase

    if hasattr(model, "set_old_class_count"):
        model.set_old_class_count(old_class_count)
    else:
        model.old_class_count = old_class_count

    return old_class_count


def _model_forward(model, patches, args, dataset, phase: int):
    """
    Evaluation forward pass.

    Critical:
    semantic_mode defaults to identity for every phase.
    """
    _set_model_phase_and_old_count(model, dataset, phase)

    if int(phase) == 0:
        classifier_mode = getattr(args, "base_classifier_mode", "geometry_only")
    else:
        classifier_mode = getattr(args, "eval_classifier_mode", "calibrated_geometry")

    semantic_mode = getattr(args, "eval_semantic_mode", "identity")

    return model(
        patches,
        semantic_mode=semantic_mode,
        classifier_mode=classifier_mode,
    )


def _build_checkpoint_payload(model, args, extra: Optional[dict] = None) -> dict:
    payload = {
        "model_state_dict": model.state_dict(),
        "memory_snapshot": model.export_memory_snapshot() if hasattr(model, "export_memory_snapshot") else None,
        "args": vars(args),
        "current_num_classes": int(getattr(model, "current_num_classes", 0)),
        "old_class_count": int(getattr(model, "old_class_count", 0)),
        "current_phase": int(getattr(model, "current_phase", 0)),
    }
    if extra is not None:
        payload.update(extra)
    return payload


@torch.no_grad()
def visualize_features_tsne(model, dataset, phase, device, save_path, args, max_samples=1000):
    model.eval()
    _set_model_phase_and_old_count(model, dataset, phase)

    loader = dataset.get_cumulative_dataloader(
        phase,
        split="test",
        batch_size=128,
        shuffle=True,
    )

    all_features, all_labels, seen = [], [], 0

    for patches, labels in loader:
        patches = patches.to(device).float()
        out = _model_forward(model, patches, args, dataset, phase=phase)

        feats = out["features"].detach().cpu().numpy()
        labs = labels.numpy()

        all_features.append(feats)
        all_labels.append(labs)

        seen += len(labels)
        if seen >= max_samples:
            break

    if not all_features:
        return None

    all_features = np.concatenate(all_features, axis=0)[:max_samples]
    all_labels = np.concatenate(all_labels, axis=0)[:max_samples]

    if len(np.unique(all_labels)) < 2 or len(all_features) < 10:
        return None

    n_samples = len(all_features)
    perplexity = min(20, max(5, n_samples // 100))

    try:
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            random_state=42,
            learning_rate="auto",
        )
        reduced = tsne.fit_transform(all_features)
    except KeyboardInterrupt:
        print("[t-SNE] Interrupted by user. Skipping visualization.")
        return None
    except Exception as e:
        print(f"[t-SNE] Skipped due to error: {e}")
        return None

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 8), facecolor="white")
    scatter = plt.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=all_labels,
        cmap="tab20",
        alpha=0.7,
        s=15,
    )
    plt.colorbar(scatter, ticks=np.unique(all_labels))
    plt.title(f"t-SNE Phase {phase}")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    return save_path


@torch.no_grad()
def get_phase_predictions(model, dataset, phase, device, args, batch_size=128):
    model.eval()
    _set_model_phase_and_old_count(model, dataset, phase)

    loader = dataset.get_cumulative_dataloader(
        phase, split="test", batch_size=batch_size, shuffle=False
    )
    all_preds, all_labels = [], []

    for patches, labels in loader:
        patches = patches.to(device).float()
        out = _model_forward(model, patches, args, dataset, phase=phase)
        all_preds.append(out["logits"].argmax(dim=1).cpu().numpy())
        all_labels.append(labels.numpy())

    if len(all_preds) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    return np.concatenate(all_preds), np.concatenate(all_labels)


@torch.no_grad()
def evaluate_model(model, dataset, device, args, batch_size=128):
    model.eval()
    last_phase = dataset.num_phases - 1
    _set_model_phase_and_old_count(model, dataset, last_phase)

    loader = dataset.get_cumulative_dataloader(
        last_phase, split="test", batch_size=batch_size, shuffle=False
    )
    all_preds, all_labels = [], []

    for patches, labels in loader:
        patches = patches.to(device).float()
        out = _model_forward(model, patches, args, dataset, phase=last_phase)
        all_preds.append(out["logits"].argmax(dim=1).cpu().numpy())
        all_labels.append(labels.numpy())

    if len(all_preds) == 0:
        return {"overall_accuracy": 0.0, "per_class_accuracy": {}}

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    overall_acc = 100.0 * (all_preds == all_labels).mean()

    per_class_acc = {}
    for cls in np.unique(all_labels):
        mask = all_labels == cls
        per_class_acc[int(cls)] = (
            100.0 * (all_preds[mask] == cls).mean() if mask.sum() > 0 else 0.0
        )

    return {
        "overall_accuracy": float(overall_acc),
        "per_class_accuracy": per_class_acc,
    }


def _call_predict_phase_grid_compat(**kwargs):
    """
    Call utils.visualize.predict_phase_grid while passing only supported keyword
    arguments. This keeps main.py aligned with both older and newer visualize.py.
    """
    sig = inspect.signature(predict_phase_grid)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return predict_phase_grid(**filtered)


def save_phase_classification_report(
    *,
    evaluator,
    y_true,
    y_pred,
    phase,
    phase_dir,
    target_names_seq,
    seen_classes,
    old_class_count,
    enabled=True,
    tr_time=None,
    te_time=None,
    dl_time=0.0,
):
    """
    Centralized report saving.

    Saves both reports when the updated utils/eval.py is installed:
    - structured report files for debugging/aggregation
    - HSI-style classification report matching common HSI report format

    Critical fix:
    never hard-code FINAL_HSI_Classification_Report.csv inside the per-phase
    report path. Each phase writes into its own phase directory.
    """
    if not enabled:
        return None

    os.makedirs(phase_dir, exist_ok=True)

    if hasattr(evaluator, "save_phase_report"):
        return evaluator.save_phase_report(
            phase=phase,
            y_true=y_true,
            y_pred=y_pred,
            target_names=target_names_seq,
            save_dir=phase_dir,
            seen_classes=seen_classes,
            old_class_count=old_class_count,
            tr_time=tr_time,
            te_time=te_time,
            dl_time=dl_time,
        )

    return save_classification_report(
        y_true=y_true,
        y_pred=y_pred,
        target_names=target_names_seq,
        save_dir=phase_dir,
        phase=phase,
        seen_classes=seen_classes,
        old_class_count=old_class_count,
        tr_time=tr_time,
        te_time=te_time,
        dl_time=dl_time,
        save_hsi_style=True,
        save_structured=True,
    )


# ============================================================
# Dataset builder
# ============================================================
def build_incremental_dataset(args, patches, labels, coords, gt_shape, gt_map, target_names=None, label_policy=None):
    base_kwargs = dict(
        patches=patches,
        labels=labels,
        coords=coords,
        gt_shape=gt_shape,
        GT=gt_map,
        base_classes=args.base_classes,
        increment=args.increment,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        device=args.device,
        min_train_per_class=args.min_train_per_class,
        strict_non_exemplar=args.strict_non_exemplar,
    )

    sig = inspect.signature(IncrementalHSIDataset.__init__)

    optional = {
        "num_workers": args.num_workers,
        "target_names": target_names,
        "label_policy": label_policy,
    }

    for key, value in optional.items():
        if key in sig.parameters:
            base_kwargs[key] = value

    return IncrementalHSIDataset(**base_kwargs)


def evaluator_update_compat(evaluator, phase, y_true, y_pred, old_class_count, seen_classes=None):
    sig = inspect.signature(evaluator.update)
    kwargs = {}

    if "old_class_count" in sig.parameters:
        kwargs["old_class_count"] = old_class_count
    if "seen_classes" in sig.parameters:
        kwargs["seen_classes"] = seen_classes

    evaluator.update(phase, y_true, y_pred, **kwargs)


def save_run_config(args, save_root):
    os.makedirs(save_root, exist_ok=True)
    config_path = os.path.join(save_root, "run_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(make_json_serializable(vars(args)), f, indent=2)
    return config_path


def aggregate_metric(metric_list):
    arr = np.asarray(metric_list, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def _metric_get(metrics: Dict[str, Any], *keys, default=0.0):
    for k in keys:
        if k in metrics:
            return metrics[k]
    return default


# ============================================================
# Experiment
# ============================================================
def run_single_experiment(args, run_idx: int, run_seed: int):
    local_args = argparse.Namespace(**vars(args))
    local_args.seed = int(run_seed)

    set_seed(local_args.seed, deterministic=local_args.deterministic)
    device = torch.device(local_args.device)

    print("\n=== GEOMETRY-NATIVE NECIL-HSI PIPELINE ===")
    print(f"Run: {run_idx + 1}/{args.num_runs} | Seed: {local_args.seed}")
    print(f"Device: {device} | Dataset: {local_args.dataset}")
    print(f"Replay: {local_args.synthetic_replay_weight} / {local_args.synthetic_replay_per_class}")
    print(
        f"Alignment: mean={local_args.align_mean_weight}, "
        f"basis={local_args.align_basis_weight}, var={local_args.align_var_weight}"
    )
    print(
        f"Structure: token={local_args.token_loss_weight}, "
        f"spectral={local_args.spectral_guidance_weight}, "
        f"band={local_args.band_guidance_weight}"
    )
    print(
        f"Geometry calibration: weight={local_args.geometry_calibration_weight}, "
        f"basis={local_args.geometry_calibrate_basis}, "
        f"hidden={local_args.geometry_calibration_hidden_dim}, "
        f"dropout={local_args.geometry_calibration_dropout}"
    )
    print(
        f"Modes: base={local_args.base_classifier_mode}, "
        f"inc={local_args.incremental_classifier_mode}, "
        f"eval={local_args.eval_classifier_mode}, "
        f"semantic={local_args.eval_semantic_mode}"
    )
    print(
        f"Freeze: classifier={local_args.freeze_classifier_during_incremental}, "
        f"semantic={local_args.freeze_semantic_encoder_during_incremental}, "
        f"last_backbone={local_args.unfreeze_last_backbone_during_incremental}, "
        f"projection={local_args.freeze_projection_during_incremental}"
    )
    print(f"Strict non-exemplar: {local_args.strict_non_exemplar}")
    print(
        f"Reports: phase={local_args.save_classification_report}, "
        f"final={local_args.save_final_classification_report}"
    )
    print(
        f"Validation/checkpoint: refresh_before_val={local_args.refresh_before_validation}, "
        f"val_refresh_every={local_args.validation_refresh_every}, "
        f"best_metric={local_args.best_state_metric}, "
        f"early_stop_patience={local_args.early_stop_patience}"
    )
    print(
        f"Risk controller: enabled={local_args.use_geometry_risk_controller}, "
        f"adaptive_update_after_val={local_args.adaptive_update_after_validation}, "
        f"replay_alpha={local_args.risk_replay_alpha}, sep_beta={local_args.risk_sep_beta}"
    )
    if not local_args.freeze_projection_during_incremental:
        print(
            "[WARN] freeze_projection_during_incremental=false. "
            "For feature-space geometry replay this can cause projection drift and old-class collapse."
        )
    print("================================================\n")

    apply_reduction = (not local_args.no_pca) and (local_args.reduction_method.lower() != "none")

    # Updated loader may return label_policy when requested. Fall back for older loader.
    try:
        load_out = LoadHSIData(
            method=local_args.dataset,
            base_dir=local_args.data_dir,
            apply_reduction=apply_reduction,
            n_components=local_args.pca_components,
            reduction_method=local_args.reduction_method,
            return_label_policy=True,
        )
        hsi, gt, num_classes, target_names, has_bg, label_policy = load_out
    except TypeError:
        hsi, gt, num_classes, target_names, has_bg = LoadHSIData(
            method=local_args.dataset,
            base_dir=local_args.data_dir,
            apply_reduction=apply_reduction,
            n_components=local_args.pca_components,
            reduction_method=local_args.reduction_method,
        )
        label_policy = None

    try:
        patches, labels, coords = ImageCubes(
            HSI=hsi,
            GT=gt,
            WS=local_args.patch_size,
            removeZeroLabels=True,
            has_background=has_bg,
            num_classes=num_classes,
            pytorch_format=True,
            label_policy=label_policy,
        )
    except TypeError:
        patches, labels, coords = ImageCubes(
            HSI=hsi,
            GT=gt,
            WS=local_args.patch_size,
            removeZeroLabels=True,
            has_background=has_bg,
            num_classes=num_classes,
            pytorch_format=True,
        )

    local_args.num_bands = int(patches.shape[1])
    local_args.max_classes = int(num_classes)

    if local_args.base_classes is None:
        local_args.base_classes = 4 if local_args.dataset in {"IP", "HC"} else max(2, num_classes // 2)

    if local_args.increment is None:
        remaining = max(1, num_classes - local_args.base_classes)
        local_args.increment = 3 if remaining >= 3 else 1

    if local_args.base_classes >= num_classes:
        raise ValueError(
            f"base_classes={local_args.base_classes} must be < total classes={num_classes}"
        )

    # Keep raw GT only for map shape/reference. Training labels already come
    # from ImageCubes and are sequential 0..K-1.
    gt_for_dataset = gt.copy().astype(np.int64)

    inc_dataset = build_incremental_dataset(
        local_args,
        patches,
        labels,
        coords,
        gt.shape,
        gt_for_dataset,
        target_names=target_names,
        label_policy=label_policy,
    )

    if hasattr(inc_dataset, "inv_label_map"):
        target_names_seq = []
        for sid in range(inc_dataset.num_classes):
            input_label = inc_dataset.inv_label_map[sid]
            if int(input_label) < len(target_names):
                target_names_seq.append(target_names[int(input_label)])
            else:
                target_names_seq.append(f"Class {sid}")
    else:
        target_names_seq = list(target_names)

    inc_dataset.target_names = target_names_seq

    # Hard guard for class-0-real datasets.
    if label_policy is not None and not bool(label_policy.get("has_background", True)):
        if 0 in label_policy.get("raw_class_values", []) and 0 not in np.unique(labels):
            raise RuntimeError(
                "Label policy says raw class 0 is real, but label 0 is missing after ImageCubes. "
                "The loader is still treating class 0 as background."
            )

    run_dir = os.path.join(
        local_args.save_dir,
        local_args.dataset,
        f"patch_{local_args.patch_size}",
        f"run_{run_idx + 1}_seed_{local_args.seed}",
    )
    os.makedirs(run_dir, exist_ok=True)

    # Main owns the active experiment folder. Expose it before Trainer creation
    # so any fallback/manual Trainer checkpoint also lands inside this run folder.
    local_args.run_dir = run_dir
    save_run_config(local_args, run_dir)
    print(f"Run directory: {run_dir}")

    model = NECILModel(local_args).to(device)
    trainer = Trainer(model, inc_dataset, local_args)
    evaluator = NECILEvaluator()

    full_history = {
        "train_loss": [],
        "train_acc": [],
        "online_train_acc": [],
        "train_eval_loss": [],
        "train_eval_acc": [],
        "train_eval_old_acc": [],
        "train_eval_new_acc": [],
        "train_eval_hm": [],
        "val_loss": [],
        "val_acc": [],
        "val_old_acc": [],
        "val_new_acc": [],
        "val_hm": [],
        "phase_boundaries": [],
    }

    phase_report_paths = {}
    start_time = time.time()

    for phase in range(inc_dataset.num_phases):
        _set_model_phase_and_old_count(model, inc_dataset, phase)

        epochs = local_args.epochs_base if phase == 0 else local_args.epochs_inc
        full_history["phase_boundaries"].append(len(full_history["train_loss"]))

        phase_train_start = time.time()
        phase_history = trainer.train_phase(
            phase=phase,
            epochs=epochs,
            batch_size=local_args.batch_size,
            lr=local_args.lr,
        )
        phase_train_time = time.time() - phase_train_start

        if isinstance(phase_history, dict):
            for key in [
                "train_loss",
                "train_acc",
                "online_train_acc",
                "train_eval_loss",
                "train_eval_acc",
                "train_eval_old_acc",
                "train_eval_new_acc",
                "train_eval_hm",
                "val_loss",
                "val_acc",
                "val_old_acc",
                "val_new_acc",
                "val_hm",
            ]:
                if key in phase_history:
                    full_history.setdefault(key, [])
                    full_history[key].extend(phase_history[key])

        print(f"\n[Eval] Phase {phase}")
        phase_eval_start = time.time()
        y_pred, y_true = get_phase_predictions(model, inc_dataset, phase, device, local_args)
        phase_eval_time = time.time() - phase_eval_start
        old_class_count = 0 if phase == 0 else len(inc_dataset.get_classes_up_to_phase(phase - 1))
        seen_classes = inc_dataset.get_classes_up_to_phase(phase)

        evaluator_update_compat(
            evaluator,
            phase,
            y_true,
            y_pred,
            old_class_count,
            seen_classes=seen_classes,
        )
        evaluator.print_summary()

        phase_dir = os.path.join(run_dir, f"phase_{phase}")
        os.makedirs(phase_dir, exist_ok=True)

        report_info = save_phase_classification_report(
            evaluator=evaluator,
            y_true=y_true,
            y_pred=y_pred,
            phase=phase,
            phase_dir=phase_dir,
            target_names_seq=target_names_seq,
            seen_classes=seen_classes,
            old_class_count=old_class_count,
            enabled=bool(local_args.save_classification_report),
            tr_time=phase_train_time,
            te_time=phase_eval_time,
            dl_time=0.0,
        )
        phase_report_paths[int(phase)] = report_info

        phase_payload = _build_checkpoint_payload(
            model=model,
            args=local_args,
            extra={
                "phase": int(phase),
                "metrics": evaluator.phase_history.get(phase, {}),
                "history": phase_history if isinstance(phase_history, dict) else None,
                "classification_report": report_info,
                "target_names_seq": target_names_seq,
                "target_names_raw": target_names,
                "class_order": getattr(inc_dataset, "class_order", None),
                "label_map": getattr(inc_dataset, "label_map", None),
                "inv_label_map": getattr(inc_dataset, "inv_label_map", None),
                "label_policy": label_policy,
            },
        )
        torch.save(phase_payload, os.path.join(phase_dir, "checkpoint.pth"))

        if not local_args.skip_phase_maps:
            phase_classifier_mode = (
                local_args.base_classifier_mode if phase == 0 else local_args.eval_classifier_mode
            )
            phase_semantic_mode = local_args.eval_semantic_mode

            _call_predict_phase_grid_compat(
                model=model,
                dataset_manager=inc_dataset,
                phase=phase,
                target_names=target_names_seq,
                save_dir=phase_dir,
                device=local_args.device,
                patch_size=local_args.patch_size,
                classifier_mode=phase_classifier_mode,
                semantic_mode=phase_semantic_mode,
                class_cmap=local_args.viz_class_cmap,
                confidence_cmap=local_args.viz_confidence_cmap,
                background_color=local_args.viz_background_color,
                save_error_map=local_args.viz_save_error_map,
                save_numpy=local_args.viz_save_numpy,
            )

    elapsed_min = (time.time() - start_time) / 60.0
    print(f"Training done. Time: {elapsed_min:.1f} min")

    final_metrics = evaluator.get_standard_metrics()
    eval_results = evaluate_model(model, inc_dataset, device, local_args)

    # Final report: useful when phase loop was skipped or when final checkpoint differs after post-processing.
    final_report_info = None
    if bool(local_args.save_final_classification_report):
        final_phase = inc_dataset.num_phases - 1
        final_eval_start = time.time()
        final_y_pred, final_y_true = get_phase_predictions(model, inc_dataset, final_phase, device, local_args)
        final_eval_time = time.time() - final_eval_start
        final_seen_classes = inc_dataset.get_classes_up_to_phase(final_phase)
        final_old_class_count = 0 if final_phase == 0 else len(inc_dataset.get_classes_up_to_phase(final_phase - 1))

        final_report_info = save_phase_classification_report(
            evaluator=evaluator,
            y_true=final_y_true,
            y_pred=final_y_pred,
            phase="final",
            phase_dir=run_dir,
            target_names_seq=target_names_seq,
            seen_classes=final_seen_classes,
            old_class_count=final_old_class_count,
            enabled=True,
            tr_time=elapsed_min * 60.0,
            te_time=final_eval_time,
            dl_time=0.0,
        )

    plot_training_history(
        full_history,
        os.path.join(run_dir, "full_training_history.png"),
    )

    if hasattr(evaluator, "phase_history"):
        try:
            from utils.visualize import plot_phase_metric_summary
            plot_phase_metric_summary(
                evaluator.phase_history,
                os.path.join(run_dir, "phase_metric_summary.png"),
            )
        except Exception as e:
            print(f"[Viz] phase metric summary skipped: {e}")

    if not local_args.skip_tsne:
        visualize_features_tsne(
            model,
            inc_dataset,
            inc_dataset.num_phases - 1,
            device,
            os.path.join(run_dir, "FINAL_cumulative_tsne.png"),
            local_args,
            max_samples=local_args.tsne_max_samples,
        )

    final_payload = _build_checkpoint_payload(
        model=model,
        args=local_args,
        extra={
            "eval_results": eval_results,
            "final_metrics": final_metrics,
            "history": full_history,
            "classification_reports": phase_report_paths,
            "final_classification_report": final_report_info,
            "target_names_seq": target_names_seq,
            "target_names_raw": target_names,
            "class_order": getattr(inc_dataset, "class_order", None),
            "label_map": getattr(inc_dataset, "label_map", None),
            "inv_label_map": getattr(inc_dataset, "inv_label_map", None),
            "label_policy": label_policy,
            "evaluator": evaluator.to_dict() if hasattr(evaluator, "to_dict") else None,
        },
    )
    torch.save(final_payload, os.path.join(run_dir, "final_model.pth"))

    report_path = os.path.join(run_dir, f"patch{local_args.patch_size}_PROTOCOL_REPORT.txt")
    write_protocol_report(
        report_path=report_path,
        local_args=local_args,
        args=args,
        run_idx=run_idx,
        final_metrics=final_metrics,
        eval_results=eval_results,
        evaluator=evaluator,
        target_names_seq=target_names_seq,
        label_policy=label_policy,
        phase_report_paths=phase_report_paths,
        final_report_info=final_report_info,
    )

    return {
        "run_idx": run_idx,
        "seed": local_args.seed,
        "run_dir": run_dir,
        "final_metrics": final_metrics,
        "eval_results": eval_results,
        "classification_reports": phase_report_paths,
        "final_classification_report": final_report_info,
    }


def write_protocol_report(
    report_path: str,
    local_args,
    args,
    run_idx: int,
    final_metrics: Dict[str, Any],
    eval_results: Dict[str, Any],
    evaluator,
    target_names_seq: List[str],
    label_policy: Optional[Dict[str, Any]] = None,
    phase_report_paths: Optional[Dict[int, Any]] = None,
    final_report_info: Optional[Dict[str, Any]] = None,
):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    a_last = _metric_get(final_metrics, "A_last (Final Accuracy)", default=0.0)
    a_avg = _metric_get(final_metrics, "A_avg (Avg Accuracy)", "A_avg (Avg Inc Accuracy)", default=0.0)
    f_avg = _metric_get(final_metrics, "F_avg (Avg Forgetting)", default=0.0)
    h_last = _metric_get(final_metrics, "H_last (Final Harmonic Mean)", default=0.0)
    old_last = _metric_get(final_metrics, "Old_last (Final Old Accuracy)", default=0.0)
    new_last = _metric_get(final_metrics, "New_last (Final New Accuracy)", default=0.0)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Geometry-Native NECIL-HSI Report - {local_args.dataset}\n")
        f.write(f"Run: {run_idx + 1}/{args.num_runs} | Seed: {local_args.seed}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n")
        f.write(f"A_last: {a_last:.2f}%\n")
        f.write(f"A_avg:  {a_avg:.2f}%\n")
        f.write(f"F_avg:  {f_avg:.2f}%\n")
        f.write(f"H_last: {h_last:.2f}%\n")
        f.write(f"Old/New: {old_last:.2f}% / {new_last:.2f}%\n")
        f.write("=" * 70 + "\n\n")

        if label_policy is not None:
            f.write("Label Policy:\n")
            f.write(json.dumps(make_json_serializable(label_policy), indent=2) + "\n\n")

        report_keys = [
            "base_classifier_mode",
            "incremental_classifier_mode",
            "eval_classifier_mode",
            "eval_semantic_mode",
            "classifier_use_bias",
            "use_geom_temperature",
            "min_temperature",
            "max_temperature",
            "debias_strength",
            "energy_normalize_by_dim",
            "reliability_energy_weight",
            "volume_energy_weight",
            "max_classifier_bias_abs",
            "max_classifier_debias_abs",
            "allow_legacy_classifier_modes",
            "use_adaptive_fusion",
            "geometry_variance_shrinkage",
            "geometry_max_variance_ratio",
            "geometry_min_reliability",
            "geometry_adjacency_temperature",
            "geometry_energy_temperature",
            "geometry_volume_temperature",
            "use_geometry_risk_controller",
            "adaptive_update_after_validation",
            "risk_replay_alpha",
            "risk_sep_beta",
            "risk_align_gamma",
            "risk_margin_delta",
            "risk_max_replay_weight",
            "risk_max_insert_weight",
            "risk_max_margin",
            "adaptive_replay_min",
            "adaptive_replay_max",
            "adaptive_replay_power",
            "adaptive_margin_strength",
            "refresh_before_validation",
            "validation_refresh_every",
            "best_state_metric",
            "early_stop_metric",
            "early_stop_patience",
            "geometry_calibrate_basis",
            "geometry_calibration_hidden_dim",
            "geometry_calibration_dropout",
            "geometry_calibration_weight",
            "geometry_max_mean_scale",
            "geometry_max_var_scale",
            "geometry_max_basis_scale",
            "synthetic_replay_weight",
            "synthetic_replay_per_class",
            "replay_subspace_scale",
            "replay_residual_scale",
            "replay_geometry_weight",
            "align_mean_weight",
            "align_basis_weight",
            "align_var_weight",
            "insert_weight",
            "insert_margin",
            "old_new_energy_margin",
            "new_volume_weight",
            "new_volume_target",
            "token_loss_weight",
            "token_spectral_weight",
            "token_spatial_weight",
            "token_cross_weight",
            "token_fused_weight",
            "token_match_distance_threshold",
            "token_reliability_threshold",
            "spectral_guidance_weight",
            "band_guidance_weight",
            "base_compact",
            "base_sep",
            "base_ortho",
            "base_margin",
            "base_center_norm",
            "base_radius",
            "calibration_epochs",
            "calibration_lr",
            "calibration_replay_weight",
            "bank_refresh_every",
            "incremental_warmup_epochs",
            "freeze_classifier_during_incremental",
            "freeze_semantic_encoder_during_incremental",
            "unfreeze_last_backbone_during_incremental",
            "freeze_projection_during_incremental",
            "strict_non_exemplar",
            "dropout",
            "semantic_dropout",
            "projection_dropout",
            "backbone_norm",
            "ssm_residual_scale_init",
            "fusion_residual_scale",
            "token_temperature",
            "loss_scale",
            "geom_var_floor",
            "epochs_base",
            "epochs_inc",
            "lr",
            "batch_size",
            "seed",
            "save_classification_report",
            "save_final_classification_report",
            "viz_class_cmap",
            "viz_confidence_cmap",
            "viz_background_color",
            "viz_save_error_map",
            "viz_save_numpy",
        ]

        f.write("Configuration:\n")
        for key in report_keys:
            if hasattr(local_args, key):
                f.write(f"{key}: {getattr(local_args, key)}\n")

        f.write("\nPhase History:\n")
        for p, m in evaluator.phase_history.items():
            f.write(
                f"Phase {p}: "
                f"OA={m.get('overall_accuracy', 0):.2f}%, "
                f"AA={m.get('average_accuracy', 0):.2f}%, "
                f"Old={m.get('old_accuracy', 0):.2f}%, "
                f"New={m.get('new_accuracy', 0):.2f}%, "
                f"H={m.get('harmonic_mean', 0):.2f}%, "
                f"Kappa={m.get('kappa', 0):.2f}%, "
                f"F1={m.get('f1_macro', 0):.2f}%\n"
            )

        if phase_report_paths:
            f.write("\nClassification Report Files:\n")
            for p, info in phase_report_paths.items():
                if not info:
                    continue
                f.write(f"Phase {p}:\n")
                for key in [
                    "txt_path",
                    "json_path",
                    "confusion_matrix_csv_path",
                    "confusion_matrix_npy_path",
                    "per_class_csv_path",
                    "hsi_style_path",
                    # Compatibility with older save_classification_report return keys.
                    "confusion_matrix_path",
                ]:
                    if key in info:
                        f.write(f"  {key}: {info[key]}\n")

        if final_report_info:
            f.write("\nFinal Classification Report Files:\n")
            for key in [
                "txt_path",
                "json_path",
                "confusion_matrix_csv_path",
                "confusion_matrix_npy_path",
                "per_class_csv_path",
                "hsi_style_path",
                "confusion_matrix_path",
            ]:
                if key in final_report_info:
                    f.write(f"  {key}: {final_report_info[key]}\n")

        if hasattr(evaluator, "get_per_class_summary"):
            f.write("\nPer-Class Forgetting Summary:\n")
            for cls, s in evaluator.get_per_class_summary().items():
                name = target_names_seq[cls] if cls < len(target_names_seq) else f"Class {cls}"
                f.write(
                    f"  {cls} ({name}): "
                    f"first={s.get('first', 0):.2f}, "
                    f"best={s.get('best', 0):.2f}, "
                    f"last={s.get('last', 0):.2f}, "
                    f"forget={s.get('forgetting', 0):.2f}\n"
                )

        f.write("\nFinal Per-Class Acc:\n")
        for cls, acc in eval_results.get("per_class_accuracy", {}).items():
            name = target_names_seq[cls] if cls < len(target_names_seq) else f"Class {cls}"
            f.write(f"  {cls} ({name}): {acc:.2f}%\n")

    print(f"[Report] Saved protocol report to: {report_path}")


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    args = validate_args(args)

    seed_list = parse_seed_list(args.seed_list)
    if seed_list is None or len(seed_list) == 0:
        seed_list = [args.seed + i for i in range(args.num_runs)]

    all_run_results = []

    for run_idx in range(args.num_runs):
        result = run_single_experiment(
            args=args,
            run_idx=run_idx,
            run_seed=seed_list[run_idx],
        )
        all_run_results.append(result)

    root_dir = os.path.join(args.save_dir, args.dataset, f"patch_{args.patch_size}")
    os.makedirs(root_dir, exist_ok=True)

    a_last_values = [_metric_get(r["final_metrics"], "A_last (Final Accuracy)", default=0.0) for r in all_run_results]
    a_avg_values = [_metric_get(r["final_metrics"], "A_avg (Avg Accuracy)", "A_avg (Avg Inc Accuracy)", default=0.0) for r in all_run_results]
    f_avg_values = [_metric_get(r["final_metrics"], "F_avg (Avg Forgetting)", default=0.0) for r in all_run_results]
    h_last_values = [_metric_get(r["final_metrics"], "H_last (Final Harmonic Mean)", default=0.0) for r in all_run_results]

    a_last_mean, a_last_std = aggregate_metric(a_last_values)
    a_avg_mean, a_avg_std = aggregate_metric(a_avg_values)
    f_avg_mean, f_avg_std = aggregate_metric(f_avg_values)
    h_last_mean, h_last_std = aggregate_metric(h_last_values)

    summary = {
        "num_runs": args.num_runs,
        "seeds": seed_list,
        "A_last_mean": a_last_mean,
        "A_last_std": a_last_std,
        "A_avg_mean": a_avg_mean,
        "A_avg_std": a_avg_std,
        "F_avg_mean": f_avg_mean,
        "F_avg_std": f_avg_std,
        "H_last_mean": h_last_mean,
        "H_last_std": h_last_std,
        "runs": all_run_results,
    }

    with open(os.path.join(root_dir, "multi_run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(make_json_serializable(summary), f, indent=2)

    report_path = os.path.join(root_dir, "MULTI_RUN_PROTOCOL_REPORT.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Geometry-Native NECIL-HSI Multi-Run Report - {args.dataset}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n")
        f.write(f"Runs: {args.num_runs}\n")
        f.write(f"Seeds: {seed_list}\n")
        f.write("=" * 70 + "\n")
        f.write(f"A_last: {a_last_mean:.2f} ± {a_last_std:.2f}\n")
        f.write(f"A_avg : {a_avg_mean:.2f} ± {a_avg_std:.2f}\n")
        f.write(f"F_avg : {f_avg_mean:.2f} ± {f_avg_std:.2f}\n")
        f.write(f"H_last: {h_last_mean:.2f} ± {h_last_std:.2f}\n")
        f.write("=" * 70 + "\n\n")

        f.write("Per-run results:\n")
        for r in all_run_results:
            fm = r["final_metrics"]
            f.write(
                f"Run {r['run_idx'] + 1} | Seed {r['seed']} | "
                f"A_last={_metric_get(fm, 'A_last (Final Accuracy)', default=0.0):.2f}, "
                f"A_avg={_metric_get(fm, 'A_avg (Avg Accuracy)', 'A_avg (Avg Inc Accuracy)', default=0.0):.2f}, "
                f"F_avg={_metric_get(fm, 'F_avg (Avg Forgetting)', default=0.0):.2f}, "
                f"H_last={_metric_get(fm, 'H_last (Final Harmonic Mean)', default=0.0):.2f}, "
                f"RunDir={r.get('run_dir', '')}\n"
            )

    print("\n=== MULTI-RUN SUMMARY ===")
    print(f"A_last: {a_last_mean:.2f} ± {a_last_std:.2f}")
    print(f"A_avg : {a_avg_mean:.2f} ± {a_avg_std:.2f}")
    print(f"F_avg : {f_avg_mean:.2f} ± {f_avg_std:.2f}")
    print(f"H_last: {h_last_mean:.2f} ± {h_last_std:.2f}")
    print("=========================\n")


if __name__ == "__main__":
    main()






# """
# Main Entry Point for Geometry-Native NECIL-HSI
# ==============================================

# Aligned with:
# - strict non-exemplar IncrementalHSIDataset
# - explicit HSI label policy where class 0 may be a real class
# - reliability-aware low-rank GeometryBank
# - geometry-centric Trainer / TrainerHelper
# - NECILModel with conservative geometry calibration
# - geometry-native classifier
# - visualization/evaluation using semantic_mode='identity'
# - multi-run evaluation with mean ± std
# - per-phase and final classification reports

# Critical policy
# ---------------
# Do not evaluate phase 0 with semantic_mode='all' while incremental phases use
# semantic_mode='identity'. Geometry memory must be built/evaluated in one feature
# manifold. The default is identity for all phases.
# """

# import argparse
# import inspect
# import json
# import os
# import random
# import sys
# import time
# from datetime import datetime
# from typing import Optional, Dict, Any, List

# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# from sklearn.manifold import TSNE

# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# from data.hsi_dataloader_pytorch import LoadHSIData, ImageCubes
# from data.incremental_dataset import IncrementalHSIDataset
# from models.necil_model import NECILModel
# from trainers.trainer import Trainer
# from utils.eval import NECILEvaluator, save_classification_report, make_json_serializable
# from utils.visualize import predict_phase_grid, plot_training_history


# DATASET_INFO = {
#     "IP": {"name": "Indian Pines", "bands": 200, "classes": 16},
#     "SA": {"name": "Salinas", "bands": 204, "classes": 16},
#     "PU": {"name": "Pavia University", "bands": 103, "classes": 9},
#     "PC": {"name": "Pavia Centre", "bands": 102, "classes": 9},
#     "BS": {"name": "Botswana", "bands": 145, "classes": 14},
#     "LK": {"name": "LongKou", "bands": 270, "classes": 9},
#     "HH": {"name": "HongHu", "bands": 270, "classes": 22},
#     "HC": {"name": "HanChuan", "bands": 274, "classes": 16},
#     "UH13": {"name": "Houston 2013", "bands": 144, "classes": 15},
#     "QUH": {"name": "QUH-Qingyun", "bands": 270, "classes": 6},
#     "PI": {"name": "QUH-Pingan", "bands": 270, "classes": 10},
#     "TH": {"name": "QUH-Tangdaowan", "bands": 270, "classes": 18},
# }

# SUPPORTED_CLASSIFIER_MODES = {
#     "geometry_only",
#     "calibrated_geometry",
# }

# SUPPORTED_EVAL_SEMANTIC_MODES = {
#     "identity",
#     "off",
#     "none",
#     "bypass",
#     "raw",
#     # "all" is kept only for explicit ablation, not for main runs.
#     "all",
#     "auto",
# }


# # ============================================================
# # Argument helpers
# # ============================================================
# def str2bool(v):
#     if isinstance(v, bool):
#         return v
#     if v is None:
#         return False
#     return str(v).lower() in ("true", "1", "yes", "y", "t")


# def parse_seed_list(seed_list_str: str):
#     if seed_list_str is None or str(seed_list_str).strip() == "":
#         return None
#     return [int(s.strip()) for s in str(seed_list_str).split(",") if s.strip() != ""]


# def parse_args():
#     parser = argparse.ArgumentParser(description="Geometry-Native NECIL-HSI")

#     # ---------------- Dataset ----------------
#     parser.add_argument("--dataset", type=str, default="IP", choices=DATASET_INFO.keys())
#     parser.add_argument("--data_dir", type=str, default="./datasets")
#     parser.add_argument("--save_dir", type=str, default="./checkpoints")
#     parser.add_argument("--patch_size", type=int, default=11)
#     parser.add_argument("--train_ratio", type=float, default=0.2)
#     parser.add_argument("--val_ratio", type=float, default=0.1)

#     # ---------------- Model ----------------
#     parser.add_argument("--d_model", type=int, default=128)
#     parser.add_argument("--subspace_rank", type=int, default=5)
#     parser.add_argument("--num_concepts_per_class", type=int, default=4)
#     parser.add_argument("--d_state", type=int, default=16)
#     parser.add_argument("--d_conv", type=int, default=4)
#     parser.add_argument("--expand", type=int, default=2)
#     parser.add_argument("--num_spectral_layers", type=int, default=3)
#     parser.add_argument("--num_layers", type=int, default=3)

#     # ---------------- Backbone / projection ----------------
#     parser.add_argument("--dropout", type=float, default=0.1)
#     parser.add_argument("--semantic_dropout", type=float, default=0.1)
#     parser.add_argument("--projection_dropout", type=float, default=0.1)
#     parser.add_argument("--backbone_norm", type=str, default="layer", choices=["layer", "rms"])
#     parser.add_argument("--stem_norm_groups", type=int, default=8)
#     parser.add_argument("--ssm_residual_scale_init", type=float, default=0.7)
#     parser.add_argument("--fusion_residual_scale", type=float, default=0.3)
#     parser.add_argument("--backbone_output_dropout", type=float, default=0.0)

#     # ---------------- Token / semantic ----------------
#     parser.add_argument("--token_temperature", type=float, default=0.07)
#     parser.add_argument("--token_topk_ratio", type=float, default=1.0)
#     parser.add_argument("--token_symmetric_affinity", type=str2bool, default=True)
#     parser.add_argument("--disable_semantic_in_incremental", type=str2bool, default=True)

#     # ---------------- Classifier ----------------
#     parser.add_argument("--cls_temperature", type=float, default=0.07)
#     parser.add_argument("--loss_scale", type=float, default=8.0)
#     parser.add_argument("--geom_var_floor", type=float, default=1e-4)
#     parser.add_argument("--classifier_use_bias", type=str2bool, default=True)
#     parser.add_argument("--use_geom_temperature", type=str2bool, default=True)
#     parser.add_argument("--min_temperature", type=float, default=0.25)
#     parser.add_argument("--max_temperature", type=float, default=4.0)
#     parser.add_argument("--debias_strength", type=float, default=0.10)
#     parser.add_argument("--energy_normalize_by_dim", type=str2bool, default=True)
#     # Accepted by strict classifier variants. Older classifier versions ignore these via getattr.
#     parser.add_argument("--reliability_energy_weight", type=float, default=0.05)
#     parser.add_argument("--volume_energy_weight", type=float, default=0.0)
#     parser.add_argument("--max_classifier_bias_abs", type=float, default=0.50)
#     parser.add_argument("--max_classifier_debias_abs", type=float, default=0.25)
#     parser.add_argument("--allow_legacy_classifier_modes", type=str2bool, default=False)

#     # Legacy compatibility knobs. Keep false for the main method.
#     parser.add_argument("--use_adaptive_fusion", type=str2bool, default=False)
#     parser.add_argument("--init_alpha_old", type=float, default=-0.5)
#     parser.add_argument("--init_alpha_new", type=float, default=-0.2)

#     # ---------------- Geometry bank ----------------
#     parser.add_argument("--geometry_variance_shrinkage", type=float, default=0.10)
#     parser.add_argument("--geometry_max_variance_ratio", type=float, default=50.0)
#     parser.add_argument("--geometry_min_reliability", type=float, default=0.05)
#     # Accepted by newer GeometryBank variants. Older banks ignore these through getattr/defaults.
#     parser.add_argument("--geometry_adjacency_temperature", type=float, default=1.0)
#     parser.add_argument("--geometry_energy_temperature", type=float, default=1.0)
#     parser.add_argument("--geometry_volume_temperature", type=float, default=1.0)

#     # ---------------- Adaptive geometry risk controller ----------------
#     # These flags are parsed and logged for compatibility with adaptive trainers.
#     # If your current trainer does not consume them, they are harmless no-op config fields.
#     parser.add_argument("--use_geometry_risk_controller", type=str2bool, default=False)
#     parser.add_argument("--adaptive_update_after_validation", type=str2bool, default=True)
#     parser.add_argument("--risk_replay_alpha", type=float, default=0.8)
#     parser.add_argument("--risk_sep_beta", type=float, default=1.2)
#     parser.add_argument("--risk_align_gamma", type=float, default=0.8)
#     parser.add_argument("--risk_margin_delta", type=float, default=0.6)
#     parser.add_argument("--risk_max_replay_weight", type=float, default=2.5)
#     parser.add_argument("--risk_max_insert_weight", type=float, default=0.04)
#     parser.add_argument("--risk_max_margin", type=float, default=5.0)
#     parser.add_argument("--adaptive_replay_min", type=int, default=32)
#     parser.add_argument("--adaptive_replay_max", type=int, default=128)
#     parser.add_argument("--adaptive_replay_power", type=float, default=1.0)
#     parser.add_argument("--adaptive_margin_strength", type=float, default=1.0)

#     # ---------------- Validation / checkpoint policy ----------------
#     parser.add_argument("--refresh_before_validation", type=str2bool, default=True)
#     parser.add_argument("--validation_refresh_every", type=int, default=1)
#     parser.add_argument("--best_state_metric", type=str, default="hm", choices=["hm", "h", "harmonic", "acc", "oa", "old", "new"])
#     parser.add_argument("--early_stop_metric", type=str, default="hm", choices=["hm", "h", "harmonic", "acc", "oa", "old", "new"])
#     parser.add_argument("--early_stop_patience", type=int, default=0)

#     # ---------------- Geometry calibration ----------------
#     parser.add_argument("--geometry_calibration_hidden_dim", type=int, default=128)
#     parser.add_argument("--geometry_calibration_dropout", type=float, default=0.1)
#     parser.add_argument("--geometry_calibration_weight", type=float, default=0.03)
#     parser.add_argument("--geometry_calibrate_basis", type=str2bool, default=False)
#     parser.add_argument("--geometry_max_mean_scale", type=float, default=0.10)
#     parser.add_argument("--geometry_max_var_scale", type=float, default=0.10)
#     parser.add_argument("--geometry_max_basis_scale", type=float, default=0.03)

#     # ---------------- Optimization ----------------
#     parser.add_argument("--lr", type=float, default=5e-5)
#     parser.add_argument("--epochs_base", type=int, default=100)
#     parser.add_argument("--epochs_inc", type=int, default=50)
#     parser.add_argument("--batch_size", type=int, default=64)
#     parser.add_argument("--label_smoothing", type=float, default=0.0)

#     # ---------------- Base shaping ----------------
#     parser.add_argument("--base_compact", type=float, default=0.05)
#     parser.add_argument("--base_sep", type=float, default=0.05)
#     parser.add_argument("--base_ortho", type=float, default=0.03)
#     parser.add_argument("--base_margin", type=float, default=1.0)
#     parser.add_argument("--base_center_norm", type=float, default=0.01)
#     parser.add_argument("--base_radius", type=float, default=0.01)

#     # ---------------- Incremental core ----------------
#     parser.add_argument("--synthetic_replay_weight", type=float, default=1.0)
#     parser.add_argument("--synthetic_replay_per_class", type=int, default=32)
#     parser.add_argument("--replay_subspace_scale", type=float, default=0.8)
#     parser.add_argument("--replay_residual_scale", type=float, default=0.25)
#     parser.add_argument("--replay_min_reliability", type=float, default=0.05)
#     parser.add_argument("--replay_geometry_weight", type=float, default=0.05)

#     parser.add_argument("--align_mean_weight", type=float, default=0.05)
#     parser.add_argument("--align_basis_weight", type=float, default=0.02)
#     parser.add_argument("--align_var_weight", type=float, default=0.01)
#     parser.add_argument("--align_spec_weight", type=float, default=0.0)

#     parser.add_argument("--incremental_warmup_epochs", type=int, default=5)
#     parser.add_argument("--bank_refresh_every", type=int, default=1)

#     # ---------------- Geometry separation / compactness ----------------
#     parser.add_argument("--insert_margin", type=float, default=5.0)
#     parser.add_argument("--insert_weight", type=float, default=0.02)
#     parser.add_argument("--old_new_energy_margin", type=float, default=5.0)
#     parser.add_argument("--new_volume_weight", type=float, default=0.001)
#     parser.add_argument("--new_volume_target", type=float, default=2.0)

#     # ---------------- Token/spectral preservation ----------------
#     parser.add_argument("--token_match_distance_threshold", type=float, default=1.5)
#     parser.add_argument("--token_reliability_threshold", type=float, default=0.35)

#     parser.add_argument("--token_loss_weight", type=float, default=0.0)
#     parser.add_argument("--token_spectral_weight", type=float, default=0.25)
#     parser.add_argument("--token_spatial_weight", type=float, default=0.25)
#     parser.add_argument("--token_cross_weight", type=float, default=0.50)
#     parser.add_argument("--token_fused_weight", type=float, default=0.0)

#     parser.add_argument("--spectral_guidance_weight", type=float, default=0.01)
#     parser.add_argument("--band_guidance_weight", type=float, default=0.005)
#     parser.add_argument("--spectral_guidance_band_loss_type", type=str, default="kl")

#     # ---------------- Auxiliary classification shaping ----------------
#     parser.add_argument("--logit_margin_value", type=float, default=0.2)
#     parser.add_argument("--logit_margin_weight", type=float, default=0.02)
#     parser.add_argument("--inc_logit_margin_weight", type=float, default=0.005)

#     parser.add_argument("--concept_sep_weight", type=float, default=0.01)
#     parser.add_argument("--feature_concept_compact_weight", type=float, default=0.03)
#     parser.add_argument("--inc_feature_concept_compact_weight", type=float, default=0.005)
#     parser.add_argument("--concept_sep_max_cosine", type=float, default=0.25)

#     parser.add_argument("--classifier_adaptation_weight", type=float, default=0.0)

#     # ---------------- Incremental setup ----------------
#     parser.add_argument("--base_classes", type=int, default=None)
#     parser.add_argument("--increment", type=int, default=None)
#     parser.add_argument("--min_train_per_class", type=int, default=20)
#     parser.add_argument("--alignment_samples_per_class", type=int, default=8)

#     # ---------------- Calibration ----------------
#     parser.add_argument("--calibration_epochs", type=int, default=3)
#     parser.add_argument("--calibration_lr", type=float, default=1e-4)
#     parser.add_argument("--calibration_replay_weight", type=float, default=1.0)

#     # ---------------- PCA / preprocessing ----------------
#     parser.add_argument("--no_pca", action="store_true")
#     parser.add_argument("--pca_components", type=int, default=30)
#     parser.add_argument("--reduction_method", type=str, default="PCA")

#     # ---------------- Modes ----------------
#     parser.add_argument("--base_classifier_mode", type=str, default="geometry_only")
#     parser.add_argument("--incremental_classifier_mode", type=str, default="calibrated_geometry")
#     parser.add_argument("--eval_classifier_mode", type=str, default="calibrated_geometry")
#     parser.add_argument("--eval_semantic_mode", type=str, default="identity")

#     # ---------------- Freezing / protocol ----------------
#     parser.add_argument("--freeze_classifier_during_incremental", type=str2bool, default=False)
#     parser.add_argument("--freeze_semantic_encoder_during_incremental", type=str2bool, default=True)
#     parser.add_argument("--unfreeze_last_backbone_during_incremental", type=str2bool, default=False)
#     parser.add_argument("--freeze_projection_during_incremental", type=str2bool, default=True)
#     parser.add_argument("--strict_non_exemplar", type=str2bool, default=True)

#     # ---------------- Reproducibility / multi-run ----------------
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--num_runs", type=int, default=1)
#     parser.add_argument("--seed_list", type=str, default="")
#     parser.add_argument("--deterministic", type=str2bool, default=False)

#     # ---------------- Visualization / reporting ----------------
#     parser.add_argument("--skip_tsne", type=str2bool, default=False)
#     parser.add_argument("--tsne_max_samples", type=int, default=1000)
#     parser.add_argument("--skip_phase_maps", type=str2bool, default=False)
#     parser.add_argument("--save_classification_report", type=str2bool, default=True)
#     parser.add_argument("--save_final_classification_report", type=str2bool, default=True)

#     # Phase-map visualization options
#     parser.add_argument("--viz_class_cmap", type=str, default="turbo")
#     parser.add_argument("--viz_confidence_cmap", type=str, default="magma")
#     parser.add_argument("--viz_background_color", type=str, default="#20252B")
#     parser.add_argument("--viz_save_error_map", type=str2bool, default=True)

#     # Save raw visualization arrays. Keep true for experiments, false for paper-only image output.
#     parser.add_argument("--viz_save_numpy", type=str2bool, default=True)

#     # ---------------- System ----------------
#     parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
#     parser.add_argument("--num_workers", type=int, default=0)
#     parser.add_argument("--subspace_extract_batch_size", type=int, default=256)
#     parser.add_argument("--debug_verbose", type=str2bool, default=False)

#     return parser.parse_args()


# def validate_args(args):
#     args.base_classifier_mode = str(args.base_classifier_mode).lower().strip()
#     args.incremental_classifier_mode = str(args.incremental_classifier_mode).lower().strip()
#     args.eval_classifier_mode = str(args.eval_classifier_mode).lower().strip()
#     args.eval_semantic_mode = str(args.eval_semantic_mode).lower().strip()

#     for mode_name, mode_value in [
#         ("base_classifier_mode", args.base_classifier_mode),
#         ("incremental_classifier_mode", args.incremental_classifier_mode),
#         ("eval_classifier_mode", args.eval_classifier_mode),
#     ]:
#         if mode_value not in SUPPORTED_CLASSIFIER_MODES:
#             raise ValueError(
#                 f"Unsupported {mode_name}='{mode_value}'. "
#                 f"Supported: {sorted(SUPPORTED_CLASSIFIER_MODES)}"
#             )

#     if args.eval_semantic_mode not in SUPPORTED_EVAL_SEMANTIC_MODES:
#         raise ValueError(
#             f"Unsupported eval_semantic_mode='{args.eval_semantic_mode}'. "
#             f"Supported: {sorted(SUPPORTED_EVAL_SEMANTIC_MODES)}"
#         )

#     if args.eval_semantic_mode == "all":
#         print(
#             "[WARN] eval_semantic_mode='all' evaluates a different feature path. "
#             "Use only for ablation. Main geometry runs should use identity."
#         )

#     if args.base_classes is not None and args.base_classes <= 0:
#         raise ValueError("--base_classes must be positive.")
#     if args.increment is not None and args.increment <= 0:
#         raise ValueError("--increment must be positive.")
#     if args.pca_components <= 0 and not args.no_pca:
#         raise ValueError("--pca_components must be positive when PCA is enabled.")
#     if args.subspace_rank <= 0:
#         raise ValueError("--subspace_rank must be positive.")
#     if args.num_concepts_per_class <= 0:
#         raise ValueError("--num_concepts_per_class must be positive.")
#     if args.batch_size <= 0:
#         raise ValueError("--batch_size must be positive.")
#     if args.epochs_base <= 0 or args.epochs_inc <= 0:
#         raise ValueError("--epochs_base and --epochs_inc must be positive.")
#     if args.synthetic_replay_weight < 0.0:
#         raise ValueError("--synthetic_replay_weight must be >= 0.")
#     if args.synthetic_replay_per_class < 0:
#         raise ValueError("--synthetic_replay_per_class must be >= 0.")
#     if args.calibration_epochs < 0:
#         raise ValueError("--calibration_epochs must be >= 0.")
#     if args.calibration_replay_weight < 0.0:
#         raise ValueError("--calibration_replay_weight must be >= 0.")
#     if args.geometry_calibration_weight < 0.0:
#         raise ValueError("--geometry_calibration_weight must be >= 0.")
#     if args.num_runs <= 0:
#         raise ValueError("--num_runs must be >= 1.")
#     if not (0.0 < args.token_topk_ratio <= 1.0):
#         raise ValueError("--token_topk_ratio must be in (0, 1].")
#     if args.bank_refresh_every < 0:
#         raise ValueError("--bank_refresh_every must be >= 0.")
#     if args.validation_refresh_every < 0:
#         raise ValueError("--validation_refresh_every must be >= 0.")
#     if args.early_stop_patience < 0:
#         raise ValueError("--early_stop_patience must be >= 0.")
#     if args.adaptive_replay_min < 0 or args.adaptive_replay_max < 0:
#         raise ValueError("--adaptive_replay_min/max must be >= 0.")
#     if args.adaptive_replay_min > args.adaptive_replay_max:
#         raise ValueError("--adaptive_replay_min must be <= --adaptive_replay_max.")
#     if args.risk_max_replay_weight <= 0.0:
#         raise ValueError("--risk_max_replay_weight must be positive.")
#     if args.risk_max_insert_weight < 0.0:
#         raise ValueError("--risk_max_insert_weight must be >= 0.")
#     if args.risk_max_margin <= 0.0:
#         raise ValueError("--risk_max_margin must be positive.")
#     if args.reliability_energy_weight < 0.0 or args.volume_energy_weight < 0.0:
#         raise ValueError("--reliability_energy_weight and --volume_energy_weight must be >= 0.")
#     if args.max_classifier_bias_abs <= 0.0 or args.max_classifier_debias_abs <= 0.0:
#         raise ValueError("--max_classifier_bias_abs and --max_classifier_debias_abs must be positive.")

#     # Validate visualization options before training. Otherwise a bad cmap name
#     # would crash after the expensive experiment finishes.
#     try:
#         plt.get_cmap(str(args.viz_class_cmap))
#     except Exception as e:
#         raise ValueError(f"Invalid --viz_class_cmap='{args.viz_class_cmap}'.") from e

#     try:
#         plt.get_cmap(str(args.viz_confidence_cmap))
#     except Exception as e:
#         raise ValueError(f"Invalid --viz_confidence_cmap='{args.viz_confidence_cmap}'.") from e

#     if not isinstance(args.viz_background_color, str) or len(args.viz_background_color.strip()) == 0:
#         raise ValueError("--viz_background_color must be a non-empty Matplotlib-compatible color string.")

#     seed_list = parse_seed_list(args.seed_list)
#     if seed_list is not None and len(seed_list) > 0 and len(seed_list) != args.num_runs:
#         raise ValueError(
#             f"--seed_list has {len(seed_list)} seeds but --num_runs={args.num_runs}. "
#             f"These must match."
#         )

#     return args


# def set_seed(seed: int, deterministic: bool = False):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)

#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

#     if deterministic:
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#         try:
#             torch.use_deterministic_algorithms(True, warn_only=True)
#         except Exception:
#             pass
#     else:
#         torch.backends.cudnn.benchmark = True


# # ============================================================
# # Model / eval helpers
# # ============================================================
# def _set_model_phase_and_old_count(model, dataset, phase: int):
#     phase = int(phase)
#     old_class_count = 0 if phase == 0 else len(dataset.get_classes_up_to_phase(phase - 1))

#     if hasattr(model, "set_phase"):
#         model.set_phase(phase)
#     else:
#         model.current_phase = phase

#     if hasattr(model, "set_old_class_count"):
#         model.set_old_class_count(old_class_count)
#     else:
#         model.old_class_count = old_class_count

#     return old_class_count


# def _model_forward(model, patches, args, dataset, phase: int):
#     """
#     Evaluation forward pass.

#     Critical:
#     semantic_mode defaults to identity for every phase.
#     """
#     _set_model_phase_and_old_count(model, dataset, phase)

#     if int(phase) == 0:
#         classifier_mode = getattr(args, "base_classifier_mode", "geometry_only")
#     else:
#         classifier_mode = getattr(args, "eval_classifier_mode", "calibrated_geometry")

#     semantic_mode = getattr(args, "eval_semantic_mode", "identity")

#     return model(
#         patches,
#         semantic_mode=semantic_mode,
#         classifier_mode=classifier_mode,
#     )


# def _build_checkpoint_payload(model, args, extra: Optional[dict] = None) -> dict:
#     payload = {
#         "model_state_dict": model.state_dict(),
#         "memory_snapshot": model.export_memory_snapshot() if hasattr(model, "export_memory_snapshot") else None,
#         "args": vars(args),
#         "current_num_classes": int(getattr(model, "current_num_classes", 0)),
#         "old_class_count": int(getattr(model, "old_class_count", 0)),
#         "current_phase": int(getattr(model, "current_phase", 0)),
#     }
#     if extra is not None:
#         payload.update(extra)
#     return payload


# @torch.no_grad()
# def visualize_features_tsne(model, dataset, phase, device, save_path, args, max_samples=1000):
#     model.eval()
#     _set_model_phase_and_old_count(model, dataset, phase)

#     loader = dataset.get_cumulative_dataloader(
#         phase,
#         split="test",
#         batch_size=128,
#         shuffle=True,
#     )

#     all_features, all_labels, seen = [], [], 0

#     for patches, labels in loader:
#         patches = patches.to(device).float()
#         out = _model_forward(model, patches, args, dataset, phase=phase)

#         feats = out["features"].detach().cpu().numpy()
#         labs = labels.numpy()

#         all_features.append(feats)
#         all_labels.append(labs)

#         seen += len(labels)
#         if seen >= max_samples:
#             break

#     if not all_features:
#         return None

#     all_features = np.concatenate(all_features, axis=0)[:max_samples]
#     all_labels = np.concatenate(all_labels, axis=0)[:max_samples]

#     if len(np.unique(all_labels)) < 2 or len(all_features) < 10:
#         return None

#     n_samples = len(all_features)
#     perplexity = min(20, max(5, n_samples // 100))

#     try:
#         tsne = TSNE(
#             n_components=2,
#             perplexity=perplexity,
#             init="pca",
#             random_state=42,
#             learning_rate="auto",
#         )
#         reduced = tsne.fit_transform(all_features)
#     except KeyboardInterrupt:
#         print("[t-SNE] Interrupted by user. Skipping visualization.")
#         return None
#     except Exception as e:
#         print(f"[t-SNE] Skipped due to error: {e}")
#         return None

#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     plt.figure(figsize=(10, 8), facecolor="white")
#     scatter = plt.scatter(
#         reduced[:, 0],
#         reduced[:, 1],
#         c=all_labels,
#         cmap="tab20",
#         alpha=0.7,
#         s=15,
#     )
#     plt.colorbar(scatter, ticks=np.unique(all_labels))
#     plt.title(f"t-SNE Phase {phase}")
#     plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
#     plt.close()
#     return save_path


# @torch.no_grad()
# def get_phase_predictions(model, dataset, phase, device, args, batch_size=128):
#     model.eval()
#     _set_model_phase_and_old_count(model, dataset, phase)

#     loader = dataset.get_cumulative_dataloader(
#         phase, split="test", batch_size=batch_size, shuffle=False
#     )
#     all_preds, all_labels = [], []

#     for patches, labels in loader:
#         patches = patches.to(device).float()
#         out = _model_forward(model, patches, args, dataset, phase=phase)
#         all_preds.append(out["logits"].argmax(dim=1).cpu().numpy())
#         all_labels.append(labels.numpy())

#     if len(all_preds) == 0:
#         return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

#     return np.concatenate(all_preds), np.concatenate(all_labels)


# @torch.no_grad()
# def evaluate_model(model, dataset, device, args, batch_size=128):
#     model.eval()
#     last_phase = dataset.num_phases - 1
#     _set_model_phase_and_old_count(model, dataset, last_phase)

#     loader = dataset.get_cumulative_dataloader(
#         last_phase, split="test", batch_size=batch_size, shuffle=False
#     )
#     all_preds, all_labels = [], []

#     for patches, labels in loader:
#         patches = patches.to(device).float()
#         out = _model_forward(model, patches, args, dataset, phase=last_phase)
#         all_preds.append(out["logits"].argmax(dim=1).cpu().numpy())
#         all_labels.append(labels.numpy())

#     if len(all_preds) == 0:
#         return {"overall_accuracy": 0.0, "per_class_accuracy": {}}

#     all_preds = np.concatenate(all_preds)
#     all_labels = np.concatenate(all_labels)

#     overall_acc = 100.0 * (all_preds == all_labels).mean()

#     per_class_acc = {}
#     for cls in np.unique(all_labels):
#         mask = all_labels == cls
#         per_class_acc[int(cls)] = (
#             100.0 * (all_preds[mask] == cls).mean() if mask.sum() > 0 else 0.0
#         )

#     return {
#         "overall_accuracy": float(overall_acc),
#         "per_class_accuracy": per_class_acc,
#     }


# def save_phase_classification_report(
#     *,
#     evaluator,
#     y_true,
#     y_pred,
#     phase,
#     phase_dir,
#     target_names_seq,
#     seen_classes,
#     old_class_count,
#     enabled=True,
#     tr_time=None,
#     te_time=None,
#     dl_time=0.0,
# ):
#     """
#     Centralized report saving.

#     Saves both reports when the updated utils/eval.py is installed:
#     - structured report files for debugging/aggregation
#     - HSI-style classification report matching common HSI report format

#     Critical fix:
#     never hard-code FINAL_HSI_Classification_Report.csv inside the per-phase
#     report path. Each phase writes into its own phase directory.
#     """
#     if not enabled:
#         return None

#     os.makedirs(phase_dir, exist_ok=True)

#     if hasattr(evaluator, "save_phase_report"):
#         return evaluator.save_phase_report(
#             phase=phase,
#             y_true=y_true,
#             y_pred=y_pred,
#             target_names=target_names_seq,
#             save_dir=phase_dir,
#             seen_classes=seen_classes,
#             old_class_count=old_class_count,
#             tr_time=tr_time,
#             te_time=te_time,
#             dl_time=dl_time,
#         )

#     return save_classification_report(
#         y_true=y_true,
#         y_pred=y_pred,
#         target_names=target_names_seq,
#         save_dir=phase_dir,
#         phase=phase,
#         seen_classes=seen_classes,
#         old_class_count=old_class_count,
#         tr_time=tr_time,
#         te_time=te_time,
#         dl_time=dl_time,
#         save_hsi_style=True,
#         save_structured=True,
#     )


# # ============================================================
# # Dataset builder
# # ============================================================
# def build_incremental_dataset(args, patches, labels, coords, gt_shape, gt_map, target_names=None, label_policy=None):
#     base_kwargs = dict(
#         patches=patches,
#         labels=labels,
#         coords=coords,
#         gt_shape=gt_shape,
#         GT=gt_map,
#         base_classes=args.base_classes,
#         increment=args.increment,
#         train_ratio=args.train_ratio,
#         val_ratio=args.val_ratio,
#         seed=args.seed,
#         device=args.device,
#         min_train_per_class=args.min_train_per_class,
#         strict_non_exemplar=args.strict_non_exemplar,
#     )

#     sig = inspect.signature(IncrementalHSIDataset.__init__)

#     optional = {
#         "num_workers": args.num_workers,
#         "target_names": target_names,
#         "label_policy": label_policy,
#     }

#     for key, value in optional.items():
#         if key in sig.parameters:
#             base_kwargs[key] = value

#     return IncrementalHSIDataset(**base_kwargs)


# def evaluator_update_compat(evaluator, phase, y_true, y_pred, old_class_count, seen_classes=None):
#     sig = inspect.signature(evaluator.update)
#     kwargs = {}

#     if "old_class_count" in sig.parameters:
#         kwargs["old_class_count"] = old_class_count
#     if "seen_classes" in sig.parameters:
#         kwargs["seen_classes"] = seen_classes

#     evaluator.update(phase, y_true, y_pred, **kwargs)


# def save_run_config(args, save_root):
#     os.makedirs(save_root, exist_ok=True)
#     config_path = os.path.join(save_root, "run_config.json")
#     with open(config_path, "w", encoding="utf-8") as f:
#         json.dump(vars(args), f, indent=2)
#     return config_path


# def aggregate_metric(metric_list):
#     arr = np.asarray(metric_list, dtype=np.float64)
#     return float(arr.mean()), float(arr.std(ddof=0))


# def _metric_get(metrics: Dict[str, Any], *keys, default=0.0):
#     for k in keys:
#         if k in metrics:
#             return metrics[k]
#     return default


# # ============================================================
# # Experiment
# # ============================================================
# def run_single_experiment(args, run_idx: int, run_seed: int):
#     local_args = argparse.Namespace(**vars(args))
#     local_args.seed = int(run_seed)

#     set_seed(local_args.seed, deterministic=local_args.deterministic)
#     device = torch.device(local_args.device)

#     print("\n=== GEOMETRY-NATIVE NECIL-HSI PIPELINE ===")
#     print(f"Run: {run_idx + 1}/{args.num_runs} | Seed: {local_args.seed}")
#     print(f"Device: {device} | Dataset: {local_args.dataset}")
#     print(f"Replay: {local_args.synthetic_replay_weight} / {local_args.synthetic_replay_per_class}")
#     print(
#         f"Alignment: mean={local_args.align_mean_weight}, "
#         f"basis={local_args.align_basis_weight}, var={local_args.align_var_weight}"
#     )
#     print(
#         f"Structure: token={local_args.token_loss_weight}, "
#         f"spectral={local_args.spectral_guidance_weight}, "
#         f"band={local_args.band_guidance_weight}"
#     )
#     print(
#         f"Geometry calibration: weight={local_args.geometry_calibration_weight}, "
#         f"basis={local_args.geometry_calibrate_basis}, "
#         f"hidden={local_args.geometry_calibration_hidden_dim}, "
#         f"dropout={local_args.geometry_calibration_dropout}"
#     )
#     print(
#         f"Modes: base={local_args.base_classifier_mode}, "
#         f"inc={local_args.incremental_classifier_mode}, "
#         f"eval={local_args.eval_classifier_mode}, "
#         f"semantic={local_args.eval_semantic_mode}"
#     )
#     print(
#         f"Freeze: classifier={local_args.freeze_classifier_during_incremental}, "
#         f"semantic={local_args.freeze_semantic_encoder_during_incremental}, "
#         f"last_backbone={local_args.unfreeze_last_backbone_during_incremental}, "
#         f"projection={local_args.freeze_projection_during_incremental}"
#     )
#     print(f"Strict non-exemplar: {local_args.strict_non_exemplar}")
#     print(
#         f"Reports: phase={local_args.save_classification_report}, "
#         f"final={local_args.save_final_classification_report}"
#     )
#     print(
#         f"Validation/checkpoint: refresh_before_val={local_args.refresh_before_validation}, "
#         f"val_refresh_every={local_args.validation_refresh_every}, "
#         f"best_metric={local_args.best_state_metric}, "
#         f"early_stop_patience={local_args.early_stop_patience}"
#     )
#     print(
#         f"Risk controller: enabled={local_args.use_geometry_risk_controller}, "
#         f"adaptive_update_after_val={local_args.adaptive_update_after_validation}, "
#         f"replay_alpha={local_args.risk_replay_alpha}, sep_beta={local_args.risk_sep_beta}"
#     )
#     print(
#         f"Visualization: class_cmap={local_args.viz_class_cmap}, "
#         f"confidence_cmap={local_args.viz_confidence_cmap}, "
#         f"background={local_args.viz_background_color}, "
#         f"save_error_map={local_args.viz_save_error_map}, "
#         f"save_numpy={local_args.viz_save_numpy}"
#     )
#     if not local_args.freeze_projection_during_incremental:
#         print(
#             "[WARN] freeze_projection_during_incremental=false. "
#             "For feature-space geometry replay this can cause projection drift and old-class collapse."
#         )
#     print("================================================\n")

#     apply_reduction = (not local_args.no_pca) and (local_args.reduction_method.lower() != "none")

#     # Updated loader may return label_policy when requested. Fall back for older loader.
#     try:
#         load_out = LoadHSIData(
#             method=local_args.dataset,
#             base_dir=local_args.data_dir,
#             apply_reduction=apply_reduction,
#             n_components=local_args.pca_components,
#             reduction_method=local_args.reduction_method,
#             return_label_policy=True,
#         )
#         hsi, gt, num_classes, target_names, has_bg, label_policy = load_out
#     except TypeError:
#         hsi, gt, num_classes, target_names, has_bg = LoadHSIData(
#             method=local_args.dataset,
#             base_dir=local_args.data_dir,
#             apply_reduction=apply_reduction,
#             n_components=local_args.pca_components,
#             reduction_method=local_args.reduction_method,
#         )
#         label_policy = None

#     try:
#         patches, labels, coords = ImageCubes(
#             HSI=hsi,
#             GT=gt,
#             WS=local_args.patch_size,
#             removeZeroLabels=True,
#             has_background=has_bg,
#             num_classes=num_classes,
#             pytorch_format=True,
#             label_policy=label_policy,
#         )
#     except TypeError:
#         patches, labels, coords = ImageCubes(
#             HSI=hsi,
#             GT=gt,
#             WS=local_args.patch_size,
#             removeZeroLabels=True,
#             has_background=has_bg,
#             num_classes=num_classes,
#             pytorch_format=True,
#         )

#     local_args.num_bands = int(patches.shape[1])
#     local_args.max_classes = int(num_classes)

#     if local_args.base_classes is None:
#         local_args.base_classes = 4 if local_args.dataset in {"IP", "HC"} else max(2, num_classes // 2)

#     if local_args.increment is None:
#         remaining = max(1, num_classes - local_args.base_classes)
#         local_args.increment = 3 if remaining >= 3 else 1

#     if local_args.base_classes >= num_classes:
#         raise ValueError(
#             f"base_classes={local_args.base_classes} must be < total classes={num_classes}"
#         )

#     # Keep raw GT only for map shape/reference. Training labels already come
#     # from ImageCubes and are sequential 0..K-1.
#     gt_for_dataset = gt.copy().astype(np.int64)

#     inc_dataset = build_incremental_dataset(
#         local_args,
#         patches,
#         labels,
#         coords,
#         gt.shape,
#         gt_for_dataset,
#         target_names=target_names,
#         label_policy=label_policy,
#     )

#     if hasattr(inc_dataset, "inv_label_map"):
#         target_names_seq = []
#         for sid in range(inc_dataset.num_classes):
#             input_label = inc_dataset.inv_label_map[sid]
#             if int(input_label) < len(target_names):
#                 target_names_seq.append(target_names[int(input_label)])
#             else:
#                 target_names_seq.append(f"Class {sid}")
#     else:
#         target_names_seq = list(target_names)

#     inc_dataset.target_names = target_names_seq

#     # Hard guard for class-0-real datasets.
#     if label_policy is not None and not bool(label_policy.get("has_background", True)):
#         if 0 in label_policy.get("raw_class_values", []) and 0 not in np.unique(labels):
#             raise RuntimeError(
#                 "Label policy says raw class 0 is real, but label 0 is missing after ImageCubes. "
#                 "The loader is still treating class 0 as background."
#             )

#     run_dir = os.path.join(
#         local_args.save_dir,
#         local_args.dataset,
#         f"patch_{local_args.patch_size}",
#         f"run_{run_idx + 1}_seed_{local_args.seed}",
#     )
#     os.makedirs(run_dir, exist_ok=True)

#     # Expose the active run directory to Trainer. Main still owns normal checkpoint
#     # saving, but this prevents fallback/manual Trainer checkpoints from going to
#     # ./checkpoints/<dataset>/phase_X.
#     local_args.run_dir = run_dir
#     save_run_config(local_args, run_dir)
#     print(f"Run directory: {run_dir}")

#     model = NECILModel(local_args).to(device)
#     trainer = Trainer(model, inc_dataset, local_args)
#     evaluator = NECILEvaluator()

#     full_history = {
#         "train_loss": [],
#         "train_acc": [],
#         "val_loss": [],
#         "val_acc": [],
#         "val_old_acc": [],
#         "val_new_acc": [],
#         "val_hm": [],
#         "phase_boundaries": [],
#     }

#     phase_report_paths = {}
#     start_time = time.time()

#     for phase in range(inc_dataset.num_phases):
#         _set_model_phase_and_old_count(model, inc_dataset, phase)

#         epochs = local_args.epochs_base if phase == 0 else local_args.epochs_inc
#         full_history["phase_boundaries"].append(len(full_history["train_loss"]))

#         phase_train_start = time.time()
#         phase_history = trainer.train_phase(
#             phase=phase,
#             epochs=epochs,
#             batch_size=local_args.batch_size,
#             lr=local_args.lr,
#         )
#         phase_train_time = time.time() - phase_train_start

#         if isinstance(phase_history, dict):
#             for key in [
#                 "train_loss",
#                 "train_acc",
#                 "val_loss",
#                 "val_acc",
#                 "val_old_acc",
#                 "val_new_acc",
#                 "val_hm",
#             ]:
#                 if key in phase_history:
#                     full_history.setdefault(key, [])
#                     full_history[key].extend(phase_history[key])

#         print(f"\n[Eval] Phase {phase}")
#         phase_eval_start = time.time()
#         y_pred, y_true = get_phase_predictions(model, inc_dataset, phase, device, local_args)
#         phase_eval_time = time.time() - phase_eval_start
#         old_class_count = 0 if phase == 0 else len(inc_dataset.get_classes_up_to_phase(phase - 1))
#         seen_classes = inc_dataset.get_classes_up_to_phase(phase)

#         evaluator_update_compat(
#             evaluator,
#             phase,
#             y_true,
#             y_pred,
#             old_class_count,
#             seen_classes=seen_classes,
#         )
#         evaluator.print_summary()

#         phase_dir = os.path.join(run_dir, f"phase_{phase}")
#         os.makedirs(phase_dir, exist_ok=True)

#         report_info = save_phase_classification_report(
#             evaluator=evaluator,
#             y_true=y_true,
#             y_pred=y_pred,
#             phase=phase,
#             phase_dir=phase_dir,
#             target_names_seq=target_names_seq,
#             seen_classes=seen_classes,
#             old_class_count=old_class_count,
#             enabled=bool(local_args.save_classification_report),
#             tr_time=phase_train_time,
#             te_time=phase_eval_time,
#             dl_time=0.0,
#         )
#         phase_report_paths[int(phase)] = report_info

#         phase_payload = _build_checkpoint_payload(
#             model=model,
#             args=local_args,
#             extra={
#                 "phase": int(phase),
#                 "metrics": evaluator.phase_history.get(phase, {}),
#                 "history": phase_history if isinstance(phase_history, dict) else None,
#                 "classification_report": report_info,
#                 "target_names_seq": target_names_seq,
#                 "target_names_raw": target_names,
#                 "class_order": getattr(inc_dataset, "class_order", None),
#                 "label_map": getattr(inc_dataset, "label_map", None),
#                 "inv_label_map": getattr(inc_dataset, "inv_label_map", None),
#                 "label_policy": label_policy,
#             },
#         )
#         torch.save(phase_payload, os.path.join(phase_dir, "checkpoint.pth"))

#         if not local_args.skip_phase_maps:
#             phase_classifier_mode = (
#                 local_args.base_classifier_mode if phase == 0 else local_args.eval_classifier_mode
#             )
#             phase_semantic_mode = local_args.eval_semantic_mode
#             predict_phase_grid(
#                 model=model,
#                 dataset_manager=inc_dataset,
#                 phase=phase,
#                 target_names=target_names_seq,
#                 save_dir=phase_dir,
#                 device=local_args.device,
#                 patch_size=local_args.patch_size,
#                 classifier_mode=phase_classifier_mode,
#                 semantic_mode=phase_semantic_mode,
#                 class_cmap=local_args.viz_class_cmap,
#                 confidence_cmap=local_args.viz_confidence_cmap,
#                 background_color=local_args.viz_background_color,
#                 save_error_map=local_args.viz_save_error_map,
#                 save_numpy=local_args.viz_save_numpy,
#             )

#     elapsed_min = (time.time() - start_time) / 60.0
#     print(f"Training done. Time: {elapsed_min:.1f} min")

#     final_metrics = evaluator.get_standard_metrics()
#     eval_results = evaluate_model(model, inc_dataset, device, local_args)

#     # Final report: useful when phase loop was skipped or when final checkpoint differs after post-processing.
#     final_report_info = None
#     if bool(local_args.save_final_classification_report):
#         final_phase = inc_dataset.num_phases - 1
#         final_eval_start = time.time()
#         final_y_pred, final_y_true = get_phase_predictions(model, inc_dataset, final_phase, device, local_args)
#         final_eval_time = time.time() - final_eval_start
#         final_seen_classes = inc_dataset.get_classes_up_to_phase(final_phase)
#         final_old_class_count = 0 if final_phase == 0 else len(inc_dataset.get_classes_up_to_phase(final_phase - 1))

#         final_report_info = save_phase_classification_report(
#             evaluator=evaluator,
#             y_true=final_y_true,
#             y_pred=final_y_pred,
#             phase="final",
#             phase_dir=run_dir,
#             target_names_seq=target_names_seq,
#             seen_classes=final_seen_classes,
#             old_class_count=final_old_class_count,
#             enabled=True,
#             tr_time=elapsed_min * 60.0,
#             te_time=final_eval_time,
#             dl_time=0.0,
#         )

#     plot_training_history(
#         full_history,
#         os.path.join(run_dir, "full_training_history.png"),
#     )

#     if hasattr(evaluator, "phase_history"):
#         try:
#             from utils.visualize import plot_phase_metric_summary
#             plot_phase_metric_summary(
#                 evaluator.phase_history,
#                 os.path.join(run_dir, "phase_metric_summary.png"),
#             )
#         except Exception as e:
#             print(f"[Viz] phase metric summary skipped: {e}")

#     if not local_args.skip_tsne:
#         visualize_features_tsne(
#             model,
#             inc_dataset,
#             inc_dataset.num_phases - 1,
#             device,
#             os.path.join(run_dir, "FINAL_cumulative_tsne.png"),
#             local_args,
#             max_samples=local_args.tsne_max_samples,
#         )

#     final_payload = _build_checkpoint_payload(
#         model=model,
#         args=local_args,
#         extra={
#             "eval_results": eval_results,
#             "final_metrics": final_metrics,
#             "history": full_history,
#             "classification_reports": phase_report_paths,
#             "final_classification_report": final_report_info,
#             "target_names_seq": target_names_seq,
#             "target_names_raw": target_names,
#             "class_order": getattr(inc_dataset, "class_order", None),
#             "label_map": getattr(inc_dataset, "label_map", None),
#             "inv_label_map": getattr(inc_dataset, "inv_label_map", None),
#             "label_policy": label_policy,
#             "evaluator": evaluator.to_dict() if hasattr(evaluator, "to_dict") else None,
#         },
#     )
#     torch.save(final_payload, os.path.join(run_dir, "final_model.pth"))

#     report_path = os.path.join(run_dir, f"patch{local_args.patch_size}_PROTOCOL_REPORT.txt")
#     write_protocol_report(
#         report_path=report_path,
#         local_args=local_args,
#         args=args,
#         run_idx=run_idx,
#         final_metrics=final_metrics,
#         eval_results=eval_results,
#         evaluator=evaluator,
#         target_names_seq=target_names_seq,
#         label_policy=label_policy,
#         phase_report_paths=phase_report_paths,
#         final_report_info=final_report_info,
#     )

#     return {
#         "run_idx": run_idx,
#         "seed": local_args.seed,
#         "run_dir": run_dir,
#         "final_metrics": final_metrics,
#         "eval_results": eval_results,
#         "classification_reports": phase_report_paths,
#         "final_classification_report": final_report_info,
#     }


# def write_protocol_report(
#     report_path: str,
#     local_args,
#     args,
#     run_idx: int,
#     final_metrics: Dict[str, Any],
#     eval_results: Dict[str, Any],
#     evaluator,
#     target_names_seq: List[str],
#     label_policy: Optional[Dict[str, Any]] = None,
#     phase_report_paths: Optional[Dict[int, Any]] = None,
#     final_report_info: Optional[Dict[str, Any]] = None,
# ):
#     os.makedirs(os.path.dirname(report_path), exist_ok=True)

#     a_last = _metric_get(final_metrics, "A_last (Final Accuracy)", default=0.0)
#     a_avg = _metric_get(final_metrics, "A_avg (Avg Accuracy)", "A_avg (Avg Inc Accuracy)", default=0.0)
#     f_avg = _metric_get(final_metrics, "F_avg (Avg Forgetting)", default=0.0)
#     h_last = _metric_get(final_metrics, "H_last (Final Harmonic Mean)", default=0.0)
#     old_last = _metric_get(final_metrics, "Old_last (Final Old Accuracy)", default=0.0)
#     new_last = _metric_get(final_metrics, "New_last (Final New Accuracy)", default=0.0)

#     with open(report_path, "w", encoding="utf-8") as f:
#         f.write(f"Geometry-Native NECIL-HSI Report - {local_args.dataset}\n")
#         f.write(f"Run: {run_idx + 1}/{args.num_runs} | Seed: {local_args.seed}\n")
#         f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#         f.write("=" * 70 + "\n")
#         f.write(f"A_last: {a_last:.2f}%\n")
#         f.write(f"A_avg:  {a_avg:.2f}%\n")
#         f.write(f"F_avg:  {f_avg:.2f}%\n")
#         f.write(f"H_last: {h_last:.2f}%\n")
#         f.write(f"Old/New: {old_last:.2f}% / {new_last:.2f}%\n")
#         f.write("=" * 70 + "\n\n")

#         if label_policy is not None:
#             f.write("Label Policy:\n")
#             f.write(json.dumps(make_json_serializable(label_policy), indent=2) + "\n\n")

#         report_keys = [
#             "base_classifier_mode",
#             "incremental_classifier_mode",
#             "eval_classifier_mode",
#             "eval_semantic_mode",
#             "classifier_use_bias",
#             "use_geom_temperature",
#             "min_temperature",
#             "max_temperature",
#             "debias_strength",
#             "energy_normalize_by_dim",
#             "reliability_energy_weight",
#             "volume_energy_weight",
#             "max_classifier_bias_abs",
#             "max_classifier_debias_abs",
#             "allow_legacy_classifier_modes",
#             "use_adaptive_fusion",
#             "geometry_variance_shrinkage",
#             "geometry_max_variance_ratio",
#             "geometry_min_reliability",
#             "geometry_adjacency_temperature",
#             "geometry_energy_temperature",
#             "geometry_volume_temperature",
#             "use_geometry_risk_controller",
#             "adaptive_update_after_validation",
#             "risk_replay_alpha",
#             "risk_sep_beta",
#             "risk_align_gamma",
#             "risk_margin_delta",
#             "risk_max_replay_weight",
#             "risk_max_insert_weight",
#             "risk_max_margin",
#             "adaptive_replay_min",
#             "adaptive_replay_max",
#             "adaptive_replay_power",
#             "adaptive_margin_strength",
#             "refresh_before_validation",
#             "validation_refresh_every",
#             "best_state_metric",
#             "early_stop_metric",
#             "early_stop_patience",
#             "geometry_calibrate_basis",
#             "geometry_calibration_hidden_dim",
#             "geometry_calibration_dropout",
#             "geometry_calibration_weight",
#             "geometry_max_mean_scale",
#             "geometry_max_var_scale",
#             "geometry_max_basis_scale",
#             "synthetic_replay_weight",
#             "synthetic_replay_per_class",
#             "replay_subspace_scale",
#             "replay_residual_scale",
#             "replay_geometry_weight",
#             "align_mean_weight",
#             "align_basis_weight",
#             "align_var_weight",
#             "insert_weight",
#             "insert_margin",
#             "old_new_energy_margin",
#             "new_volume_weight",
#             "new_volume_target",
#             "token_loss_weight",
#             "token_spectral_weight",
#             "token_spatial_weight",
#             "token_cross_weight",
#             "token_fused_weight",
#             "token_match_distance_threshold",
#             "token_reliability_threshold",
#             "spectral_guidance_weight",
#             "band_guidance_weight",
#             "base_compact",
#             "base_sep",
#             "base_ortho",
#             "base_margin",
#             "base_center_norm",
#             "base_radius",
#             "calibration_epochs",
#             "calibration_lr",
#             "calibration_replay_weight",
#             "bank_refresh_every",
#             "incremental_warmup_epochs",
#             "freeze_classifier_during_incremental",
#             "freeze_semantic_encoder_during_incremental",
#             "unfreeze_last_backbone_during_incremental",
#             "freeze_projection_during_incremental",
#             "strict_non_exemplar",
#             "dropout",
#             "semantic_dropout",
#             "projection_dropout",
#             "backbone_norm",
#             "ssm_residual_scale_init",
#             "fusion_residual_scale",
#             "token_temperature",
#             "loss_scale",
#             "geom_var_floor",
#             "epochs_base",
#             "epochs_inc",
#             "lr",
#             "batch_size",
#             "seed",
#             "save_classification_report",
#             "save_final_classification_report",
#             "viz_class_cmap",
#             "viz_confidence_cmap",
#             "viz_background_color",
#             "viz_save_error_map",
#             "viz_save_numpy",
#         ]

#         f.write("Configuration:\n")
#         for key in report_keys:
#             if hasattr(local_args, key):
#                 f.write(f"{key}: {getattr(local_args, key)}\n")

#         f.write("\nPhase History:\n")
#         for p, m in evaluator.phase_history.items():
#             f.write(
#                 f"Phase {p}: "
#                 f"OA={m.get('overall_accuracy', 0):.2f}%, "
#                 f"AA={m.get('average_accuracy', 0):.2f}%, "
#                 f"Old={m.get('old_accuracy', 0):.2f}%, "
#                 f"New={m.get('new_accuracy', 0):.2f}%, "
#                 f"H={m.get('harmonic_mean', 0):.2f}%, "
#                 f"Kappa={m.get('kappa', 0):.2f}%, "
#                 f"F1={m.get('f1_macro', 0):.2f}%\n"
#             )

#         if phase_report_paths:
#             f.write("\nClassification Report Files:\n")
#             for p, info in phase_report_paths.items():
#                 if not info:
#                     continue
#                 f.write(f"Phase {p}:\n")
#                 for key in [
#                     "txt_path",
#                     "json_path",
#                     "confusion_matrix_csv_path",
#                     "confusion_matrix_npy_path",
#                     "per_class_csv_path",
#                     "hsi_style_path",
#                     # Compatibility with older save_classification_report return keys.
#                     "confusion_matrix_path",
#                 ]:
#                     if key in info:
#                         f.write(f"  {key}: {info[key]}\n")

#         if final_report_info:
#             f.write("\nFinal Classification Report Files:\n")
#             for key in [
#                 "txt_path",
#                 "json_path",
#                 "confusion_matrix_csv_path",
#                 "confusion_matrix_npy_path",
#                 "per_class_csv_path",
#                 "hsi_style_path",
#                 "confusion_matrix_path",
#             ]:
#                 if key in final_report_info:
#                     f.write(f"  {key}: {final_report_info[key]}\n")

#         if hasattr(evaluator, "get_per_class_summary"):
#             f.write("\nPer-Class Forgetting Summary:\n")
#             for cls, s in evaluator.get_per_class_summary().items():
#                 name = target_names_seq[cls] if cls < len(target_names_seq) else f"Class {cls}"
#                 f.write(
#                     f"  {cls} ({name}): "
#                     f"first={s.get('first', 0):.2f}, "
#                     f"best={s.get('best', 0):.2f}, "
#                     f"last={s.get('last', 0):.2f}, "
#                     f"forget={s.get('forgetting', 0):.2f}\n"
#                 )

#         f.write("\nFinal Per-Class Acc:\n")
#         for cls, acc in eval_results.get("per_class_accuracy", {}).items():
#             name = target_names_seq[cls] if cls < len(target_names_seq) else f"Class {cls}"
#             f.write(f"  {cls} ({name}): {acc:.2f}%\n")

#     print(f"[Report] Saved protocol report to: {report_path}")


# # ============================================================
# # Main
# # ============================================================
# def main():
#     args = parse_args()
#     args = validate_args(args)

#     seed_list = parse_seed_list(args.seed_list)
#     if seed_list is None or len(seed_list) == 0:
#         seed_list = [args.seed + i for i in range(args.num_runs)]

#     all_run_results = []

#     for run_idx in range(args.num_runs):
#         result = run_single_experiment(
#             args=args,
#             run_idx=run_idx,
#             run_seed=seed_list[run_idx],
#         )
#         all_run_results.append(result)

#     root_dir = os.path.join(args.save_dir, args.dataset, f"patch_{args.patch_size}")
#     os.makedirs(root_dir, exist_ok=True)

#     a_last_values = [_metric_get(r["final_metrics"], "A_last (Final Accuracy)", default=0.0) for r in all_run_results]
#     a_avg_values = [_metric_get(r["final_metrics"], "A_avg (Avg Accuracy)", "A_avg (Avg Inc Accuracy)", default=0.0) for r in all_run_results]
#     f_avg_values = [_metric_get(r["final_metrics"], "F_avg (Avg Forgetting)", default=0.0) for r in all_run_results]
#     h_last_values = [_metric_get(r["final_metrics"], "H_last (Final Harmonic Mean)", default=0.0) for r in all_run_results]

#     a_last_mean, a_last_std = aggregate_metric(a_last_values)
#     a_avg_mean, a_avg_std = aggregate_metric(a_avg_values)
#     f_avg_mean, f_avg_std = aggregate_metric(f_avg_values)
#     h_last_mean, h_last_std = aggregate_metric(h_last_values)

#     summary = {
#         "num_runs": args.num_runs,
#         "seeds": seed_list,
#         "A_last_mean": a_last_mean,
#         "A_last_std": a_last_std,
#         "A_avg_mean": a_avg_mean,
#         "A_avg_std": a_avg_std,
#         "F_avg_mean": f_avg_mean,
#         "F_avg_std": f_avg_std,
#         "H_last_mean": h_last_mean,
#         "H_last_std": h_last_std,
#         "runs": all_run_results,
#     }

#     with open(os.path.join(root_dir, "multi_run_summary.json"), "w", encoding="utf-8") as f:
#         json.dump(make_json_serializable(summary), f, indent=2)

#     report_path = os.path.join(root_dir, "MULTI_RUN_PROTOCOL_REPORT.txt")
#     with open(report_path, "w", encoding="utf-8") as f:
#         f.write(f"Geometry-Native NECIL-HSI Multi-Run Report - {args.dataset}\n")
#         f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#         f.write("=" * 70 + "\n")
#         f.write(f"Runs: {args.num_runs}\n")
#         f.write(f"Seeds: {seed_list}\n")
#         f.write("=" * 70 + "\n")
#         f.write(f"A_last: {a_last_mean:.2f} ± {a_last_std:.2f}\n")
#         f.write(f"A_avg : {a_avg_mean:.2f} ± {a_avg_std:.2f}\n")
#         f.write(f"F_avg : {f_avg_mean:.2f} ± {f_avg_std:.2f}\n")
#         f.write(f"H_last: {h_last_mean:.2f} ± {h_last_std:.2f}\n")
#         f.write("=" * 70 + "\n\n")

#         f.write("Per-run results:\n")
#         for r in all_run_results:
#             fm = r["final_metrics"]
#             f.write(
#                 f"Run {r['run_idx'] + 1} | Seed {r['seed']} | "
#                 f"A_last={_metric_get(fm, 'A_last (Final Accuracy)', default=0.0):.2f}, "
#                 f"A_avg={_metric_get(fm, 'A_avg (Avg Accuracy)', 'A_avg (Avg Inc Accuracy)', default=0.0):.2f}, "
#                 f"F_avg={_metric_get(fm, 'F_avg (Avg Forgetting)', default=0.0):.2f}, "
#                 f"H_last={_metric_get(fm, 'H_last (Final Harmonic Mean)', default=0.0):.2f}, "
#                 f"RunDir={r.get('run_dir', '')}\n"
#             )

#     print("\n=== MULTI-RUN SUMMARY ===")
#     print(f"A_last: {a_last_mean:.2f} ± {a_last_std:.2f}")
#     print(f"A_avg : {a_avg_mean:.2f} ± {a_avg_std:.2f}")
#     print(f"F_avg : {f_avg_mean:.2f} ± {f_avg_std:.2f}")
#     print(f"H_last: {h_last_mean:.2f} ± {h_last_std:.2f}")
#     print("=========================\n")


# if __name__ == "__main__":
#     main()


