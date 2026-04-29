import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from matplotlib.colors import ListedColormap


# ============================================================
# Colormap helpers
# ============================================================
def _build_cmap(
    num_classes_needed: int,
    cmap_name: str = "nipy_spectral",
    background_color: str = "#20252B",
) -> ListedColormap:
    """
    Build a discrete class colormap from a Matplotlib colormap.

    Important:
    - Map value 0 is reserved for background / unseen / suppressed pixels.
    - Actual class ids are visualized as cls + 1.
    - The base cmap is sampled discretely, so class colors stay fixed within a phase.

    Do not pass cmap="turbo" directly to imshow for class maps, because raw
    continuous mapping can make categorical class colors unstable.
    """
    num_classes_needed = int(max(num_classes_needed, 2))

    if num_classes_needed <= 1:
        return ListedColormap([background_color])

    class_count = num_classes_needed - 1

    # For <=20 classes, tab20 is often cleaner for categorical maps.
    # For larger class counts, turbo gives stronger separation.
    if cmap_name is None or str(cmap_name).strip() == "":
        cmap_name = "tab20" if class_count <= 20 else "turbo"

    base = plt.get_cmap(cmap_name)

    if cmap_name.lower() in {"tab10", "tab20", "tab20b", "tab20c"}:
        # Categorical matplotlib maps have a fixed finite palette. Sample by integer index.
        n = getattr(base, "N", class_count)
        class_colors = [base(i % n) for i in range(class_count)]
    else:
        # Avoid endpoints because they are frequently too dark/too bright.
        samples = np.linspace(0.05, 0.95, class_count)
        class_colors = [base(float(s)) for s in samples]

    colors = [background_color] + class_colors
    return ListedColormap(colors)


def _safe_target_name(target_names: Optional[List[str]], cls: int) -> str:
    if target_names is not None and int(cls) < len(target_names):
        return str(target_names[int(cls)])
    return f"Class {int(cls)}"


def _set_model_phase_and_old_count(model, dataset_manager, phase: int) -> int:
    phase = int(phase)
    old_class_count = 0 if phase == 0 else len(dataset_manager.get_classes_up_to_phase(phase - 1))

    if hasattr(model, "set_phase"):
        model.set_phase(phase)
    else:
        model.current_phase = phase

    if hasattr(model, "set_old_class_count"):
        model.set_old_class_count(old_class_count)
    else:
        model.old_class_count = old_class_count

    return old_class_count


def _resolve_viz_modes(
    phase: int,
    classifier_mode: Optional[str],
    semantic_mode: Optional[str],
) -> Tuple[str, str]:
    """
    Critical geometry-native policy:
    - phase 0 uses geometry_only
    - incremental phases use calibrated_geometry
    - semantic mode defaults to identity for every phase

    Do not use semantic_mode='all' in visualization. It evaluates a different
    feature manifold from the one used to build the geometry bank.
    """
    if classifier_mode is None:
        classifier_mode = "geometry_only" if int(phase) == 0 else "calibrated_geometry"
    if semantic_mode is None:
        semantic_mode = "identity"
    return str(classifier_mode).lower(), str(semantic_mode).lower()


@torch.no_grad()
def _viz_model_forward(
    model: torch.nn.Module,
    dataset_manager,
    batch: torch.Tensor,
    phase: int,
    classifier_mode: Optional[str] = None,
    semantic_mode: Optional[str] = None,
):
    _set_model_phase_and_old_count(model, dataset_manager, phase)
    classifier_mode, semantic_mode = _resolve_viz_modes(
        phase=phase,
        classifier_mode=classifier_mode,
        semantic_mode=semantic_mode,
    )

    return model(
        batch,
        semantic_mode=semantic_mode,
        classifier_mode=classifier_mode,
    )


def _get_true_labels_and_coords(dataset_manager):
    if hasattr(dataset_manager, "remapped_labels"):
        true_labels = dataset_manager.remapped_labels
    else:
        true_labels = dataset_manager.labels

    coords = dataset_manager.coords
    true_labels = np.asarray(true_labels).reshape(-1)
    return true_labels, coords


def _safe_seen_classes(dataset_manager, phase: int) -> List[int]:
    seen_classes = dataset_manager.get_classes_up_to_phase(int(phase))
    return [int(c) for c in seen_classes]


def _clean_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


def _legend_items(cmap: ListedColormap, seen_classes: List[int], target_names: Optional[List[str]]):
    items = [
        mpatches.Patch(color=cmap.colors[0], label="0: Background / Unseen / Suppressed")
    ]
    for cls in seen_classes:
        color_idx = int(cls) + 1
        if color_idx < len(cmap.colors):
            name = _safe_target_name(target_names, int(cls))
            items.append(mpatches.Patch(color=cmap.colors[color_idx], label=f"{cls}: {name}"))
    return items


def _save_gt_pred_figure(
    gt_map: np.ndarray,
    pred_map: np.ndarray,
    cmap: ListedColormap,
    phase: int,
    save_path: str,
    seen_classes: List[int],
    target_names: Optional[List[str]],
    stats_text: str,
):
    fig = plt.figure(figsize=(16.5, 8.5), facecolor="white")
    gs = fig.add_gridspec(1, 2, wspace=0.08)

    ax_gt = fig.add_subplot(gs[0, 0])
    ax_pred = fig.add_subplot(gs[0, 1])

    ax_gt.imshow(gt_map, cmap=cmap, interpolation="nearest")
    ax_gt.set_title(f"Phase {phase} Ground Truth", fontsize=18, fontweight="bold", pad=14)
    _clean_axis(ax_gt)

    ax_pred.imshow(pred_map, cmap=cmap, interpolation="nearest")
    ax_pred.set_title(f"Phase {phase} Prediction", fontsize=18, fontweight="bold", pad=14)
    _clean_axis(ax_pred)

    ax_pred.text(
        0.98,
        0.02,
        stats_text,
        transform=ax_pred.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color="white",
        ha="right",
        va="bottom",
        bbox=dict(facecolor="#111111", alpha=0.72, edgecolor="none", boxstyle="round,pad=0.50"),
    )

    handles = _legend_items(cmap, seen_classes, target_names)
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(4, len(handles)),
        bbox_to_anchor=(0.5, 0.01),
        title="Displayed Labels",
        title_fontsize=12,
        fontsize=9,
        frameon=False,
    )

    fig.suptitle(
        f"Geometry-Calibrated NECIL-HSI — Phase {phase}",
        fontsize=20,
        fontweight="bold",
        y=0.98,
    )

    plt.subplots_adjust(bottom=0.18, top=0.90, left=0.03, right=0.97)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_confidence_figure(
    conf_map: np.ndarray,
    phase: int,
    save_path: str,
    confidence_cmap: str = "magma",
    background_color: str = "#20252B",
):
    conf_vis = conf_map.copy()
    conf_vis[conf_vis <= 0.0] = np.nan

    cmap = plt.get_cmap(confidence_cmap).copy()
    cmap.set_bad(color=background_color)

    fig, ax = plt.subplots(figsize=(8.4, 8.0), facecolor="white")
    im = ax.imshow(conf_vis, cmap=cmap, interpolation="nearest", vmin=0.0, vmax=1.0)
    ax.set_title(f"Phase {phase} Confidence Map", fontsize=18, fontweight="bold", pad=14)
    _clean_axis(ax)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.set_ylabel("Softmax Confidence", rotation=270, labelpad=20, fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_error_figure(
    error_map: np.ndarray,
    phase: int,
    save_path: str,
):
    # 0: background/unseen, 1: correct, 2: wrong, 3: suppressed
    colors = ["#20252B", "#38B000", "#D00000", "#FFB703"]
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(8.4, 8.0), facecolor="white")
    ax.imshow(error_map, cmap=cmap, interpolation="nearest", vmin=0, vmax=3)
    ax.set_title(f"Phase {phase} Error Map", fontsize=18, fontweight="bold", pad=14)
    _clean_axis(ax)

    handles = [
        mpatches.Patch(color=colors[0], label="Background / Unseen"),
        mpatches.Patch(color=colors[1], label="Correct"),
        mpatches.Patch(color=colors[2], label="Incorrect"),
        mpatches.Patch(color=colors[3], label="Suppressed"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


@torch.no_grad()
def predict_phase_grid(
    model: torch.nn.Module,
    dataset_manager,
    phase: int,
    target_names: Optional[List[str]],
    save_dir: str = "./results/phase_visuals",
    device: str = "cuda",
    patch_size=None,
    chunk_size: int = 512,
    classifier_mode: Optional[str] = None,
    semantic_mode: Optional[str] = None,
    save_numpy: bool = True,
    return_outputs: bool = False,
    class_cmap: str = "turbo",
    confidence_cmap: str = "magma",
    background_color: str = "#20252B",
    save_error_map: bool = True,
    save_legacy_publication_name: bool = False,
):
    """
    Publication-style HSI phase visualization aligned with the geometry-native
    NECIL pipeline.

    Behavior:
    - predicts only currently seen classes
    - suppresses invalid/unseen predictions
    - defaults to semantic_mode='identity'
    - reports OA over visible seen-class pixels
    - uses a discrete Matplotlib cmap for categorical maps
    - saves confidence/error maps separately
    - does not save duplicate legacy *_publication.png unless explicitly requested
    """
    model.eval()
    _set_model_phase_and_old_count(model, dataset_manager, phase)

    gt_shape = dataset_manager.gt_shape
    true_labels, coords = _get_true_labels_and_coords(dataset_manager)

    patches = dataset_manager.patches
    num_samples = len(patches)
    if num_samples != len(coords):
        raise ValueError(f"patch/coord length mismatch: {num_samples} vs {len(coords)}")

    all_preds = []
    all_conf = []

    classifier_mode, semantic_mode = _resolve_viz_modes(phase, classifier_mode, semantic_mode)

    for i in range(0, num_samples, int(chunk_size)):
        end = min(i + int(chunk_size), num_samples)

        batch_np = patches[i:end]
        if isinstance(batch_np, torch.Tensor):
            batch = batch_np.to(device).float()
        else:
            batch = torch.from_numpy(batch_np).float().to(device)

        out = _viz_model_forward(
            model=model,
            dataset_manager=dataset_manager,
            batch=batch,
            phase=phase,
            classifier_mode=classifier_mode,
            semantic_mode=semantic_mode,
        )

        logits = out["logits"]
        probs = torch.softmax(logits, dim=1)
        conf, preds = probs.max(dim=1)

        all_preds.append(preds.detach().cpu().numpy())
        all_conf.append(conf.detach().cpu().numpy())

    if len(all_preds) == 0:
        print(f"[Viz] No predictions generated for phase {phase}.")
        return None if return_outputs else None

    all_preds = np.concatenate(all_preds, axis=0).astype(np.int64)
    all_conf = np.concatenate(all_conf, axis=0).astype(np.float32)

    seen_classes = _safe_seen_classes(dataset_manager, phase)
    seen_set = set(seen_classes)
    max_valid_cls = max(seen_classes) if len(seen_classes) > 0 else -1

    pred_map = np.zeros(gt_shape, dtype=np.int32)
    gt_map = np.zeros(gt_shape, dtype=np.int32)
    conf_map = np.zeros(gt_shape, dtype=np.float32)
    error_map = np.zeros(gt_shape, dtype=np.int32)

    correct_count = 0
    total_seen_pixels = 0
    suppressed_predictions = 0

    old_class_count = 0 if int(phase) == 0 else len(dataset_manager.get_classes_up_to_phase(int(phase) - 1))
    old_correct = old_total = 0
    new_correct = new_total = 0

    for i, (r, c) in enumerate(coords):
        y_true = int(true_labels[i])

        if y_true not in seen_set:
            pred_map[r, c] = 0
            gt_map[r, c] = 0
            conf_map[r, c] = 0.0
            error_map[r, c] = 0
            continue

        y_pred = int(all_preds[i])
        conf = float(all_conf[i])
        valid_pred = (0 <= y_pred <= max_valid_cls) and (y_pred in seen_set)

        gt_map[r, c] = y_true + 1
        total_seen_pixels += 1

        if valid_pred:
            pred_map[r, c] = y_pred + 1
            conf_map[r, c] = conf
            is_correct = y_pred == y_true
            if is_correct:
                correct_count += 1
                error_map[r, c] = 1
            else:
                error_map[r, c] = 2

            if y_true < old_class_count:
                old_total += 1
                old_correct += int(is_correct)
            else:
                new_total += 1
                new_correct += int(is_correct)
        else:
            pred_map[r, c] = 0
            conf_map[r, c] = 0.0
            suppressed_predictions += 1
            error_map[r, c] = 3

            if y_true < old_class_count:
                old_total += 1
            else:
                new_total += 1

    phase_acc = 100.0 * correct_count / max(total_seen_pixels, 1)
    suppression_ratio = 100.0 * suppressed_predictions / max(total_seen_pixels, 1)
    old_acc = 100.0 * old_correct / max(old_total, 1)
    new_acc = 100.0 * new_correct / max(new_total, 1)
    hm = 0.0 if old_acc + new_acc <= 0 else 2.0 * old_acc * new_acc / (old_acc + new_acc)

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.titleweight"] = "bold"

    num_display = int(max(gt_map.max(), pred_map.max()) + 1)
    cmap = _build_cmap(num_display, cmap_name=class_cmap, background_color=background_color)

    stats_text = (
        f"OA: {phase_acc:.2f}%\n"
        f"Old: {old_acc:.2f}% | New: {new_acc:.2f}%\n"
        f"H: {hm:.2f}%\n"
        f"Seen classes: {len(seen_classes)}\n"
        f"Visible pixels: {total_seen_pixels}\n"
        f"Suppressed: {suppression_ratio:.2f}%\n"
        f"Mode: {classifier_mode}/{semantic_mode}"
    )

    os.makedirs(save_dir, exist_ok=True)
    p_str = f"_ps{patch_size}" if patch_size else ""

    gt_pred_path = os.path.join(save_dir, f"phase_{phase}{p_str}_gt_pred.png")
    confidence_path = os.path.join(save_dir, f"phase_{phase}{p_str}_confidence.png")
    error_path = os.path.join(save_dir, f"phase_{phase}{p_str}_error.png")

    _save_gt_pred_figure(
        gt_map=gt_map,
        pred_map=pred_map,
        cmap=cmap,
        phase=phase,
        save_path=gt_pred_path,
        seen_classes=seen_classes,
        target_names=target_names,
        stats_text=stats_text,
    )

    _save_confidence_figure(
        conf_map=conf_map,
        phase=phase,
        save_path=confidence_path,
        confidence_cmap=confidence_cmap,
        background_color=background_color,
    )

    if save_error_map:
        _save_error_figure(error_map=error_map, phase=phase, save_path=error_path)

    # Optional compatibility name for any older code expecting *_publication.png.
    publication_path = None
    if save_legacy_publication_name:
        publication_path = os.path.join(save_dir, f"phase_{phase}{p_str}_publication.png")
        _save_gt_pred_figure(
            gt_map=gt_map,
            pred_map=pred_map,
            cmap=cmap,
            phase=phase,
            save_path=publication_path,
            seen_classes=seen_classes,
            target_names=target_names,
            stats_text=stats_text,
        )

    if save_numpy:
        np.save(os.path.join(save_dir, f"phase_{phase}{p_str}_pred_map.npy"), pred_map)
        np.save(os.path.join(save_dir, f"phase_{phase}{p_str}_gt_map.npy"), gt_map)
        np.save(os.path.join(save_dir, f"phase_{phase}{p_str}_confidence_map.npy"), conf_map)
        np.save(os.path.join(save_dir, f"phase_{phase}{p_str}_error_map.npy"), error_map)

    print(
        f"[Viz] Saved Phase {phase} GT/Prediction to: {gt_pred_path}\n"
        f"[Viz] Saved Phase {phase} confidence map to: {confidence_path}\n"
        + (f"[Viz] Saved Phase {phase} error map to: {error_path}\n" if save_error_map else "")
        + (f"[Viz] Saved legacy publication map to: {publication_path}\n" if publication_path else "")
        + f"[Viz] Metrics — OA: {phase_acc:.2f}%, Old: {old_acc:.2f}%, New: {new_acc:.2f}%, H: {hm:.2f}%, Suppressed: {suppression_ratio:.2f}%"
    )

    outputs = {
        "pred_map": pred_map,
        "gt_map": gt_map,
        "confidence_map": conf_map,
        "error_map": error_map,
        "metrics": {
            "overall_accuracy": phase_acc,
            "old_accuracy": old_acc,
            "new_accuracy": new_acc,
            "harmonic_mean": hm,
            "suppression_ratio": suppression_ratio,
            "visible_pixels": total_seen_pixels,
            "seen_classes": seen_classes,
            "classifier_mode": classifier_mode,
            "semantic_mode": semantic_mode,
        },
        "save_path": gt_pred_path,
        "gt_pred_path": gt_pred_path,
        "confidence_path": confidence_path,
        "error_path": error_path if save_error_map else None,
        "publication_path": publication_path,
    }

    return outputs if return_outputs else gt_pred_path



def _save_single_history_plot(
    *,
    x,
    series,
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: str,
    phase_boundaries=None,
    ylim=None,
):
    """
    Save one clean standalone training-history figure.

    Parameters
    ----------
    x:
        Epoch indices.
    series:
        List of (name, values) tuples.
    title:
        Figure title.
    xlabel / ylabel:
        Axis labels.
    save_path:
        Output path.
    phase_boundaries:
        Optional global epoch indices where phase boundaries should be drawn.
    ylim:
        Optional y-axis limits, e.g. (0, 100).
    """
    plt.rcParams["font.family"] = "DejaVu Sans"

    fig, ax = plt.subplots(figsize=(10, 5.8), facecolor="white")

    for name, values in series:
        if values is None:
            continue
        if len(values) == 0:
            continue
        ax.plot(x, values, linewidth=2.2, label=name)

    if phase_boundaries is not None:
        for phase_start in phase_boundaries:
            if phase_start > 0:
                ax.axvline(x=phase_start, color="k", linestyle="--", alpha=0.35)

    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Viz] Saved {title} to: {save_path}")
    return save_path


def plot_training_history(
    history: dict,
    save_path: str = "./results/training_history.png",
    save_separate: bool = False,
):
    """
    Save training-history plots.

    Default behavior:
        saves only the combined figure:
            full_training_history.png

    Optional behavior:
        if save_separate=True, also saves:
            full_training_history_loss.png
            full_training_history_accuracy.png
            full_training_history_old_new_hm.png

    Notes
    -----
    This function expects the cleaned trainer history:
        train_loss, train_acc, val_loss, val_acc, val_old_acc, val_new_acc, val_hm
    It ignores removed online/stale-bank metrics.
    """
    if not history or "train_loss" not in history:
        raise ValueError("history must contain at least train_loss.")

    epochs = range(1, len(history["train_loss"]) + 1)
    plt.rcParams["font.family"] = "DejaVu Sans"

    has_old_new = "val_old_acc" in history and "val_new_acc" in history and "val_hm" in history
    nrows = 3 if has_old_new else 2

    fig, axes = plt.subplots(nrows, 1, figsize=(10, 4.8 * nrows), facecolor="white")
    if nrows == 1:
        axes = [axes]

    # ---------------- Combined loss panel ----------------
    ax1 = axes[0]
    ax1.plot(epochs, history.get("train_loss", []), linewidth=2.0, label="Train Loss")
    if "val_loss" in history:
        ax1.plot(epochs, history["val_loss"], linewidth=2.0, label="Val Loss")
    ax1.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(frameon=False)
    ax1.grid(True, alpha=0.25)

    # ---------------- Combined accuracy panel ----------------
    ax2 = axes[1]
    if "train_acc" in history:
        ax2.plot(epochs, history["train_acc"], linewidth=2.0, label="Train Acc")
    if "val_acc" in history:
        ax2.plot(epochs, history["val_acc"], linewidth=2.0, label="Val Acc")
    ax2.set_title("Overall Accuracy", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend(frameon=False)
    ax2.grid(True, alpha=0.25)

    # ---------------- Combined old/new/H panel ----------------
    if has_old_new:
        ax3 = axes[2]
        ax3.plot(epochs, history["val_old_acc"], linewidth=2.0, label="Old Acc")
        ax3.plot(epochs, history["val_new_acc"], linewidth=2.0, label="New Acc")
        ax3.plot(epochs, history["val_hm"], linewidth=2.0, label="Harmonic Mean")
        ax3.set_title("Old/New Stability-Plasticity Balance", fontsize=14, fontweight="bold")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Accuracy / H (%)")
        ax3.legend(frameon=False)
        ax3.grid(True, alpha=0.25)

    phase_boundaries = history.get("phase_boundaries", [])
    for phase_start in phase_boundaries:
        if phase_start > 0:
            for ax in axes:
                ax.axvline(x=phase_start, color="k", linestyle="--", alpha=0.35)

    plt.tight_layout()

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plt.savefig(save_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Viz] Saved training history to: {save_path}")

    saved_paths = {"combined": save_path}

    if save_separate:
        root, ext = os.path.splitext(save_path)
        ext = ext if ext else ".png"

        loss_path = f"{root}_loss{ext}"
        acc_path = f"{root}_accuracy{ext}"
        old_new_path = f"{root}_old_new_hm{ext}"

        saved_paths["loss"] = _save_single_history_plot(
            x=epochs,
            series=[
                ("Train Loss", history.get("train_loss", [])),
                ("Val Loss", history.get("val_loss", [])),
            ],
            title="Training and Validation Loss",
            xlabel="Epoch",
            ylabel="Loss",
            save_path=loss_path,
            phase_boundaries=phase_boundaries,
        )

        saved_paths["accuracy"] = _save_single_history_plot(
            x=epochs,
            series=[
                ("Train Acc", history.get("train_acc", [])),
                ("Val Acc", history.get("val_acc", [])),
            ],
            title="Overall Accuracy",
            xlabel="Epoch",
            ylabel="Accuracy (%)",
            save_path=acc_path,
            phase_boundaries=phase_boundaries,
            ylim=(0, 100),
        )

        if has_old_new:
            saved_paths["old_new_hm"] = _save_single_history_plot(
                x=epochs,
                series=[
                    ("Old Acc", history.get("val_old_acc", [])),
                    ("New Acc", history.get("val_new_acc", [])),
                    ("Harmonic Mean", history.get("val_hm", [])),
                ],
                title="Old/New Stability-Plasticity Balance",
                xlabel="Epoch",
                ylabel="Accuracy / H (%)",
                save_path=old_new_path,
                phase_boundaries=phase_boundaries,
                ylim=(0, 100),
            )

    return saved_paths


def plot_phase_metric_summary(
    phase_history: Dict[int, Dict],
    save_path: str = "./results/phase_metric_summary.png",
):
    if not phase_history:
        raise ValueError("phase_history is empty.")

    phases = sorted(int(p) for p in phase_history.keys())
    oa = [float(phase_history[p].get("overall_accuracy", 0.0)) for p in phases]
    old = [float(phase_history[p].get("old_accuracy", 0.0)) for p in phases]
    new = [float(phase_history[p].get("new_accuracy", 0.0)) for p in phases]
    hm = [float(phase_history[p].get("harmonic_mean", 0.0)) for p in phases]

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")

    ax.plot(phases, oa, marker="o", linewidth=2.0, label="OA")
    ax.plot(phases, old, marker="o", linewidth=2.0, label="Old")
    ax.plot(phases, new, marker="o", linewidth=2.0, label="New")
    ax.plot(phases, hm, marker="o", linewidth=2.0, label="H")

    ax.set_title("NECIL-HSI Phase Metrics", fontsize=14, fontweight="bold")
    ax.set_xlabel("Phase")
    ax.set_ylabel("Metric (%)")
    ax.set_xticks(phases)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[Viz] Saved phase metric summary to: {save_path}")
    return save_path






# import os
# from typing import Dict, List, Optional, Tuple

# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import numpy as np
# import torch
# from matplotlib.colors import ListedColormap


# # ============================================================
# # Colormap helpers
# # ============================================================
# def _build_cmap(
#     num_classes_needed: int,
#     cmap_name: str = "nipy_spectral",
#     background_color: str = "#20252B",
# ) -> ListedColormap:
#     """
#     Build a discrete class colormap from a Matplotlib colormap.

#     Important:
#     - Map value 0 is reserved for background / unseen / suppressed pixels.
#     - Actual class ids are visualized as cls + 1.
#     - The base cmap is sampled discretely, so class colors stay fixed within a phase.

#     Do not pass cmap="turbo" directly to imshow for class maps, because raw
#     continuous mapping can make categorical class colors unstable.
#     """
#     num_classes_needed = int(max(num_classes_needed, 2))

#     if num_classes_needed <= 1:
#         return ListedColormap([background_color])

#     class_count = num_classes_needed - 1

#     # For <=20 classes, tab20 is often cleaner for categorical maps.
#     # For larger class counts, turbo gives stronger separation.
#     if cmap_name is None or str(cmap_name).strip() == "":
#         cmap_name = "tab20" if class_count <= 20 else "turbo"

#     base = plt.get_cmap(cmap_name)

#     if cmap_name.lower() in {"tab10", "tab20", "tab20b", "tab20c"}:
#         # Categorical matplotlib maps have a fixed finite palette. Sample by integer index.
#         n = getattr(base, "N", class_count)
#         class_colors = [base(i % n) for i in range(class_count)]
#     else:
#         # Avoid endpoints because they are frequently too dark/too bright.
#         samples = np.linspace(0.05, 0.95, class_count)
#         class_colors = [base(float(s)) for s in samples]

#     colors = [background_color] + class_colors
#     return ListedColormap(colors)


# def _safe_target_name(target_names: Optional[List[str]], cls: int) -> str:
#     if target_names is not None and int(cls) < len(target_names):
#         return str(target_names[int(cls)])
#     return f"Class {int(cls)}"


# def _set_model_phase_and_old_count(model, dataset_manager, phase: int) -> int:
#     phase = int(phase)
#     old_class_count = 0 if phase == 0 else len(dataset_manager.get_classes_up_to_phase(phase - 1))

#     if hasattr(model, "set_phase"):
#         model.set_phase(phase)
#     else:
#         model.current_phase = phase

#     if hasattr(model, "set_old_class_count"):
#         model.set_old_class_count(old_class_count)
#     else:
#         model.old_class_count = old_class_count

#     return old_class_count


# def _resolve_viz_modes(
#     phase: int,
#     classifier_mode: Optional[str],
#     semantic_mode: Optional[str],
# ) -> Tuple[str, str]:
#     """
#     Critical geometry-native policy:
#     - phase 0 uses geometry_only
#     - incremental phases use calibrated_geometry
#     - semantic mode defaults to identity for every phase

#     Do not use semantic_mode='all' in visualization. It evaluates a different
#     feature manifold from the one used to build the geometry bank.
#     """
#     if classifier_mode is None:
#         classifier_mode = "geometry_only" if int(phase) == 0 else "calibrated_geometry"
#     if semantic_mode is None:
#         semantic_mode = "identity"
#     return str(classifier_mode).lower(), str(semantic_mode).lower()


# @torch.no_grad()
# def _viz_model_forward(
#     model: torch.nn.Module,
#     dataset_manager,
#     batch: torch.Tensor,
#     phase: int,
#     classifier_mode: Optional[str] = None,
#     semantic_mode: Optional[str] = None,
# ):
#     _set_model_phase_and_old_count(model, dataset_manager, phase)
#     classifier_mode, semantic_mode = _resolve_viz_modes(
#         phase=phase,
#         classifier_mode=classifier_mode,
#         semantic_mode=semantic_mode,
#     )

#     return model(
#         batch,
#         semantic_mode=semantic_mode,
#         classifier_mode=classifier_mode,
#     )


# def _get_true_labels_and_coords(dataset_manager):
#     if hasattr(dataset_manager, "remapped_labels"):
#         true_labels = dataset_manager.remapped_labels
#     else:
#         true_labels = dataset_manager.labels

#     coords = dataset_manager.coords
#     true_labels = np.asarray(true_labels).reshape(-1)
#     return true_labels, coords


# def _safe_seen_classes(dataset_manager, phase: int) -> List[int]:
#     seen_classes = dataset_manager.get_classes_up_to_phase(int(phase))
#     return [int(c) for c in seen_classes]


# def _clean_axis(ax):
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_frame_on(False)


# def _legend_items(cmap: ListedColormap, seen_classes: List[int], target_names: Optional[List[str]]):
#     items = [
#         mpatches.Patch(color=cmap.colors[0], label="0: Background / Unseen / Suppressed")
#     ]
#     for cls in seen_classes:
#         color_idx = int(cls) + 1
#         if color_idx < len(cmap.colors):
#             name = _safe_target_name(target_names, int(cls))
#             items.append(mpatches.Patch(color=cmap.colors[color_idx], label=f"{cls}: {name}"))
#     return items


# def _save_gt_pred_figure(
#     gt_map: np.ndarray,
#     pred_map: np.ndarray,
#     cmap: ListedColormap,
#     phase: int,
#     save_path: str,
#     seen_classes: List[int],
#     target_names: Optional[List[str]],
#     stats_text: str,
# ):
#     fig = plt.figure(figsize=(16.5, 8.5), facecolor="white")
#     gs = fig.add_gridspec(1, 2, wspace=0.08)

#     ax_gt = fig.add_subplot(gs[0, 0])
#     ax_pred = fig.add_subplot(gs[0, 1])

#     ax_gt.imshow(gt_map, cmap=cmap, interpolation="nearest")
#     ax_gt.set_title(f"Phase {phase} Ground Truth", fontsize=18, fontweight="bold", pad=14)
#     _clean_axis(ax_gt)

#     ax_pred.imshow(pred_map, cmap=cmap, interpolation="nearest")
#     ax_pred.set_title(f"Phase {phase} Prediction", fontsize=18, fontweight="bold", pad=14)
#     _clean_axis(ax_pred)

#     ax_pred.text(
#         0.98,
#         0.02,
#         stats_text,
#         transform=ax_pred.transAxes,
#         fontsize=10.5,
#         fontweight="bold",
#         color="white",
#         ha="right",
#         va="bottom",
#         bbox=dict(facecolor="#111111", alpha=0.72, edgecolor="none", boxstyle="round,pad=0.50"),
#     )

#     handles = _legend_items(cmap, seen_classes, target_names)
#     fig.legend(
#         handles=handles,
#         loc="lower center",
#         ncol=min(4, len(handles)),
#         bbox_to_anchor=(0.5, 0.01),
#         title="Displayed Labels",
#         title_fontsize=12,
#         fontsize=9,
#         frameon=False,
#     )

#     fig.suptitle(
#         f"Geometry-Calibrated NECIL-HSI — Phase {phase}",
#         fontsize=20,
#         fontweight="bold",
#         y=0.98,
#     )

#     plt.subplots_adjust(bottom=0.18, top=0.90, left=0.03, right=0.97)
#     plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
#     plt.close(fig)


# def _save_confidence_figure(
#     conf_map: np.ndarray,
#     phase: int,
#     save_path: str,
#     confidence_cmap: str = "magma",
#     background_color: str = "#20252B",
# ):
#     conf_vis = conf_map.copy()
#     conf_vis[conf_vis <= 0.0] = np.nan

#     cmap = plt.get_cmap(confidence_cmap).copy()
#     cmap.set_bad(color=background_color)

#     fig, ax = plt.subplots(figsize=(8.4, 8.0), facecolor="white")
#     im = ax.imshow(conf_vis, cmap=cmap, interpolation="nearest", vmin=0.0, vmax=1.0)
#     ax.set_title(f"Phase {phase} Confidence Map", fontsize=18, fontweight="bold", pad=14)
#     _clean_axis(ax)

#     cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
#     cbar.ax.set_ylabel("Softmax Confidence", rotation=270, labelpad=20, fontsize=11)
#     cbar.ax.tick_params(labelsize=10)

#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
#     plt.close(fig)


# def _save_error_figure(
#     error_map: np.ndarray,
#     phase: int,
#     save_path: str,
# ):
#     # 0: background/unseen, 1: correct, 2: wrong, 3: suppressed
#     colors = ["#20252B", "#38B000", "#D00000", "#FFB703"]
#     cmap = ListedColormap(colors)

#     fig, ax = plt.subplots(figsize=(8.4, 8.0), facecolor="white")
#     ax.imshow(error_map, cmap=cmap, interpolation="nearest", vmin=0, vmax=3)
#     ax.set_title(f"Phase {phase} Error Map", fontsize=18, fontweight="bold", pad=14)
#     _clean_axis(ax)

#     handles = [
#         mpatches.Patch(color=colors[0], label="Background / Unseen"),
#         mpatches.Patch(color=colors[1], label="Correct"),
#         mpatches.Patch(color=colors[2], label="Incorrect"),
#         mpatches.Patch(color=colors[3], label="Suppressed"),
#     ]
#     ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)

#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
#     plt.close(fig)


# @torch.no_grad()
# def predict_phase_grid(
#     model: torch.nn.Module,
#     dataset_manager,
#     phase: int,
#     target_names: Optional[List[str]],
#     save_dir: str = "./results/phase_visuals",
#     device: str = "cuda",
#     patch_size=None,
#     chunk_size: int = 512,
#     classifier_mode: Optional[str] = None,
#     semantic_mode: Optional[str] = None,
#     save_numpy: bool = True,
#     return_outputs: bool = False,
#     class_cmap: str = "turbo",
#     confidence_cmap: str = "magma",
#     background_color: str = "#20252B",
#     save_error_map: bool = True,
#     save_legacy_publication_name: bool = True,
# ):
#     """
#     Publication-style HSI phase visualization aligned with the geometry-native
#     NECIL pipeline.

#     Behavior:
#     - predicts only currently seen classes
#     - suppresses invalid/unseen predictions
#     - defaults to semantic_mode='identity'
#     - reports OA over visible seen-class pixels
#     - uses a discrete Matplotlib cmap for categorical maps
#     - saves confidence/error maps separately
#     """
#     model.eval()
#     _set_model_phase_and_old_count(model, dataset_manager, phase)

#     gt_shape = dataset_manager.gt_shape
#     true_labels, coords = _get_true_labels_and_coords(dataset_manager)

#     patches = dataset_manager.patches
#     num_samples = len(patches)
#     if num_samples != len(coords):
#         raise ValueError(f"patch/coord length mismatch: {num_samples} vs {len(coords)}")

#     all_preds = []
#     all_conf = []

#     classifier_mode, semantic_mode = _resolve_viz_modes(phase, classifier_mode, semantic_mode)

#     for i in range(0, num_samples, int(chunk_size)):
#         end = min(i + int(chunk_size), num_samples)

#         batch_np = patches[i:end]
#         if isinstance(batch_np, torch.Tensor):
#             batch = batch_np.to(device).float()
#         else:
#             batch = torch.from_numpy(batch_np).float().to(device)

#         out = _viz_model_forward(
#             model=model,
#             dataset_manager=dataset_manager,
#             batch=batch,
#             phase=phase,
#             classifier_mode=classifier_mode,
#             semantic_mode=semantic_mode,
#         )

#         logits = out["logits"]
#         probs = torch.softmax(logits, dim=1)
#         conf, preds = probs.max(dim=1)

#         all_preds.append(preds.detach().cpu().numpy())
#         all_conf.append(conf.detach().cpu().numpy())

#     if len(all_preds) == 0:
#         print(f"[Viz] No predictions generated for phase {phase}.")
#         return None if return_outputs else None

#     all_preds = np.concatenate(all_preds, axis=0).astype(np.int64)
#     all_conf = np.concatenate(all_conf, axis=0).astype(np.float32)

#     seen_classes = _safe_seen_classes(dataset_manager, phase)
#     seen_set = set(seen_classes)
#     max_valid_cls = max(seen_classes) if len(seen_classes) > 0 else -1

#     pred_map = np.zeros(gt_shape, dtype=np.int32)
#     gt_map = np.zeros(gt_shape, dtype=np.int32)
#     conf_map = np.zeros(gt_shape, dtype=np.float32)
#     error_map = np.zeros(gt_shape, dtype=np.int32)

#     correct_count = 0
#     total_seen_pixels = 0
#     suppressed_predictions = 0

#     old_class_count = 0 if int(phase) == 0 else len(dataset_manager.get_classes_up_to_phase(int(phase) - 1))
#     old_correct = old_total = 0
#     new_correct = new_total = 0

#     for i, (r, c) in enumerate(coords):
#         y_true = int(true_labels[i])

#         if y_true not in seen_set:
#             pred_map[r, c] = 0
#             gt_map[r, c] = 0
#             conf_map[r, c] = 0.0
#             error_map[r, c] = 0
#             continue

#         y_pred = int(all_preds[i])
#         conf = float(all_conf[i])
#         valid_pred = (0 <= y_pred <= max_valid_cls) and (y_pred in seen_set)

#         gt_map[r, c] = y_true + 1
#         total_seen_pixels += 1

#         if valid_pred:
#             pred_map[r, c] = y_pred + 1
#             conf_map[r, c] = conf
#             is_correct = y_pred == y_true
#             if is_correct:
#                 correct_count += 1
#                 error_map[r, c] = 1
#             else:
#                 error_map[r, c] = 2

#             if y_true < old_class_count:
#                 old_total += 1
#                 old_correct += int(is_correct)
#             else:
#                 new_total += 1
#                 new_correct += int(is_correct)
#         else:
#             pred_map[r, c] = 0
#             conf_map[r, c] = 0.0
#             suppressed_predictions += 1
#             error_map[r, c] = 3

#             if y_true < old_class_count:
#                 old_total += 1
#             else:
#                 new_total += 1

#     phase_acc = 100.0 * correct_count / max(total_seen_pixels, 1)
#     suppression_ratio = 100.0 * suppressed_predictions / max(total_seen_pixels, 1)
#     old_acc = 100.0 * old_correct / max(old_total, 1)
#     new_acc = 100.0 * new_correct / max(new_total, 1)
#     hm = 0.0 if old_acc + new_acc <= 0 else 2.0 * old_acc * new_acc / (old_acc + new_acc)

#     plt.rcParams["font.family"] = "DejaVu Sans"
#     plt.rcParams["axes.titleweight"] = "bold"

#     num_display = int(max(gt_map.max(), pred_map.max()) + 1)
#     cmap = _build_cmap(num_display, cmap_name=class_cmap, background_color=background_color)

#     stats_text = (
#         f"OA: {phase_acc:.2f}%\n"
#         f"Old: {old_acc:.2f}% | New: {new_acc:.2f}%\n"
#         f"H: {hm:.2f}%\n"
#         f"Seen classes: {len(seen_classes)}\n"
#         f"Visible pixels: {total_seen_pixels}\n"
#         f"Suppressed: {suppression_ratio:.2f}%\n"
#         f"Mode: {classifier_mode}/{semantic_mode}"
#     )

#     os.makedirs(save_dir, exist_ok=True)
#     p_str = f"_ps{patch_size}" if patch_size else ""

#     gt_pred_path = os.path.join(save_dir, f"phase_{phase}{p_str}_gt_pred.png")
#     confidence_path = os.path.join(save_dir, f"phase_{phase}{p_str}_confidence.png")
#     error_path = os.path.join(save_dir, f"phase_{phase}{p_str}_error.png")

#     _save_gt_pred_figure(
#         gt_map=gt_map,
#         pred_map=pred_map,
#         cmap=cmap,
#         phase=phase,
#         save_path=gt_pred_path,
#         seen_classes=seen_classes,
#         target_names=target_names,
#         stats_text=stats_text,
#     )

#     _save_confidence_figure(
#         conf_map=conf_map,
#         phase=phase,
#         save_path=confidence_path,
#         confidence_cmap=confidence_cmap,
#         background_color=background_color,
#     )

#     if save_error_map:
#         _save_error_figure(error_map=error_map, phase=phase, save_path=error_path)

#     # Optional compatibility name for any older code expecting *_publication.png.
#     publication_path = None
#     if save_legacy_publication_name:
#         publication_path = os.path.join(save_dir, f"phase_{phase}{p_str}_publication.png")
#         _save_gt_pred_figure(
#             gt_map=gt_map,
#             pred_map=pred_map,
#             cmap=cmap,
#             phase=phase,
#             save_path=publication_path,
#             seen_classes=seen_classes,
#             target_names=target_names,
#             stats_text=stats_text,
#         )

#     if save_numpy:
#         np.save(os.path.join(save_dir, f"phase_{phase}{p_str}_pred_map.npy"), pred_map)
#         np.save(os.path.join(save_dir, f"phase_{phase}{p_str}_gt_map.npy"), gt_map)
#         np.save(os.path.join(save_dir, f"phase_{phase}{p_str}_confidence_map.npy"), conf_map)
#         np.save(os.path.join(save_dir, f"phase_{phase}{p_str}_error_map.npy"), error_map)

#     print(
#         f"[Viz] Saved Phase {phase} GT/Prediction to: {gt_pred_path}\n"
#         f"[Viz] Saved Phase {phase} confidence map to: {confidence_path}\n"
#         + (f"[Viz] Saved Phase {phase} error map to: {error_path}\n" if save_error_map else "")
#         + (f"[Viz] Saved legacy publication map to: {publication_path}\n" if publication_path else "")
#         + f"[Viz] Metrics — OA: {phase_acc:.2f}%, Old: {old_acc:.2f}%, New: {new_acc:.2f}%, H: {hm:.2f}%, Suppressed: {suppression_ratio:.2f}%"
#     )

#     outputs = {
#         "pred_map": pred_map,
#         "gt_map": gt_map,
#         "confidence_map": conf_map,
#         "error_map": error_map,
#         "metrics": {
#             "overall_accuracy": phase_acc,
#             "old_accuracy": old_acc,
#             "new_accuracy": new_acc,
#             "harmonic_mean": hm,
#             "suppression_ratio": suppression_ratio,
#             "visible_pixels": total_seen_pixels,
#             "seen_classes": seen_classes,
#             "classifier_mode": classifier_mode,
#             "semantic_mode": semantic_mode,
#         },
#         "save_path": gt_pred_path,
#         "gt_pred_path": gt_pred_path,
#         "confidence_path": confidence_path,
#         "error_path": error_path if save_error_map else None,
#         "publication_path": publication_path,
#     }

#     return outputs if return_outputs else gt_pred_path



# def _save_single_history_plot(
#     *,
#     x,
#     series,
#     title: str,
#     xlabel: str,
#     ylabel: str,
#     save_path: str,
#     phase_boundaries=None,
#     ylim=None,
# ):
#     """
#     Save one clean standalone training-history figure.

#     Parameters
#     ----------
#     x:
#         Epoch indices.
#     series:
#         List of (name, values) tuples.
#     title:
#         Figure title.
#     xlabel / ylabel:
#         Axis labels.
#     save_path:
#         Output path.
#     phase_boundaries:
#         Optional global epoch indices where phase boundaries should be drawn.
#     ylim:
#         Optional y-axis limits, e.g. (0, 100).
#     """
#     plt.rcParams["font.family"] = "DejaVu Sans"

#     fig, ax = plt.subplots(figsize=(10, 5.8), facecolor="white")

#     for name, values in series:
#         if values is None:
#             continue
#         if len(values) == 0:
#             continue
#         ax.plot(x, values, linewidth=2.2, label=name)

#     if phase_boundaries is not None:
#         for phase_start in phase_boundaries:
#             if phase_start > 0:
#                 ax.axvline(x=phase_start, color="k", linestyle="--", alpha=0.35)

#     ax.set_title(title, fontsize=15, fontweight="bold")
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     if ylim is not None:
#         ax.set_ylim(*ylim)
#     ax.legend(frameon=False)
#     ax.grid(True, alpha=0.25)

#     save_dir = os.path.dirname(save_path)
#     if save_dir:
#         os.makedirs(save_dir, exist_ok=True)

#     plt.tight_layout()
#     plt.savefig(save_path, dpi=260, bbox_inches="tight", facecolor="white")
#     plt.close(fig)
#     print(f"[Viz] Saved {title} to: {save_path}")
#     return save_path


# def plot_training_history(
#     history: dict,
#     save_path: str = "./results/training_history.png",
#     save_separate: bool = True,
# ):
#     """
#     Save training-history plots.

#     Always saves the original combined figure:
#         full_training_history.png

#     If save_separate=True, also saves:
#         full_training_history_loss.png
#         full_training_history_accuracy.png
#         full_training_history_old_new_hm.png

#     Notes
#     -----
#     This function expects the cleaned trainer history:
#         train_loss, train_acc, val_loss, val_acc, val_old_acc, val_new_acc, val_hm
#     It ignores removed online/stale-bank metrics.
#     """
#     if not history or "train_loss" not in history:
#         raise ValueError("history must contain at least train_loss.")

#     epochs = range(1, len(history["train_loss"]) + 1)
#     plt.rcParams["font.family"] = "DejaVu Sans"

#     has_old_new = "val_old_acc" in history and "val_new_acc" in history and "val_hm" in history
#     nrows = 3 if has_old_new else 2

#     fig, axes = plt.subplots(nrows, 1, figsize=(10, 4.8 * nrows), facecolor="white")
#     if nrows == 1:
#         axes = [axes]

#     # ---------------- Combined loss panel ----------------
#     ax1 = axes[0]
#     ax1.plot(epochs, history.get("train_loss", []), linewidth=2.0, label="Train Loss")
#     if "val_loss" in history:
#         ax1.plot(epochs, history["val_loss"], linewidth=2.0, label="Val Loss")
#     ax1.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
#     ax1.set_xlabel("Epoch")
#     ax1.set_ylabel("Loss")
#     ax1.legend(frameon=False)
#     ax1.grid(True, alpha=0.25)

#     # ---------------- Combined accuracy panel ----------------
#     ax2 = axes[1]
#     if "train_acc" in history:
#         ax2.plot(epochs, history["train_acc"], linewidth=2.0, label="Train Acc")
#     if "val_acc" in history:
#         ax2.plot(epochs, history["val_acc"], linewidth=2.0, label="Val Acc")
#     ax2.set_title("Overall Accuracy", fontsize=14, fontweight="bold")
#     ax2.set_xlabel("Epoch")
#     ax2.set_ylabel("Accuracy (%)")
#     ax2.legend(frameon=False)
#     ax2.grid(True, alpha=0.25)

#     # ---------------- Combined old/new/H panel ----------------
#     if has_old_new:
#         ax3 = axes[2]
#         ax3.plot(epochs, history["val_old_acc"], linewidth=2.0, label="Old Acc")
#         ax3.plot(epochs, history["val_new_acc"], linewidth=2.0, label="New Acc")
#         ax3.plot(epochs, history["val_hm"], linewidth=2.0, label="Harmonic Mean")
#         ax3.set_title("Old/New Stability-Plasticity Balance", fontsize=14, fontweight="bold")
#         ax3.set_xlabel("Epoch")
#         ax3.set_ylabel("Accuracy / H (%)")
#         ax3.legend(frameon=False)
#         ax3.grid(True, alpha=0.25)

#     phase_boundaries = history.get("phase_boundaries", [])
#     for phase_start in phase_boundaries:
#         if phase_start > 0:
#             for ax in axes:
#                 ax.axvline(x=phase_start, color="k", linestyle="--", alpha=0.35)

#     plt.tight_layout()

#     save_dir = os.path.dirname(save_path)
#     if save_dir:
#         os.makedirs(save_dir, exist_ok=True)

#     plt.savefig(save_path, dpi=220, bbox_inches="tight", facecolor="white")
#     plt.close(fig)
#     print(f"[Viz] Saved training history to: {save_path}")

#     saved_paths = {"combined": save_path}

#     if save_separate:
#         root, ext = os.path.splitext(save_path)
#         ext = ext if ext else ".png"

#         loss_path = f"{root}_loss{ext}"
#         acc_path = f"{root}_accuracy{ext}"
#         old_new_path = f"{root}_old_new_hm{ext}"

#         saved_paths["loss"] = _save_single_history_plot(
#             x=epochs,
#             series=[
#                 ("Train Loss", history.get("train_loss", [])),
#                 ("Val Loss", history.get("val_loss", [])),
#             ],
#             title="Training and Validation Loss",
#             xlabel="Epoch",
#             ylabel="Loss",
#             save_path=loss_path,
#             phase_boundaries=phase_boundaries,
#         )

#         saved_paths["accuracy"] = _save_single_history_plot(
#             x=epochs,
#             series=[
#                 ("Train Acc", history.get("train_acc", [])),
#                 ("Val Acc", history.get("val_acc", [])),
#             ],
#             title="Overall Accuracy",
#             xlabel="Epoch",
#             ylabel="Accuracy (%)",
#             save_path=acc_path,
#             phase_boundaries=phase_boundaries,
#             ylim=(0, 100),
#         )

#         if has_old_new:
#             saved_paths["old_new_hm"] = _save_single_history_plot(
#                 x=epochs,
#                 series=[
#                     ("Old Acc", history.get("val_old_acc", [])),
#                     ("New Acc", history.get("val_new_acc", [])),
#                     ("Harmonic Mean", history.get("val_hm", [])),
#                 ],
#                 title="Old/New Stability-Plasticity Balance",
#                 xlabel="Epoch",
#                 ylabel="Accuracy / H (%)",
#                 save_path=old_new_path,
#                 phase_boundaries=phase_boundaries,
#                 ylim=(0, 100),
#             )

#     return saved_paths


# def plot_phase_metric_summary(
#     phase_history: Dict[int, Dict],
#     save_path: str = "./results/phase_metric_summary.png",
# ):
#     if not phase_history:
#         raise ValueError("phase_history is empty.")

#     phases = sorted(int(p) for p in phase_history.keys())
#     oa = [float(phase_history[p].get("overall_accuracy", 0.0)) for p in phases]
#     old = [float(phase_history[p].get("old_accuracy", 0.0)) for p in phases]
#     new = [float(phase_history[p].get("new_accuracy", 0.0)) for p in phases]
#     hm = [float(phase_history[p].get("harmonic_mean", 0.0)) for p in phases]

#     plt.rcParams["font.family"] = "DejaVu Sans"
#     fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")

#     ax.plot(phases, oa, marker="o", linewidth=2.0, label="OA")
#     ax.plot(phases, old, marker="o", linewidth=2.0, label="Old")
#     ax.plot(phases, new, marker="o", linewidth=2.0, label="New")
#     ax.plot(phases, hm, marker="o", linewidth=2.0, label="H")

#     ax.set_title("NECIL-HSI Phase Metrics", fontsize=14, fontweight="bold")
#     ax.set_xlabel("Phase")
#     ax.set_ylabel("Metric (%)")
#     ax.set_xticks(phases)
#     ax.set_ylim(0, 100)
#     ax.grid(True, alpha=0.25)
#     ax.legend(frameon=False)

#     save_dir = os.path.dirname(save_path)
#     if save_dir:
#         os.makedirs(save_dir, exist_ok=True)

#     plt.tight_layout()
#     plt.savefig(save_path, dpi=220, bbox_inches="tight", facecolor="white")
#     plt.close()
#     print(f"[Viz] Saved phase metric summary to: {save_path}")
#     return save_path
