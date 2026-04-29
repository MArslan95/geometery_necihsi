"""
Evaluation Metrics & Reports for Geometry-Native NECIL-HSI
==========================================================

Torch-native metric core:
- confusion matrix
- OA / AA / Kappa / Macro-F1
- precision / recall / F1 / support
- old/new split accuracy
- harmonic mean

Report outputs:
- phase_X_classification_report.txt
- phase_X_classification_report.json
- phase_X_confusion_matrix.csv
- phase_X_confusion_matrix.npy
- phase_X_per_class_metrics.csv
- phase_X_HSI_Classification_Report.csv

Important:
This file assumes labels are already sequential class IDs: 0..K-1.
"""

import csv
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Iterable, Any

import numpy as np
import torch
from sklearn.metrics import classification_report


# =========================================================
# Core utilities
# =========================================================
def _as_1d_np(x, name: str) -> np.ndarray:
    arr = np.asarray(x).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} is empty.")
    return arr


def _to_1d_long_tensor(x, device: Optional[torch.device] = None) -> torch.Tensor:
    if torch.is_tensor(x):
        t = x.detach()
    else:
        t = torch.as_tensor(x)
    if device is not None:
        t = t.to(device)
    return t.long().view(-1)


def _safe_class_name(target_names: Optional[List[str]], cls: int) -> str:
    cls = int(cls)
    if target_names is not None and 0 <= cls < len(target_names):
        return str(target_names[cls])
    return f"Class {cls}"


def _filter_valid_labels_np(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    seen_classes: Optional[Iterable[int]] = None,
    ignore_index: Optional[int] = None,
):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"y_true and y_pred must have same length. Got {len(y_true)} vs {len(y_pred)}"
        )

    valid = np.ones_like(y_true, dtype=bool)

    if ignore_index is not None:
        valid &= y_true != int(ignore_index)

    if seen_classes is not None:
        seen = set(int(c) for c in seen_classes)
        valid &= np.array([int(v) in seen for v in y_true], dtype=bool)

    return y_true[valid], y_pred[valid]


def _filter_valid_labels_torch(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    seen_classes: Optional[Iterable[int]] = None,
    ignore_index: Optional[int] = None,
):
    if y_true.numel() != y_pred.numel():
        raise ValueError(
            f"y_true and y_pred must have same length. Got {y_true.numel()} vs {y_pred.numel()}"
        )

    valid = torch.ones_like(y_true, dtype=torch.bool)

    if ignore_index is not None:
        valid &= y_true != int(ignore_index)

    if seen_classes is not None:
        seen = torch.as_tensor(list(seen_classes), device=y_true.device, dtype=torch.long)
        if hasattr(torch, "isin"):
            valid &= torch.isin(y_true, seen)
        else:
            seen_mask = torch.zeros_like(valid)
            for c in seen:
                seen_mask |= y_true == c
            valid &= seen_mask

    return y_true[valid], y_pred[valid]


def _labels_for_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    seen_classes: Optional[Iterable[int]] = None,
    include_predicted_unseen: bool = False,
) -> List[int]:
    if seen_classes is not None:
        labels = [int(c) for c in seen_classes]
    else:
        labels = sorted(int(c) for c in np.unique(y_true).tolist())

    if include_predicted_unseen:
        label_set = set(labels)
        extra = sorted(int(c) for c in np.unique(y_pred).tolist() if int(c) not in label_set)
        labels = labels + extra

    return labels


def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]

    if isinstance(obj, tuple):
        return [make_json_serializable(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()

    return obj


# =========================================================
# Torch-native metrics
# =========================================================
@torch.no_grad()
def torch_confusion_matrix(
    y_true,
    y_pred,
    num_classes: int,
    device: Optional[str] = "cpu",
) -> torch.Tensor:
    """
    Fast multiclass confusion matrix.

    Rows    = true class
    Columns = predicted class
    """
    dev = torch.device(device) if device is not None else None
    y_true = _to_1d_long_tensor(y_true, dev)
    y_pred = _to_1d_long_tensor(y_pred, dev)

    valid = (
        (y_true >= 0)
        & (y_true < int(num_classes))
        & (y_pred >= 0)
        & (y_pred < int(num_classes))
    )

    y_true = y_true[valid]
    y_pred = y_pred[valid]

    indices = y_true * int(num_classes) + y_pred
    cm = torch.bincount(
        indices,
        minlength=int(num_classes) * int(num_classes),
    ).reshape(int(num_classes), int(num_classes))

    return cm


@torch.no_grad()
def torch_metrics_from_confusion_matrix(
    cm: torch.Tensor,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """
    Compute multiclass metrics from a confusion matrix.

    cm[row, col]
    row = true label
    col = predicted label
    """
    cm = cm.float()

    tp = torch.diag(cm)
    support = cm.sum(dim=1)
    predicted = cm.sum(dim=0)
    total = cm.sum()

    recall = tp / (support + eps)
    precision = tp / (predicted + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)

    valid_classes = support > 0

    oa = tp.sum() / (total + eps) * 100.0
    aa = recall[valid_classes].mean() * 100.0 if valid_classes.any() else torch.tensor(0.0, device=cm.device)
    macro_f1 = f1[valid_classes].mean() * 100.0 if valid_classes.any() else torch.tensor(0.0, device=cm.device)

    po = tp.sum() / (total + eps)
    pe = (cm.sum(dim=1) * cm.sum(dim=0)).sum() / ((total * total) + eps)
    kappa = (po - pe) / (1.0 - pe + eps) * 100.0

    per_class_accuracy = recall * 100.0
    precision_pct = precision * 100.0
    recall_pct = recall * 100.0
    f1_pct = f1 * 100.0

    return {
        "overall_accuracy": float(oa.item()),
        "balanced_accuracy": float(aa.item()),
        "average_accuracy": float(aa.item()),
        "kappa": float(kappa.item()),
        "f1_macro": float(macro_f1.item()),
        "per_class_accuracy": {
            int(i): float(per_class_accuracy[i].item())
            for i in range(cm.shape[0])
        },
        "precision": {
            int(i): float(precision_pct[i].item())
            for i in range(cm.shape[0])
        },
        "recall": {
            int(i): float(recall_pct[i].item())
            for i in range(cm.shape[0])
        },
        "f1_per_class": {
            int(i): float(f1_pct[i].item())
            for i in range(cm.shape[0])
        },
        "support": {
            int(i): int(support[i].item())
            for i in range(cm.shape[0])
        },
        "predicted_count": {
            int(i): int(predicted[i].item())
            for i in range(cm.shape[0])
        },
        "confusion_matrix": cm.detach().cpu(),
    }


@torch.no_grad()
def torch_old_new_metrics(
    y_true,
    y_pred,
    old_class_count: Optional[int] = None,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    y_true = _to_1d_long_tensor(y_true, None)
    y_pred = _to_1d_long_tensor(y_pred, y_true.device)

    overall_acc = (y_true == y_pred).float().mean() * 100.0 if y_true.numel() > 0 else torch.tensor(0.0)

    if old_class_count is None or int(old_class_count) <= 0:
        return {
            "old_accuracy": 0.0,
            "new_accuracy": float(overall_acc.item()),
            "harmonic_mean": 0.0,
            "old_count": 0,
            "new_count": int(y_true.numel()),
        }

    old_class_count = int(old_class_count)

    old_mask = y_true < old_class_count
    new_mask = y_true >= old_class_count

    old_total = int(old_mask.sum().item())
    new_total = int(new_mask.sum().item())

    if old_total > 0:
        old_acc = (y_pred[old_mask] == y_true[old_mask]).float().mean() * 100.0
    else:
        old_acc = torch.tensor(0.0, device=y_true.device)

    if new_total > 0:
        new_acc = (y_pred[new_mask] == y_true[new_mask]).float().mean() * 100.0
    else:
        new_acc = torch.tensor(0.0, device=y_true.device)

    h = 2.0 * old_acc * new_acc / (old_acc + new_acc + eps)

    return {
        "old_accuracy": float(old_acc.item()),
        "new_accuracy": float(new_acc.item()),
        "harmonic_mean": float(h.item()),
        "old_count": old_total,
        "new_count": new_total,
    }


@torch.no_grad()
def calculate_metrics_torch(
    y_true,
    y_pred,
    num_classes: Optional[int] = None,
    old_class_count: Optional[int] = None,
    seen_classes: Optional[Iterable[int]] = None,
    ignore_index: Optional[int] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Torch-native snapshot metrics for one phase evaluation.

    y_true/y_pred can be:
    - torch.Tensor
    - numpy.ndarray
    - list
    """
    dev = torch.device(device)
    y_true_t = _to_1d_long_tensor(y_true, dev)
    y_pred_t = _to_1d_long_tensor(y_pred, dev)

    y_true_t, y_pred_t = _filter_valid_labels_torch(
        y_true=y_true_t,
        y_pred=y_pred_t,
        seen_classes=seen_classes,
        ignore_index=ignore_index,
    )

    if y_true_t.numel() == 0:
        raise ValueError("No valid labels remain after filtering.")

    if seen_classes is not None:
        labels = [int(c) for c in seen_classes]
        # Critical: include predicted unseen classes in the confusion-matrix width.
        # Otherwise invalid/unseen predictions are silently dropped by torch_confusion_matrix,
        # inflating OA/AA and hiding classifier leakage.
        max_seen = int(max(labels)) if labels else -1
        max_true = int(y_true_t.max().item()) if y_true_t.numel() > 0 else -1
        max_pred = int(y_pred_t.max().item()) if y_pred_t.numel() > 0 else -1
        if num_classes is not None:
            num_classes_eff = max(int(num_classes), max_seen + 1, max_true + 1, max_pred + 1)
        else:
            num_classes_eff = max(max_seen, max_true, max_pred) + 1
    elif num_classes is not None:
        max_true = int(y_true_t.max().item()) if y_true_t.numel() > 0 else -1
        max_pred = int(y_pred_t.max().item()) if y_pred_t.numel() > 0 else -1
        num_classes_eff = max(int(num_classes), max_true + 1, max_pred + 1)
        labels = list(range(num_classes_eff))
    else:
        max_label = int(torch.cat([y_true_t, y_pred_t]).max().item())
        num_classes_eff = max_label + 1
        labels = list(range(num_classes_eff))

    cm_full = torch_confusion_matrix(
        y_true=y_true_t,
        y_pred=y_pred_t,
        num_classes=num_classes_eff,
        device=dev,
    )

    metrics = torch_metrics_from_confusion_matrix(cm_full)

    # Keep only seen/present classes for phase history if seen_classes was provided.
    if seen_classes is not None:
        seen_set = set(int(c) for c in seen_classes)
        metrics["per_class_accuracy"] = {
            int(k): float(v)
            for k, v in metrics["per_class_accuracy"].items()
            if int(k) in seen_set
        }
        metrics["precision"] = {
            int(k): float(v)
            for k, v in metrics["precision"].items()
            if int(k) in seen_set
        }
        metrics["recall"] = {
            int(k): float(v)
            for k, v in metrics["recall"].items()
            if int(k) in seen_set
        }
        metrics["f1_per_class"] = {
            int(k): float(v)
            for k, v in metrics["f1_per_class"].items()
            if int(k) in seen_set
        }
        metrics["support"] = {
            int(k): int(v)
            for k, v in metrics["support"].items()
            if int(k) in seen_set
        }

    split_metrics = torch_old_new_metrics(
        y_true=y_true_t,
        y_pred=y_pred_t,
        old_class_count=old_class_count,
    )
    metrics.update(split_metrics)

    if seen_classes is not None:
        seen_list = [int(c) for c in seen_classes]
        seen = torch.as_tensor(seen_list, device=dev, dtype=torch.long)
        if hasattr(torch, "isin"):
            invalid = ~torch.isin(y_pred_t, seen)
        else:
            valid_pred = torch.zeros_like(y_pred_t, dtype=torch.bool)
            for c in seen:
                valid_pred |= y_pred_t == c
            invalid = ~valid_pred
        metrics["invalid_prediction_rate"] = float(invalid.float().mean().item() * 100.0)
        invalid_classes = sorted(int(c) for c in torch.unique(y_pred_t[invalid]).detach().cpu().tolist()) if invalid.any() else []
        metrics["predicted_unseen_classes"] = invalid_classes
        metrics["predicted_unseen_count"] = int(invalid.sum().item())
    else:
        metrics["invalid_prediction_rate"] = 0.0
        metrics["predicted_unseen_classes"] = []
        metrics["predicted_unseen_count"] = 0

    present_classes = sorted(int(c) for c in torch.unique(y_true_t).detach().cpu().tolist())

    metrics["num_samples"] = int(y_true_t.numel())
    metrics["num_classes"] = int(len(present_classes) if seen_classes is None else len(list(seen_classes)))
    metrics["classes"] = [int(c) for c in (seen_classes if seen_classes is not None else present_classes)]
    metrics["confusion_matrix"] = cm_full.detach().cpu()

    return metrics


# Backward-compatible name used by NECILEvaluator.update
def calculate_metrics(
    y_true,
    y_pred,
    class_names: Optional[List[str]] = None,
    old_class_count: Optional[int] = None,
    seen_classes: Optional[Iterable[int]] = None,
    ignore_index: Optional[int] = None,
) -> Dict[str, Any]:
    del class_names

    if seen_classes is not None:
        seen_list = [int(c) for c in seen_classes]
        num_classes = int(max(seen_list)) + 1 if seen_list else None
    else:
        yt = np.asarray(y_true).reshape(-1)
        yp = np.asarray(y_pred).reshape(-1)
        num_classes = int(max(yt.max(), yp.max())) + 1 if yt.size > 0 else None

    return calculate_metrics_torch(
        y_true=y_true,
        y_pred=y_pred,
        num_classes=num_classes,
        old_class_count=old_class_count,
        seen_classes=seen_classes,
        ignore_index=ignore_index,
        device="cpu",
    )


# =========================================================
# Report savers
# =========================================================
def save_structured_classification_report(
    y_true,
    y_pred,
    target_names: Optional[List[str]] = None,
    save_dir: str = "./results",
    phase=0,
    seen_classes: Optional[Iterable[int]] = None,
    old_class_count: Optional[int] = None,
    ignore_index: Optional[int] = None,
    include_predicted_unseen: bool = True,
) -> Dict[str, Any]:
    """
    Save machine-readable class-wise classification report and Torch confusion matrix.

    Saves:
    - phase_X_classification_report.txt
    - phase_X_classification_report.json
    - phase_X_confusion_matrix.csv
    - phase_X_confusion_matrix.npy
    - phase_X_per_class_metrics.csv
    """

    os.makedirs(save_dir, exist_ok=True)

    y_true_np, y_pred_np = _filter_valid_labels_np(
        y_true=np.asarray(y_true).reshape(-1),
        y_pred=np.asarray(y_pred).reshape(-1),
        seen_classes=seen_classes,
        ignore_index=ignore_index,
    )

    if y_true_np.size == 0:
        raise ValueError("No valid labels remain after filtering for classification report.")

    labels = _labels_for_report(
        y_true=y_true_np,
        y_pred=y_pred_np,
        seen_classes=seen_classes,
        include_predicted_unseen=include_predicted_unseen,
    )
    names = [_safe_class_name(target_names, c) for c in labels]

    num_classes_eff = int(max(labels)) + 1 if labels else 0
    cm_full = torch_confusion_matrix(y_true_np, y_pred_np, num_classes_eff, device="cpu")
    cm_np = cm_full.detach().cpu().numpy()
    cm_report = cm_np[np.ix_(labels, labels)]

    torch_metrics = calculate_metrics_torch(
        y_true=y_true_np,
        y_pred=y_pred_np,
        num_classes=num_classes_eff,
        old_class_count=old_class_count,
        seen_classes=seen_classes,
        ignore_index=None,
        device="cpu",
    )

    report_dict = classification_report(
        y_true_np,
        y_pred_np,
        labels=labels,
        target_names=names,
        zero_division=0,
        output_dict=True,
    )

    # Override sklearn scalar/per-class values with Torch-derived values for consistency.
    for cls, name in zip(labels, names):
        report_dict.setdefault(name, {})
        report_dict[name]["precision"] = torch_metrics.get("precision", {}).get(cls, 0.0) / 100.0
        report_dict[name]["recall"] = torch_metrics.get("recall", {}).get(cls, 0.0) / 100.0
        report_dict[name]["f1-score"] = torch_metrics.get("f1_per_class", {}).get(cls, 0.0) / 100.0
        report_dict[name]["support"] = torch_metrics.get("support", {}).get(cls, 0)

    report_dict["torch_metrics"] = make_json_serializable(torch_metrics)

    report_text = classification_report(
        y_true_np,
        y_pred_np,
        labels=labels,
        target_names=names,
        zero_division=0,
        digits=4,
    )

    old_new_text = ""
    if old_class_count is not None and int(old_class_count) > 0:
        split = torch_old_new_metrics(y_true_np, y_pred_np, old_class_count=int(old_class_count))
        old_new_text = (
            "\n\nOld/New Split\n"
            "-------------\n"
            f"Old Accuracy: {split['old_accuracy']:.4f}%\n"
            f"New Accuracy: {split['new_accuracy']:.4f}%\n"
            f"Harmonic Mean: {split['harmonic_mean']:.4f}%\n"
            f"Old Samples: {int(split['old_count'])}\n"
            f"New Samples: {int(split['new_count'])}\n"
        )
        report_dict["old_new_split"] = split

    invalid_prediction_rate = 0.0
    if seen_classes is not None:
        seen = set(int(c) for c in seen_classes)
        invalid_prediction_rate = float(np.mean([int(v) not in seen for v in y_pred_np]) * 100.0)
        report_dict["invalid_prediction_rate"] = invalid_prediction_rate

    base_name = f"phase_{phase}"
    txt_path = os.path.join(save_dir, f"{base_name}_classification_report.txt")
    json_path = os.path.join(save_dir, f"{base_name}_classification_report.json")
    cm_csv_path = os.path.join(save_dir, f"{base_name}_confusion_matrix.csv")
    cm_npy_path = os.path.join(save_dir, f"{base_name}_confusion_matrix.npy")
    per_class_csv_path = os.path.join(save_dir, f"{base_name}_per_class_metrics.csv")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Classification Report - Phase {phase}\n")
        f.write("=" * 90 + "\n\n")
        f.write(f"OA: {torch_metrics['overall_accuracy']:.4f}%\n")
        f.write(f"AA: {torch_metrics['average_accuracy']:.4f}%\n")
        f.write(f"Kappa: {torch_metrics['kappa']:.4f}%\n")
        f.write(f"Macro-F1: {torch_metrics['f1_macro']:.4f}%\n\n")
        f.write(report_text)
        f.write(old_new_text)
        if seen_classes is not None:
            f.write("\nInvalid Prediction Rate\n")
            f.write("-----------------------\n")
            f.write(f"{invalid_prediction_rate:.4f}%\n")
            f.write(f"Predicted unseen classes: {torch_metrics.get('predicted_unseen_classes', [])}\n")
            f.write(f"Predicted unseen count: {torch_metrics.get('predicted_unseen_count', 0)}\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(make_json_serializable(report_dict), f, indent=2)

    with open(cm_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + names)
        for i, name in enumerate(names):
            writer.writerow([name] + [int(v) for v in cm_report[i].tolist()])

    np.save(cm_npy_path, cm_report)

    with open(per_class_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "class_id",
            "class_name",
            "precision_percent",
            "recall_percent",
            "f1_percent",
            "accuracy_percent",
            "support",
            "predicted_count",
        ])
        for cls, name in zip(labels, names):
            writer.writerow([
                int(cls),
                name,
                float(torch_metrics.get("precision", {}).get(cls, 0.0)),
                float(torch_metrics.get("recall", {}).get(cls, 0.0)),
                float(torch_metrics.get("f1_per_class", {}).get(cls, 0.0)),
                float(torch_metrics.get("per_class_accuracy", {}).get(cls, 0.0)),
                int(torch_metrics.get("support", {}).get(cls, 0)),
                int(torch_metrics.get("predicted_count", {}).get(cls, 0)),
            ])

    print(f"[Report] Saved structured classification report: {txt_path}")
    print(f"[Report] Saved classification report JSON: {json_path}")
    print(f"[Report] Saved confusion matrix CSV: {cm_csv_path}")
    print(f"[Report] Saved confusion matrix NPY: {cm_npy_path}")
    print(f"[Report] Saved per-class metrics CSV: {per_class_csv_path}")

    return {
        "txt_path": txt_path,
        "json_path": json_path,
        "confusion_matrix_csv_path": cm_csv_path,
        "confusion_matrix_npy_path": cm_npy_path,
        "per_class_csv_path": per_class_csv_path,
        "report": report_dict,
        "torch_metrics": torch_metrics,
        "confusion_matrix": cm_report,
        "labels": labels,
        "names": names,
    }


def save_hsi_style_classification_report(
    y_true,
    y_pred,
    target_names: Optional[List[str]] = None,
    save_path: str = "./classification_report.csv",
    tr_time: Optional[float] = None,
    te_time: Optional[float] = None,
    dl_time: Optional[float] = None,
    seen_classes: Optional[Iterable[int]] = None,
    old_class_count: Optional[int] = None,
    ignore_index: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Save HSI-style classification report:

        Tr_Time
        Te_Time
        DL_Time
        Kappa accuracy (%)
        Overall accuracy (%)
        Average accuracy (%)
        sklearn-style classification_report
        per-class accuracy vector
        confusion matrix
        old/new/harmonic summary when applicable
    """

    save_parent = os.path.dirname(save_path)
    if save_parent:
        os.makedirs(save_parent, exist_ok=True)

    y_true_np, y_pred_np = _filter_valid_labels_np(
        y_true=np.asarray(y_true).reshape(-1),
        y_pred=np.asarray(y_pred).reshape(-1),
        seen_classes=seen_classes,
        ignore_index=ignore_index,
    )

    if y_true_np.size == 0:
        raise ValueError("No valid labels remain after filtering.")

    if seen_classes is None:
        labels = sorted(np.unique(y_true_np).astype(int).tolist())
    else:
        labels = [int(c) for c in seen_classes]

    names = [_safe_class_name(target_names, c) for c in labels]

    num_classes_eff = int(max(labels)) + 1 if labels else 0
    torch_metrics = calculate_metrics_torch(
        y_true=y_true_np,
        y_pred=y_pred_np,
        num_classes=num_classes_eff,
        old_class_count=old_class_count,
        seen_classes=seen_classes,
        ignore_index=None,
        device="cpu",
    )

    cm_full = torch_metrics["confusion_matrix"].detach().cpu().numpy()
    cm_report = cm_full[np.ix_(labels, labels)]

    per_class_acc = [
        float(torch_metrics.get("per_class_accuracy", {}).get(int(cls), 0.0))
        for cls in labels
    ]

    report_text = classification_report(
        y_true_np,
        y_pred_np,
        labels=labels,
        target_names=names,
        digits=4,
        zero_division=0,
    )

    old_new_text = ""
    old_new_split = None
    if old_class_count is not None and int(old_class_count) > 0:
        old_new_split = {
            "old_accuracy": torch_metrics.get("old_accuracy", 0.0),
            "new_accuracy": torch_metrics.get("new_accuracy", 0.0),
            "harmonic_mean": torch_metrics.get("harmonic_mean", 0.0),
            "old_count": torch_metrics.get("old_count", 0),
            "new_count": torch_metrics.get("new_count", 0),
        }

        old_new_text = (
            f"\n{old_new_split['old_accuracy']} Old accuracy (%)\n"
            f"{old_new_split['new_accuracy']} New accuracy (%)\n"
            f"{old_new_split['harmonic_mean']} Harmonic mean (%)\n"
            f"{int(old_new_split['old_count'])} Old samples\n"
            f"{int(old_new_split['new_count'])} New samples\n"
        )

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"{0.0 if tr_time is None else float(tr_time)} Tr_Time\n")
        f.write(f"{0.0 if te_time is None else float(te_time)} Te_Time\n")
        f.write(f"{0.0 if dl_time is None else float(dl_time)} DL_Time\n")
        f.write(f"{torch_metrics['kappa']} Kappa accuracy (%)\n")
        f.write(f"{torch_metrics['overall_accuracy']} Overall accuracy (%)\n")
        f.write(f"{torch_metrics['average_accuracy']} Average accuracy (%)\n")
        f.write(f"{torch_metrics['f1_macro']} Macro F1 (%)\n")
        f.write(report_text)
        f.write("\n")
        f.write(str(np.asarray(per_class_acc)))
        f.write("\n")
        f.write(str(cm_report))
        f.write("\n")
        f.write(old_new_text)

    print(f"[Report] Saved HSI-style classification report: {save_path}")

    return {
        "save_path": save_path,
        "overall_accuracy": torch_metrics["overall_accuracy"],
        "average_accuracy": torch_metrics["average_accuracy"],
        "kappa": torch_metrics["kappa"],
        "f1_macro": torch_metrics["f1_macro"],
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": cm_report,
        "old_new_split": old_new_split,
        "torch_metrics": torch_metrics,
    }


def save_classification_report(
    y_true,
    y_pred,
    target_names: Optional[List[str]] = None,
    save_dir: str = "./results",
    phase=0,
    seen_classes: Optional[Iterable[int]] = None,
    old_class_count: Optional[int] = None,
    ignore_index: Optional[int] = None,
    include_predicted_unseen: bool = True,
    save_hsi_style: bool = True,
    save_structured: bool = True,
    tr_time: Optional[float] = None,
    te_time: Optional[float] = None,
    dl_time: Optional[float] = None,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Backward-compatible wrapper.

    Existing main.py call:
        save_classification_report(..., save_dir=phase_dir, phase=phase)

    Direct HSI-style call:
        save_classification_report(..., save_path="...csv")
    """

    if save_path is not None:
        return save_hsi_style_classification_report(
            y_true=y_true,
            y_pred=y_pred,
            target_names=target_names,
            save_path=save_path,
            tr_time=tr_time,
            te_time=te_time,
            dl_time=dl_time,
            seen_classes=seen_classes,
            old_class_count=old_class_count,
            ignore_index=ignore_index,
        )

    os.makedirs(save_dir, exist_ok=True)
    output: Dict[str, Any] = {}

    if save_structured:
        output["structured"] = save_structured_classification_report(
            y_true=y_true,
            y_pred=y_pred,
            target_names=target_names,
            save_dir=save_dir,
            phase=phase,
            seen_classes=seen_classes,
            old_class_count=old_class_count,
            ignore_index=ignore_index,
            include_predicted_unseen=include_predicted_unseen,
        )

    if save_hsi_style:
        hsi_path = os.path.join(save_dir, f"phase_{phase}_HSI_Classification_Report.csv")
        output["hsi_style"] = save_hsi_style_classification_report(
            y_true=y_true,
            y_pred=y_pred,
            target_names=target_names,
            save_path=hsi_path,
            tr_time=tr_time,
            te_time=te_time,
            dl_time=dl_time,
            seen_classes=seen_classes,
            old_class_count=old_class_count,
            ignore_index=ignore_index,
        )

    if "structured" in output:
        output.update({
            k: v
            for k, v in output["structured"].items()
            if k in {
                "txt_path",
                "json_path",
                "confusion_matrix_csv_path",
                "confusion_matrix_npy_path",
                "per_class_csv_path",
            }
        })

    if "hsi_style" in output:
        output["hsi_style_path"] = output["hsi_style"].get("save_path")

    return output


# =========================================================
# NECIL Evaluator
# =========================================================
class NECILEvaluator:
    """
    Tracks CIL performance across phases.
    """

    def __init__(self):
        self.phase_history: Dict[int, Dict] = {}
        self.class_acc_history = defaultdict(list)
        self.class_presence_history = defaultdict(list)
        self.phases_seen: List[int] = []

    def _sanity_check_labels(self, y_true, y_pred):
        y_true = _as_1d_np(y_true, "y_true")
        y_pred = _as_1d_np(y_pred, "y_pred")

        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError(f"y_true/y_pred length mismatch: {len(y_true)} vs {len(y_pred)}")

        if y_true.min() < 0:
            raise ValueError(f"Negative true labels detected: min true={y_true.min()}.")

        return y_true, y_pred

    def update(
        self,
        phase: int,
        y_true,
        y_pred,
        old_class_count: Optional[int] = None,
        seen_classes: Optional[Iterable[int]] = None,
        ignore_index: Optional[int] = None,
    ):
        phase = int(phase)
        y_true, y_pred = self._sanity_check_labels(y_true, y_pred)

        if seen_classes is not None:
            seen_list = [int(c) for c in seen_classes]
            max_seen = max(seen_list) if seen_list else -1
            max_pred = int(np.asarray(y_pred).reshape(-1).max()) if np.asarray(y_pred).size > 0 else -1
            max_true = int(np.asarray(y_true).reshape(-1).max()) if np.asarray(y_true).size > 0 else -1
            num_classes_eff = max(max_seen, max_pred, max_true) + 1
        else:
            num_classes_eff = None

        current_metrics = calculate_metrics_torch(
            y_true=y_true,
            y_pred=y_pred,
            num_classes=num_classes_eff,
            old_class_count=old_class_count,
            seen_classes=seen_classes,
            ignore_index=ignore_index,
            device="cpu",
        )

        self.phase_history[phase] = current_metrics

        if phase not in self.phases_seen:
            self.phases_seen.append(phase)
            self.phases_seen.sort()

        self._rebuild_class_history()

    def save_phase_report(
        self,
        phase: int,
        y_true,
        y_pred,
        target_names: Optional[List[str]] = None,
        save_dir: str = "./results",
        seen_classes: Optional[Iterable[int]] = None,
        old_class_count: Optional[int] = None,
        ignore_index: Optional[int] = None,
        tr_time: Optional[float] = None,
        te_time: Optional[float] = None,
        dl_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        return save_classification_report(
            y_true=y_true,
            y_pred=y_pred,
            target_names=target_names,
            save_dir=save_dir,
            phase=phase,
            seen_classes=seen_classes,
            old_class_count=old_class_count,
            ignore_index=ignore_index,
            tr_time=tr_time,
            te_time=te_time,
            dl_time=dl_time,
            save_hsi_style=True,
            save_structured=True,
        )

    def _rebuild_class_history(self):
        all_classes = set()
        for p in self.phases_seen:
            per_class = self.phase_history[p].get("per_class_accuracy", {})
            all_classes.update(int(k) for k in per_class.keys())

        new_hist = defaultdict(list)
        presence = defaultdict(list)

        for cls in sorted(all_classes):
            for p in self.phases_seen:
                per_class = self.phase_history[p].get("per_class_accuracy", {})
                if cls in per_class:
                    new_hist[cls].append(float(per_class[cls]))
                    presence[cls].append(True)
                else:
                    new_hist[cls].append(np.nan)
                    presence[cls].append(False)

        self.class_acc_history = new_hist
        self.class_presence_history = presence

    def calculate_forgetting_per_class(self) -> Dict[int, float]:
        if len(self.phases_seen) < 2:
            return {}

        forgetting: Dict[int, float] = {}
        for cls, history in self.class_acc_history.items():
            vals = np.asarray(history, dtype=float)
            vals = vals[~np.isnan(vals)]

            if vals.size < 2:
                continue

            peak_before_last = float(np.max(vals[:-1]))
            current = float(vals[-1])
            forgetting[int(cls)] = float(max(0.0, peak_before_last - current))

        return forgetting

    def calculate_backward_transfer(self) -> float:
        transfers = []
        for _, history in self.class_acc_history.items():
            vals = np.asarray(history, dtype=float)
            vals = vals[~np.isnan(vals)]
            if vals.size >= 2:
                transfers.append(float(vals[-1] - vals[0]))

        return float(np.mean(transfers)) if transfers else 0.0

    def get_standard_metrics(self) -> Dict[str, float]:
        if not self.phases_seen:
            return {}

        last_phase = self.phases_seen[-1]
        all_oas = [float(self.phase_history[p]["overall_accuracy"]) for p in self.phases_seen]
        all_hm = [float(self.phase_history[p].get("harmonic_mean", 0.0)) for p in self.phases_seen]

        forgetting_per_class = self.calculate_forgetting_per_class()
        avg_forgetting = (
            float(np.mean(list(forgetting_per_class.values())))
            if forgetting_per_class
            else 0.0
        )

        last_metrics = self.phase_history[last_phase]

        return {
            "A_last (Final Accuracy)": float(last_metrics["overall_accuracy"]),
            "A_avg (Avg Accuracy)": float(np.mean(all_oas)),
            "H_last (Final Harmonic Mean)": float(last_metrics.get("harmonic_mean", 0.0)),
            "H_avg (Avg Harmonic Mean)": float(np.mean(all_hm)),
            "F_avg (Avg Forgetting)": avg_forgetting,
            "BWT (Backward Transfer)": self.calculate_backward_transfer(),
            "Old_last (Final Old Accuracy)": float(last_metrics.get("old_accuracy", 0.0)),
            "New_last (Final New Accuracy)": float(last_metrics.get("new_accuracy", 0.0)),
            "AA_last (Final Avg Accuracy)": float(last_metrics.get("average_accuracy", 0.0)),
            "Kappa_last": float(last_metrics.get("kappa", 0.0)),
            "F1_last": float(last_metrics.get("f1_macro", 0.0)),
            "Phases": len(self.phases_seen),
        }

    def get_phase_table(self) -> List[Dict[str, Any]]:
        rows = []
        for p in self.phases_seen:
            m = self.phase_history[p]
            rows.append({
                "phase": int(p),
                "OA": float(m.get("overall_accuracy", 0.0)),
                "AA": float(m.get("average_accuracy", 0.0)),
                "Kappa": float(m.get("kappa", 0.0)),
                "F1": float(m.get("f1_macro", 0.0)),
                "Old": float(m.get("old_accuracy", 0.0)),
                "New": float(m.get("new_accuracy", 0.0)),
                "H": float(m.get("harmonic_mean", 0.0)),
                "Samples": int(m.get("num_samples", 0)),
            })
        return rows

    def get_per_class_summary(self) -> Dict[int, Dict[str, float]]:
        forgetting = self.calculate_forgetting_per_class()
        summary = {}

        for cls, history in self.class_acc_history.items():
            vals = np.asarray(history, dtype=float)
            valid = vals[~np.isnan(vals)]
            if valid.size == 0:
                continue

            summary[int(cls)] = {
                "first": float(valid[0]),
                "best": float(np.max(valid)),
                "last": float(valid[-1]),
                "forgetting": float(forgetting.get(int(cls), 0.0)),
            }

        return summary

    def print_summary(self):
        if not self.phases_seen:
            print("[NECILEvaluator] No phases evaluated yet.")
            return

        metrics = self.get_standard_metrics()
        last_phase = self.phases_seen[-1]
        phase_metrics = self.phase_history[last_phase]

        print("\n" + "=" * 58)
        print(f" NECIL-HSI Evaluation Report (Phase {last_phase})")
        print("=" * 58)
        print(f" 1. Final Accuracy (A_last):      {metrics.get('A_last (Final Accuracy)', 0):.2f}%")
        print(f" 2. Avg Accuracy (A_avg):         {metrics.get('A_avg (Avg Accuracy)', 0):.2f}%")
        print(f" 3. Avg Forgetting (F_avg):       {metrics.get('F_avg (Avg Forgetting)', 0):.2f}%")
        print(f" 4. Backward Transfer (BWT):      {metrics.get('BWT (Backward Transfer)', 0):.2f}%")
        print(f" 5. Old Accuracy:                 {phase_metrics.get('old_accuracy', 0):.2f}%")
        print(f" 6. New Accuracy:                 {phase_metrics.get('new_accuracy', 0):.2f}%")
        print(f" 7. Harmonic Mean:                {phase_metrics.get('harmonic_mean', 0):.2f}%")
        print(
            f" 8. AA / Kappa / F1:              "
            f"{phase_metrics.get('average_accuracy', 0):.2f}% / "
            f"{phase_metrics.get('kappa', 0):.2f}% / "
            f"{phase_metrics.get('f1_macro', 0):.2f}%"
        )
        print("-" * 58)


    def save_phase_table_csv(self, save_path: str) -> str:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        rows = self.get_phase_table()
        fieldnames = ["phase", "OA", "AA", "Kappa", "F1", "Old", "New", "H", "Samples"]
        with open(save_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return save_path

    def to_dict(self) -> Dict[str, Any]:
        return make_json_serializable({
            "phase_history": self.phase_history,
            "standard_metrics": self.get_standard_metrics(),
            "phase_table": self.get_phase_table(),
            "per_class_summary": self.get_per_class_summary(),
        })













# """
# Evaluation Metrics & Reports for Geometry-Native NECIL-HSI
# ==========================================================

# Torch-native metric core:
# - confusion matrix
# - OA / AA / Kappa / Macro-F1
# - precision / recall / F1 / support
# - old/new split accuracy
# - harmonic mean

# Report outputs:
# - phase_X_classification_report.txt
# - phase_X_classification_report.json
# - phase_X_confusion_matrix.csv
# - phase_X_confusion_matrix.npy
# - phase_X_per_class_metrics.csv
# - phase_X_HSI_Classification_Report.csv

# Important:
# This file assumes labels are already sequential class IDs: 0..K-1.
# """

# import csv
# import json
# import os
# from collections import defaultdict
# from typing import Dict, List, Optional, Iterable, Any

# import numpy as np
# import torch
# from sklearn.metrics import classification_report


# # =========================================================
# # Core utilities
# # =========================================================
# def _as_1d_np(x, name: str) -> np.ndarray:
#     arr = np.asarray(x).reshape(-1)
#     if arr.size == 0:
#         raise ValueError(f"{name} is empty.")
#     return arr


# def _to_1d_long_tensor(x, device: Optional[torch.device] = None) -> torch.Tensor:
#     if torch.is_tensor(x):
#         t = x.detach()
#     else:
#         t = torch.as_tensor(x)
#     if device is not None:
#         t = t.to(device)
#     return t.long().view(-1)


# def _safe_class_name(target_names: Optional[List[str]], cls: int) -> str:
#     cls = int(cls)
#     if target_names is not None and 0 <= cls < len(target_names):
#         return str(target_names[cls])
#     return f"Class {cls}"


# def _filter_valid_labels_np(
#     y_true: np.ndarray,
#     y_pred: np.ndarray,
#     seen_classes: Optional[Iterable[int]] = None,
#     ignore_index: Optional[int] = None,
# ):
#     y_true = np.asarray(y_true).reshape(-1)
#     y_pred = np.asarray(y_pred).reshape(-1)

#     if y_true.shape[0] != y_pred.shape[0]:
#         raise ValueError(
#             f"y_true and y_pred must have same length. Got {len(y_true)} vs {len(y_pred)}"
#         )

#     valid = np.ones_like(y_true, dtype=bool)

#     if ignore_index is not None:
#         valid &= y_true != int(ignore_index)

#     if seen_classes is not None:
#         seen = set(int(c) for c in seen_classes)
#         valid &= np.array([int(v) in seen for v in y_true], dtype=bool)

#     return y_true[valid], y_pred[valid]


# def _filter_valid_labels_torch(
#     y_true: torch.Tensor,
#     y_pred: torch.Tensor,
#     seen_classes: Optional[Iterable[int]] = None,
#     ignore_index: Optional[int] = None,
# ):
#     if y_true.numel() != y_pred.numel():
#         raise ValueError(
#             f"y_true and y_pred must have same length. Got {y_true.numel()} vs {y_pred.numel()}"
#         )

#     valid = torch.ones_like(y_true, dtype=torch.bool)

#     if ignore_index is not None:
#         valid &= y_true != int(ignore_index)

#     if seen_classes is not None:
#         seen = torch.as_tensor(list(seen_classes), device=y_true.device, dtype=torch.long)
#         if hasattr(torch, "isin"):
#             valid &= torch.isin(y_true, seen)
#         else:
#             seen_mask = torch.zeros_like(valid)
#             for c in seen:
#                 seen_mask |= y_true == c
#             valid &= seen_mask

#     return y_true[valid], y_pred[valid]


# def _labels_for_report(
#     y_true: np.ndarray,
#     y_pred: np.ndarray,
#     seen_classes: Optional[Iterable[int]] = None,
#     include_predicted_unseen: bool = False,
# ) -> List[int]:
#     if seen_classes is not None:
#         labels = [int(c) for c in seen_classes]
#     else:
#         labels = sorted(int(c) for c in np.unique(y_true).tolist())

#     if include_predicted_unseen:
#         label_set = set(labels)
#         extra = sorted(int(c) for c in np.unique(y_pred).tolist() if int(c) not in label_set)
#         labels = labels + extra

#     return labels


# def make_json_serializable(obj):
#     if isinstance(obj, dict):
#         return {str(k): make_json_serializable(v) for k, v in obj.items()}

#     if isinstance(obj, list):
#         return [make_json_serializable(v) for v in obj]

#     if isinstance(obj, tuple):
#         return [make_json_serializable(v) for v in obj]

#     if isinstance(obj, np.ndarray):
#         return obj.tolist()

#     if isinstance(obj, np.integer):
#         return int(obj)

#     if isinstance(obj, np.floating):
#         return float(obj)

#     if isinstance(obj, np.bool_):
#         return bool(obj)

#     if torch.is_tensor(obj):
#         return obj.detach().cpu().tolist()

#     return obj


# # =========================================================
# # Torch-native metrics
# # =========================================================
# @torch.no_grad()
# def torch_confusion_matrix(
#     y_true,
#     y_pred,
#     num_classes: int,
#     device: Optional[str] = "cpu",
# ) -> torch.Tensor:
#     """
#     Fast multiclass confusion matrix.

#     Rows    = true class
#     Columns = predicted class
#     """
#     dev = torch.device(device) if device is not None else None
#     y_true = _to_1d_long_tensor(y_true, dev)
#     y_pred = _to_1d_long_tensor(y_pred, dev)

#     valid = (
#         (y_true >= 0)
#         & (y_true < int(num_classes))
#         & (y_pred >= 0)
#         & (y_pred < int(num_classes))
#     )

#     y_true = y_true[valid]
#     y_pred = y_pred[valid]

#     indices = y_true * int(num_classes) + y_pred
#     cm = torch.bincount(
#         indices,
#         minlength=int(num_classes) * int(num_classes),
#     ).reshape(int(num_classes), int(num_classes))

#     return cm


# @torch.no_grad()
# def torch_metrics_from_confusion_matrix(
#     cm: torch.Tensor,
#     eps: float = 1e-12,
# ) -> Dict[str, Any]:
#     """
#     Compute multiclass metrics from a confusion matrix.

#     cm[row, col]
#     row = true label
#     col = predicted label
#     """
#     cm = cm.float()

#     tp = torch.diag(cm)
#     support = cm.sum(dim=1)
#     predicted = cm.sum(dim=0)
#     total = cm.sum()

#     recall = tp / (support + eps)
#     precision = tp / (predicted + eps)
#     f1 = 2.0 * precision * recall / (precision + recall + eps)

#     valid_classes = support > 0

#     oa = tp.sum() / (total + eps) * 100.0
#     aa = recall[valid_classes].mean() * 100.0 if valid_classes.any() else torch.tensor(0.0, device=cm.device)
#     macro_f1 = f1[valid_classes].mean() * 100.0 if valid_classes.any() else torch.tensor(0.0, device=cm.device)

#     po = tp.sum() / (total + eps)
#     pe = (cm.sum(dim=1) * cm.sum(dim=0)).sum() / ((total * total) + eps)
#     kappa = (po - pe) / (1.0 - pe + eps) * 100.0

#     per_class_accuracy = recall * 100.0
#     precision_pct = precision * 100.0
#     recall_pct = recall * 100.0
#     f1_pct = f1 * 100.0

#     return {
#         "overall_accuracy": float(oa.item()),
#         "balanced_accuracy": float(aa.item()),
#         "average_accuracy": float(aa.item()),
#         "kappa": float(kappa.item()),
#         "f1_macro": float(macro_f1.item()),
#         "per_class_accuracy": {
#             int(i): float(per_class_accuracy[i].item())
#             for i in range(cm.shape[0])
#         },
#         "precision": {
#             int(i): float(precision_pct[i].item())
#             for i in range(cm.shape[0])
#         },
#         "recall": {
#             int(i): float(recall_pct[i].item())
#             for i in range(cm.shape[0])
#         },
#         "f1_per_class": {
#             int(i): float(f1_pct[i].item())
#             for i in range(cm.shape[0])
#         },
#         "support": {
#             int(i): int(support[i].item())
#             for i in range(cm.shape[0])
#         },
#         "predicted_count": {
#             int(i): int(predicted[i].item())
#             for i in range(cm.shape[0])
#         },
#         "confusion_matrix": cm.detach().cpu(),
#     }


# @torch.no_grad()
# def torch_old_new_metrics(
#     y_true,
#     y_pred,
#     old_class_count: Optional[int] = None,
#     eps: float = 1e-12,
# ) -> Dict[str, Any]:
#     y_true = _to_1d_long_tensor(y_true, None)
#     y_pred = _to_1d_long_tensor(y_pred, y_true.device)

#     overall_acc = (y_true == y_pred).float().mean() * 100.0 if y_true.numel() > 0 else torch.tensor(0.0)

#     if old_class_count is None or int(old_class_count) <= 0:
#         return {
#             "old_accuracy": 0.0,
#             "new_accuracy": float(overall_acc.item()),
#             "harmonic_mean": 0.0,
#             "old_count": 0,
#             "new_count": int(y_true.numel()),
#         }

#     old_class_count = int(old_class_count)

#     old_mask = y_true < old_class_count
#     new_mask = y_true >= old_class_count

#     old_total = int(old_mask.sum().item())
#     new_total = int(new_mask.sum().item())

#     if old_total > 0:
#         old_acc = (y_pred[old_mask] == y_true[old_mask]).float().mean() * 100.0
#     else:
#         old_acc = torch.tensor(0.0, device=y_true.device)

#     if new_total > 0:
#         new_acc = (y_pred[new_mask] == y_true[new_mask]).float().mean() * 100.0
#     else:
#         new_acc = torch.tensor(0.0, device=y_true.device)

#     h = 2.0 * old_acc * new_acc / (old_acc + new_acc + eps)

#     return {
#         "old_accuracy": float(old_acc.item()),
#         "new_accuracy": float(new_acc.item()),
#         "harmonic_mean": float(h.item()),
#         "old_count": old_total,
#         "new_count": new_total,
#     }


# @torch.no_grad()
# def calculate_metrics_torch(
#     y_true,
#     y_pred,
#     num_classes: Optional[int] = None,
#     old_class_count: Optional[int] = None,
#     seen_classes: Optional[Iterable[int]] = None,
#     ignore_index: Optional[int] = None,
#     device: str = "cpu",
# ) -> Dict[str, Any]:
#     """
#     Torch-native snapshot metrics for one phase evaluation.

#     y_true/y_pred can be:
#     - torch.Tensor
#     - numpy.ndarray
#     - list
#     """
#     dev = torch.device(device)
#     y_true_t = _to_1d_long_tensor(y_true, dev)
#     y_pred_t = _to_1d_long_tensor(y_pred, dev)

#     y_true_t, y_pred_t = _filter_valid_labels_torch(
#         y_true=y_true_t,
#         y_pred=y_pred_t,
#         seen_classes=seen_classes,
#         ignore_index=ignore_index,
#     )

#     if y_true_t.numel() == 0:
#         raise ValueError("No valid labels remain after filtering.")

#     if seen_classes is not None:
#         labels = [int(c) for c in seen_classes]
#         num_classes_eff = int(max(labels)) + 1 if labels else 0
#     elif num_classes is not None:
#         num_classes_eff = int(num_classes)
#         labels = list(range(num_classes_eff))
#     else:
#         max_label = int(torch.cat([y_true_t, y_pred_t]).max().item())
#         num_classes_eff = max_label + 1
#         labels = list(range(num_classes_eff))

#     cm_full = torch_confusion_matrix(
#         y_true=y_true_t,
#         y_pred=y_pred_t,
#         num_classes=num_classes_eff,
#         device=dev,
#     )

#     metrics = torch_metrics_from_confusion_matrix(cm_full)

#     # Keep only seen/present classes for phase history if seen_classes was provided.
#     if seen_classes is not None:
#         seen_set = set(int(c) for c in seen_classes)
#         metrics["per_class_accuracy"] = {
#             int(k): float(v)
#             for k, v in metrics["per_class_accuracy"].items()
#             if int(k) in seen_set
#         }
#         metrics["precision"] = {
#             int(k): float(v)
#             for k, v in metrics["precision"].items()
#             if int(k) in seen_set
#         }
#         metrics["recall"] = {
#             int(k): float(v)
#             for k, v in metrics["recall"].items()
#             if int(k) in seen_set
#         }
#         metrics["f1_per_class"] = {
#             int(k): float(v)
#             for k, v in metrics["f1_per_class"].items()
#             if int(k) in seen_set
#         }
#         metrics["support"] = {
#             int(k): int(v)
#             for k, v in metrics["support"].items()
#             if int(k) in seen_set
#         }

#     split_metrics = torch_old_new_metrics(
#         y_true=y_true_t,
#         y_pred=y_pred_t,
#         old_class_count=old_class_count,
#     )
#     metrics.update(split_metrics)

#     if seen_classes is not None:
#         seen = torch.as_tensor(list(seen_classes), device=dev, dtype=torch.long)
#         if hasattr(torch, "isin"):
#             invalid = ~torch.isin(y_pred_t, seen)
#         else:
#             valid_pred = torch.zeros_like(y_pred_t, dtype=torch.bool)
#             for c in seen:
#                 valid_pred |= y_pred_t == c
#             invalid = ~valid_pred
#         metrics["invalid_prediction_rate"] = float(invalid.float().mean().item() * 100.0)
#     else:
#         metrics["invalid_prediction_rate"] = 0.0

#     present_classes = sorted(int(c) for c in torch.unique(y_true_t).detach().cpu().tolist())

#     metrics["num_samples"] = int(y_true_t.numel())
#     metrics["num_classes"] = int(len(present_classes) if seen_classes is None else len(list(seen_classes)))
#     metrics["classes"] = [int(c) for c in (seen_classes if seen_classes is not None else present_classes)]
#     metrics["confusion_matrix"] = cm_full.detach().cpu()

#     return metrics


# # Backward-compatible name used by NECILEvaluator.update
# def calculate_metrics(
#     y_true,
#     y_pred,
#     class_names: Optional[List[str]] = None,
#     old_class_count: Optional[int] = None,
#     seen_classes: Optional[Iterable[int]] = None,
#     ignore_index: Optional[int] = None,
# ) -> Dict[str, Any]:
#     del class_names

#     if seen_classes is not None:
#         seen_list = [int(c) for c in seen_classes]
#         num_classes = int(max(seen_list)) + 1 if seen_list else None
#     else:
#         yt = np.asarray(y_true).reshape(-1)
#         yp = np.asarray(y_pred).reshape(-1)
#         num_classes = int(max(yt.max(), yp.max())) + 1 if yt.size > 0 else None

#     return calculate_metrics_torch(
#         y_true=y_true,
#         y_pred=y_pred,
#         num_classes=num_classes,
#         old_class_count=old_class_count,
#         seen_classes=seen_classes,
#         ignore_index=ignore_index,
#         device="cpu",
#     )


# # =========================================================
# # Report savers
# # =========================================================
# def save_structured_classification_report(
#     y_true,
#     y_pred,
#     target_names: Optional[List[str]] = None,
#     save_dir: str = "./results",
#     phase=0,
#     seen_classes: Optional[Iterable[int]] = None,
#     old_class_count: Optional[int] = None,
#     ignore_index: Optional[int] = None,
#     include_predicted_unseen: bool = True,
# ) -> Dict[str, Any]:
#     """
#     Save machine-readable class-wise classification report and Torch confusion matrix.

#     Saves:
#     - phase_X_classification_report.txt
#     - phase_X_classification_report.json
#     - phase_X_confusion_matrix.csv
#     - phase_X_confusion_matrix.npy
#     - phase_X_per_class_metrics.csv
#     """

#     os.makedirs(save_dir, exist_ok=True)

#     y_true_np, y_pred_np = _filter_valid_labels_np(
#         y_true=np.asarray(y_true).reshape(-1),
#         y_pred=np.asarray(y_pred).reshape(-1),
#         seen_classes=seen_classes,
#         ignore_index=ignore_index,
#     )

#     if y_true_np.size == 0:
#         raise ValueError("No valid labels remain after filtering for classification report.")

#     labels = _labels_for_report(
#         y_true=y_true_np,
#         y_pred=y_pred_np,
#         seen_classes=seen_classes,
#         include_predicted_unseen=include_predicted_unseen,
#     )
#     names = [_safe_class_name(target_names, c) for c in labels]

#     num_classes_eff = int(max(labels)) + 1 if labels else 0
#     cm_full = torch_confusion_matrix(y_true_np, y_pred_np, num_classes_eff, device="cpu")
#     cm_np = cm_full.detach().cpu().numpy()
#     cm_report = cm_np[np.ix_(labels, labels)]

#     torch_metrics = calculate_metrics_torch(
#         y_true=y_true_np,
#         y_pred=y_pred_np,
#         num_classes=num_classes_eff,
#         old_class_count=old_class_count,
#         seen_classes=seen_classes,
#         ignore_index=None,
#         device="cpu",
#     )

#     report_dict = classification_report(
#         y_true_np,
#         y_pred_np,
#         labels=labels,
#         target_names=names,
#         zero_division=0,
#         output_dict=True,
#     )

#     # Override sklearn scalar/per-class values with Torch-derived values for consistency.
#     for cls, name in zip(labels, names):
#         report_dict.setdefault(name, {})
#         report_dict[name]["precision"] = torch_metrics.get("precision", {}).get(cls, 0.0) / 100.0
#         report_dict[name]["recall"] = torch_metrics.get("recall", {}).get(cls, 0.0) / 100.0
#         report_dict[name]["f1-score"] = torch_metrics.get("f1_per_class", {}).get(cls, 0.0) / 100.0
#         report_dict[name]["support"] = torch_metrics.get("support", {}).get(cls, 0)

#     report_dict["torch_metrics"] = make_json_serializable(torch_metrics)

#     report_text = classification_report(
#         y_true_np,
#         y_pred_np,
#         labels=labels,
#         target_names=names,
#         zero_division=0,
#         digits=4,
#     )

#     old_new_text = ""
#     if old_class_count is not None and int(old_class_count) > 0:
#         split = torch_old_new_metrics(y_true_np, y_pred_np, old_class_count=int(old_class_count))
#         old_new_text = (
#             "\n\nOld/New Split\n"
#             "-------------\n"
#             f"Old Accuracy: {split['old_accuracy']:.4f}%\n"
#             f"New Accuracy: {split['new_accuracy']:.4f}%\n"
#             f"Harmonic Mean: {split['harmonic_mean']:.4f}%\n"
#             f"Old Samples: {int(split['old_count'])}\n"
#             f"New Samples: {int(split['new_count'])}\n"
#         )
#         report_dict["old_new_split"] = split

#     invalid_prediction_rate = 0.0
#     if seen_classes is not None:
#         seen = set(int(c) for c in seen_classes)
#         invalid_prediction_rate = float(np.mean([int(v) not in seen for v in y_pred_np]) * 100.0)
#         report_dict["invalid_prediction_rate"] = invalid_prediction_rate

#     base_name = f"phase_{phase}"
#     txt_path = os.path.join(save_dir, f"{base_name}_classification_report.txt")
#     json_path = os.path.join(save_dir, f"{base_name}_classification_report.json")
#     cm_csv_path = os.path.join(save_dir, f"{base_name}_confusion_matrix.csv")
#     cm_npy_path = os.path.join(save_dir, f"{base_name}_confusion_matrix.npy")
#     per_class_csv_path = os.path.join(save_dir, f"{base_name}_per_class_metrics.csv")

#     with open(txt_path, "w", encoding="utf-8") as f:
#         f.write(f"Classification Report - Phase {phase}\n")
#         f.write("=" * 90 + "\n\n")
#         f.write(f"OA: {torch_metrics['overall_accuracy']:.4f}%\n")
#         f.write(f"AA: {torch_metrics['average_accuracy']:.4f}%\n")
#         f.write(f"Kappa: {torch_metrics['kappa']:.4f}%\n")
#         f.write(f"Macro-F1: {torch_metrics['f1_macro']:.4f}%\n\n")
#         f.write(report_text)
#         f.write(old_new_text)
#         if seen_classes is not None:
#             f.write("\nInvalid Prediction Rate\n")
#             f.write("-----------------------\n")
#             f.write(f"{invalid_prediction_rate:.4f}%\n")

#     with open(json_path, "w", encoding="utf-8") as f:
#         json.dump(make_json_serializable(report_dict), f, indent=2)

#     with open(cm_csv_path, "w", encoding="utf-8", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["true\\pred"] + names)
#         for i, name in enumerate(names):
#             writer.writerow([name] + [int(v) for v in cm_report[i].tolist()])

#     np.save(cm_npy_path, cm_report)

#     with open(per_class_csv_path, "w", encoding="utf-8", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow([
#             "class_id",
#             "class_name",
#             "precision_percent",
#             "recall_percent",
#             "f1_percent",
#             "accuracy_percent",
#             "support",
#             "predicted_count",
#         ])
#         for cls, name in zip(labels, names):
#             writer.writerow([
#                 int(cls),
#                 name,
#                 float(torch_metrics.get("precision", {}).get(cls, 0.0)),
#                 float(torch_metrics.get("recall", {}).get(cls, 0.0)),
#                 float(torch_metrics.get("f1_per_class", {}).get(cls, 0.0)),
#                 float(torch_metrics.get("per_class_accuracy", {}).get(cls, 0.0)),
#                 int(torch_metrics.get("support", {}).get(cls, 0)),
#                 int(torch_metrics.get("predicted_count", {}).get(cls, 0)),
#             ])

#     print(f"[Report] Saved structured classification report: {txt_path}")
#     print(f"[Report] Saved classification report JSON: {json_path}")
#     print(f"[Report] Saved confusion matrix CSV: {cm_csv_path}")
#     print(f"[Report] Saved confusion matrix NPY: {cm_npy_path}")
#     print(f"[Report] Saved per-class metrics CSV: {per_class_csv_path}")

#     return {
#         "txt_path": txt_path,
#         "json_path": json_path,
#         "confusion_matrix_csv_path": cm_csv_path,
#         "confusion_matrix_npy_path": cm_npy_path,
#         "per_class_csv_path": per_class_csv_path,
#         "report": report_dict,
#         "torch_metrics": torch_metrics,
#         "confusion_matrix": cm_report,
#         "labels": labels,
#         "names": names,
#     }


# def save_hsi_style_classification_report(
#     y_true,
#     y_pred,
#     target_names: Optional[List[str]] = None,
#     save_path: str = "./classification_report.csv",
#     tr_time: Optional[float] = None,
#     te_time: Optional[float] = None,
#     dl_time: Optional[float] = None,
#     seen_classes: Optional[Iterable[int]] = None,
#     old_class_count: Optional[int] = None,
#     ignore_index: Optional[int] = None,
# ) -> Dict[str, Any]:
#     """
#     Save HSI-style classification report:

#         Tr_Time
#         Te_Time
#         DL_Time
#         Kappa accuracy (%)
#         Overall accuracy (%)
#         Average accuracy (%)
#         sklearn-style classification_report
#         per-class accuracy vector
#         confusion matrix
#         old/new/harmonic summary when applicable
#     """

#     save_parent = os.path.dirname(save_path)
#     if save_parent:
#         os.makedirs(save_parent, exist_ok=True)

#     y_true_np, y_pred_np = _filter_valid_labels_np(
#         y_true=np.asarray(y_true).reshape(-1),
#         y_pred=np.asarray(y_pred).reshape(-1),
#         seen_classes=seen_classes,
#         ignore_index=ignore_index,
#     )

#     if y_true_np.size == 0:
#         raise ValueError("No valid labels remain after filtering.")

#     if seen_classes is None:
#         labels = sorted(np.unique(y_true_np).astype(int).tolist())
#     else:
#         labels = [int(c) for c in seen_classes]

#     names = [_safe_class_name(target_names, c) for c in labels]

#     num_classes_eff = int(max(labels)) + 1 if labels else 0
#     torch_metrics = calculate_metrics_torch(
#         y_true=y_true_np,
#         y_pred=y_pred_np,
#         num_classes=num_classes_eff,
#         old_class_count=old_class_count,
#         seen_classes=seen_classes,
#         ignore_index=None,
#         device="cpu",
#     )

#     cm_full = torch_metrics["confusion_matrix"].detach().cpu().numpy()
#     cm_report = cm_full[np.ix_(labels, labels)]

#     per_class_acc = [
#         float(torch_metrics.get("per_class_accuracy", {}).get(int(cls), 0.0))
#         for cls in labels
#     ]

#     report_text = classification_report(
#         y_true_np,
#         y_pred_np,
#         labels=labels,
#         target_names=names,
#         digits=4,
#         zero_division=0,
#     )

#     old_new_text = ""
#     old_new_split = None
#     if old_class_count is not None and int(old_class_count) > 0:
#         old_new_split = {
#             "old_accuracy": torch_metrics.get("old_accuracy", 0.0),
#             "new_accuracy": torch_metrics.get("new_accuracy", 0.0),
#             "harmonic_mean": torch_metrics.get("harmonic_mean", 0.0),
#             "old_count": torch_metrics.get("old_count", 0),
#             "new_count": torch_metrics.get("new_count", 0),
#         }

#         old_new_text = (
#             f"\n{old_new_split['old_accuracy']} Old accuracy (%)\n"
#             f"{old_new_split['new_accuracy']} New accuracy (%)\n"
#             f"{old_new_split['harmonic_mean']} Harmonic mean (%)\n"
#             f"{int(old_new_split['old_count'])} Old samples\n"
#             f"{int(old_new_split['new_count'])} New samples\n"
#         )

#     with open(save_path, "w", encoding="utf-8") as f:
#         f.write(f"{0.0 if tr_time is None else float(tr_time)} Tr_Time\n")
#         f.write(f"{0.0 if te_time is None else float(te_time)} Te_Time\n")
#         f.write(f"{0.0 if dl_time is None else float(dl_time)} DL_Time\n")
#         f.write(f"{torch_metrics['kappa']} Kappa accuracy (%)\n")
#         f.write(f"{torch_metrics['overall_accuracy']} Overall accuracy (%)\n")
#         f.write(f"{torch_metrics['average_accuracy']} Average accuracy (%)\n")
#         f.write(f"{torch_metrics['f1_macro']} Macro F1 (%)\n")
#         f.write(report_text)
#         f.write("\n")
#         f.write(str(np.asarray(per_class_acc)))
#         f.write("\n")
#         f.write(str(cm_report))
#         f.write("\n")
#         f.write(old_new_text)

#     print(f"[Report] Saved HSI-style classification report: {save_path}")

#     return {
#         "save_path": save_path,
#         "overall_accuracy": torch_metrics["overall_accuracy"],
#         "average_accuracy": torch_metrics["average_accuracy"],
#         "kappa": torch_metrics["kappa"],
#         "f1_macro": torch_metrics["f1_macro"],
#         "per_class_accuracy": per_class_acc,
#         "confusion_matrix": cm_report,
#         "old_new_split": old_new_split,
#         "torch_metrics": torch_metrics,
#     }


# def save_classification_report(
#     y_true,
#     y_pred,
#     target_names: Optional[List[str]] = None,
#     save_dir: str = "./results",
#     phase=0,
#     seen_classes: Optional[Iterable[int]] = None,
#     old_class_count: Optional[int] = None,
#     ignore_index: Optional[int] = None,
#     include_predicted_unseen: bool = True,
#     save_hsi_style: bool = True,
#     save_structured: bool = True,
#     tr_time: Optional[float] = None,
#     te_time: Optional[float] = None,
#     dl_time: Optional[float] = None,
#     save_path: Optional[str] = None,
# ) -> Dict[str, Any]:
#     """
#     Backward-compatible wrapper.

#     Existing main.py call:
#         save_classification_report(..., save_dir=phase_dir, phase=phase)

#     Direct HSI-style call:
#         save_classification_report(..., save_path="...csv")
#     """

#     if save_path is not None:
#         return save_hsi_style_classification_report(
#             y_true=y_true,
#             y_pred=y_pred,
#             target_names=target_names,
#             save_path=save_path,
#             tr_time=tr_time,
#             te_time=te_time,
#             dl_time=dl_time,
#             seen_classes=seen_classes,
#             old_class_count=old_class_count,
#             ignore_index=ignore_index,
#         )

#     os.makedirs(save_dir, exist_ok=True)
#     output: Dict[str, Any] = {}

#     if save_structured:
#         output["structured"] = save_structured_classification_report(
#             y_true=y_true,
#             y_pred=y_pred,
#             target_names=target_names,
#             save_dir=save_dir,
#             phase=phase,
#             seen_classes=seen_classes,
#             old_class_count=old_class_count,
#             ignore_index=ignore_index,
#             include_predicted_unseen=include_predicted_unseen,
#         )

#     if save_hsi_style:
#         hsi_path = os.path.join(save_dir, f"phase_{phase}_HSI_Classification_Report.csv")
#         output["hsi_style"] = save_hsi_style_classification_report(
#             y_true=y_true,
#             y_pred=y_pred,
#             target_names=target_names,
#             save_path=hsi_path,
#             tr_time=tr_time,
#             te_time=te_time,
#             dl_time=dl_time,
#             seen_classes=seen_classes,
#             old_class_count=old_class_count,
#             ignore_index=ignore_index,
#         )

#     if "structured" in output:
#         output.update({
#             k: v
#             for k, v in output["structured"].items()
#             if k in {
#                 "txt_path",
#                 "json_path",
#                 "confusion_matrix_csv_path",
#                 "confusion_matrix_npy_path",
#                 "per_class_csv_path",
#             }
#         })

#     if "hsi_style" in output:
#         output["hsi_style_path"] = output["hsi_style"].get("save_path")

#     return output


# # =========================================================
# # NECIL Evaluator
# # =========================================================
# class NECILEvaluator:
#     """
#     Tracks CIL performance across phases.
#     """

#     def __init__(self):
#         self.phase_history: Dict[int, Dict] = {}
#         self.class_acc_history = defaultdict(list)
#         self.class_presence_history = defaultdict(list)
#         self.phases_seen: List[int] = []

#     def _sanity_check_labels(self, y_true, y_pred):
#         y_true = _as_1d_np(y_true, "y_true")
#         y_pred = _as_1d_np(y_pred, "y_pred")

#         if y_true.shape[0] != y_pred.shape[0]:
#             raise ValueError(f"y_true/y_pred length mismatch: {len(y_true)} vs {len(y_pred)}")

#         if y_true.min() < 0:
#             raise ValueError(f"Negative true labels detected: min true={y_true.min()}.")

#         return y_true, y_pred

#     def update(
#         self,
#         phase: int,
#         y_true,
#         y_pred,
#         old_class_count: Optional[int] = None,
#         seen_classes: Optional[Iterable[int]] = None,
#         ignore_index: Optional[int] = None,
#     ):
#         phase = int(phase)
#         y_true, y_pred = self._sanity_check_labels(y_true, y_pred)

#         current_metrics = calculate_metrics_torch(
#             y_true=y_true,
#             y_pred=y_pred,
#             num_classes=(max(seen_classes) + 1 if seen_classes is not None else None),
#             old_class_count=old_class_count,
#             seen_classes=seen_classes,
#             ignore_index=ignore_index,
#             device="cpu",
#         )

#         self.phase_history[phase] = current_metrics

#         if phase not in self.phases_seen:
#             self.phases_seen.append(phase)
#             self.phases_seen.sort()

#         self._rebuild_class_history()

#     def save_phase_report(
#         self,
#         phase: int,
#         y_true,
#         y_pred,
#         target_names: Optional[List[str]] = None,
#         save_dir: str = "./results",
#         seen_classes: Optional[Iterable[int]] = None,
#         old_class_count: Optional[int] = None,
#         ignore_index: Optional[int] = None,
#         tr_time: Optional[float] = None,
#         te_time: Optional[float] = None,
#         dl_time: Optional[float] = None,
#     ) -> Dict[str, Any]:
#         return save_classification_report(
#             y_true=y_true,
#             y_pred=y_pred,
#             target_names=target_names,
#             save_dir=save_dir,
#             phase=phase,
#             seen_classes=seen_classes,
#             old_class_count=old_class_count,
#             ignore_index=ignore_index,
#             tr_time=tr_time,
#             te_time=te_time,
#             dl_time=dl_time,
#             save_hsi_style=True,
#             save_structured=True,
#         )

#     def _rebuild_class_history(self):
#         all_classes = set()
#         for p in self.phases_seen:
#             per_class = self.phase_history[p].get("per_class_accuracy", {})
#             all_classes.update(int(k) for k in per_class.keys())

#         new_hist = defaultdict(list)
#         presence = defaultdict(list)

#         for cls in sorted(all_classes):
#             for p in self.phases_seen:
#                 per_class = self.phase_history[p].get("per_class_accuracy", {})
#                 if cls in per_class:
#                     new_hist[cls].append(float(per_class[cls]))
#                     presence[cls].append(True)
#                 else:
#                     new_hist[cls].append(np.nan)
#                     presence[cls].append(False)

#         self.class_acc_history = new_hist
#         self.class_presence_history = presence

#     def calculate_forgetting_per_class(self) -> Dict[int, float]:
#         if len(self.phases_seen) < 2:
#             return {}

#         forgetting: Dict[int, float] = {}
#         for cls, history in self.class_acc_history.items():
#             vals = np.asarray(history, dtype=float)
#             vals = vals[~np.isnan(vals)]

#             if vals.size < 2:
#                 continue

#             peak_before_last = float(np.max(vals[:-1]))
#             current = float(vals[-1])
#             forgetting[int(cls)] = float(max(0.0, peak_before_last - current))

#         return forgetting

#     def calculate_backward_transfer(self) -> float:
#         transfers = []
#         for _, history in self.class_acc_history.items():
#             vals = np.asarray(history, dtype=float)
#             vals = vals[~np.isnan(vals)]
#             if vals.size >= 2:
#                 transfers.append(float(vals[-1] - vals[0]))

#         return float(np.mean(transfers)) if transfers else 0.0

#     def get_standard_metrics(self) -> Dict[str, float]:
#         if not self.phases_seen:
#             return {}

#         last_phase = self.phases_seen[-1]
#         all_oas = [float(self.phase_history[p]["overall_accuracy"]) for p in self.phases_seen]
#         all_hm = [float(self.phase_history[p].get("harmonic_mean", 0.0)) for p in self.phases_seen]

#         forgetting_per_class = self.calculate_forgetting_per_class()
#         avg_forgetting = (
#             float(np.mean(list(forgetting_per_class.values())))
#             if forgetting_per_class
#             else 0.0
#         )

#         last_metrics = self.phase_history[last_phase]

#         return {
#             "A_last (Final Accuracy)": float(last_metrics["overall_accuracy"]),
#             "A_avg (Avg Accuracy)": float(np.mean(all_oas)),
#             "H_last (Final Harmonic Mean)": float(last_metrics.get("harmonic_mean", 0.0)),
#             "H_avg (Avg Harmonic Mean)": float(np.mean(all_hm)),
#             "F_avg (Avg Forgetting)": avg_forgetting,
#             "BWT (Backward Transfer)": self.calculate_backward_transfer(),
#             "Old_last (Final Old Accuracy)": float(last_metrics.get("old_accuracy", 0.0)),
#             "New_last (Final New Accuracy)": float(last_metrics.get("new_accuracy", 0.0)),
#             "AA_last (Final Avg Accuracy)": float(last_metrics.get("average_accuracy", 0.0)),
#             "Kappa_last": float(last_metrics.get("kappa", 0.0)),
#             "F1_last": float(last_metrics.get("f1_macro", 0.0)),
#             "Phases": len(self.phases_seen),
#         }

#     def get_phase_table(self) -> List[Dict[str, Any]]:
#         rows = []
#         for p in self.phases_seen:
#             m = self.phase_history[p]
#             rows.append({
#                 "phase": int(p),
#                 "OA": float(m.get("overall_accuracy", 0.0)),
#                 "AA": float(m.get("average_accuracy", 0.0)),
#                 "Kappa": float(m.get("kappa", 0.0)),
#                 "F1": float(m.get("f1_macro", 0.0)),
#                 "Old": float(m.get("old_accuracy", 0.0)),
#                 "New": float(m.get("new_accuracy", 0.0)),
#                 "H": float(m.get("harmonic_mean", 0.0)),
#                 "Samples": int(m.get("num_samples", 0)),
#             })
#         return rows

#     def get_per_class_summary(self) -> Dict[int, Dict[str, float]]:
#         forgetting = self.calculate_forgetting_per_class()
#         summary = {}

#         for cls, history in self.class_acc_history.items():
#             vals = np.asarray(history, dtype=float)
#             valid = vals[~np.isnan(vals)]
#             if valid.size == 0:
#                 continue

#             summary[int(cls)] = {
#                 "first": float(valid[0]),
#                 "best": float(np.max(valid)),
#                 "last": float(valid[-1]),
#                 "forgetting": float(forgetting.get(int(cls), 0.0)),
#             }

#         return summary

#     def print_summary(self):
#         if not self.phases_seen:
#             print("[NECILEvaluator] No phases evaluated yet.")
#             return

#         metrics = self.get_standard_metrics()
#         last_phase = self.phases_seen[-1]
#         phase_metrics = self.phase_history[last_phase]

#         print("\n" + "=" * 58)
#         print(f" NECIL-HSI Evaluation Report (Phase {last_phase})")
#         print("=" * 58)
#         print(f" 1. Final Accuracy (A_last):      {metrics.get('A_last (Final Accuracy)', 0):.2f}%")
#         print(f" 2. Avg Accuracy (A_avg):         {metrics.get('A_avg (Avg Accuracy)', 0):.2f}%")
#         print(f" 3. Avg Forgetting (F_avg):       {metrics.get('F_avg (Avg Forgetting)', 0):.2f}%")
#         print(f" 4. Backward Transfer (BWT):      {metrics.get('BWT (Backward Transfer)', 0):.2f}%")
#         print(f" 5. Old Accuracy:                 {phase_metrics.get('old_accuracy', 0):.2f}%")
#         print(f" 6. New Accuracy:                 {phase_metrics.get('new_accuracy', 0):.2f}%")
#         print(f" 7. Harmonic Mean:                {phase_metrics.get('harmonic_mean', 0):.2f}%")
#         print(
#             f" 8. AA / Kappa / F1:              "
#             f"{phase_metrics.get('average_accuracy', 0):.2f}% / "
#             f"{phase_metrics.get('kappa', 0):.2f}% / "
#             f"{phase_metrics.get('f1_macro', 0):.2f}%"
#         )
#         print("-" * 58)

#     def to_dict(self) -> Dict[str, Any]:
#         return make_json_serializable({
#             "phase_history": self.phase_history,
#             "standard_metrics": self.get_standard_metrics(),
#             "phase_table": self.get_phase_table(),
#             "per_class_summary": self.get_per_class_summary(),
#         })
