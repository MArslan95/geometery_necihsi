"""
Evaluation Metrics
==================
Metrics for evaluating HSI classification and incremental learning.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Optional list of class names
        
    Returns:
        Dictionary with various metrics
    """
    metrics = {}
    
    # Overall accuracy
    metrics['overall_accuracy'] = accuracy_score(y_true, y_pred) * 100
    
    # Precision, Recall, F1 (macro average)
    metrics['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0) * 100
    metrics['recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0) * 100
    metrics['f1_score'] = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100
    
    # Per-class accuracy
    metrics['per_class_accuracy'] = per_class_accuracy(y_true, y_pred)
    
    # Average accuracy (mean of per-class accuracies)
    metrics['average_accuracy'] = np.mean(list(metrics['per_class_accuracy'].values()))
    
    # Kappa coefficient
    metrics['kappa'] = calculate_kappa(y_true, y_pred)
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    return metrics


def per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[int, float]:
    """
    Calculate accuracy for each class.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary mapping class ID to accuracy percentage
    """
    classes = np.unique(y_true)
    accuracies = {}
    
    for cls in classes:
        mask = y_true == cls
        if mask.sum() > 0:
            accuracies[int(cls)] = (y_pred[mask] == cls).mean() * 100
        else:
            accuracies[int(cls)] = 0.0
    
    return accuracies


def calculate_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Cohen's Kappa coefficient.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Kappa coefficient
    """
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    n_samples = cm.sum()
    
    if n_samples == 0:
        return 0.0
    
    p_o = np.diag(cm).sum() / n_samples  # Observed agreement
    
    # Expected agreement
    p_e = 0
    for i in range(n_classes):
        p_e += (cm[i, :].sum() / n_samples) * (cm[:, i].sum() / n_samples)
    
    if p_e == 1:
        return 1.0
    
    kappa = (p_o - p_e) / (1 - p_e)
    return kappa * 100


def calculate_forgetting(
    accuracies_after_each_phase: List[Dict[int, float]]
) -> Dict[int, float]:
    """
    Calculate forgetting for each class.
    
    Forgetting = peak accuracy - final accuracy
    
    Args:
        accuracies_after_each_phase: List of per-class accuracy dicts after each phase
        
    Returns:
        Dictionary mapping class ID to forgetting percentage
    """
    if len(accuracies_after_each_phase) < 2:
        return {}
    
    forgetting = {}
    
    # Get all classes that appeared
    all_classes = set()
    for acc_dict in accuracies_after_each_phase:
        all_classes.update(acc_dict.keys())
    
    for cls in all_classes:
        # Find peak accuracy
        peak = 0
        for acc_dict in accuracies_after_each_phase:
            if cls in acc_dict:
                peak = max(peak, acc_dict[cls])
        
        # Final accuracy
        final = accuracies_after_each_phase[-1].get(cls, 0)
        
        forgetting[cls] = peak - final
    
    return forgetting


def calculate_backward_transfer(
    accuracies_after_each_phase: List[Dict[int, float]],
    phase_to_classes: Dict[int, List[int]]
) -> float:
    """
    Calculate backward transfer (average forgetting of old classes).
    
    Args:
        accuracies_after_each_phase: Per-class accuracies after each phase
        phase_to_classes: Mapping from phase to classes introduced
        
    Returns:
        Average backward transfer (negative = forgetting)
    """
    if len(accuracies_after_each_phase) < 2:
        return 0.0
    
    backward_transfers = []
    
    for phase, classes in phase_to_classes.items():
        if phase == len(accuracies_after_each_phase) - 1:
            continue  # Skip last phase
        
        # Accuracy right after learning
        acc_after = accuracies_after_each_phase[phase]
        # Final accuracy
        acc_final = accuracies_after_each_phase[-1]
        
        for cls in classes:
            if cls in acc_after and cls in acc_final:
                bt = acc_final[cls] - acc_after[cls]
                backward_transfers.append(bt)
    
    return np.mean(backward_transfers) if backward_transfers else 0.0


def calculate_forward_transfer(
    accuracies_zero_shot: Dict[int, float],
    accuracies_trained: Dict[int, float]
) -> float:
    """
    Calculate forward transfer (how well old knowledge helps new classes).
    
    Args:
        accuracies_zero_shot: Accuracy on new classes before training
        accuracies_trained: Accuracy on new classes after training on them
        
    Returns:
        Average forward transfer
    """
    forward_transfers = []
    
    for cls in accuracies_zero_shot:
        if cls in accuracies_trained:
            ft = accuracies_trained[cls] - accuracies_zero_shot[cls]
            forward_transfers.append(ft)
    
    return np.mean(forward_transfers) if forward_transfers else 0.0


if __name__ == "__main__":
    # Test metrics
    np.random.seed(42)
    n_samples = 100
    n_classes = 5
    
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = y_true.copy()
    # Add some errors
    error_idx = np.random.choice(n_samples, 20, replace=False)
    y_pred[error_idx] = np.random.randint(0, n_classes, len(error_idx))
    
    metrics = calculate_metrics(y_true, y_pred)
    
    print("Metrics:")
    for k, v in metrics.items():
        if k != 'confusion_matrix':
            print(f"  {k}: {v}")
