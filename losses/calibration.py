"""
Classifier Calibration Loss
=============================
Handles bias between old and new classes in incremental learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

class ClassifierCalibrationLoss(nn.Module):
    """
    Classifier Calibration Loss for DSDMA Architecture.
    
    Since the semantic concept weights are strictly L2 normalized, 
    traditional Weight Alignment (WA) is mathematically obsolete.
    Instead, we apply Logit Margin Calibration to prevent the network 
    from being overconfident in new classes.
    """
    
    def __init__(
        self,
        calibration_type: str = 'logit_margin',  # 'logit_margin', 'margin'
        margin: float = 0.1,  # Small margin for cosine logits
    ):
        """
        Args:
            calibration_type: Type of calibration to apply
            margin: Desired margin constraint
        """
        super().__init__()
        self.calibration_type = calibration_type
        self.margin = margin
        
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        old_classes: List[int],
        new_classes: List[int],
        classifier_weights: Optional[torch.Tensor] = None # Kept for API compatibility, but unused
    ) -> torch.Tensor:
        """
        Compute calibration loss on the logits.
        """
        device = logits.device
        
        # Only apply calibration if we have both old and new classes
        if not old_classes or not new_classes:
            return torch.tensor(0.0, device=device, requires_grad=True)
            
        if self.calibration_type == 'logit_margin':
            return self._logit_margin_calibration(logits, old_classes, new_classes)
            
        elif self.calibration_type == 'margin':
            return self._standard_margin_loss(logits, labels)
            
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    def _logit_margin_calibration(
        self,
        logits: torch.Tensor,
        old_classes: List[int],
        new_classes: List[int]
    ) -> torch.Tensor:
        """
        Penalizes the network if the mean confidence for new classes 
        exceeds the mean confidence for old classes by more than the margin.
        This directly fights task-recency bias.
        """
        old_logits = logits[:, old_classes]
        new_logits = logits[:, new_classes]
        
        # Mean activation for old vs new classes across the batch
        mean_old = old_logits.mean()
        mean_new = new_logits.mean()
        
        # Loss activates only if new classes dominate old classes beyond the margin
        # F.relu(x) = max(0, x)
        loss = F.relu((mean_new - mean_old) + self.margin)
        
        return loss
    
    def _standard_margin_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Standard margin-based calibration for general class separation.
        """
        # For each sample, compute margin between correct class and max of others
        correct_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        
        mask = torch.ones_like(logits)
        mask.scatter_(1, labels.unsqueeze(1), 0)
        other_logits = (logits * mask).max(dim=1)[0]
        
        margin_diff = correct_logits - other_logits
        loss = F.relu(self.margin - margin_diff).mean()
        
        return loss


# class ClassifierCalibrationLoss(nn.Module):
#     """
#     Classifier Calibration Loss for incremental learning.
    
#     Addresses the bias where the model tends to prefer new classes
#     because they are trained on while old classes are not.
    
#     Techniques:
#     1. Weight normalization
#     2. Bias correction
#     3. Temperature scaling
#     4. Class-balanced regularization
#     """
    
#     def __init__(
#         self,
#         calibration_type: str = 'weight_align',  # 'weight_align', 'bias_correct', 'margin'
#         margin: float = 0.5,
#         temperature: float = 1.0
#     ):
#         """
#         Args:
#             calibration_type: Type of calibration
#             margin: Margin for margin-based calibration
#             temperature: Temperature for scaling
#         """
#         super().__init__()
#         self.calibration_type = calibration_type
#         self.margin = margin
#         self.temperature = temperature
        
#     def forward(
#         self,
#         logits: torch.Tensor,
#         labels: torch.Tensor,
#         old_classes: List[int],
#         new_classes: List[int],
#         classifier_weights: Optional[torch.Tensor] = None
#     ) -> torch.Tensor:
#         """
#         Compute calibration loss.
        
#         Args:
#             logits: Model predictions (B, num_classes)
#             labels: Ground truth labels (B,)
#             old_classes: List of old class indices
#             new_classes: List of new class indices
#             classifier_weights: Optional classifier weight matrix (num_classes, D)
            
#         Returns:
#             Calibration loss
#         """
#         device = logits.device
        
#         if self.calibration_type == 'weight_align':
#             return self._weight_alignment_loss(classifier_weights, old_classes, new_classes)
        
#         elif self.calibration_type == 'bias_correct':
#             return self._bias_correction_loss(logits, labels, old_classes, new_classes)
        
#         elif self.calibration_type == 'margin':
#             return self._margin_loss(logits, labels, old_classes, new_classes)
        
#         return torch.tensor(0.0, device=device, requires_grad=True)
    
#     def _weight_alignment_loss(
#         self,
#         weights: Optional[torch.Tensor],
#         old_classes: List[int],
#         new_classes: List[int]
#     ) -> torch.Tensor:
#         """
#         Align weight norms between old and new classes.
#         Prevents new classes from having much larger weights.
#         """
#         if weights is None or len(old_classes) == 0 or len(new_classes) == 0:
#             return torch.tensor(0.0, device=weights.device if weights is not None else 'cpu', 
#                               requires_grad=True)
        
#         # Compute weight norms
#         weight_norms = weights.norm(p=2, dim=1)
        
#         old_norms = weight_norms[old_classes]
#         new_norms = weight_norms[new_classes]
        
#         # Loss: difference in mean norms
#         old_mean = old_norms.mean()
#         new_mean = new_norms.mean()
        
#         loss = F.mse_loss(new_mean, old_mean)
        
#         return loss
    
#     def _bias_correction_loss(
#         self,
#         logits: torch.Tensor,
#         labels: torch.Tensor,
#         old_classes: List[int],
#         new_classes: List[int]
#     ) -> torch.Tensor:
#         """
#         Bias correction to balance predictions.
#         """
#         B, C = logits.shape
        
#         # Compute average prediction confidence for old vs new classes
#         probs = F.softmax(logits / self.temperature, dim=1)
        
#         if len(old_classes) > 0:
#             old_probs = probs[:, old_classes].mean()
#         else:
#             old_probs = torch.tensor(0.5, device=logits.device)
            
#         if len(new_classes) > 0:
#             new_probs = probs[:, new_classes].mean()
#         else:
#             new_probs = torch.tensor(0.5, device=logits.device)
        
#         # Encourage balanced predictions
#         target = torch.tensor(0.5, device=logits.device)
#         loss = F.mse_loss(old_probs, target) + F.mse_loss(new_probs, target)
        
#         return loss
    
#     def _margin_loss(
#         self,
#         logits: torch.Tensor,
#         labels: torch.Tensor,
#         old_classes: List[int],
#         new_classes: List[int]
#     ) -> torch.Tensor:
#         """
#         Margin-based calibration for better class separation.
#         """
#         B = logits.shape[0]
        
#         # For each sample, compute margin between correct class and others
#         correct_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        
#         # Compute max of other classes
#         mask = torch.ones_like(logits)
#         mask.scatter_(1, labels.unsqueeze(1), 0)
#         other_logits = (logits * mask).max(dim=1)[0]
        
#         # Margin loss: encourage margin > threshold
#         margin = correct_logits - other_logits
#         loss = F.relu(self.margin - margin).mean()
        
#         return loss


# class WA_AlignLoss(nn.Module):
#     """
#     Weight Alignment loss from "Maintaining Discrimination and Fairness 
#     in Class Incremental Learning" (WA).
    
#     Scales old class weights to match new class weight norms.
#     """
    
#     def __init__(self):
#         super().__init__()
        
#     def compute_alignment_factor(
#         self,
#         weights: torch.Tensor,
#         num_old_classes: int
#     ) -> float:
#         """
#         Compute alignment factor gamma.
        
#         gamma = mean(||w_new||) / mean(||w_old||)
#         """
#         if num_old_classes <= 0 or num_old_classes >= weights.shape[0]:
#             return 1.0
        
#         with torch.no_grad():
#             weight_norms = weights.norm(p=2, dim=1)
#             old_mean = weight_norms[:num_old_classes].mean()
#             new_mean = weight_norms[num_old_classes:].mean()
            
#             if old_mean > 0:
#                 gamma = (new_mean / old_mean).item()
#             else:
#                 gamma = 1.0
        
#         return gamma
    
#     def forward(
#         self,
#         weights: torch.Tensor,
#         num_old_classes: int
#     ) -> torch.Tensor:
#         """
#         Compute weight alignment loss.
        
#         Encourages old and new class weights to have similar norms.
#         """
#         if num_old_classes <= 0 or num_old_classes >= weights.shape[0]:
#             return torch.tensor(0.0, device=weights.device, requires_grad=True)
        
#         weight_norms = weights.norm(p=2, dim=1)
#         old_norms = weight_norms[:num_old_classes]
#         new_norms = weight_norms[num_old_classes:]
        
#         # L2 distance between mean norms
#         loss = (old_norms.mean() - new_norms.mean()) ** 2
        
#         return loss


# if __name__ == "__main__":
#     # Test calibration loss
#     batch_size = 16
#     num_classes = 10
#     feature_dim = 256
    
#     logits = torch.randn(batch_size, num_classes)
#     labels = torch.randint(0, num_classes, (batch_size,))
#     weights = torch.randn(num_classes, feature_dim)
    
#     old_classes = list(range(8))
#     new_classes = list(range(8, 10))
    
#     # Test weight alignment
#     cal_loss = ClassifierCalibrationLoss(calibration_type='weight_align')
#     loss = cal_loss(logits, labels, old_classes, new_classes, weights)
#     print(f"Weight Alignment Loss: {loss.item():.4f}")
    
#     # Test bias correction
#     cal_loss2 = ClassifierCalibrationLoss(calibration_type='bias_correct')
#     loss2 = cal_loss2(logits, labels, old_classes, new_classes)
#     print(f"Bias Correction Loss: {loss2.item():.4f}")
    
#     # Test margin
#     cal_loss3 = ClassifierCalibrationLoss(calibration_type='margin')
#     loss3 = cal_loss3(logits, labels, old_classes, new_classes)
#     print(f"Margin Loss: {loss3.item():.4f}")
