import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedCrossEntropyLoss(nn.Module):
    """
    CE applied only to selected classes (useful for incremental phase)
    """

    def __init__(self, label_smoothing=0.0):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits, labels, class_mask=None):
        labels = labels.view(-1).long()

        if class_mask is not None:
            mask = torch.isin(labels, torch.tensor(class_mask, device=labels.device))
            if not mask.any():
                return torch.tensor(0.0, device=logits.device, requires_grad=True)

            logits = logits[mask]
            labels = labels[mask]

        return F.cross_entropy(
            logits,
            labels,
            label_smoothing=self.label_smoothing
        )


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, labels):
        ce = F.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()
# """
# Cross Entropy Loss for New Classes
# ===================================
# Standard CE loss applied only to new class samples during incremental phases.
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Optional, List


# """
# Cross Entropy Loss for New Classes (Hardened)
# =============================================
# Standard CE loss applied only to specific class samples (typically new classes).
# Ensures strictly 1D LongTensor targets to avoid PyTorch shape mismatches.
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Optional, List

# class MaskedCrossEntropyLoss(nn.Module):
#     """
#     Cross entropy loss that can be masked to specific classes.
#     Vectorized for performance and hardened against shape-mismatch errors.
#     """
    
#     def __init__(
#         self,
#         label_smoothing: float = 0.0,
#         reduction: str = 'mean'
#     ):
#         super().__init__()
#         self.label_smoothing = label_smoothing
#         self.reduction = reduction
        
#     def forward(
#         self,
#         logits: torch.Tensor,
#         labels: torch.Tensor,
#         class_mask: Optional[List[int]] = None
#     ) -> torch.Tensor:
#         """
#         Args:
#             logits: (B, num_classes)
#             labels: (B,) - Hard class indices
#             class_mask: Optional list of class IDs to compute loss for.
#         """
#         device = logits.device
        
#         # 1. Hardening: Ensure labels are 1D Long indices
#         # This prevents the "Expected target size [B, C], got [B]" error
#         labels = labels.view(-1).long()     

#         # 2. Vectorized Class Masking
#         if class_mask is not None:
#             # Convert list to tensor for fast GPU-side comparison
#             mask_tensor = torch.tensor(class_mask, device=device)
#             # Find samples where labels are IN the class_mask
#             mask = torch.isin(labels, mask_tensor)
            
#             if not mask.any():
#                 # Return 0 with gradient if no samples match the mask
#                 return torch.tensor(0.0, device=device, requires_grad=True)
            
#             # Filter batch to only include samples from the mask
#             logits = logits[mask]
#             labels = labels[mask]
            
#         if labels.max() >= logits.shape[1]:
#             raise RuntimeError(f"Label {labels.max()} is out of bounds for logits with {logits.shape[1]} classes. "
#                                f"Check your Sequential Mapping in IncrementalHSIDataset.")
#         # 3. Final Cross Entropy
#         # PyTorch F.cross_entropy handles smoothing and reduction internally
#         return F.cross_entropy(
#             logits, 
#             labels, 
#             label_smoothing=self.label_smoothing,
#             reduction=self.reduction
#         )

# class FocalLoss(nn.Module):
#     """
#     Focal loss for handling class imbalance.
#     Useful when some classes have many more samples than others.
#     """
    
#     def __init__(
#         self,
#         gamma: float = 2.0,
#         alpha: Optional[torch.Tensor] = None,
#         reduction: str = 'mean'
#     ):
#         """
#         Args:
#             gamma: Focusing parameter (higher = more focus on hard examples)
#             alpha: Class weights (optional)
#             reduction: 'mean', 'sum', or 'none'
#         """
#         super().__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.reduction = reduction
        
#     def forward(
#         self,
#         logits: torch.Tensor,
#         labels: torch.Tensor
#     ) -> torch.Tensor:
#         """
#         Compute focal loss.
        
#         Args:
#             logits: (B, num_classes)
#             labels: (B,)
#         """
#         ce_loss = F.cross_entropy(logits, labels, reduction='none')
#         pt = torch.exp(-ce_loss)  # Probability of correct class
#         focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
#         if self.alpha is not None:
#             alpha = self.alpha.to(logits.device)
#             alpha_t = alpha[labels]
#             focal_loss = alpha_t * focal_loss
        
#         if self.reduction == 'mean':
#             return focal_loss.mean()
#         elif self.reduction == 'sum':
#             return focal_loss.sum()
#         return focal_loss


# if __name__ == "__main__":
#     # Test cross entropy loss
#     batch_size = 16
#     num_classes = 10
    
#     logits = torch.randn(batch_size, num_classes)
#     labels = torch.randint(0, num_classes, (batch_size,))
    
#     # Standard CE
#     loss_fn = MaskedCrossEntropyLoss()
#     loss = loss_fn(logits, labels)
#     print(f"Standard CE loss: {loss.item():.4f}")
    
#     # Masked CE (only classes 8, 9)
#     loss_masked = loss_fn(logits, labels, class_mask=[8, 9])
#     print(f"Masked CE loss (classes 8,9): {loss_masked.item():.4f}")
    
#     # Focal loss
#     focal_fn = FocalLoss(gamma=2.0)
#     focal_loss = focal_fn(logits, labels)
#     print(f"Focal loss: {focal_loss.item():.4f}")
