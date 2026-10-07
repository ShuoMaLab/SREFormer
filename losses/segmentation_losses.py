"""Segmentation loss placeholders."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute a simple soft Dice loss for placeholder experiments."""
    num_classes = logits.shape[1]
    probs = logits.softmax(dim=1)
    target_one_hot = F.one_hot(target.long(), num_classes=num_classes).movedim(-1, 1).float()
    dims = tuple(range(2, logits.ndim))
    intersection = torch.sum(probs * target_one_hot, dim=dims)
    denominator = torch.sum(probs + target_one_hot, dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()
