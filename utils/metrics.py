"""Metric placeholders for volumetric segmentation."""

from __future__ import annotations

import torch


def dice_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1e-6) -> float:
    """Return the mean foreground Dice score."""
    scores: list[torch.Tensor] = []
    for cls_idx in range(1, num_classes):
        pred_cls = pred == cls_idx
        target_cls = target == cls_idx
        intersection = (pred_cls & target_cls).sum().float()
        denominator = pred_cls.sum().float() + target_cls.sum().float()
        scores.append((2.0 * intersection + eps) / (denominator + eps))
    if not scores:
        return 0.0
    return torch.stack(scores).mean().item()
