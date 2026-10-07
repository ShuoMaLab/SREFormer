"""Boundary-Structure Refinement Module placeholder."""

from __future__ import annotations

import torch
from torch import nn


class BSRM(nn.Module):
    """Placeholder BSRM block with the expected module interface."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=1),
            nn.InstanceNorm3d(channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.refine(x)
