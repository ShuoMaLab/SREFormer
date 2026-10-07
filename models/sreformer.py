"""Placeholder SREFormer model.

The finalized network implementation will replace this file in the official
release. The current class preserves the public constructor expected by the
training and evaluation entry points.
"""

from __future__ import annotations

import torch
from torch import nn

from .bsrm import BSRM
from .raca import RACA


class SREFormer(nn.Module):
    """Minimal placeholder for the SREFormer architecture."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_channels: int = 32,
        num_register_tokens: int = 8,
        use_raca: bool = True,
        use_bsrm: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(base_channels),
            nn.GELU(),
        )
        self.raca = RACA(base_channels, num_register_tokens) if use_raca else nn.Identity()
        self.bsrm = BSRM(base_channels) if use_bsrm else nn.Identity()
        self.head = nn.Conv3d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.raca(x)
        x = self.bsrm(x)
        return self.head(x)
