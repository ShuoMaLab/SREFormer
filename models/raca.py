"""Register-Augmented Context Attention placeholder."""

from __future__ import annotations

import torch
from torch import nn


class RACA(nn.Module):
    """Placeholder RACA block with the expected module interface."""

    def __init__(self, channels: int, num_register_tokens: int = 8) -> None:
        super().__init__()
        self.num_register_tokens = num_register_tokens
        self.proj = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.proj(x)
