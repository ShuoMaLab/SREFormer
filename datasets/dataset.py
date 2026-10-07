"""Dataset placeholders for SREFormer."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class PlaceholderVolumeDataset(Dataset):
    """A tiny synthetic dataset used only to validate the code skeleton."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        input_size: tuple[int, int, int] = (64, 64, 64),
        num_classes: int = 2,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.input_size = input_size
        self.num_classes = num_classes
        self.length = 2

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image = torch.zeros((1, *self.input_size), dtype=torch.float32)
        label = torch.zeros(self.input_size, dtype=torch.long)
        return {"image": image, "label": label, "case_id": f"{self.split}_{index:04d}"}
