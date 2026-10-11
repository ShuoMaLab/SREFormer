import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from typing import Optional, Dict, Any
import warnings


class Brats2017Task1Dataset(Dataset):
    """
    BraTS2017 segmentation dataset.
    CSV columns:
        - data_path
        - case_name
    """

    def __init__(
        self,
        root_dir: str,
        is_train: bool = True,
        transform: Optional[Any] = None,
        fold_id: Optional[int] = None
    ) -> None:
        super().__init__()

        self.root_dir = os.path.abspath(root_dir)

        if fold_id is not None:
            csv_name = f"train_fold_{fold_id}.csv" if is_train else f"validation_fold_{fold_id}.csv"
        else:
            csv_name = "train.csv" if is_train else "validation.csv"

        csv_fp = os.path.join(self.root_dir, csv_name)
        assert os.path.exists(csv_fp), f"CSV file not found: {csv_fp}"

        self.csv = pd.read_csv(csv_fp)
        self.transform = transform
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.csv)

    def _resolve_data_path(self, data_path: str) -> str:
        """
        Resolve CSV data_path to an absolute path.

        Cases handled:
        1. data_path already absolute -> use directly
        2. data_path relative to project root or current cwd
        3. data_path relative to dataset root_dir
        """
        data_path = str(data_path).replace("\\", os.sep).replace("/", os.sep)

        if os.path.isabs(data_path):
            return data_path

        # 尝试 1：相对于当前工作目录
        abs_from_cwd = os.path.abspath(data_path)
        if os.path.exists(abs_from_cwd):
            return abs_from_cwd

        # 尝试 2：相对于 root_dir
        abs_from_root = os.path.abspath(os.path.join(self.root_dir, data_path))
        if os.path.exists(abs_from_root):
            return abs_from_root

        # 尝试 3：只取最后一级 case 名，拼到标准目录
        case_dir = os.path.basename(os.path.normpath(data_path))
        fallback = os.path.join(self.root_dir, "BraTS2017_Training_Data", case_dir)
        if os.path.exists(fallback):
            return fallback

        # 返回一个最可能的路径，便于报错时看清楚
        return abs_from_root

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data_path = self.csv["data_path"].iloc[idx]
        case_name = self.csv["case_name"].iloc[idx]

        data_path = self._resolve_data_path(data_path)

        volume_fp = os.path.join(data_path, f"{case_name}_modalities.pt")
        label_fp = os.path.join(data_path, f"{case_name}_label.pt")

        try:
            volume = torch.load(volume_fp, map_location="cpu", weights_only=False)
            label = torch.load(label_fp, map_location="cpu", weights_only=False)

            if not isinstance(volume, torch.Tensor):
                volume = torch.from_numpy(volume)
            if not isinstance(label, torch.Tensor):
                label = torch.from_numpy(label)

            data = {
                "image": volume.float(),
                "label": label.float()
            }
        except Exception as e:
            warnings.warn(
                f"Error loading data at index {idx} ({case_name}): {str(e)}\n"
                f"Resolved data_path: {data_path}\n"
                f"volume_fp: {volume_fp}\n"
                f"label_fp: {label_fp}"
            )
            raise

        if self.transform:
            data = self.transform(data)

        return data