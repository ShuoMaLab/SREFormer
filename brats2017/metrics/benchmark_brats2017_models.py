#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Unified benchmark for 3-D BraTS2017 segmentation models.

Metrics:
  1) Parameters (M)
  2) FLOPs for one 4 x ROI_D x ROI_H x ROI_W patch (GFLOPs, fvcore convention)
  3) Inference time per full validation volume (s/volume)
  4) Training time per epoch (s/epoch)
  5) Peak GPU memory during training and inference (GiB, max_memory_allocated)

The benchmark uses one shared BraTS2017 data pipeline for every model. Model-specific
code is isolated in an adapter Python file supplied through --adapter.

Adapter contract:
  def build_model(args) -> torch.nn.Module
Optional:
  def forward(model, x, args)
  def unwrap_output(output, args) -> torch.Tensor
  def compute_loss(logits, target, args) -> torch.Tensor
  def compute_loss_from_output(output, target, args) -> torch.Tensor
  def predict(model, x, args) -> torch.Tensor

Example:
  python benchmark_brats2017_models.py \
      --adapter .\adapter_segformer3d.py \
      --model_name Ours \
      --project_root <SREFORMER_BRATS_ROOT> \
      --arch_file <SREFORMER_BRATS_ROOT>\architectures\segformer3d_stage34_register_bs2.py \
      --config <SREFORMER_BRATS_ROOT>\experiments\brats_2017\exp_agent_stage34_register_bs2\config.yaml \
      --checkpoint <SREFORMER_BRATS_ROOT>\experiments\brats_2017\stage34_register_metric_snapshot_20260712_140454\stage34_register_checkpoint_best.pth \
      --data_root <BRATS2017_DATA_ROOT>\BraTS2017_Training_Data \
      --roi_size 128 128 128 --batch_size 1 --sliding_window \
      --output_csv .\brats2017_efficiency.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import inspect
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from fvcore.nn import FlopCountAnalysis
except Exception:
    FlopCountAnalysis = None

try:
    from monai.inferers import sliding_window_inference
except Exception:
    sliding_window_inference = None


# ----------------------------- reproducibility -----------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ------------------------------- data I/O ----------------------------------

def _first_tensor(obj: Any, preferred_keys: Sequence[str]) -> torch.Tensor:
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)
    if isinstance(obj, dict):
        for key in preferred_keys:
            if key in obj:
                return _first_tensor(obj[key], preferred_keys)
        for value in obj.values():
            try:
                return _first_tensor(value, preferred_keys)
            except (TypeError, ValueError):
                pass
    if isinstance(obj, (list, tuple)):
        for value in obj:
            try:
                return _first_tensor(value, preferred_keys)
            except (TypeError, ValueError):
                pass
    raise TypeError(f"Cannot extract a tensor from object of type {type(obj)!r}")


def _safe_torch_load(path: Path) -> Any:
    # Compatible with both older and newer PyTorch versions.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _normalize_image_layout(x: torch.Tensor, in_channels: int) -> torch.Tensor:
    x = x.squeeze()
    if x.ndim != 4:
        raise ValueError(f"Expected a 4-D image tensor, got shape {tuple(x.shape)}")
    if x.shape[0] == in_channels:
        return x.float().contiguous()
    if x.shape[-1] == in_channels:
        return x.permute(3, 0, 1, 2).float().contiguous()
    raise ValueError(
        f"Cannot identify channel axis for image shape {tuple(x.shape)}; "
        f"expected {in_channels} channels."
    )


def _normalize_label_layout(y: torch.Tensor) -> torch.Tensor:
    y = y.squeeze()
    if y.ndim == 3:
        return y.contiguous()
    if y.ndim == 4:
        # Prefer channel-first. If channel-last is obvious, move it.
        if y.shape[0] <= 8:
            return y.contiguous()
        if y.shape[-1] <= 8:
            return y.permute(3, 0, 1, 2).contiguous()
    raise ValueError(f"Expected a 3-D or 4-D label tensor, got shape {tuple(y.shape)}")


def _find_label_file(case_dir: Path, case_name: str) -> Path:
    candidates = [
        case_dir / f"{case_name}_seg.pt",
        case_dir / f"{case_name}_segmentation.pt",
        case_dir / f"{case_name}_label.pt",
        case_dir / f"{case_name}_labels.pt",
        case_dir / "seg.pt",
        case_dir / "label.pt",
    ]
    for p in candidates:
        if p.exists():
            return p

    broad = []
    for pattern in ("*seg*.pt", "*label*.pt", "*mask*.pt"):
        broad.extend(case_dir.glob(pattern))
    broad = sorted({p.resolve() for p in broad})
    if len(broad) == 1:
        return Path(broad[0])
    if not broad:
        raise FileNotFoundError(f"No label tensor found in {case_dir}")
    raise RuntimeError(
        f"Multiple possible label files in {case_dir}: {[str(p) for p in broad]}. "
        "Please rename the desired file to <case>_seg.pt or use an adapter-specific dataset."
    )


def discover_cases(data_root: Path) -> List[Tuple[str, Path, Path]]:
    image_files = sorted(data_root.rglob("*_modalities.pt"))
    if not image_files:
        image_files = sorted(data_root.rglob("*modalit*.pt"))
    cases: List[Tuple[str, Path, Path]] = []
    for image_path in image_files:
        case_dir = image_path.parent
        case_name = image_path.stem.replace("_modalities", "")
        label_path = _find_label_file(case_dir, case_name)
        cases.append((case_name, image_path, label_path))
    if not cases:
        raise FileNotFoundError(
            f"No '*_modalities.pt' files were found under: {data_root}"
        )
    return cases


def read_case_ids(path: Optional[str]) -> Optional[set[str]]:
    if not path:
        return None
    p = Path(path)
    ids = {
        line.strip().split()[0]
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    return ids


def split_cases(
    cases: Sequence[Tuple[str, Path, Path]],
    train_list: Optional[str],
    val_list: Optional[str],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[str, Path, Path]], List[Tuple[str, Path, Path]]]:
    train_ids = read_case_ids(train_list)
    val_ids = read_case_ids(val_list)
    if train_ids is not None or val_ids is not None:
        if train_ids is None or val_ids is None:
            raise ValueError("Provide both --train_list and --val_list, or neither.")
        train_cases = [c for c in cases if c[0] in train_ids]
        val_cases = [c for c in cases if c[0] in val_ids]
        missing_train = train_ids - {c[0] for c in train_cases}
        missing_val = val_ids - {c[0] for c in val_cases}
        if missing_train or missing_val:
            raise ValueError(
                f"Missing case IDs. train={sorted(missing_train)}, val={sorted(missing_val)}"
            )
        return train_cases, val_cases

    shuffled = list(cases)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_ratio)))
    return shuffled[n_val:], shuffled[:n_val]


def _pad_spatial(x: torch.Tensor, target: Sequence[int]) -> torch.Tensor:
    spatial = x.shape[-3:]
    pads: List[int] = []
    for size, need in reversed(list(zip(spatial, target))):
        total = max(0, need - size)
        left = total // 2
        right = total - left
        pads.extend([left, right])
    return F.pad(x, pads, mode="constant", value=0)


def _crop_pair(
    image: torch.Tensor,
    label: torch.Tensor,
    roi: Sequence[int],
    random_crop: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    image = _pad_spatial(image, roi)
    label = _pad_spatial(label, roi)
    spatial = image.shape[-3:]
    starts = []
    for size, need in zip(spatial, roi):
        if random_crop:
            starts.append(random.randint(0, max(0, size - need)))
        else:
            starts.append(max(0, (size - need) // 2))
    d, h, w = starts
    rd, rh, rw = roi
    image = image[..., d:d + rd, h:h + rh, w:w + rw]
    label = label[..., d:d + rd, h:h + rh, w:w + rw]
    return image.contiguous(), label.contiguous()


class BratsPtDataset(Dataset):
    def __init__(
        self,
        cases: Sequence[Tuple[str, Path, Path]],
        in_channels: int,
        roi_size: Sequence[int],
        train: bool,
        full_volume: bool,
        augment: bool,
    ) -> None:
        self.cases = list(cases)
        self.in_channels = in_channels
        self.roi_size = tuple(int(v) for v in roi_size)
        self.train = train
        self.full_volume = full_volume
        self.augment = augment

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        case_id, image_path, label_path = self.cases[index]
        image_obj = _safe_torch_load(image_path)
        label_obj = _safe_torch_load(label_path)
        image = _first_tensor(image_obj, ("image", "images", "modalities", "data", "x"))
        label = _first_tensor(label_obj, ("label", "labels", "seg", "mask", "target", "y"))
        image = _normalize_image_layout(image, self.in_channels)
        label = _normalize_label_layout(label)

        if self.train or not self.full_volume:
            image, label = _crop_pair(
                image, label, self.roi_size, random_crop=self.train
            )

        if self.train and self.augment:
            # Same family of augmentations commonly used by SegFormer3D/BraTS.
            for axis in (-3, -2, -1):
                if random.random() < 0.5:
                    image = torch.flip(image, dims=(axis,))
                    label = torch.flip(label, dims=(axis,))
            if random.random() < 0.2:
                image = image * random.uniform(0.9, 1.1)
            if random.random() < 0.2:
                image = image + random.uniform(-0.1, 0.1)

        return {"image": image.float(), "label": label, "case_id": case_id}


def collate_single(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(batch) != 1:
        raise ValueError("Full-volume validation requires batch size 1.")
    item = batch[0]
    return {
        "image": item["image"].unsqueeze(0),
        "label": item["label"].unsqueeze(0),
        "case_id": [item["case_id"]],
    }


# ----------------------------- adapter helpers -----------------------------

def load_adapter(path: str):
    adapter_path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("benchmark_model_adapter", adapter_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import adapter from {adapter_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "build_model"):
        raise AttributeError("Adapter must define build_model(args).")
    return module


def adapter_forward(adapter, model: nn.Module, x: torch.Tensor, args) -> Any:
    if hasattr(adapter, "forward"):
        return adapter.forward(model, x, args)
    return model(x)


def default_unwrap_output(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, dict):
        for key in ("out", "logits", "pred", "prediction", "seg", "seg_logits"):
            if key in output:
                return default_unwrap_output(output[key])
        for value in output.values():
            try:
                return default_unwrap_output(value)
            except TypeError:
                pass
    if isinstance(output, (list, tuple)):
        for value in output:
            try:
                return default_unwrap_output(value)
            except TypeError:
                pass
    raise TypeError(f"Cannot unwrap a tensor from model output type {type(output)!r}")


def adapter_unwrap(adapter, output: Any, args) -> torch.Tensor:
    if hasattr(adapter, "unwrap_output"):
        return adapter.unwrap_output(output, args)
    return default_unwrap_output(output)


class TensorOutputWrapper(nn.Module):
    def __init__(self, adapter, model: nn.Module, args) -> None:
        super().__init__()
        self.adapter = adapter
        self.model = model
        self.args = args

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = adapter_forward(self.adapter, self.model, x, self.args)
        return adapter_unwrap(self.adapter, out, self.args)


def brats_regions_from_label(target: torch.Tensor) -> torch.Tensor:
    """Convert BraTS label map to [TC, WT, ET] overlapping binary channels."""
    if target.ndim == 5 and target.shape[1] == 3:
        return target.float()
    if target.ndim == 5 and target.shape[1] == 1:
        target = target[:, 0]
    if target.ndim != 4:
        raise ValueError(f"Cannot convert target shape {tuple(target.shape)} to BraTS regions")
    y = target.long()
    wt = y > 0
    # Supports both original BraTS labels {0,1,2,4} and remapped {0,1,2,3}.
    et = (y == 4) | (y == 3)
    tc = (y == 1) | et
    return torch.stack([tc, wt, et], dim=1).float()


def default_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if logits.shape[-3:] != target.shape[-3:]:
        logits = F.interpolate(
            logits, size=target.shape[-3:], mode="trilinear", align_corners=False
        )

    if logits.shape[1] == 3:
        target_regions = brats_regions_from_label(target).to(logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, target_regions)
        prob = torch.sigmoid(logits)
        dims = tuple(range(2, prob.ndim))
        inter = (prob * target_regions).sum(dims)
        denom = prob.sum(dims) + target_regions.sum(dims)
        dice_loss = 1.0 - ((2.0 * inter + 1e-5) / (denom + 1e-5)).mean()
        return bce + dice_loss

    # Generic multiclass fallback.
    if target.ndim == 5:
        if target.shape[1] == 1:
            target = target[:, 0]
        else:
            target = target.argmax(dim=1)
    return F.cross_entropy(logits, target.long())


def adapter_loss(
    adapter, output: Any, logits: torch.Tensor, target: torch.Tensor, args
) -> torch.Tensor:
    # Use this hook for nnU-Net/nnFormer-style deep supervision, where the
    # official loss consumes every output scale rather than only the main head.
    if hasattr(adapter, "compute_loss_from_output"):
        return adapter.compute_loss_from_output(output, target, args)
    if hasattr(adapter, "compute_loss"):
        return adapter.compute_loss(logits, target, args)
    return default_loss(logits, target)


def adapter_predict(adapter, model: nn.Module, x: torch.Tensor, args) -> torch.Tensor:
    if hasattr(adapter, "predict"):
        return adapter.predict(model, x, args)

    wrapper = TensorOutputWrapper(adapter, model, args)
    if args.sliding_window:
        if sliding_window_inference is None:
            raise ImportError("MONAI is required for --sliding_window.")
        return sliding_window_inference(
            inputs=x,
            roi_size=tuple(args.roi_size),
            sw_batch_size=args.sw_batch_size,
            predictor=wrapper,
            overlap=args.sw_overlap,
            mode="gaussian",
        )
    return wrapper(x)


# ------------------------------- profiling ---------------------------------

def count_params_m(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def profile_flops_g(adapter, model: nn.Module, args, device: torch.device) -> Tuple[float, str]:
    if FlopCountAnalysis is None:
        return float("nan"), "fvcore_not_installed"
    wrapper = TensorOutputWrapper(adapter, model, args).eval()
    dummy = torch.randn(
        1, args.in_channels, *args.roi_size, device=device, dtype=torch.float32
    )
    try:
        with torch.no_grad():
            analysis = FlopCountAnalysis(wrapper, dummy)
            total = float(analysis.total()) / 1e9
            unsupported = analysis.unsupported_ops()
        note = ""
        if unsupported:
            note = "unsupported_ops=" + ";".join(
                f"{k}:{v}" for k, v in sorted(unsupported.items())
            )
        return total, note
    except Exception as exc:
        return float("nan"), f"flops_failed:{type(exc).__name__}:{exc}"
    finally:
        del dummy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _autocast_context(enabled: bool):
    if not enabled:
        return torch.autocast(device_type="cuda", enabled=False)
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)


@dataclass
class TrainStats:
    mean_step_s: float
    model_s_per_epoch: float
    wall_s_per_epoch: float
    peak_alloc_gib: float
    peak_reserved_gib: float
    measured_steps: int


def benchmark_training(
    adapter,
    model: nn.Module,
    loader: DataLoader,
    args,
    device: torch.device,
) -> TrainStats:
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    def run_step(batch: Dict[str, Any]) -> float:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with _autocast_context(args.amp):
            output = adapter_forward(adapter, model, x, args)
            logits = adapter_unwrap(adapter, output, args)
            loss = adapter_loss(adapter, output, logits, y, args)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize(device)
        return time.perf_counter() - t0

    # Warm-up creates cudnn kernels and AdamW states before peak measurement.
    warm_iter = iter(loader)
    for _ in range(args.train_warmup_steps):
        try:
            batch = next(warm_iter)
        except StopIteration:
            warm_iter = iter(loader)
            batch = next(warm_iter)
        run_step(batch)

    torch.cuda.reset_peak_memory_stats(device)
    measured: List[float] = []
    wall_t0 = time.perf_counter()
    for step, batch in enumerate(loader):
        measured.append(run_step(batch))
        if not args.full_train_epoch and len(measured) >= args.train_steps:
            break
    torch.cuda.synchronize(device)
    wall_elapsed = time.perf_counter() - wall_t0

    if not measured:
        raise RuntimeError("No training batches were measured.")
    mean_step = float(np.mean(measured))
    steps_per_epoch = len(loader)
    model_s_epoch = mean_step * steps_per_epoch
    if args.full_train_epoch and len(measured) == steps_per_epoch:
        wall_s_epoch = wall_elapsed
    else:
        wall_s_epoch = (wall_elapsed / len(measured)) * steps_per_epoch

    peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)

    optimizer.zero_grad(set_to_none=True)
    del optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()

    return TrainStats(
        mean_step_s=mean_step,
        model_s_per_epoch=model_s_epoch,
        wall_s_per_epoch=wall_s_epoch,
        peak_alloc_gib=peak_alloc,
        peak_reserved_gib=peak_reserved,
        measured_steps=len(measured),
    )


@dataclass
class InferStats:
    mean_s_per_volume: float
    std_s_per_volume: float
    peak_alloc_gib: float
    peak_reserved_gib: float
    measured_volumes: int


def benchmark_inference(
    adapter,
    model: nn.Module,
    loader: DataLoader,
    args,
    device: torch.device,
) -> InferStats:
    model.eval()
    if len(loader) == 0:
        raise RuntimeError("Validation loader is empty.")

    # Data loading is excluded: each CPU batch is obtained and copied to CUDA
    # before the timed predictor section begins. No full-volume cache is kept.
    with torch.inference_mode():
        warm_iter = iter(loader)
        for _ in range(args.infer_warmup_volumes):
            try:
                batch = next(warm_iter)
            except StopIteration:
                warm_iter = iter(loader)
                batch = next(warm_iter)
            x = batch["image"].to(device, non_blocking=True)
            with _autocast_context(args.amp):
                _ = adapter_predict(adapter, model, x, args)
            torch.cuda.synchronize(device)
            del x

        torch.cuda.reset_peak_memory_stats(device)
        times: List[float] = []
        count = min(args.infer_volumes, len(loader))
        infer_iter = iter(loader)
        for _ in range(count):
            batch = next(infer_iter)
            x = batch["image"].to(device, non_blocking=True)
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            with _autocast_context(args.amp):
                _ = adapter_predict(adapter, model, x, args)
            torch.cuda.synchronize(device)
            times.append(time.perf_counter() - t0)
            del x

    if not times:
        raise RuntimeError("No validation volumes were measured.")

    peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
    gc.collect()
    torch.cuda.empty_cache()

    return InferStats(
        mean_s_per_volume=float(np.mean(times)),
        std_s_per_volume=float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
        peak_alloc_gib=peak_alloc,
        peak_reserved_gib=peak_reserved,
        measured_volumes=len(times),
    )


def append_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def fmt(v: float, digits: int = 4) -> str:
    return "nan" if not math.isfinite(v) else f"{v:.{digits}f}"


# ---------------------------------- CLI -------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--adapter", required=True, help="Model adapter Python file")
    p.add_argument("--model_name", required=True)
    p.add_argument("--project_root", default="")
    p.add_argument("--arch_file", default="")
    p.add_argument("--config", default="")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--data_root", required=True)
    p.add_argument("--train_list", default="")
    p.add_argument("--val_list", default="")
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--in_channels", type=int, default=4)
    p.add_argument("--num_classes", type=int, default=3)
    p.add_argument("--roi_size", type=int, nargs=3, default=(128, 128, 128))
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--augment", action="store_true")

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--train_warmup_steps", type=int, default=5)
    p.add_argument("--train_steps", type=int, default=30)
    p.add_argument("--full_train_epoch", action="store_true")

    p.add_argument("--sliding_window", action="store_true")
    p.add_argument("--sw_overlap", type=float, default=0.5)
    p.add_argument("--sw_batch_size", type=int, default=1)
    p.add_argument("--infer_warmup_volumes", type=int, default=2)
    p.add_argument("--infer_volumes", type=int, default=20)

    p.add_argument("--output_csv", default="brats2017_efficiency.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for timing and memory benchmarking.")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")

    seed_everything(args.seed)
    device = torch.device("cuda:0")
    adapter = load_adapter(args.adapter)

    data_root = Path(args.data_root)
    cases = discover_cases(data_root)
    train_cases, val_cases = split_cases(
        cases,
        args.train_list or None,
        args.val_list or None,
        args.val_ratio,
        args.seed,
    )
    print(f"[Data] total={len(cases)}, train={len(train_cases)}, val={len(val_cases)}")

    train_ds = BratsPtDataset(
        train_cases,
        in_channels=args.in_channels,
        roi_size=args.roi_size,
        train=True,
        full_volume=False,
        augment=args.augment,
    )
    val_ds = BratsPtDataset(
        val_cases,
        in_channels=args.in_channels,
        roi_size=args.roi_size,
        train=False,
        full_volume=args.sliding_window,
        augment=False,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_single,
        persistent_workers=args.num_workers > 0,
    )

    if args.project_root:
        os.chdir(args.project_root)

    model = adapter.build_model(args)
    if not isinstance(model, nn.Module):
        raise TypeError(f"adapter.build_model(args) returned {type(model)!r}, expected nn.Module")
    model = model.to(device)

    params_m = count_params_m(model)
    flops_g, flops_note = profile_flops_g(adapter, model, args, device)
    train_stats = benchmark_training(adapter, model, train_loader, args, device)
    infer_stats = benchmark_inference(adapter, model, val_loader, args, device)

    row: Dict[str, Any] = {
        "Model": args.model_name,
        "Param_M": round(params_m, 6),
        "GFLOPs_fvcore_1x4xROI": round(flops_g, 6) if math.isfinite(flops_g) else "nan",
        "Infer_s_per_volume": round(infer_stats.mean_s_per_volume, 6),
        "Infer_std_s": round(infer_stats.std_s_per_volume, 6),
        "Train_model_s_per_epoch": round(train_stats.model_s_per_epoch, 6),
        "Train_wall_s_per_epoch": round(train_stats.wall_s_per_epoch, 6),
        "Train_step_s": round(train_stats.mean_step_s, 6),
        "Train_peak_alloc_GiB": round(train_stats.peak_alloc_gib, 6),
        "Train_peak_reserved_GiB": round(train_stats.peak_reserved_gib, 6),
        "Infer_peak_alloc_GiB": round(infer_stats.peak_alloc_gib, 6),
        "Infer_peak_reserved_GiB": round(infer_stats.peak_reserved_gib, 6),
        "Train_cases": len(train_cases),
        "Val_cases": len(val_cases),
        "Train_steps_per_epoch": len(train_loader),
        "Measured_train_steps": train_stats.measured_steps,
        "Measured_infer_volumes": infer_stats.measured_volumes,
        "Batch_size": args.batch_size,
        "ROI": "x".join(map(str, args.roi_size)),
        "Sliding_window": int(args.sliding_window),
        "SW_overlap": args.sw_overlap if args.sliding_window else "",
        "AMP": int(args.amp),
        "FLOPs_note": flops_note,
        "Checkpoint": args.checkpoint,
    }
    append_csv(Path(args.output_csv), row)

    print("\n" + "=" * 88)
    print(f"Model                         : {args.model_name}")
    print(f"Parameters                    : {params_m:.3f} M")
    print(f"FLOPs, 1x4x{'x'.join(map(str,args.roi_size))} : {fmt(flops_g, 3)} G")
    if flops_note:
        print(f"FLOPs note                    : {flops_note}")
    print(f"Inference time                : {infer_stats.mean_s_per_volume:.4f} ± {infer_stats.std_s_per_volume:.4f} s/volume")
    print(f"Training time, model-only     : {train_stats.model_s_per_epoch:.2f} s/epoch")
    print(f"Training time, wall-clock     : {train_stats.wall_s_per_epoch:.2f} s/epoch")
    print(f"Peak training memory allocated: {train_stats.peak_alloc_gib:.3f} GiB")
    print(f"Peak inference memory allocated: {infer_stats.peak_alloc_gib:.3f} GiB")
    print(f"CSV                           : {Path(args.output_csv).resolve()}")
    print("=" * 88)


if __name__ == "__main__":
    main()

