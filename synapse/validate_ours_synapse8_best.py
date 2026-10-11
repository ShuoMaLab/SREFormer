# -*- coding: utf-8 -*-
"""Validate a SREFormer Synapse8 checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "brats2017"))

import train_segformer3d_synapse8_patch_sw as base
from train_ours_synapse8_patch_sw import build_model


DEFAULT_ROOT = Path(".")
DEFAULT_DATA_DIR = DEFAULT_ROOT / "data" / "synapse8_original"
DEFAULT_CHECKPOINT = (
    DEFAULT_ROOT
    / "checkpoints"
    / "synapse_best.pth"
)

# Original validation order returned by the training script.
ORIGINAL_NAMES = [
    "Aorta",
    "Gallbladder",
    "Left Kidney",
    "Right Kidney",
    "Liver",
    "Pancreas",
    "Spleen",
    "Stomach",
]

# Requested paper-table order.
REQUESTED_NAMES = ["Aorta", "Liver", "KL", "KR", "GB", "PC", "SP", "SM"]
REQUESTED_INDEX = [0, 4, 2, 3, 1, 5, 6, 7]


def load_torch_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate current Ours Synapse8 checkpoint_best.pth."
    )
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--roi_size", type=int, nargs=3, default=None)
    parser.add_argument("--overlap", type=float, default=None)
    parser.add_argument("--sw_batch_size", type=int, default=None)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable AMP. By default, use the checkpoint training setting.",
    )
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    val_dir = args.data_dir / "val"
    if not val_dir.is_dir():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    val_files = sorted(val_dir.glob("*.npz"))
    if not val_files:
        raise RuntimeError(f"No validation npz files found in: {val_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_torch_checkpoint(args.checkpoint, device)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

    roi_size = (
        list(args.roi_size)
        if args.roi_size is not None
        else list(ckpt_args.get("patch_size", [64, 128, 128]))
    )
    overlap = (
        float(args.overlap)
        if args.overlap is not None
        else float(ckpt_args.get("overlap", 0.5))
    )
    sw_batch_size = (
        int(args.sw_batch_size)
        if args.sw_batch_size is not None
        else int(ckpt_args.get("sw_batch_size", 2))
    )
    num_classes = (
        int(args.num_classes)
        if args.num_classes is not None
        else int(ckpt_args.get("num_classes", 9))
    )
    amp = False if args.no_amp else bool(ckpt_args.get("amp", True))

    model = build_model(num_classes=num_classes).to(device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a valid model state_dict.")

    # Tolerate DataParallel/DistributedDataParallel prefix only.
    state_dict = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }
    load_result = model.load_state_dict(state_dict, strict=True)

    eval_args = SimpleNamespace(
        patch_size=roi_size,
        overlap=overlap,
        sw_batch_size=sw_batch_size,
        num_classes=num_classes,
        amp=amp,
    )

    params = sum(p.numel() for p in model.parameters())
    saved_epoch = ckpt.get("epoch", "N/A") if isinstance(ckpt, dict) else "N/A"
    saved_best = ckpt.get("best_dice", float("nan")) if isinstance(ckpt, dict) else float("nan")

    print("=" * 110)
    print("OURS SYNAPSE8 BEST-CHECKPOINT VALIDATION")
    print("=" * 110)
    print("Device                 :", device)
    print("Data dir               :", args.data_dir)
    print("Validation dir         :", val_dir)
    print("Validation cases       :", len(val_files))
    print("Checkpoint             :", args.checkpoint)
    print("Checkpoint epoch       :", saved_epoch)
    print("Checkpoint best_dice   :", saved_best)
    print("State-dict load        :", load_result)
    print("Model parameters       :", f"{params:,} ({params / 1e6:.6f} M)")
    print("ROI size               :", tuple(roi_size))
    print("Overlap                :", overlap)
    print("SW batch size          :", sw_batch_size)
    print("AMP                    :", amp)
    print("Original organ order   :", ORIGINAL_NAMES)
    print("=" * 110)

    mean_dice, per_class = base.validate(model, val_dir, device, eval_args)
    per_class = np.asarray(per_class, dtype=np.float64)
    reordered = per_class[REQUESTED_INDEX]

    print("\n" + "=" * 110)
    print("FINAL BEST-CHECKPOINT RESULT")
    print("=" * 110)
    print(f"Revalidated mean Dice  : {mean_dice:.6f}")
    if np.isfinite(float(saved_best)):
        print(f"Saved checkpoint Dice  : {float(saved_best):.6f}")
        print(f"Absolute difference    : {abs(mean_dice - float(saved_best)):.8f}")

    print("\nRequested order:")
    print("\t".join(REQUESTED_NAMES))
    print("\t".join(f"{float(x):.2f}" for x in reordered))

    print("\nPaper-table row:")
    print(
        "Mean Dice\t"
        + "\t".join(REQUESTED_NAMES)
        + "\n"
        + f"{mean_dice:.2f}\t"
        + "\t".join(f"{float(x):.2f}" for x in reordered)
    )
    print("=" * 110)


if __name__ == "__main__":
    main()
