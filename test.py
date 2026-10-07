"""Evaluation entry point placeholder for SREFormer."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SREFormer")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config file.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a model checkpoint.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    checkpoint = args.checkpoint or cfg.get("evaluation", {}).get("checkpoint", "checkpoints/placeholder_best.pth")
    dataset_name = cfg.get("data", {}).get("dataset", "PLACEHOLDER")
    print(f"[SREFormer] Evaluation placeholder for dataset: {dataset_name}")
    print(f"[SREFormer] Checkpoint: {checkpoint}")
    print("[SREFormer] Replace test.py with the finalized evaluation pipeline before release.")


if __name__ == "__main__":
    main()
