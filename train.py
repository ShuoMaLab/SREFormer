"""Training entry point placeholder for SREFormer."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SREFormer")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config file.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    dataset_name = cfg.get("data", {}).get("dataset", "PLACEHOLDER")
    experiment_name = cfg.get("experiment", {}).get("name", "sreformer_placeholder")
    print(f"[SREFormer] Training placeholder for experiment: {experiment_name}")
    print(f"[SREFormer] Dataset: {dataset_name}")
    print("[SREFormer] Replace train.py with the finalized training loop before release.")


if __name__ == "__main__":
    main()
