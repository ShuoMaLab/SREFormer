"""Unified training entry for SREFormer."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def infer_dataset(config: str | None, dataset: str | None) -> str:
    if dataset:
        return dataset.lower()
    if config:
        name = Path(config).stem.lower()
        if "brats" in name:
            return "brats2017"
        if "acdc" in name:
            return "acdc"
        if "synapse" in name:
            return "synapse"
    return "brats2017"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SREFormer")
    parser.add_argument("--dataset", choices=["brats2017", "synapse", "acdc"], default=None)
    parser.add_argument("--config", default=None)
    args, extra = parser.parse_known_args()

    dataset = infer_dataset(args.config, args.dataset)

    if dataset == "brats2017":
        target = ROOT / "run_experiment.py"
        config = args.config or "./experiments/brats_2017/exp_agent_stage34_register_bs2/config.yaml"
        sys.argv = [str(target), "--config", config, *extra]
        runpy.run_path(str(target), run_name="__main__")
        return

    target = ROOT / "synapse" / "train_ours_synapse8_patch_sw.py"
    sys.argv = [str(target), *extra]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
