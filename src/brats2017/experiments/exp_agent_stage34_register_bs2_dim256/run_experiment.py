"""Compatibility entry for the BraTS2017 experiment configuration."""

from pathlib import Path
import runpy
import sys


def main() -> None:
    root = Path(__file__).resolve().parents[4]
    target = root / "run_experiment.py"
    config_path = Path(__file__).with_name("config.yaml")
    sys.argv = [str(target), "--config", str(config_path)]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
