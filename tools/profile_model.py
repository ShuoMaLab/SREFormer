"""Model profiling placeholder."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile SREFormer complexity")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    print(f"[SREFormer] Profiling placeholder for config: {args.config}")
    print("[SREFormer] Final params and GFLOPs scripts will be released with the model code.")


if __name__ == "__main__":
    main()
