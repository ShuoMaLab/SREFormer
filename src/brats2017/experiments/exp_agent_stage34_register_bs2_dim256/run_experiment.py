"""BraTS2017 experiment entry placeholder.

The imported BraTS2017 training code is being cleaned for the public release.
This entry keeps the experiment location stable while the final trainer script is
standardized.
"""

from pathlib import Path


def main() -> None:
    config_path = Path(__file__).with_name("config.yaml")
    print("SREFormer BraTS2017 experiment placeholder")
    print(f"Config: {config_path}")
    print("The finalized training entry will be released after code cleanup.")


if __name__ == "__main__":
    main()
