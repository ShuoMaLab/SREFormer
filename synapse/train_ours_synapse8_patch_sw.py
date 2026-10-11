# -*- coding: utf-8 -*-
"""Synapse8 training wrapper for SREFormer."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "brats2017"))

import train_segformer3d_synapse8_patch_sw as base
try:
    from architectures.segformer3d_stage34_register_synapse_non_cubic import (
        build_segformer3d_model,
    )
except ModuleNotFoundError:
    from architectures.segformer3d import build_segformer3d_model


def build_model(num_classes=9):
    config = {
        "model_name": "SREFormer",
        "model_parameters": {
            "in_channels": 1,
            "num_classes": num_classes,
            "sr_ratios": [4, 2, 1, 1],
            "embed_dims": [32, 64, 160, 256],
            "patch_kernel_size": [7, 3, 3, 3],
            "patch_stride": [4, 2, 2, 2],
            "patch_padding": [3, 1, 1, 1],
            "mlp_ratios": [4, 4, 4, 4],
            "num_heads": [1, 2, 5, 8],
            "depths": [2, 2, 2, 2],
            "agent_num_per_stage": [8, 8, 8, 8],
            "agent_use_dwc": True,
            "agent_register_num_per_stage": [0, 0, 4, 4],
            "decoder_head_embedding_dim": 256,
            "decoder_dropout": 0.0,
        },
    }
    return build_segformer3d_model(config)


base.build_model = build_model


if __name__ == "__main__":
    base.main()
