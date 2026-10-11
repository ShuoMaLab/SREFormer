#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Adapter template for SegFormer3D baseline/ours.

Usually only MODEL FACTORY section needs adjustment. The benchmark core calls
build_model(args). Keep this file beside benchmark_brats2017_models.py.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import yaml


def _load_module(py_file: str):
    path = Path(py_file).resolve()
    spec = importlib.util.spec_from_file_location("bench_architecture", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import architecture file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_yaml(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _strip_prefix(state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if state and all(k.startswith(prefix) for k in state):
        return {k[len(prefix):]: v for k, v in state.items()}
    return state


def _load_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    if not checkpoint_path:
        return
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(ckpt, dict):
        for key in (
            "model_state_dict",
            "state_dict",
            "model",
            "network_weights",
            "network",
        ):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint object: {type(ckpt)!r}")

    ckpt = _strip_prefix(ckpt, "module.")
    ckpt = _strip_prefix(ckpt, "model.")
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print(f"[Checkpoint] missing={len(missing)}, unexpected={len(unexpected)}")
    if missing[:10]:
        print("  missing examples   :", missing[:10])
    if unexpected[:10]:
        print("  unexpected examples:", unexpected[:10])


# --------------------------- MODEL FACTORY SECTION --------------------------

def build_model(args) -> nn.Module:
    """
    Required CLI arguments:
      --project_root, --arch_file, --config, --checkpoint

    This automatic factory supports common function names:
      build_architecture(config)
      build_model(config)
      get_model(config)
      create_model(config)

    If your repository uses another constructor, replace the middle part of
    this function with the exact import and constructor from the repository.
    """
    if args.project_root:
        root = str(Path(args.project_root).resolve())
        if root not in sys.path:
            sys.path.insert(0, root)

    cfg = _load_yaml(args.config)
    arch = _load_module(args.arch_file)

    factory = None
    for name in ("build_segformer3d_model", "build_architecture", "build_model", "get_model", "create_model"):
        if hasattr(arch, name) and callable(getattr(arch, name)):
            factory = getattr(arch, name)
            print(f"[Factory] using {name} from {args.arch_file}")
            break
    if factory is None:
        raise AttributeError(
            "No model factory found. Add a direct constructor in "
            "adapter_segformer3d_template.py::build_model()."
        )

    # Try the common signatures in a controlled order.
    errors = []
    model = None
    for call in (
        lambda: factory(cfg),
        lambda: factory(config=cfg),
        lambda: factory(args.config),
        lambda: factory(),
    ):
        try:
            candidate = call()
            if isinstance(candidate, nn.Module):
                model = candidate
                break
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")

    if model is None:
        raise RuntimeError(
            "Automatic model construction failed. Replace the factory block "
            "with the repository's exact constructor. Attempts:\n- "
            + "\n- ".join(errors)
        )

    _load_checkpoint(model, args.checkpoint)
    return model


# Optional hooks. Keep them only when the model needs special handling.

def forward(model: nn.Module, x: torch.Tensor, args):
    return model(x)


def unwrap_output(output, args) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, dict):
        for key in ("out", "logits", "pred", "prediction", "seg_logits"):
            if key in output and torch.is_tensor(output[key]):
                return output[key]
    if isinstance(output, (list, tuple)):
        for item in output:
            if torch.is_tensor(item):
                return item
    raise TypeError(f"Please customize unwrap_output(); got {type(output)!r}")

