"""Renderer selection.  A renderer is `fn(g, viewmat, K, width, height, **kw) -> (rgb, alpha)`."""
from __future__ import annotations

import importlib.util

import torch


def available_backends() -> list[str]:
    out = ["torch"]
    if importlib.util.find_spec("gsplat") is not None and torch.cuda.is_available():
        out.insert(0, "gsplat")
    return out


def pick_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_renderer(backend: str = "auto"):
    """Return (name, render_fn)."""
    if backend == "auto":
        backend = available_backends()[0]
    if backend == "gsplat":
        from .gsplat_backend import render_gsplat
        return "gsplat", render_gsplat
    if backend == "torch":
        from .torch_raster import render_torch
        return "torch", render_torch
    raise ValueError(f"unknown render backend {backend!r}; choose from gsplat, torch, auto")
