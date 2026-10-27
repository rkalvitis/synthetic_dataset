"""YAML config loading with dotted-key overrides."""
from __future__ import annotations

import copy
import os

import yaml

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "default.yaml")


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _load_yaml(path: str) -> dict:
    """Load a YAML file; a `_base: other.yaml` key (relative to the file) is loaded first and overridden."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("_base", None)
    if base:
        base_path = base if os.path.isabs(base) else os.path.join(os.path.dirname(os.path.abspath(path)), base)
        cfg = _deep_merge(_load_yaml(base_path), cfg)
    return cfg


def load_config(path: str | None = None, overrides: list[str] | None = None) -> dict:
    cfg = _load_yaml(path or DEFAULT_CONFIG)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must look like key.sub=value, got {ov!r}")
        key, val = ov.split("=", 1)
        set_dotted(cfg, key, yaml.safe_load(val))
    return cfg


def set_dotted(cfg: dict, key: str, value) -> None:
    parts = key.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def merged(cfg: dict, **kw) -> dict:
    out = copy.deepcopy(cfg)
    for k, v in kw.items():
        if v is not None:
            set_dotted(out, k, v)
    return out
