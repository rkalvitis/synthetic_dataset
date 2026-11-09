#!/usr/bin/env python
"""Analysis of the rendered sprites (beetle views rendered from the gaussian splat) of several datasets:
rendered size, rendered vs pasted size, resampling and oversampling, re-use per sprite, viewpoints, alpha solidity.

Usage:
  python scripts/compare_sprites.py --out outputs/analysis/sprites_v1_v2.png --separate outputs/analysis/plots \\
      "synth v1 (20k)=/Volumes/Extreme SSD/synth/outputs/dataset_20k" "synth v2 (40k)=/Volumes/Extreme SSD/synth_v2"
Alpha statistics are read from the sprite PNGs once and cached in <root>/stats/sprite_alpha.npz (--refresh to redo).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from popsynth.sprite_stats import plot_sprite_compare, sprite_table  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="+", help='"label=path" pairs')
    ap.add_argument("--out", default="outputs/analysis/sprites_compare.png")
    ap.add_argument("--bins", type=int, default=30)
    ap.add_argument("--no-alpha", action="store_true", help="skip reading the sprite PNGs")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--separate", default=None, help="also write every panel as its own PNG into this folder")
    a = ap.parse_args()
    tables = []
    for spec in a.datasets:
        label, root = spec.split("=", 1) if "=" in spec else (os.path.basename(spec.rstrip("/")), spec)
        tables.append((label, sprite_table(root, alpha=not a.no_alpha, refresh=a.refresh)))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    res = plot_sprite_compare(tables, a.out, a.bins, separate_dir=a.separate)
    print(f"comparison: {a.out}")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "files"} for k, v in res.items() if not k.startswith("_")}, indent=2))
    if a.separate:
        files = [f for k, v in res.items() if not k.startswith("_") for f in v.get("files", [])] + res.get("_compare_files", [])
        print("single panels:", *sorted(files), sep="\n  ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
