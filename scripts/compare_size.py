#!/usr/bin/env python
"""Beetle (sprite) size distribution for several generated datasets side by side: longest box side as a share
of the short image side, in pixels, and box area / image area.

Usage:
  python scripts/compare_size.py --out outputs/analysis/size_v1_v2.png --separate outputs/analysis/plots \\
      "synth v1 (20k)=/Volumes/Extreme SSD/synth/outputs/dataset_20k" "synth v2 (40k)=/Volumes/Extreme SSD/synth_v2"
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from popsynth.labels import read_metadata  # noqa: E402
from popsynth.stats import object_table, plot_size_compare  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="+", help='"label=path" pairs')
    ap.add_argument("--out", default="outputs/analysis/size_compare.png")
    ap.add_argument("--bins", type=int, default=30)
    ap.add_argument("--separate", default=None, help="also write every panel as its own PNG into this folder")
    a = ap.parse_args()
    tables = []
    for spec in a.datasets:
        label, root = spec.split("=", 1) if "=" in spec else (os.path.basename(spec.rstrip("/")), spec)
        tables.append((label, object_table(read_metadata(root))))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    res = plot_size_compare(tables, a.out, a.bins, separate_dir=a.separate)
    print(f"comparison: {a.out}")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "files"} for k, v in res.items()}, indent=2))
    if a.separate:
        print("single panels:", *sorted(f for v in res.values() for f in v.get("files", [])), sep="\n  ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
