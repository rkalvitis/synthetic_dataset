#!/usr/bin/env python
"""Visibility graphs for several generated datasets side by side (how much of a beetle stays visible when
another beetle, a leaf or the frame edge is on top of / cuts it).

Usage:
  python scripts/compare_visibility.py --out outputs/analysis/visibility_v1_v2.png \\
      "synth v1 (20k)=/Volumes/Extreme SSD/synth/outputs/dataset_20k" "synth v2 (40k)=/Volumes/Extreme SSD/synth_v2"
Also writes <dataset>/stats/visibility.png for each dataset (--no-per-dataset to skip) and prints the numbers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from popsynth.labels import read_metadata  # noqa: E402
from popsynth.stats import object_table, plot_visibility, plot_visibility_compare  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="+", help='"label=path" pairs')
    ap.add_argument("--out", default="outputs/analysis/visibility_compare.png")
    ap.add_argument("--min-visible", type=float, default=0.33)
    ap.add_argument("--no-per-dataset", action="store_true")
    ap.add_argument("--separate", default=None, help="also write every panel as its own PNG into this folder")
    a = ap.parse_args()
    tables = []
    for spec in a.datasets:
        label, root = spec.split("=", 1) if "=" in spec else (os.path.basename(spec.rstrip("/")), spec)
        recs = read_metadata(root)
        t = object_table(recs)
        tables.append((label, t))
        if not a.no_per_dataset:
            d = os.path.join(root, "stats"); os.makedirs(d, exist_ok=True)
            p = os.path.join(d, "visibility.png")
            sm = plot_visibility(t, p, a.min_visible, title=f"{label}: beetle visibility, n = {len(t['occluded'])}")
            print(f"{label}: {p}\n  {json.dumps(sm)}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    res = plot_visibility_compare(tables, a.out, a.min_visible, separate_dir=a.separate)
    print(f"comparison: {a.out}")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "files"} for k, v in res.items()}, indent=2))
    if a.separate:
        print("single panels:", *sorted(f for v in res.values() for f in v.get("files", [])), sep="\n  ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
