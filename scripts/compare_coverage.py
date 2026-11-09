#!/usr/bin/env python
"""Per-pixel beetle coverage of several generated datasets side by side: is every pixel equally likely to
carry a beetle, or are beetles biased toward the centre?

Usage:
  python scripts/compare_coverage.py --out outputs/analysis/coverage_v1_v2.png \\
      "synth v1 (20k)=/path/to/dataset_20k" "synth v2 (40k)=/path/to/synth_v2"
The per-pixel density of each dataset is cached in <dataset>/stats/pixel_density.npz (masks are read once).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from popsynth.labels import read_metadata  # noqa: E402
from popsynth.stats import pixel_density, plot_coverage_by_size, plot_coverage_compare  # noqa: E402


def density_cached(root: str, size: int, refresh: bool) -> tuple[np.ndarray, np.ndarray, int]:
    cache = os.path.join(root, "stats", f"pixel_density_{size}.npz")
    if os.path.exists(cache) and not refresh:
        z = np.load(cache)
        return z["dm"], z["db"], int(z["n"])
    recs = read_metadata(root)
    dm, db, n = pixel_density(root, recs, size)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, dm=dm, db=db, n=n)
    return dm, db, n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="+", help='"label=path" pairs')
    ap.add_argument("--out", default="outputs/analysis/coverage_compare.png")
    ap.add_argument("--size", type=int, default=256, help="grid resolution of the density map")
    ap.add_argument("--refresh", action="store_true", help="ignore cached densities")
    ap.add_argument("--separate", default=None, help="also write every panel / map as its own PNG into this folder")
    ap.add_argument("--by-size", default=None, help="also write this figure: coverage with each beetle weighted equally, per size class (boxes, metadata only)")
    a = ap.parse_args()
    rows, by_size = [], []
    for spec in a.datasets:
        label, root = spec.split("=", 1) if "=" in spec else (os.path.basename(spec.rstrip("/")), spec)
        print(f"{label}: reading masks from {root} ...", flush=True)
        dm, db, n = density_cached(root, a.size, a.refresh)
        rows.append((label, dm, db, n))
        if a.by_size:
            by_size.append((label, read_metadata(root)))
    if a.by_size:
        res = plot_coverage_by_size(by_size, a.by_size, separate_dir=a.separate)
        print(f"by size: {a.by_size}")
        print(json.dumps({k: {c: round(v["center_edge_ratio"], 3) for c, v in row.items() if c != "files"} for k, row in res.items()}, indent=2))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    res = plot_coverage_compare(rows, a.out, separate_dir=a.separate)
    print(f"comparison: {a.out}")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "files"} for k, v in res.items()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
