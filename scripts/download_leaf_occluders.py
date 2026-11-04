#!/usr/bin/env python
"""Build a folder of RGBA leaf cutouts for the occlusion step (beetle hidden behind a leaf).

Source: the Flavia leaf dataset (Wu et al., "A leaf recognition algorithm for plant classification using
probabilistic neural network", 2007; http://flavia.sourceforge.net) - 1 907 leaves of 32 species scanned on a
plain white background (~1 GB archive). The white background makes the cutout trivial (see
popsynth.occlude.cutout_from_plain_background). Please cite the paper if the cutouts end up in a publication.

Any other folder of photos with a plain light background can be converted the same way with --from DIR,
e.g. your own leaves/petals/twigs photographed on white paper.

Usage:
  python scripts/download_leaf_occluders.py                    # download + convert -> data/occluders/flavia
  python scripts/download_leaf_occluders.py --from my_leaves/ --out data/occluders/mine
  options: --max-side 1024 (cutout resolution), --limit N, --white-thresh 225
"""
from __future__ import annotations

import argparse
import os
import sys
import tarfile
import urllib.request

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from popsynth.occlude import cutout_from_plain_background  # noqa: E402

FLAVIA_URL = "https://sourceforge.net/projects/flavia/files/Leaf%20Image%20Dataset/1.0/Leaves.tar.bz2/download"
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def download(url: str, dst: str) -> None:
    if os.path.exists(dst):
        print(f"exists: {dst}")
        return
    print(f"downloading {url} -> {dst}")
    tmp = dst + ".part"

    def hook(n, bs, total):
        if total > 0 and n % 200 == 0:
            print(f"\r  {n * bs / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)

    req = urllib.request.Request(url, headers={"User-Agent": "popsynth-leaf-downloader"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        n = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk); n += 1; hook(n, 1 << 20, total)
    print()
    os.replace(tmp, dst)


def convert(src_dir: str, out_dir: str, max_side: int, white_thresh: int, limit: int | None) -> int:
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(os.path.join(r, f) for r, _, fs in os.walk(src_dir) for f in fs if f.lower().endswith(EXTS))
    if limit:
        files = files[:limit]
    n_ok = 0
    for i, p in enumerate(files):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        alpha = cutout_from_plain_background(img, white_thresh=white_thresh)
        if alpha is None:
            continue
        ys, xs = (alpha > 2).nonzero()
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        rgba = cv2.merge([img[y0:y1, x0:x1, 0], img[y0:y1, x0:x1, 1], img[y0:y1, x0:x1, 2], alpha[y0:y1, x0:x1]])
        s = max_side / max(rgba.shape[:2])
        if s < 1:
            rgba = cv2.resize(rgba, (max(2, int(rgba.shape[1] * s)), max(2, int(rgba.shape[0] * s))), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(out_dir, os.path.splitext(os.path.basename(p))[0] + ".png"), rgba)
        n_ok += 1
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(files)} converted ({n_ok} ok)")
    print(f"wrote {n_ok} cutouts to {out_dir} (of {len(files)} files)")
    return n_ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", default=None, help="convert this folder instead of downloading Flavia")
    ap.add_argument("--out", default="data/occluders/flavia")
    ap.add_argument("--raw", default="data/_raw/flavia", help="where the archive is downloaded/extracted")
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--white-thresh", type=int, default=225, help="min-channel value above which a pixel is background")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    src = a.src
    if src is None:
        os.makedirs(a.raw, exist_ok=True)
        archive = os.path.join(a.raw, "Leaves.tar.bz2")
        download(FLAVIA_URL, archive)
        src = os.path.join(a.raw, "Leaves")
        if not os.path.isdir(src):
            print(f"extracting {archive} ...")
            with tarfile.open(archive, "r:bz2") as tf:
                try:
                    tf.extractall(a.raw, filter="data")
                except TypeError:                   # Python < 3.10.12 has no `filter` argument
                    tf.extractall(a.raw)
            if not os.path.isdir(src):          # archive layout may differ: use the extraction root
                src = a.raw
    convert(src, a.out, a.max_side, a.white_thresh, a.limit)
    with open(os.path.join(a.out, "ATTRIBUTION.txt"), "w") as f:
        f.write("Leaf cutouts derived from the Flavia leaf dataset (Wu et al. 2007, http://flavia.sourceforge.net)\n"
                if a.src is None else f"Leaf cutouts converted from {a.src}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
