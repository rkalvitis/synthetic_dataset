#!/usr/bin/env python
"""Download the Places365 validation set and keep only outdoor/garden-like categories as backgrounds.

Places365 (Zhou et al., 2017), http://places2.csail.mit.edu  -- free for research use, see the
Places365 terms.  The 256x256 validation archive is ~500 MB and holds 100 images per category;
`--size large` fetches the full-resolution archive instead (~2 GB, better for big output images).

Also: `--source flowers102` fetches the Oxford 102 Flowers set (Nilsback & Zisserman, 2008; ~330 MB,
8 189 close-up photos of flowers and foliage, research use) into data/backgrounds/flowers102.  Close-up
plant photos match the scale of a 1 cm beetle much better than scene-level photos.

Usage:
  python scripts/download_backgrounds.py [--source places365|flowers102] [--size 256|large]
                                         [--out DIR] [--categories configs/places365_categories.txt]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import urllib.request

BASE = "https://data.csail.mit.edu/places/places365/"
FLOWERS_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
FILELIST = "filelist_places365-standard.tar"
MD5 = {"val_256.tar": "e27b17d8d44f4af9a78502beb927f808", FILELIST: "35a0585fee1fa656440f3ab298f8479c"}


def download(url: str, dst: str) -> None:
    if os.path.exists(dst):
        print(f"exists: {dst}")
        return
    print(f"downloading {url} -> {dst}")
    tmp = dst + ".part"

    def hook(n, bs, total):
        if total > 0 and n % 200 == 0:
            print(f"\r  {n * bs / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, tmp, hook)
    print()
    os.replace(tmp, dst)


def md5sum(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_categories(path: str) -> list[str]:
    out = []
    for line in open(path):
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def fetch_flowers102(args) -> int:
    archive = os.path.join(args.raw, "102flowers.tgz")
    download(FLOWERS_URL, archive)
    n = 0
    with tarfile.open(archive) as tf:
        for m in tf:
            if m.isfile() and m.name.lower().endswith(".jpg"):
                with tf.extractfile(m) as src, open(os.path.join(args.out, os.path.basename(m.name)), "wb") as dst:
                    shutil.copyfileobj(src, dst)
                n += 1
    print(f"extracted {n} images into {args.out}")
    if not args.keep_archives:
        os.remove(archive)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="places365", choices=["places365", "flowers102"])
    ap.add_argument("--size", default="256", choices=["256", "large"])
    ap.add_argument("--out", default=None, help="default: data/backgrounds/<source>")
    ap.add_argument("--raw", default="data/backgrounds/_raw", help="where archives are kept")
    ap.add_argument("--categories", default="configs/places365_categories.txt")
    ap.add_argument("--keep-archives", action="store_true")
    args = ap.parse_args()

    args.out = args.out or os.path.join("data", "backgrounds", args.source)
    os.makedirs(args.raw, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    if args.source == "flowers102":
        return fetch_flowers102(args)
    archive = "val_256.tar" if args.size == "256" else "val_large.tar"
    download(BASE + FILELIST, os.path.join(args.raw, FILELIST))
    download(BASE + archive, os.path.join(args.raw, archive))
    if archive in MD5:
        got = md5sum(os.path.join(args.raw, archive))
        if got != MD5[archive]:
            print(f"MD5 mismatch for {archive}: {got} != {MD5[archive]}", file=sys.stderr)
            return 1

    # category index -> name
    with tarfile.open(os.path.join(args.raw, FILELIST)) as tf:
        cats = tf.extractfile("categories_places365.txt").read().decode().splitlines()
        val = tf.extractfile("places365_val.txt").read().decode().splitlines()
    idx_to_name = {int(l.split()[1]): l.split()[0] for l in cats}
    wanted = set(read_categories(args.categories))
    unknown = wanted - set(idx_to_name.values())
    if unknown:
        print(f"warning: unknown categories ignored: {sorted(unknown)}")
    file_to_cat = {}
    for l in val:
        fn, idx = l.split()
        name = idx_to_name[int(idx)]
        if name in wanted:
            file_to_cat[fn] = name.strip("/").replace("/", "_")

    n = 0
    with tarfile.open(os.path.join(args.raw, archive)) as tf:
        for m in tf:
            base = os.path.basename(m.name)
            cat = file_to_cat.get(base)
            if cat is None or not m.isfile():
                continue
            d = os.path.join(args.out, cat)
            os.makedirs(d, exist_ok=True)
            with tf.extractfile(m) as src, open(os.path.join(d, base), "wb") as dst:
                shutil.copyfileobj(src, dst)
            n += 1
    print(f"extracted {n} images from {len(set(file_to_cat.values()))} categories into {args.out}")
    if not args.keep_archives:
        for a in (archive, FILELIST):
            os.remove(os.path.join(args.raw, a))
    return 0


if __name__ == "__main__":
    sys.exit(main())
