"""Background image pool with random crops."""
from __future__ import annotations

import os
import re

import cv2
import numpy as np

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def load_crop(path: str, width: int, height: int, scale: float = 1.0) -> np.ndarray:
    """Deterministic centre crop of one background file with the output aspect, as float32 RGB [0,1]."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    aspect = width / height
    cw = min(W, H * aspect) * scale
    ch = cw / aspect
    cw, ch = max(8, int(cw)), max(8, int(ch))
    x0, y0 = (W - cw) // 2, (H - ch) // 2
    crop = img[y0:y0 + ch, x0:x0 + cw]
    interp = cv2.INTER_AREA if cw > width else cv2.INTER_CUBIC
    return cv2.resize(crop, (width, height), interpolation=interp).astype(np.float32) / 255.0


def _display_path(path: str) -> str:
    """Path as stored in metadata: relative to the working directory when possible, always with '/'."""
    try:
        rel = os.path.relpath(path)
        if not rel.startswith(".."):
            path = rel
    except ValueError:          # Windows: different drive than the working directory
        pass
    return path.replace(os.sep, "/")


def describe_background(path: str, group_dirs: list[str] | None = None) -> dict:
    """Name the source and category of a background photo from its location.

    source   = the background folder it came from (e.g. inat_hostplants, flowers102, places365)
    category = sub-folder below the source (Places365 scene, 'r_river' -> 'river'), or, for flat folders,
               the alphabetic file-name prefix before the last '_' (iNaturalist taxon, 'acer_123.jpg' -> 'acer');
               a `categories.csv` (file,category) in the source folder overrides both.
    """
    path = os.path.abspath(path)
    src_dir = None
    for d in group_dirs or []:
        d_abs = os.path.abspath(d)
        if path.startswith(d_abs + os.sep):
            src_dir = d_abs
            break
    if src_dir is None:
        src_dir = os.path.dirname(path)
    source = os.path.basename(src_dir.rstrip(os.sep))
    rel = os.path.relpath(path, src_dir)
    parts = rel.split(os.sep)
    fname = parts[-1]
    override = _category_overrides(src_dir)
    if fname in override:
        return {"background_source": source, "background_category": override[fname]}
    if len(parts) > 1:
        cat = parts[0]
        cat = re.sub(r"^[a-z]_", "", cat)          # Places365 layout: r_river -> river
        return {"background_source": source, "background_category": cat}
    stem = os.path.splitext(fname)[0]
    if "_" in stem:
        prefix = stem.rsplit("_", 1)[0]
        if re.fullmatch(r"[A-Za-z][A-Za-z_\- ]*", prefix) and prefix.lower() != "image":
            return {"background_source": source, "background_category": prefix.lower()}
    return {"background_source": source, "background_category": source}


_OVERRIDES: dict[str, dict[str, str]] = {}


def _category_overrides(src_dir: str) -> dict[str, str]:
    if src_dir not in _OVERRIDES:
        table = {}
        p = os.path.join(src_dir, "categories.csv")
        if os.path.exists(p):
            import csv
            with open(p, newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("file") and row.get("category"):
                        table[row["file"]] = row["category"]
        _OVERRIDES[src_dir] = table
    return _OVERRIDES[src_dir]


class BackgroundPool:
    """Pool of background photos from several folders.

    dirs: list of folder paths, or dicts {path, weight}.  A folder is picked with probability
    proportional to its weight (default 1), then a file uniformly inside it, so a small
    high-quality folder can still dominate a large generic one.
    """

    def __init__(self, dirs: list, min_size: int = 64):
        self.groups: list[tuple[str, float, list[str]]] = []
        self.min_size = min_size
        dropped = 0
        for d in dirs:
            path, weight = (d["path"], float(d.get("weight", 1.0))) if isinstance(d, dict) else (d, 1.0)
            if not os.path.isdir(path):
                print(f"warning: background dir not found, skipped: {path} (see scripts/download_*backgrounds.py)")
                continue
            files = sorted(os.path.join(r, f) for r, _, fs in os.walk(path) for f in fs if f.lower().endswith(EXTS))
            if min_size > 1:   # drop files whose shorter side is too small (header read only, no decode)
                from PIL import Image
                kept = []
                for p in files:
                    try:
                        with Image.open(p) as im:
                            if min(im.size) >= min_size:
                                kept.append(p)
                    except Exception:  # noqa: BLE001
                        pass
                dropped += len(files) - len(kept)
                files = kept
            if files and weight > 0:
                self.groups.append((path, weight, files))
        if dropped:
            print(f"backgrounds: skipped {dropped} files with shorter side < {min_size}px")
        if not self.groups:
            raise RuntimeError(f"no background images (>= {min_size}px) found in {dirs}")
        w = np.array([g[1] for g in self.groups], float)
        self.probs = w / w.sum()
        self.paths = [p for g in self.groups for p in g[2]]
        self.group_dirs = [g[0] for g in self.groups]

    def __len__(self) -> int:
        return len(self.paths)

    def describe(self) -> str:
        return ", ".join(f"{os.path.basename(p.rstrip('/'))}: {len(f)} ({pr:.0%})"
                         for (p, _, f), pr in zip(self.groups, self.probs))

    def _pick(self, rng: np.random.Generator) -> str:
        files = self.groups[int(rng.choice(len(self.groups), p=self.probs))][2]
        return files[int(rng.integers(len(files)))]

    def sample(self, rng: np.random.Generator, width: int, height: int,
               crop_scale=(0.4, 1.0), hflip: float = 0.5) -> tuple[np.ndarray, dict]:
        """Random crop (random size, same aspect as output) resized to (height,width). Returns float32 RGB in [0,1]."""
        for _ in range(10):
            path = self._pick(rng)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None and min(img.shape[:2]) >= self.min_size:
                break
        else:
            raise RuntimeError("could not load a background image")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        s = float(rng.uniform(*crop_scale))
        # crop as large as possible with the output aspect, then shrink by s
        aspect = width / height
        cw = min(W, H * aspect) * s
        ch = cw / aspect
        cw, ch = max(8, int(cw)), max(8, int(ch))
        x0 = int(rng.integers(0, W - cw + 1))
        y0 = int(rng.integers(0, H - ch + 1))
        crop = img[y0:y0 + ch, x0:x0 + cw]
        interp = cv2.INTER_AREA if cw > width else cv2.INTER_CUBIC
        out = cv2.resize(crop, (width, height), interpolation=interp)
        flipped = bool(rng.random() < hflip)
        if flipped:
            out = out[:, ::-1]
        meta = {"background": _display_path(path), **describe_background(path, self.group_dirs),
                "bg_crop_px": [x0, y0, x0 + cw, y0 + ch], "bg_crop_scale": round(s, 3), "bg_hflip": flipped}
        return np.ascontiguousarray(out).astype(np.float32) / 255.0, meta
