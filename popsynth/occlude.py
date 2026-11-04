"""Partial occlusion: a beetle hidden behind a leaf or another foreground object.

Two occluder sources, mixed by weight (`compose.occlusion.sources`):
  leaf        an RGBA leaf cutout from a folder of PNGs (e.g. the Flavia leaf set converted with
              scripts/download_leaf_occluders.py, or any cutouts of your own)
  background  a patch cut out of a background photo (the same photo by default, so the colours match)
              with a procedurally generated leaf-shaped soft mask - no extra data needed
The occluder is pasted over the finished composite, in front of the beetle(s), optionally defocused
(a leaf closer to the camera than the focus plane) and darkened/brightened.  Its placement is chosen so
that a target fraction of the beetle's pixels is hidden (`cover`), and the achieved coverage is recorded.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

EXTS = (".png", ".webp", ".tif", ".tiff")


def _u(rng, r):
    return float(rng.uniform(r[0], r[1])) if isinstance(r, (list, tuple)) else float(r)


def _gaussian(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.05:
        return img
    k = int(2 * round(3 * sigma) + 1)
    return cv2.GaussianBlur(np.ascontiguousarray(img), (k, k), sigma)


# ----------------------------------------------------------------------------- leaf cutouts
class OccluderBank:
    """Folders of RGBA PNG cutouts (leaves, petals, twigs ...). Missing folders are skipped with a warning."""

    def __init__(self, dirs: list | None, cache_gb: float = 0.5):
        self.files: list[str] = []
        self.cache_bytes = int(cache_gb * 1024 ** 3)
        self._used = 0
        for d in dirs or []:
            path = d["path"] if isinstance(d, dict) else d
            if not os.path.isdir(path):
                print(f"warning: occluder dir not found, skipped: {path} (see scripts/download_leaf_occluders.py)")
                continue
            self.files += sorted(os.path.join(r, f) for r, _, fs in os.walk(path) for f in fs if f.lower().endswith(EXTS))
        from collections import OrderedDict
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()     # uint8 RGBA, LRU, byte-capped

    def __len__(self) -> int:
        return len(self.files)

    def load(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        rgba8 = self._cache.get(path)
        if rgba8 is None:
            im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if im is None or im.ndim != 3 or im.shape[2] != 4:
                raise ValueError(f"occluder must be an RGBA image: {path}")
            rgba8 = np.ascontiguousarray(cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA))
            if rgba8.nbytes <= self.cache_bytes:
                while self._cache and self._used + rgba8.nbytes > self.cache_bytes:
                    _, old = self._cache.popitem(last=False)
                    self._used -= old.nbytes
                self._cache[path] = rgba8
                self._used += rgba8.nbytes
        else:
            self._cache.move_to_end(path)
        rgba = rgba8.astype(np.float32) / 255
        return np.ascontiguousarray(rgba[..., :3]), np.ascontiguousarray(rgba[..., 3])

    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, str]:
        path = self.files[int(rng.integers(len(self.files)))]
        rgb, a = self.load(path)
        return rgb, a, path


def cutout_from_plain_background(bgr_or_rgb: np.ndarray, white_thresh: int = 225, feather: float = 1.5,
                                 min_area_frac: float = 0.01) -> np.ndarray | None:
    """Alpha mask (uint8 0..255) of an object photographed on a plain white background (Flavia-style scans).

    Pixels darker than `white_thresh` in the min-channel are foreground; the largest connected component is
    kept, holes filled, the edge feathered.  Returns None if nothing usable is found."""
    low = bgr_or_rgb.min(axis=2)
    fg = (low < white_thresh).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n < 2:
        return None
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[best, cv2.CC_STAT_AREA] < min_area_frac * fg.size:
        return None
    comp = (lab == best).astype(np.uint8)
    # fill holes: everything not reachable from the border is inside the leaf
    flood = comp.copy()
    h, w = comp.shape
    ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 1)
    filled = comp | (1 - flood)
    alpha = filled.astype(np.float32)
    if feather > 0:
        alpha = _gaussian(alpha, feather)
    return (np.clip(alpha, 0, 1) * 255 + 0.5).astype(np.uint8)


# ----------------------------------------------------------------------------- procedural leaf shape
def leaf_shape_mask(h: int, w: int, rng: np.random.Generator, softness: float = 0.05) -> np.ndarray:
    """A leaf-like soft mask (h, w) float32 in [0,1]: an ellipse with pointed tips whose radius is modulated
    by a few low-frequency harmonics, so every call gives a slightly different lobed outline."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2, (h - 1) / 2
    u, v = (xx - cx) / (w / 2), (yy - cy) / (h / 2)
    theta = np.arctan2(v, u)
    r = np.sqrt(u * u + v * v)
    # pointed tips along the long axis: shrink the radius near theta = +-pi/2 ... use |cos| pinch
    pinch = _u(rng, (0.1, 0.35))
    base = 1.0 - pinch * np.abs(np.sin(theta)) ** 3
    harm = np.zeros_like(theta)
    for k in range(2, 6):
        harm += _u(rng, (0.0, 0.06)) * np.cos(k * theta + rng.uniform(0, 2 * np.pi))
    radius = base + harm
    edge = (radius - r) / max(softness, 1e-3)
    return np.clip(0.5 + 0.5 * edge, 0, 1).astype(np.float32)


def background_patch(src: np.ndarray, rng: np.random.Generator, long_px: int,
                     aspect=(0.45, 0.8)) -> tuple[np.ndarray, np.ndarray]:
    """Cut a random patch from `src` (float RGB) and give it a leaf-shaped alpha. Returns (rgb, alpha)."""
    H, W = src.shape[:2]
    asp = _u(rng, aspect)
    pw, ph = long_px, max(4, int(round(long_px * asp)))
    # crop at (almost) the same scale as the destination so the texture frequency matches the surroundings
    s = _u(rng, (0.85, 1.2))
    cw = max(8, min(W, int(pw * s)))
    ch = max(8, min(H, int(cw * ph / pw)))
    x0 = int(rng.integers(0, W - cw + 1)); y0 = int(rng.integers(0, H - ch + 1))
    crop = src[y0:y0 + ch, x0:x0 + cw]
    interp = cv2.INTER_AREA if cw > pw else cv2.INTER_LINEAR
    rgb = cv2.resize(np.ascontiguousarray(crop), (pw, ph), interpolation=interp)
    return rgb, leaf_shape_mask(ph, pw, rng)


def _rotate_rgba(rgb: np.ndarray, a: np.ndarray, deg: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = a.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos) + 2, int(h * cos + w * sin) + 2
    M[0, 2] += nw / 2 - w / 2; M[1, 2] += nh / 2 - h / 2
    rgb = cv2.warpAffine(rgb, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    a = cv2.warpAffine(a, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rgb, a


def _crop_alpha(rgb, a, thresh=1 / 255):
    ys, xs = np.where(a > thresh)
    if len(ys) == 0:
        return rgb, a
    return rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1], a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


# ----------------------------------------------------------------------------- placing one occluder
def place_occluder(img: np.ndarray, target_mask: np.ndarray, occ_rgb: np.ndarray, occ_a: np.ndarray,
                   rng: np.random.Generator, *, cover: float, tries: int = 12,
                   blur_sigma: float = 0.0, brightness: float = 1.0, contrast: float = 1.0,
                   match_color: float = 0.0, shadow: dict | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Paste (occ_rgb, occ_a) over `img` so that about `cover` of `target_mask` (float, (H,W)) is hidden.

    Several candidate positions around the target are scored and the one whose coverage is closest to
    `cover` wins.  Before pasting, the occluder's mean colour and brightness are pulled toward the region it
    covers (`match_color`, 0..1), its contrast scaled, and a soft contact shadow is cast on the image under
    its edge (`shadow`: strength, blur, offset as fractions of the occluder size), so a foreground leaf does
    not look like a bright sticker.  Returns (image, occluder alpha (H,W), achieved coverage of the target)."""
    H, W = img.shape[:2]
    ys, xs = np.where(target_mask > 0.5)
    if len(ys) == 0:
        return img, np.zeros((H, W), np.float32), 0.0
    tx0, tx1, ty0, ty1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    tcx, tcy = (tx0 + tx1) / 2, (ty0 + ty1) / 2
    oh, ow = occ_a.shape
    if blur_sigma > 0:
        occ_rgb = _gaussian(occ_rgb, blur_sigma)
        occ_a = _gaussian(occ_a, blur_sigma)
    if brightness != 1.0:
        occ_rgb = np.clip(occ_rgb * brightness, 0, 1)
    target_sum = max(float(target_mask.sum()), 1e-6)

    best = None
    for _ in range(tries):
        # occluder centre: around the beetle, further out when less coverage is wanted
        spread = 0.5 * (1.0 - cover) + 0.15
        cx = tcx + rng.normal(0, spread * (tx1 - tx0 + ow) / 2)
        cy = tcy + rng.normal(0, spread * (ty1 - ty0 + oh) / 2)
        px, py = int(round(cx - ow / 2)), int(round(cy - oh / 2))
        sx0, sy0 = max(0, -px), max(0, -py)
        sx1, sy1 = min(ow, W - px), min(oh, H - py)
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        dx0, dy0, dx1, dy1 = px + sx0, py + sy0, px + sx1, py + sy1
        a_sub = occ_a[sy0:sy1, sx0:sx1]
        hidden = float((target_mask[dy0:dy1, dx0:dx1] * a_sub).sum()) / target_sum
        score = abs(hidden - cover)
        if best is None or score < best[0]:
            best = (score, hidden, (sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1))
    if best is None:
        return img, np.zeros((H, W), np.float32), 0.0
    _, hidden, (sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1) = best
    out = img.copy()
    a_sub = occ_a[sy0:sy1, sx0:sx1][..., None]
    rgb_sub = occ_rgb[sy0:sy1, sx0:sx1]
    region = out[dy0:dy1, dx0:dx1]
    if match_color > 0 or contrast != 1.0:
        wsum = max(float(a_sub.sum()), 1e-6)
        occ_mean = (rgb_sub * a_sub).sum((0, 1)) / wsum + 1e-3
        if contrast != 1.0:
            rgb_sub = (rgb_sub - occ_mean) * contrast + occ_mean
        if match_color > 0:
            bg_mean = (region * a_sub).sum((0, 1)) / wsum + 1e-3
            gain = bg_mean / occ_mean                       # per-channel: tint and brightness of the covered spot
            rgb_sub = rgb_sub * ((1 - match_color) + match_color * gain)
        rgb_sub = np.clip(rgb_sub, 0, 1)
    full = np.zeros((H, W), np.float32)
    full[dy0:dy1, dx0:dx1] = a_sub[..., 0]
    if shadow and np.max(shadow.get("strength", 0)) > 0:
        size = max(oh, ow)
        strength = _u(rng, shadow.get("strength", (0.15, 0.35)))
        sblur = _u(rng, shadow.get("blur", (0.03, 0.08))) * size
        off = _u(rng, shadow.get("offset", (0.0, 0.05))) * size
        ang = rng.uniform(0, 2 * np.pi)
        M = np.float32([[1, 0, off * np.cos(ang)], [0, 1, off * np.sin(ang)]])
        sh = cv2.warpAffine(full, M, (W, H))
        sh = _gaussian(sh, sblur)
        out *= (1 - strength * sh)[..., None]
        region = out[dy0:dy1, dx0:dx1]
    out[dy0:dy1, dx0:dx1] = rgb_sub * a_sub + region * (1 - a_sub)
    return out, full, round(hidden, 3)


def make_occluder(rng: np.random.Generator, cfg: dict, long_px: int, bank: OccluderBank | None,
                  bg_photo: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Pick a source by weight and build one occluder with its long side = long_px. Returns (rgb, alpha, meta)."""
    sources = dict(cfg.get("sources") or {"leaf": 0.5, "background": 0.5})
    if not bank or len(bank) == 0:
        sources.pop("leaf", None)
    if bg_photo is None:
        sources.pop("background", None)
    if not sources:
        raise RuntimeError("no occluder source available: need occlusion.dirs with cutouts or a background photo")
    names = list(sources)
    w = np.array([float(sources[n]) for n in names]); w = w / w.sum()
    kind = names[int(rng.choice(len(names), p=w))]
    if kind == "leaf":
        rgb, a, path = bank.sample(rng)
        rgb, a = _crop_alpha(rgb, a)
        scale = long_px / max(a.shape)
        nw, nh = max(2, int(round(a.shape[1] * scale))), max(2, int(round(a.shape[0] * scale)))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        rgb = cv2.resize(rgb, (nw, nh), interpolation=interp)
        a = cv2.resize(a, (nw, nh), interpolation=interp)
        if rng.random() < 0.5:
            rgb, a = rgb[:, ::-1], a[:, ::-1]
        meta = {"occluder": "leaf", "occluder_file": path.replace(os.sep, "/")}
    else:
        rgb, a = background_patch(bg_photo, rng, long_px, tuple(cfg.get("patch_aspect", (0.45, 0.8))))
        meta = {"occluder": "background"}
    deg = _u(rng, cfg.get("rotate_deg", (0, 360)))
    rgb, a = _rotate_rgba(np.ascontiguousarray(rgb), np.ascontiguousarray(a), deg)
    rgb, a = _crop_alpha(rgb, a)
    meta["occluder_rotate_deg"] = round(deg, 1)
    return np.ascontiguousarray(rgb), np.ascontiguousarray(a), meta


def occlude_objects(img: np.ndarray, obj_masks: list[np.ndarray], rng: np.random.Generator, cfg: dict,
                    bank: OccluderBank | None = None, bg_photo: np.ndarray | None = None
                    ) -> tuple[np.ndarray, list[np.ndarray], list[dict]]:
    """Apply the configured occlusion to a composite with beetles `obj_masks` (float (H,W) each).

    cfg keys: prob (per positive image), count [lo,hi] occluders, cover [lo,hi] target hidden fraction of
    the chosen beetle, size [lo,hi] occluder long side relative to the beetle's long side, blur_sigma_px,
    brightness, sources {leaf, background}, rotate_deg, max_hidden (skip an occluder that would hide
    more than this of any beetle).
    Returns (image, visible masks (beetle alpha x (1 - occluder alpha)), list of occluder meta dicts)."""
    if not cfg or not obj_masks or rng.random() >= float(cfg.get("prob", 0.0)):
        return img, obj_masks, []
    lo, hi = cfg.get("count", [1, 1])
    n = int(rng.integers(int(lo), int(hi) + 1))
    visible = [m.copy() for m in obj_masks]
    occ_total = np.zeros(img.shape[:2], np.float32)
    out = img
    metas = []
    max_hidden = float(cfg.get("max_hidden", 0.85))
    for _ in range(n):
        j = int(rng.integers(len(obj_masks)))
        ys, xs = np.where(visible[j] > 0.5)
        if len(ys) == 0:
            continue
        long_side = max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
        long_px = max(6, int(round(long_side * _u(rng, cfg.get("size", (0.6, 1.6))))))
        try:
            o_rgb, o_a, om = make_occluder(rng, cfg, long_px, bank, bg_photo)
        except RuntimeError as e:
            print(f"warning: {e}")
            break
        cover = _u(rng, cfg.get("cover", (0.1, 0.6)))
        blur = _u(rng, cfg.get("blur_sigma_px", (0.0, 0.0)))
        bright = _u(rng, cfg.get("brightness", (1.0, 1.0)))
        contrast = _u(rng, cfg.get("contrast", (1.0, 1.0)))
        match = _u(rng, cfg.get("match_color", (0.0, 0.0)))
        cand, o_full, hidden = place_occluder(out, visible[j], o_rgb, o_a, rng, cover=cover,
                                              blur_sigma=blur, brightness=bright, contrast=contrast,
                                              match_color=match, shadow=cfg.get("shadow"))
        # reject if any beetle would end up almost completely hidden
        too_much = False
        for m in obj_masks:
            tot = max(float(m.sum()), 1e-6)
            if float((m * np.clip(occ_total + o_full, 0, 1)).sum()) / tot > max_hidden:
                too_much = True
        if too_much:
            continue
        out = cand
        occ_total = np.clip(occ_total + o_full, 0, 1)
        for k in range(len(visible)):
            visible[k] = visible[k] * (1 - o_full)
        om.update({"target_object": j, "cover_target": round(cover, 3), "cover_achieved": hidden,
                   "occluder_px": [int(o_a.shape[1]), int(o_a.shape[0])],
                   "occluder_blur_sigma": round(blur, 2), "occluder_brightness": round(bright, 3),
                   "occluder_contrast": round(contrast, 3), "occluder_match_color": round(match, 3)})
        metas.append(om)
    return out, visible, metas
