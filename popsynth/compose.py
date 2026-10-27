"""Paste a rendered RGBA sprite onto a background with scale, position, colour matching and a shadow."""
from __future__ import annotations

import cv2
import numpy as np


def crop_to_alpha(rgb: np.ndarray, alpha: np.ndarray, pad: int = 1, thresh: float = 1 / 255):
    ys, xs = np.where(alpha > thresh)
    if len(ys) == 0:
        return rgb, alpha
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, alpha.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, alpha.shape[1])
    return rgb[y0:y1, x0:x1], alpha[y0:y1, x0:x1]


def _resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    interp = cv2.INTER_AREA if w < img.shape[1] else cv2.INTER_LINEAR
    return cv2.resize(img, (w, h), interpolation=interp)


def _gaussian(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.05:
        return img
    k = int(2 * round(3 * sigma) + 1)
    return cv2.GaussianBlur(img, (k, k), sigma)


def _u(rng, r):
    return float(rng.uniform(r[0], r[1])) if isinstance(r, (list, tuple)) else float(r)


def paste_sprite(
    bg: np.ndarray,
    sprite_rgb: np.ndarray,
    sprite_alpha: np.ndarray,
    rng: np.random.Generator,
    *,
    object_size=(0.15, 0.9),
    min_visible: float = 0.6,
    color_match=(0.0, 0.5),
    shadow: dict | None = None,
    edge_blur_px=(0.3, 1.0),
    defocus: dict | None = None,
    center: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Composite one sprite. bg/sprite_rgb float32 RGB in [0,1]; sprite_alpha float32 [0,1].

    Returns (image, mask (H,W) float alpha of the pasted object, meta with bbox etc.).
    object_size: longest side of the object as a fraction of the shorter background side.
    center: optional (x, y) in image fractions where the object centre is placed (overrides min_visible).
    """
    H, W = bg.shape[:2]
    rgb, a = crop_to_alpha(sprite_rgb, sprite_alpha)
    sh, sw = a.shape
    size_frac = _u(rng, object_size)
    target_long = max(4, int(round(size_frac * min(H, W))))
    scale = target_long / max(sh, sw)
    nw, nh = max(2, int(round(sw * scale))), max(2, int(round(sh * scale)))
    rgb = _resize(rgb, nw, nh)
    a = _resize(a, nw, nh)

    # optional defocus (depth-of-field look) on the sprite only
    df_sigma = 0.0
    if defocus and rng.random() < defocus.get("prob", 0):
        df_sigma = _u(rng, defocus.get("sigma_px", (0.5, 2.0)))
        rgb = _gaussian(rgb, df_sigma)
        a = _gaussian(a, df_sigma)
    eb = _u(rng, edge_blur_px)
    if eb > 0:
        a = _gaussian(a, eb)

    # position: keep at least min_visible of the object's bbox inside the frame
    vis = float(min_visible)
    x_min = -int((1 - vis) * nw); x_max = W - nw + int((1 - vis) * nw)
    y_min = -int((1 - vis) * nh); y_max = H - nh + int((1 - vis) * nh)
    if center is not None:
        px, py = int(round(center[0] * W - nw / 2)), int(round(center[1] * H - nh / 2))
    else:
        px = int(rng.integers(min(x_min, x_max), max(x_min, x_max) + 1))
        py = int(rng.integers(min(y_min, y_max), max(y_min, y_max) + 1))

    # clip sprite to the frame
    sx0, sy0 = max(0, -px), max(0, -py)
    sx1, sy1 = min(nw, W - px), min(nh, H - py)
    if sx1 <= sx0 or sy1 <= sy0:
        return bg, np.zeros((H, W), np.float32), {"bbox": None}
    dx0, dy0 = px + sx0, py + sy0
    dx1, dy1 = px + sx1, py + sy1
    rgb = rgb[sy0:sy1, sx0:sx1]
    a = a[sy0:sy1, sx0:sx1]

    # colour matching: pull the sprite's mean colour / brightness toward the local background
    cm = _u(rng, color_match)
    if cm > 0:
        region = bg[dy0:dy1, dx0:dx1]
        bg_mean = region.reshape(-1, 3).mean(0) + 1e-3
        w = a[..., None]
        sp_mean = (rgb * w).sum((0, 1)) / max(w.sum(), 1e-6) + 1e-3
        gain = bg_mean / sp_mean
        gain = gain / gain.mean()                          # tint only (keep sprite's own brightness)
        lum_gain = (bg_mean.mean() / sp_mean.mean()) ** 0.5  # half-way brightness match
        g = (1 - cm) + cm * gain * lum_gain
        rgb = np.clip(rgb * g, 0, 1)

    out = bg.copy()
    # soft shadow under the object
    sh_meta = {}
    if shadow and rng.random() < shadow.get("prob", 0):
        strength = _u(rng, shadow.get("strength", (0.2, 0.6)))
        blur = _u(rng, shadow.get("blur", (0.03, 0.12))) * max(nw, nh)
        off = _u(rng, shadow.get("offset", (0.0, 0.08))) * max(nw, nh)
        ang = rng.uniform(0, 2 * np.pi)
        ox, oy = int(round(off * np.cos(ang))), int(round(off * np.sin(ang)))
        full = np.zeros((H, W), np.float32)
        full[dy0:dy1, dx0:dx1] = a
        M = np.float32([[1, 0, ox], [0, 1, oy]])
        full = cv2.warpAffine(full, M, (W, H))
        full = _gaussian(full, blur)
        out *= (1 - strength * full)[..., None]
        sh_meta = {"shadow_strength": round(strength, 3), "shadow_blur_px": round(blur, 1)}

    region = out[dy0:dy1, dx0:dx1]
    w = a[..., None]
    out[dy0:dy1, dx0:dx1] = rgb * w + region * (1 - w)

    mask = np.zeros((H, W), np.float32)
    mask[dy0:dy1, dx0:dx1] = a
    ys, xs = np.where(mask > 0.5)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(ys) else None
    meta = {
        "bbox": bbox, "object_size_frac": round(size_frac, 3), "object_px": [nw, nh],
        "visible_frac": round(float((sx1 - sx0) * (sy1 - sy0)) / (nw * nh), 3),
        "color_match": round(cm, 3), "defocus_sigma": round(df_sigma, 2), **sh_meta,
    }
    return out, mask, meta
