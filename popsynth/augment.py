"""Whole-image photometric augmentation: lighting conditions (day/overcast/dusk/night), blur, noise, JPEG."""
from __future__ import annotations

import cv2
import numpy as np

# Each lighting mode: exposure multiplier, RGB tint (white balance), contrast, saturation, noise sigma.
DEFAULT_MODES = {
    "day":      {"weight": 0.55, "exposure": [0.8, 1.3],  "tint": [[0.95, 1.05], [0.95, 1.05], [0.9, 1.05]],
                 "contrast": [0.9, 1.2],  "saturation": [0.9, 1.2],  "noise": [0.0, 0.02], "vignette": [0.0, 0.2]},
    "overcast": {"weight": 0.20, "exposure": [0.6, 1.0],  "tint": [[0.9, 1.0],  [0.95, 1.05], [1.0, 1.1]],
                 "contrast": [0.7, 1.0],  "saturation": [0.6, 0.95], "noise": [0.0, 0.03], "vignette": [0.0, 0.2]},
    "dusk":     {"weight": 0.10, "exposure": [0.35, 0.7], "tint": [[1.0, 1.15], [0.85, 1.0],  [0.7, 0.95]],
                 "contrast": [0.8, 1.1],  "saturation": [0.7, 1.1],  "noise": [0.01, 0.05], "vignette": [0.1, 0.4]},
    "night":    {"weight": 0.15, "exposure": [0.1, 0.35], "tint": [[0.75, 0.95], [0.85, 1.0],  [1.0, 1.25]],
                 "contrast": [0.7, 1.0],  "saturation": [0.3, 0.8],  "noise": [0.03, 0.10], "vignette": [0.2, 0.6],
                 "flash_prob": 0.4},
}


def _u(rng, r):
    return float(rng.uniform(r[0], r[1])) if isinstance(r, (list, tuple)) else float(r)


def apply_lighting(img: np.ndarray, rng: np.random.Generator, modes: dict | None = None,
                   mode: str | None = None) -> tuple[np.ndarray, dict]:
    """img float32 RGB [0,1] -> (img, meta). Picks a lighting mode by weight (or the given one) and applies it."""
    modes = modes or DEFAULT_MODES
    names = list(modes)
    w = np.array([modes[n].get("weight", 1.0) for n in names], float)
    mode = mode or names[int(rng.choice(len(names), p=w / w.sum()))]
    m = modes[mode]
    H, W = img.shape[:2]
    out = img.copy()

    exposure = _u(rng, m.get("exposure", 1.0))
    tint = np.array([_u(rng, t) for t in m.get("tint", [1, 1, 1])], np.float32)
    out = out * exposure * tint[None, None]

    # camera flash at night: bright centre, falls off radially
    flash = False
    if rng.random() < m.get("flash_prob", 0.0):
        flash = True
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r2 = ((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2
        out = out * (1 + _u(rng, (2.0, 5.0)) * np.exp(-r2 / 0.08))[..., None]

    contrast = _u(rng, m.get("contrast", 1.0))
    out = (out - 0.5 * exposure) * contrast + 0.5 * exposure
    saturation = _u(rng, m.get("saturation", 1.0))
    gray = out @ np.array([0.299, 0.587, 0.114], np.float32)
    out = gray[..., None] + (out - gray[..., None]) * saturation

    vig = _u(rng, m.get("vignette", 0.0))
    if vig > 0:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r2 = ((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2
        out = out * (1 - vig * np.clip(r2 / 2, 0, 1))[..., None]

    noise = _u(rng, m.get("noise", 0.0))
    if noise > 0:
        out = out + rng.normal(0, noise, out.shape).astype(np.float32)
    out = np.clip(out, 0, 1)
    meta = {"lighting": mode, "exposure": round(exposure, 3), "contrast": round(contrast, 3),
            "saturation": round(saturation, 3), "noise": round(noise, 3), "flash": flash}
    return out, meta


def _motion_blur(img: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    k = max(3, int(k) | 1)
    kern = np.zeros((k, k), np.float32)
    kern[k // 2] = 1.0 / k
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), float(rng.uniform(0, 180)), 1)
    kern = cv2.warpAffine(kern, M, (k, k))
    kern /= max(kern.sum(), 1e-6)
    return cv2.filter2D(np.ascontiguousarray(img), -1, kern)


def _gauss(img: np.ndarray, sigma: float) -> np.ndarray:
    k = int(2 * round(3 * sigma) + 1)
    return cv2.GaussianBlur(np.ascontiguousarray(img), (k, k), sigma)


def _jpeg(img: np.ndarray, q: int) -> np.ndarray:
    u8 = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(u8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255


def _lowres(img: np.ndarray, factor: float) -> np.ndarray:
    H, W = img.shape[:2]
    small = cv2.resize(img, (max(8, int(W / factor)), max(8, int(H / factor))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)


def apply_imperfections(img: np.ndarray, rng: np.random.Generator, tiers: dict,
                        tier: str | None = None) -> tuple[np.ndarray, dict]:
    """Real-photo imperfections. A quality tier is drawn by weight (clean / mild / bad ...), then each
    effect of that tier is applied with its own probability: motion blur, defocus, low resolution
    (down+up-sampling, i.e. a far-away phone shot), haze / dirty lens, over- or under-exposure,
    sensor noise and JPEG compression. Returns float32 [0,1] and meta."""
    names = list(tiers)
    w = np.array([tiers[n].get("weight", 1.0) for n in names], float)
    tier = tier or names[int(rng.choice(len(names), p=w / w.sum()))]
    t = tiers[tier]
    out = np.ascontiguousarray(img, dtype=np.float32)
    meta = {"quality_tier": tier}
    if rng.random() < t.get("blur_prob", 0):
        k = int(rng.integers(int(t.get("motion_px", [3, 9])[0]), int(t.get("motion_px", [3, 9])[1]) + 1))
        out = _motion_blur(out, k, rng); meta["motion_blur_px"] = k
    if rng.random() < t.get("defocus_prob", 0):
        sg = _u(rng, t.get("defocus_sigma", (0.5, 1.5))); out = _gauss(out, sg); meta["defocus_sigma"] = round(sg, 2)
    if rng.random() < t.get("lowres_prob", 0):
        f = _u(rng, t.get("lowres_factor", (1.5, 3.0))); out = _lowres(out, f); meta["lowres_factor"] = round(f, 2)
    if rng.random() < t.get("haze_prob", 0):
        h = _u(rng, t.get("haze", (0.1, 0.4)))
        out = out * (1 - h) + h * np.array([0.78, 0.78, 0.8], np.float32); meta["haze"] = round(h, 2)
    if rng.random() < t.get("exposure_fault_prob", 0):
        if rng.random() < 0.5:
            g = _u(rng, t.get("overexposure", (1.6, 2.6))); meta["overexposure"] = round(g, 2)
        else:
            g = _u(rng, t.get("underexposure", (0.25, 0.5))); meta["underexposure"] = round(g, 2)
        out = out * g
    out = np.clip(out, 0, 1)
    n = _u(rng, t.get("noise", (0.0, 0.0)))
    if n > 0:
        out = np.clip(out + rng.normal(0, n, out.shape).astype(np.float32), 0, 1); meta["sensor_noise"] = round(n, 3)
    jq = t.get("jpeg_quality")
    if jq:
        q = int(rng.integers(int(jq[0]), int(jq[1]) + 1)); out = _jpeg(out, q); meta["jpeg_quality"] = q
    return np.ascontiguousarray(out), meta


def apply_camera_effects(img: np.ndarray, rng: np.random.Generator, cfg: dict) -> tuple[np.ndarray, dict]:
    """Camera-side degradation. With `tiers` in cfg this delegates to apply_imperfections; otherwise the
    legacy flat form: optional blur (gaussian or motion) and a JPEG round-trip. Returns float32 [0,1] and meta."""
    if cfg.get("tiers"):
        return apply_imperfections(img, rng, cfg["tiers"])
    meta = {}
    out = img
    blur = cfg.get("blur", {})
    if rng.random() < blur.get("prob", 0.0):
        if rng.random() < blur.get("motion_frac", 0.5):
            k = int(rng.integers(3, max(4, int(blur.get("motion_max_px", 9)) + 1)))
            kern = np.zeros((k, k), np.float32)
            kern[k // 2] = 1.0 / k
            ang = float(rng.uniform(0, 180))
            M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), ang, 1)
            kern = cv2.warpAffine(kern, M, (k, k))
            kern /= max(kern.sum(), 1e-6)
            out = cv2.filter2D(np.ascontiguousarray(out), -1, kern)
            meta["motion_blur_px"] = k
        else:
            s = _u(rng, blur.get("gaussian_sigma", (0.5, 1.5)))
            k = int(2 * round(3 * s) + 1)
            out = cv2.GaussianBlur(np.ascontiguousarray(out), (k, k), s)
            meta["gaussian_blur_sigma"] = round(s, 2)
    jq = cfg.get("jpeg_quality")
    if jq:
        q = int(rng.integers(int(jq[0]), int(jq[1]) + 1))
        u8 = (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(u8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
        out = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        meta["jpeg_quality"] = q
    return np.ascontiguousarray(out), meta


# ----------------------------------------------------------------------------- separate augmentations
# Two independent augmentation stages run BEFORE compositing: one on the background crop and one on the
# beetle sprite.  They decorrelate the two layers (the beetle does not always share the background's tint,
# exposure and sharpness), on top of the whole-image lighting + camera effects applied afterwards.

def _color_jitter(img: np.ndarray, rng: np.random.Generator, c: dict, alpha: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Photometric jitter of float RGB [0,1]: brightness, contrast, saturation, hue rotation, gamma, grayscale.
    Each parameter is applied only if its range is configured. Returns (img, meta of the drawn values)."""
    meta = {}
    out = img.astype(np.float32, copy=True)
    if "brightness" in c:
        b = _u(rng, c["brightness"]); out = out * b; meta["brightness"] = round(b, 3)
    if "contrast" in c:
        k = _u(rng, c["contrast"])
        mean = float((out * alpha[..., None]).sum() / max(alpha.sum() * 3, 1)) if alpha is not None else float(out.mean())
        out = (out - mean) * k + mean; meta["contrast"] = round(k, 3)
    if "gamma" in c:
        g = _u(rng, c["gamma"]); out = np.power(np.clip(out, 0, 1), g); meta["gamma"] = round(g, 3)
    hue = _u(rng, c["hue_deg"]) if "hue_deg" in c else 0.0
    sat = _u(rng, c["saturation"]) if "saturation" in c else 1.0
    if abs(hue) > 0.01 or abs(sat - 1) > 1e-3:
        hsv = cv2.cvtColor(np.clip(np.ascontiguousarray(out), 0, 1), cv2.COLOR_RGB2HSV)   # H in [0,360)
        hsv[..., 0] = (hsv[..., 0] + hue) % 360.0
        hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0, 1)
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        if "hue_deg" in c:
            meta["hue_deg"] = round(hue, 1)
        if "saturation" in c:
            meta["saturation"] = round(sat, 3)
    if rng.random() < c.get("grayscale_prob", 0.0):
        gray = out @ np.array([0.299, 0.587, 0.114], np.float32)
        out = np.repeat(gray[..., None], 3, 2); meta["grayscale"] = True
    return np.clip(out, 0, 1).astype(np.float32), meta


def augment_background(img: np.ndarray, rng: np.random.Generator, cfg: dict | None) -> tuple[np.ndarray, dict]:
    """Background-only augmentation (float RGB [0,1] crop -> same size).  cfg: prob + any of
    brightness, contrast, saturation, hue_deg, gamma, grayscale_prob, blur_sigma_px, rotate_deg, vflip.
    Rotation uses reflected borders so no black corners appear.  Meta keys are prefixed with 'bg_'."""
    if not cfg or rng.random() >= float(cfg.get("prob", 0.0)):
        return img, {"bg_aug": []}
    out, m = _color_jitter(img, rng, cfg)
    H, W = out.shape[:2]
    if "rotate_deg" in cfg:
        deg = _u(rng, cfg["rotate_deg"])
        if abs(deg) > 0.05:
            M = cv2.getRotationMatrix2D((W / 2, H / 2), deg, 1.0)
            out = cv2.warpAffine(np.ascontiguousarray(out), M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            m["rotate_deg"] = round(deg, 1)
    if rng.random() < cfg.get("vflip", 0.0):
        out = np.ascontiguousarray(out[::-1]); m["vflip"] = True
    if "blur_sigma_px" in cfg:
        s = _u(rng, cfg["blur_sigma_px"])
        if s > 0.05:
            out = _gauss(out, s); m["blur_sigma_px"] = round(s, 2)
    meta = {f"bg_{k}": v for k, v in m.items()}
    meta["bg_aug"] = sorted(m)
    return np.ascontiguousarray(out), meta


def augment_sprite(rgb: np.ndarray, alpha: np.ndarray, rng: np.random.Generator, cfg: dict | None
                   ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Beetle-only augmentation of an RGBA sprite (float [0,1]).  cfg: prob + any of brightness, contrast,
    saturation, hue_deg, gamma, hflip (mirror: the beetle is bilaterally symmetric), aspect (anisotropic
    scale range, e.g. [0.9, 1.1]), blur_sigma_px, specular (prob of an extra highlight: {prob, strength}).
    Never modifies the inputs. Meta keys are prefixed with 'aug_'."""
    if not cfg or rng.random() >= float(cfg.get("prob", 0.0)):
        return rgb, alpha, {"sprite_aug": []}
    out, m = _color_jitter(rgb, rng, cfg, alpha)
    a = alpha.astype(np.float32, copy=True)
    if rng.random() < cfg.get("hflip", 0.0):
        out, a = out[:, ::-1], a[:, ::-1]; m["hflip"] = True
    if "aspect" in cfg:
        sx, sy = _u(rng, cfg["aspect"]), _u(rng, cfg["aspect"])
        h, w = a.shape
        nw, nh = max(2, int(round(w * sx))), max(2, int(round(h * sy)))
        out = cv2.resize(np.ascontiguousarray(out), (nw, nh), interpolation=cv2.INTER_LINEAR)
        a = cv2.resize(np.ascontiguousarray(a), (nw, nh), interpolation=cv2.INTER_LINEAR)
        m["aspect"] = [round(sx, 3), round(sy, 3)]
    sp = cfg.get("specular") or {}
    if rng.random() < sp.get("prob", 0.0):
        # one soft elliptical highlight somewhere on the body (sun glint on the elytra)
        h, w = a.shape
        ys, xs = np.where(a > 0.5)
        if len(ys):
            i = int(rng.integers(len(ys)))
            cy, cx = ys[i], xs[i]
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            r = max(w, h) * _u(rng, (0.04, 0.12))
            g = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)))
            s = _u(rng, sp.get("strength", (0.2, 0.6)))
            out = np.clip(out + (s * g * a)[..., None], 0, 1); m["specular"] = round(s, 3)
    if "blur_sigma_px" in cfg:
        s = _u(rng, cfg["blur_sigma_px"])
        if s > 0.05:
            out = _gauss(out, s); a = _gauss(a, s); m["blur_sigma_px"] = round(s, 2)
    meta = {f"aug_{k}": v for k, v in m.items()}
    meta["sprite_aug"] = sorted(m)
    return np.ascontiguousarray(out), np.ascontiguousarray(a), meta
