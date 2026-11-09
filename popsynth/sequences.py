"""Video sequences for tracker training (`popsynth sequences`).

The single-image generator (generate.py) writes independent scenes; a short-range template matcher such as
PatchNet needs consecutive frames of the SAME beetle instead. Here one scene is simulated over time:

  * a background photo on an oversized canvas; a virtual camera pans / shakes / zooms / rolls a little over it
    (so the background moves too) or stays still for some sequences,
  * a target beetle, re-rendered from the gaussian splat EVERY frame at its current pose (azimuth, elevation,
    roll) and pixel size, walking a smooth path (constant speed + random acceleration + pauses + turns; the
    in-plane rotation follows the heading), growing / shrinking slowly, entering and leaving the frame,
  * 0..n distractor beetles with their own paths (often aimed to cross the target),
  * static leaf / background-patch occluders placed on the target's path, so it passes behind them
    (partial, sometimes complete occlusion for a few frames),
  * per-frame appearance drift: lighting interpolated between keyframes, motion blur proportional to the
    apparent speed (object and camera), one camera-quality tier per sequence (defocus, low-res, haze,
    exposure fault, sensor noise, JPEG).

Everything is seeded per sequence ((seed, 23, index)); the complete motion is simulated first (cheap) and the
frames are rendered afterwards, so the boxes come from the simulation state, never from re-detection.

Output (GOT-10k layout, one folder per scene; object 0 is the tracked target):
    <root>/<split>/seq_000001/00000001.jpg ...      frames
                            groundtruth.txt         x,y,w,h of the target per frame (pixels, top-left + size)
                            absence.label           1 = target (almost) invisible: fully outside / hidden
                            cover.label             0..8 = visible share of the target's whole body x 8
                            cut_by_image.label      1 = target cut by the frame edge
                            tracks.json             every object: boxes xyxy (visible + full), absence, visible share
                            frames.jsonl            one record per frame: camera, lighting, every object's state
                            meta.json               seed, split, background, drawn parameters, motion summary
    <root>/<split>/list.txt, <root>/sequences.jsonl, <root>/summary.json
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np
from tqdm import tqdm

from .augment import DEFAULT_MODES, _gauss, _jpeg, _lowres, _u, augment_background, augment_sprite, is_legible, overexpose, underexpose
from .backgrounds import BackgroundPool
from .cameras import fixed_camera, projected_extent, resized
from .compose import crop_to_alpha
from .occlude import OccluderBank, _gaussian, make_occluder

LUM = np.array([0.299, 0.587, 0.114], np.float32)

# Defaults of the `sequences:` config section (configs/default.yaml documents every key). Pixel values refer
# to the frame size below at the nominal fps; re-tune `motion` / `camera` once the real camera is known.
DEFAULTS = {
    "num_sequences": 20, "length": [40, 100], "val_fraction": 0.15, "frame_size": [640, 640], "fps": 30,
    "jpeg_quality": 90, "object_size": [0.06, 0.40], "absent_below": 0.05, "own_background": True,
    "min_render_px": 512, "lighting_keyframe_frames": [20, 50],
    "start": {"empty": True, "entry_gap_px": [2, 20], "distractor_gap_px": [0, 80], "entry_frames": [2, 15],
              "distractor_entry_frames": [5, 45]},
    "distractors": {"count": [0, 2], "cross_prob": 0.5},
    "motion": {"speed_px": [0.3, 5.0], "speed_tail_px": [5.0, 15.0], "tail_prob": 0.2, "speed_sigma": 0.15,
               "speed_relax": 0.05, "heading_sigma_deg": 6.0, "pause_prob": 0.01, "pause_frames": [5, 30],
               "turn_back_margin": 0.15, "pose_rate_deg": {"azimuth": 3.0, "elevation": 1.5, "roll": 1.0},
               "elevation_deg": [0, 90], "roll_follows_heading": True, "scale_step": 0.02, "scale_sigma": 0.008},
    "gif": {"every": 1, "max_side": 240, "frame_step": 2},
    "camera": {"static_prob": 0.2, "pan_px": [0.0, 3.0], "pan_relax": 0.05, "shake_px": [0.0, 1.5],
               "zoom_rate": 0.01, "zoom_range": [0.85, 1.2], "rot_rate_deg": 0.2, "rot_range_deg": [-5, 5],
               "canvas_margin": 0.2, "exposure": [0.2, 0.6]},
    "occlusion": {"prob": 0.4, "count": [1, 2], "size": [0.6, 1.8], "on_path_prob": 0.8},
}


def _merge(base: dict, over: dict | None) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in (over or {}).items():
        out[k] = {**out[k], **v} if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


# ----------------------------------------------------------------------------- beetle views
class BeetleViews:
    """Renders the beetle from any pose at a wanted pixel size: one GPU pass per call (no sprite reuse)."""

    def __init__(self, cfg: dict, min_render_px: int | None = None):
        from .generate import prepare_splat
        from .render import get_renderer
        from .render.base import pick_device
        rc = cfg["render"]
        self.splat = prepare_splat(cfg)
        self.device = pick_device(rc.get("device", "auto"))
        self.name, self.render = get_renderer(rc.get("backend", "auto"))
        if self.name == "gsplat" and self.device != "cuda":
            raise RuntimeError("gsplat backend needs a CUDA device")
        self.g = self.splat.to_torch(self.device)
        self.up = cfg["splat"].get("up", [0, 0, 1])
        self.radius = self.splat.max_extent()
        vis = self.splat.opacities >= 0.05
        self.means_v, self.scales_v = self.splat.means[vis], self.splat.scales[vis]
        self.ss = max(1, int(rc.get("supersample", 1)))
        self.min_render = int(min_render_px if min_render_px is not None else rc.get("min_render_px", 1024))
        self.aa = bool(rc.get("antialiased", False))
        self.count = 0
        self.seconds = 0.0

    def describe(self) -> str:
        return (f"{len(self.splat)} gaussians, renderer={self.name} device={self.device}, "
                f"supersample {self.ss}, >= {self.min_render}px per view")

    def view(self, elevation: float, azimuth: float, roll: float, fov: float, long_px: int,
             fill: float = 0.85) -> tuple[np.ndarray, np.ndarray]:
        """(rgb, alpha) float32, cropped to the beetle, longest side == long_px."""
        t0 = time.time()
        cam0 = fixed_camera(1024, 1024, up=self.up, object_radius=self.radius, elevation_deg=elevation,
                            azimuth_deg=azimuth, roll_deg=roll, fov_deg=fov, fill=fill)
        x0, y0, x1, y1 = projected_extent(self.means_v, self.scales_v, cam0.viewmat, cam0.K)
        long0 = max(x1 - x0, y1 - y0, 1.0)
        out_size = max(8, int(np.ceil(1024 * long_px / long0)))
        ss = max(self.ss, int(np.ceil(self.min_render / out_size)))
        rsize = out_size * ss
        cam = resized(cam0, rsize, rsize)
        rgb, a = self.render(self.g, cam.viewmat, cam.K, rsize, rsize, antialiased=self.aa)
        rgb, a = rgb.numpy(), a.numpy()
        if ss > 1:
            prem = cv2.resize(rgb * a[..., None], (out_size, out_size), interpolation=cv2.INTER_AREA)
            a = cv2.resize(a, (out_size, out_size), interpolation=cv2.INTER_AREA)
            rgb = np.where(a[..., None] > 1e-4, prem / np.maximum(a[..., None], 1e-4), 0).astype(np.float32)
        rgb, a = _exact_size(rgb, a, long_px)
        self.count += 1
        self.seconds += time.time() - t0
        return rgb, a


def _exact_size(rgb: np.ndarray, a: np.ndarray, long_px: int) -> tuple[np.ndarray, np.ndarray]:
    rgb, a = crop_to_alpha(rgb, a, pad=1)
    s = long_px / max(a.shape)
    if abs(s - 1) > 1e-3:
        nw, nh = max(2, int(round(a.shape[1] * s))), max(2, int(round(a.shape[0] * s)))
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
        prem = cv2.resize(rgb * a[..., None], (nw, nh), interpolation=interp)
        a = cv2.resize(a, (nw, nh), interpolation=interp)
        rgb = np.where(a[..., None] > 1e-4, prem / np.maximum(a[..., None], 1e-4), 0).astype(np.float32)
    return np.ascontiguousarray(rgb), np.ascontiguousarray(a)


class DiscViews:
    """Stand-in renderer for tests and dry runs: a shaded disc with a dark 'head' marking the roll angle."""

    def __init__(self, *_, **__):
        self.count = 0
        self.seconds = 0.0

    def describe(self) -> str:
        return "disc stand-in renderer"

    def view(self, elevation, azimuth, roll, fov, long_px, fill=0.85):
        n = int(long_px)
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
        c = (n - 1) / 2
        r = np.hypot(xx - c, yy - c) / (n / 2)
        a = np.clip((1.0 - r) * n / 2, 0, 1).astype(np.float32)
        rgb = np.stack([0.2 + 0.3 * (1 - r), (0.5 + 0.2 * np.cos(np.deg2rad(elevation))) * np.ones_like(r), 0.2 * np.ones_like(r)], -1)
        th = np.deg2rad(roll)
        hx, hy = c + 0.6 * c * np.cos(th), c + 0.6 * c * np.sin(th)
        head = np.hypot(xx - hx, yy - hy) < n * 0.15
        rgb[head] = (0.05, 0.05, 0.05)
        self.count += 1
        return np.clip(rgb, 0, 1).astype(np.float32), a


# ----------------------------------------------------------------------------- state
@dataclass
class Beetle:
    x: float; y: float                   # centre, canvas px
    heading: float                       # rad, direction of travel
    speed: float; speed_target: float
    el: float; az: float; roll: float
    d_el: float = 0.0; d_az: float = 0.0; d_roll: float = 0.0
    size: float = 100.0; size0: float = 100.0
    fov: float = 40.0
    move: float = 1.0; pause_left: int = 0
    aug_seed: int = 0
    defocus_sigma: float = 0.0
    vx: float = 0.0; vy: float = 0.0
    entered: bool = True                 # False until the beetle has been inside the view once (entry phase)
    entry_speed: float = 0.0             # speed that brings it into the view within its drawn entry time


@dataclass
class Camera:
    cx: float; cy: float
    vx: float = 0.0; vy: float = 0.0
    logz: float = 0.0; dz: float = 0.0
    rot: float = 0.0; drot: float = 0.0
    sx: float = 0.0; sy: float = 0.0      # shake offset (low-pass jitter)
    pan_sigma: float = 0.0; shake_sigma: float = 0.0
    static: bool = False
    psx: float = 0.0; psy: float = 0.0    # previous shake, for the per-frame apparent camera shift
    pcx: float = 0.0; pcy: float = 0.0

    def affine(self, W: int, H: int) -> np.ndarray:
        """canvas -> frame: frame = z R (p - c) + (W/2, H/2)."""
        z, th = math.exp(self.logz), math.radians(self.rot)
        c, s = math.cos(th), math.sin(th)
        x, y = self.cx + self.sx, self.cy + self.sy
        return np.float32([[z * c, -z * s, W / 2 - z * (c * x - s * y)], [z * s, z * c, H / 2 - z * (s * x + c * y)]])

    def apparent_shift(self) -> tuple[float, float]:
        """Shift of the scene in frame px since the previous frame (what the camera blur is made of)."""
        z = math.exp(self.logz)
        return -z * (self.cx + self.sx - self.pcx - self.psx), -z * (self.cy + self.sy - self.pcy - self.psy)


def _step_beetle(b: Beetle, rng: np.random.Generator, mo: dict, view_c: tuple[float, float], view_half: tuple[float, float]) -> None:
    b.speed = max(0.0, b.speed + mo["speed_relax"] * (b.speed_target - b.speed) + float(rng.normal(0, mo["speed_sigma"])))
    if not b.entered:
        if abs(b.x - view_c[0]) < view_half[0] and abs(b.y - view_c[1]) < view_half[1]:
            b.entered = True
        else:                                        # walking in: no pauses, at least its entry speed
            b.speed = max(b.speed, b.entry_speed)
            b.pause_left = 0
    if b.pause_left > 0:
        b.pause_left -= 1
        b.move = max(0.0, b.move - 0.25)
    else:
        if rng.random() < mo["pause_prob"]:
            lo, hi = mo["pause_frames"]
            b.pause_left = int(rng.integers(int(lo), int(hi) + 1))
        b.move = min(1.0, b.move + 0.25)
    dh = float(rng.normal(0, math.radians(mo["heading_sigma_deg"])))
    # outside the view (plus a margin) the beetle turns back toward the view centre
    m = mo["turn_back_margin"]
    if abs(b.x - view_c[0]) > view_half[0] * (1 + m) or abs(b.y - view_c[1]) > view_half[1] * (1 + m):
        want = math.atan2(view_c[1] - b.y, view_c[0] - b.x)
        dh += float(np.clip(_wrap(want - b.heading), -0.3, 0.3))
    b.heading = _wrap(b.heading + dh)
    if mo["roll_follows_heading"]:
        b.roll += math.degrees(dh)
    v = b.speed * b.move
    b.vx, b.vy = v * math.cos(b.heading), v * math.sin(b.heading)
    b.x += b.vx
    b.y += b.vy
    pr = mo["pose_rate_deg"]
    b.d_az += -0.1 * b.d_az + float(rng.normal(0, 0.3 * pr["azimuth"]))
    b.d_el += -0.1 * b.d_el + float(rng.normal(0, 0.3 * pr["elevation"]))
    b.d_roll += -0.1 * b.d_roll + float(rng.normal(0, 0.3 * pr["roll"]))
    b.d_az = float(np.clip(b.d_az, -3 * pr["azimuth"], 3 * pr["azimuth"]))
    b.d_el = float(np.clip(b.d_el, -3 * pr["elevation"], 3 * pr["elevation"]))
    b.d_roll = float(np.clip(b.d_roll, -3 * pr["roll"], 3 * pr["roll"]))
    b.az = (b.az + b.d_az) % 360.0
    b.el += b.d_el
    lo, hi = mo["elevation_deg"]
    if b.el > hi:
        b.el, b.d_el = 2 * hi - b.el, -b.d_el
    if b.el < lo:
        b.el, b.d_el = 2 * lo - b.el, -b.d_el
    b.roll = (b.roll + b.d_roll + 180.0) % 360.0 - 180.0
    step = float(np.clip(rng.normal(0, mo["scale_sigma"]) + 0.02 * (math.log(b.size0) - math.log(b.size)),
                         -mo["scale_step"], mo["scale_step"]))
    b.size = float(np.clip(b.size * math.exp(step), 0.5 * b.size0, 2.0 * b.size0))


def _step_camera(cam: Camera, rng: np.random.Generator, cc: dict, W: int, H: int, Cw: int, Ch: int) -> None:
    cam.pcx, cam.pcy, cam.psx, cam.psy = cam.cx, cam.cy, cam.sx, cam.sy
    if cam.static:
        return
    cam.vx += -cc["pan_relax"] * cam.vx + float(rng.normal(0, cc["pan_relax"] * cam.pan_sigma * 2))
    cam.vy += -cc["pan_relax"] * cam.vy + float(rng.normal(0, cc["pan_relax"] * cam.pan_sigma * 2))
    cam.cx += cam.vx
    cam.cy += cam.vy
    cam.sx = 0.7 * cam.sx + float(rng.normal(0, cam.shake_sigma))
    cam.sy = 0.7 * cam.sy + float(rng.normal(0, cam.shake_sigma))
    cam.dz += -0.1 * cam.dz + float(rng.normal(0, 0.3 * cc["zoom_rate"]))
    cam.logz += cam.dz
    zlo, zhi = math.log(cc["zoom_range"][0]), math.log(cc["zoom_range"][1])
    if cam.logz > zhi:
        cam.logz, cam.dz = 2 * zhi - cam.logz, -cam.dz
    if cam.logz < zlo:
        cam.logz, cam.dz = 2 * zlo - cam.logz, -cam.dz
    cam.drot += -0.1 * cam.drot + float(rng.normal(0, 0.3 * cc["rot_rate_deg"]))
    cam.rot += cam.drot
    rlo, rhi = cc["rot_range_deg"]
    if cam.rot > rhi:
        cam.rot, cam.drot = 2 * rhi - cam.rot, -cam.drot
    if cam.rot < rlo:
        cam.rot, cam.drot = 2 * rlo - cam.rot, -cam.drot
    # keep the whole view inside the canvas: bounce the pan at the limits
    hx, hy = _view_half(cam, W, H)
    for attr, v_attr, half, size in (("cx", "vx", hx, Cw), ("cy", "vy", hy, Ch)):
        p = getattr(cam, attr)
        if p - half < 0:
            setattr(cam, attr, 2 * half - p); setattr(cam, v_attr, abs(getattr(cam, v_attr)))
        elif p + half > size:
            setattr(cam, attr, 2 * (size - half) - p); setattr(cam, v_attr, -abs(getattr(cam, v_attr)))


def _view_half(cam: Camera, W: int, H: int) -> tuple[float, float]:
    """Half extent (canvas px) of the rotated view plus the shake, so nothing outside the canvas is ever shown."""
    z, th = math.exp(cam.logz), math.radians(cam.rot)
    c, s = abs(math.cos(th)), abs(math.sin(th))
    sh = 3 * cam.shake_sigma + 2
    return (W / 2 * c + H / 2 * s) / z + sh, (W / 2 * s + H / 2 * c) / z + sh


def canvas_size(W: int, H: int, cc: dict) -> tuple[int, int]:
    """Canvas big enough for the most zoomed-out, rotated view plus the pan room (`canvas_margin` x frame)."""
    zmin = float(cc["zoom_range"][0])
    rmax = math.radians(max(abs(float(r)) for r in cc["rot_range_deg"]))
    c, s = math.cos(rmax), math.sin(rmax)
    vw, vh = (W * c + H * s) / zmin, (W * s + H * c) / zmin
    m = float(cc["canvas_margin"])
    return int(math.ceil(vw + 2 * m * W)) + 8, int(math.ceil(vh + 2 * m * H)) + 8


# ----------------------------------------------------------------------------- photometrics over time
def _light_params(rng: np.random.Generator, m: dict) -> dict:
    return {"exposure": _u(rng, m.get("exposure", 1.0)), "tint": [_u(rng, t) for t in m.get("tint", [1, 1, 1])],
            "contrast": _u(rng, m.get("contrast", 1.0)), "saturation": _u(rng, m.get("saturation", 1.0)),
            "vignette": _u(rng, m.get("vignette", 0.0))}


def _lerp(a: dict, b: dict, t: float) -> dict:
    out = {}
    for k in a:
        out[k] = [x + (y - x) * t for x, y in zip(a[k], b[k])] if isinstance(a[k], list) else a[k] + (b[k] - a[k]) * t
    return out


def _apply_light(img: np.ndarray, p: dict, clamp: float, noise: float, rng: np.random.Generator) -> np.ndarray:
    H, W = img.shape[:2]
    out = img * (p["exposure"] * np.asarray(p["tint"], np.float32))[None, None]
    out = (out - 0.5 * p["exposure"]) * p["contrast"] + 0.5 * p["exposure"]
    gray = out @ LUM
    out = gray[..., None] + (out - gray[..., None]) * p["saturation"]
    if p["vignette"] > 0:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r2 = ((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2
        out = out * (1 - p["vignette"] * np.clip(r2 / 2, 0, 1))[..., None]
    out = out * clamp
    if noise > 0:
        out = out + rng.normal(0, noise, out.shape).astype(np.float32)
    return np.clip(out, 0, 1).astype(np.float32)


def _light_clamp(img: np.ndarray, p: dict, mean_range) -> float:
    """Exposure rescale that puts the frame's mean luminance inside the lighting mode's range (frame 0 decides,
    then it is kept, so the sequence does not flicker)."""
    if not mean_range:
        return 1.0
    lum = float((_apply_light(img, p, 1.0, 0.0, np.random.default_rng(0)) @ LUM).mean())
    if lum < mean_range[0]:
        return mean_range[0] / max(lum, 1e-4)
    if lum > mean_range[1]:
        return mean_range[1] / max(lum, 1e-4)
    return 1.0


def _draw_tier(rng: np.random.Generator, tiers: dict | None) -> dict:
    """One camera-quality tier per sequence; its random blur is NOT used (motion blur is physical here)."""
    if not tiers:
        return {"quality_tier": "none"}
    names = list(tiers)
    w = np.array([tiers[n].get("weight", 1.0) for n in names], float)
    tier = names[int(rng.choice(len(names), p=w / w.sum()))]
    t = tiers[tier]
    p = {"quality_tier": tier}
    if rng.random() < t.get("defocus_prob", 0):
        p["defocus_sigma"] = round(_u(rng, t.get("defocus_sigma", (0.5, 1.5))), 2)
    if rng.random() < t.get("lowres_prob", 0):
        p["lowres_factor"] = round(_u(rng, t.get("lowres_factor", (1.5, 3.0))), 2)
    if rng.random() < t.get("haze_prob", 0):
        p["haze"] = round(_u(rng, t.get("haze", (0.1, 0.4))), 2)
    elif rng.random() < t.get("exposure_fault_prob", 0):
        if rng.random() < 0.5:
            p["overexposure"] = round(_u(rng, t.get("overexposure", (1.6, 2.6))), 2)
        else:
            p["underexposure"] = round(_u(rng, t.get("underexposure", (0.25, 0.5))), 2)
    p["sensor_noise"] = round(_u(rng, t.get("noise", (0.0, 0.0))), 3)
    jq = t.get("jpeg_quality")
    if jq:
        p["jpeg_quality"] = int(rng.integers(int(jq[0]), int(jq[1]) + 1))
    p["max_mean"], p["min_mean"] = float(t.get("max_mean", 0.78)), float(t.get("min_mean", 0.13))
    return p


def _apply_tier(img: np.ndarray, p: dict, rng: np.random.Generator) -> np.ndarray:
    out = img
    if p.get("defocus_sigma"):
        out = _gauss(out, p["defocus_sigma"])
    if p.get("lowres_factor"):
        out = _lowres(out, p["lowres_factor"])
    if p.get("haze"):
        out = out * (1 - p["haze"]) + p["haze"] * np.array([0.78, 0.78, 0.8], np.float32)
    if p.get("overexposure"):
        out, _ = overexpose(out, p["overexposure"], max_mean=p["max_mean"])
    if p.get("underexposure"):
        out, _ = underexpose(out, p["underexposure"], min_mean=p["min_mean"])
    if p.get("sensor_noise", 0) > 0:
        out = out + rng.normal(0, p["sensor_noise"], out.shape).astype(np.float32)
    out = np.clip(out, 0, 1).astype(np.float32)
    if p.get("jpeg_quality"):
        out = _jpeg(out, p["jpeg_quality"])
    return np.ascontiguousarray(out)


def line_kernel(dx: float, dy: float) -> np.ndarray | None:
    """Motion-blur kernel for a shift of (dx, dy) px during the exposure; None when shorter than a pixel."""
    L = math.hypot(dx, dy)
    if L < 1.0:
        return None
    k = int(math.ceil(L)) | 1
    k += 2
    kern = np.zeros((k, k), np.float32)
    c = (k - 1) / 2
    ux, uy = dx / L, dy / L
    cv2.line(kern, (int(round(c - ux * L / 2)), int(round(c - uy * L / 2))),
             (int(round(c + ux * L / 2)), int(round(c + uy * L / 2))), 1.0, 1, cv2.LINE_AA)
    s = kern.sum()
    return kern / s if s > 0 else None


def _blur_rgba(rgb: np.ndarray, a: np.ndarray, kern: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pad = kern.shape[0] // 2
    rgb = cv2.copyMakeBorder(rgb, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    a = cv2.copyMakeBorder(a, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    prem = cv2.filter2D(rgb * a[..., None], -1, kern)
    a2 = cv2.filter2D(a, -1, kern)
    rgb2 = np.where(a2[..., None] > 1e-4, prem / np.maximum(a2[..., None], 1e-4), 0).astype(np.float32)
    return rgb2, np.clip(a2, 0, 1).astype(np.float32)


def _place(shape: tuple[int, int], rgb: np.ndarray, a: np.ndarray, cx: float, cy: float) -> tuple[np.ndarray, np.ndarray]:
    """Sub-pixel placement of a sprite (centre at cx, cy) as canvas-sized layers."""
    h, w = a.shape
    M = np.float32([[1, 0, cx - w / 2], [0, 1, cy - h / 2]])
    H, W = shape
    A = cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    R = cv2.warpAffine(rgb, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return R, A


def _box(mask: np.ndarray, thresh: float = 0.5) -> list[int] | None:
    ys, xs = np.where(mask > thresh)
    if len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _xywh(b: list[int] | None) -> list[int]:
    return [0, 0, 0, 0] if b is None else [b[0], b[1], b[2] - b[0], b[3] - b[1]]


# ----------------------------------------------------------------------------- one sequence
def simulate(cfg: dict, sq: dict, rng: np.random.Generator, n_frames: int, W: int, H: int, Cw: int, Ch: int) -> dict:
    """Draw the scene and step the motion for every frame. Returns the per-frame states (no pixels)."""
    mo, cc = sq["motion"], sq["camera"]
    short = min(W, H)
    cam = Camera(cx=Cw / 2 + float(rng.uniform(-0.3, 0.3)) * (Cw - W) / 2,
                 cy=Ch / 2 + float(rng.uniform(-0.3, 0.3)) * (Ch - H) / 2,
                 logz=float(rng.uniform(math.log(cc["zoom_range"][0]), math.log(cc["zoom_range"][1]))) * 0.5,
                 rot=float(rng.uniform(*cc["rot_range_deg"])) * 0.5,
                 pan_sigma=_u(rng, cc["pan_px"]), shake_sigma=_u(rng, cc["shake_px"]),
                 static=bool(rng.random() < cc["static_prob"]))
    if cam.static:
        cam.pan_sigma = cam.shake_sigma = 0.0
    cam.pcx, cam.pcy = cam.cx, cam.cy
    fov = _u(rng, cfg["render"]["camera"].get("fov_deg", (25, 70)))
    el_lo, el_hi = mo["elevation_deg"]
    lo, hi = sq["distractors"]["count"]
    n_obj = 1 + int(rng.integers(int(lo), int(hi) + 1))
    beetles = []
    z0 = math.exp(cam.logz)
    hx, hy = W / (2 * z0), H / (2 * z0)
    st = sq["start"]
    for k in range(n_obj):
        size = _u(rng, sq["object_size"]) * short / z0
        speed = _u(rng, mo["speed_tail_px"] if rng.random() < mo["tail_prob"] else mo["speed_px"])
        heading = float(rng.uniform(-math.pi, math.pi))
        entered = True
        if st["empty"]:
            # every beetle starts completely outside the view (first frame empty) and walks in from any side:
            # the whole body (a disc of radius size/sqrt(2) covers any pose) plus a gap lies beyond the edge
            gap = _u(rng, st["entry_gap_px"] if k == 0 else st["distractor_gap_px"])
            r = size / math.sqrt(2) + gap + 2
            side = int(rng.integers(4))
            x = cam.cx + (-hx - r if side == 0 else hx + r if side == 1 else float(rng.uniform(-hx, hx)))
            y = cam.cy + (-hy - r if side == 2 else hy + r if side == 3 else float(rng.uniform(-hy, hy)))
            # aim at a random point well inside the view
            tx, ty = cam.cx + float(rng.uniform(-0.6, 0.6)) * hx, cam.cy + float(rng.uniform(-0.6, 0.6)) * hy
            heading = math.atan2(ty - y, tx - x)
            entered = False
            # the centre must cover r + a bit more to be inside the view: within entry_frames frames
            n_entry = float(rng.uniform(*(st["entry_frames"] if k == 0 else st["distractor_entry_frames"])))
            entry_speed = (r + 0.2 * size) / max(n_entry, 1.0)
        else:
            x = cam.cx + float(rng.uniform(-hx, hx)) * (0.9 if k == 0 else 1.3)
            y = cam.cy + float(rng.uniform(-hy, hy)) * (0.9 if k == 0 else 1.3)
            entry_speed = 0.0
        b = Beetle(x=x, y=y, heading=heading, speed=max(speed, entry_speed) if not entered else speed, speed_target=speed,
                   el=float(rng.uniform(el_lo, el_hi)), az=float(rng.uniform(0, 360)), roll=float(rng.uniform(-180, 180)),
                   size=size, size0=size, fov=fov, aug_seed=int(rng.integers(2 ** 31)), entered=entered, entry_speed=entry_speed)
        df = cfg["compose"].get("defocus") or {}
        if rng.random() < df.get("prob", 0):
            b.defocus_sigma = _u(rng, df.get("sigma_px", (0.5, 2.0)))
        beetles.append(b)
    # distractors aimed to cross the target's current position in a few dozen frames
    for b in beetles[1:]:
        if b.entered and rng.random() < sq["distractors"]["cross_prob"] and b.speed > 1e-3:
            t_cross = beetles[0]
            b.heading = math.atan2(t_cross.y - b.y, t_cross.x - b.x) + float(rng.normal(0, 0.2))
    order = [int(k) for k in rng.permutation(n_obj)]   # paste order: later = in front (depth, fixed per sequence)
    frames = []
    for f in range(n_frames):
        if f > 0:
            _step_camera(cam, rng, cc, W, H, Cw, Ch)
            z = math.exp(cam.logz)
            for b in beetles:
                _step_beetle(b, rng, mo, (cam.cx, cam.cy), (W / (2 * z), H / (2 * z)))
        frames.append({"camera": asdict(cam), "beetles": [asdict(b) for b in beetles]})
    return {"frames": frames, "order": order, "fov": fov, "camera_static": cam.static,
            "pan_sigma": round(cam.pan_sigma, 3), "shake_sigma": round(cam.shake_sigma, 3), "n_objects": n_obj}


class SequenceWriter:
    def __init__(self, root: str, jpeg_quality: int = 90):
        self.root = root
        self.q = int(jpeg_quality)
        os.makedirs(root, exist_ok=True)

    def frame(self, i: int, img: np.ndarray) -> str:
        fn = f"{i + 1:08d}.jpg"
        u8 = cv2.cvtColor((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.root, fn), u8, [cv2.IMWRITE_JPEG_QUALITY, self.q])
        return fn

    def finish(self, tracks: dict, frames: list[dict], meta: dict) -> None:
        t = tracks["0"]
        with open(os.path.join(self.root, "groundtruth.txt"), "w") as f:
            f.writelines(",".join(str(v) for v in _xywh(b)) + "\n" for b in t["boxes"])
        for name, key in (("absence.label", "absence"), ("cover.label", "cover"), ("cut_by_image.label", "cut_by_image")):
            with open(os.path.join(self.root, name), "w") as f:
                f.writelines(f"{v}\n" for v in t[key])
        with open(os.path.join(self.root, "tracks.json"), "w") as f:
            json.dump(tracks, f, default=_json_default)
        with open(os.path.join(self.root, "frames.jsonl"), "w") as f:
            f.writelines(json.dumps(r, default=_json_default) + "\n" for r in frames)
        with open(os.path.join(self.root, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1, default=_json_default)


def render_sequence(cfg: dict, sq: dict, index: int, split: str, pool: BackgroundPool, leaf_bank: OccluderBank | None,
                    views, out_root: str, verbose: bool = False, bg_path: str | None = None) -> dict:
    """Simulate + render one sequence into <out_root>/<split>/seq_<index>. Returns its summary record."""
    rng = np.random.default_rng([int(cfg.get("seed", 0)), 23, index])
    W, H = int(sq["frame_size"][0]), int(sq["frame_size"][1])
    Cw, Ch = canvas_size(W, H, sq["camera"])
    n_frames = int(rng.integers(int(sq["length"][0]), int(sq["length"][1]) + 1))
    name = f"seq_{index:06d}"
    seq_dir = os.path.join(out_root, split, name)
    sim = simulate(cfg, sq, rng, n_frames, W, H, Cw, Ch)
    cc_cfg = cfg["compose"]
    bcfg = cfg["backgrounds"]

    # --- static scene: background canvas (+ constant background augmentation), occluders on the target's path
    bg, bg_meta = pool.sample(rng, Cw, Ch, tuple(cc_cfg.get("crop_scale", bcfg.get("crop_scale", (0.4, 1.0)))),
                              bcfg.get("hflip", 0.5), path=bg_path)
    bg_photo = bg
    bg, bam = augment_background(bg, rng, cfg.get("background_aug"))
    occ_rgb = np.zeros_like(bg)
    occ_a = np.zeros((Ch, Cw), np.float32)
    occ_meta = []
    oc = {**(cc_cfg.get("occlusion") or {}), **sq["occlusion"]}
    if rng.random() < float(oc.get("prob", 0)):
        lo, hi = oc.get("count", [1, 1])
        for _ in range(int(rng.integers(int(lo), int(hi) + 1))):
            f_at = int(rng.integers(n_frames))
            tb = sim["frames"][f_at]["beetles"][0]
            long_px = max(6, int(round(tb["size"] * _u(rng, oc.get("size", (0.6, 1.8))))))
            try:
                o_rgb, o_a, om = make_occluder(rng, oc, long_px, leaf_bank, bg_photo)
            except RuntimeError as e:
                print(f"warning: {e}")
                break
            if rng.random() < float(oc.get("on_path_prob", 0.8)):
                cx = tb["x"] + float(rng.normal(0, 0.3 * tb["size"]))
                cy = tb["y"] + float(rng.normal(0, 0.3 * tb["size"]))
            else:
                cam0 = sim["frames"][0]["camera"]
                cx, cy = cam0["cx"] + float(rng.uniform(-W / 2, W / 2)), cam0["cy"] + float(rng.uniform(-H / 2, H / 2))
            blur = _u(rng, oc.get("blur_sigma_px", (0.0, 0.0)))
            if blur > 0:
                o_rgb, o_a = _gaussian(o_rgb, blur), _gaussian(o_a, blur)
            o_rgb = np.clip(o_rgb * _u(rng, oc.get("brightness", (1.0, 1.0))), 0, 1)
            R, A = _place((Ch, Cw), o_rgb, o_a, cx, cy)
            wsum = max(float(A.sum()), 1e-6)
            o_mean = (R * A[..., None]).sum((0, 1)) / wsum + 1e-3
            contrast = _u(rng, oc.get("contrast", (1.0, 1.0)))
            R = (R - o_mean) * contrast + o_mean
            match = _u(rng, oc.get("match_color", (0.0, 0.0)))
            if match > 0:
                bg_mean = (bg * A[..., None]).sum((0, 1)) / wsum + 1e-3
                R = R * ((1 - match) + match * bg_mean / o_mean)
            R = np.clip(R, 0, 1)
            sh = oc.get("shadow")
            if sh and np.max(sh.get("strength", 0)) > 0:          # contact shadow baked into the background
                size = max(o_a.shape)
                M = np.float32([[1, 0, _u(rng, sh.get("offset", (0, 0.05))) * size], [0, 1, 0]])
                shd = _gaussian(cv2.warpAffine(A, M, (Cw, Ch)), _u(rng, sh.get("blur", (0.03, 0.08))) * size)
                bg = bg * (1 - _u(rng, sh.get("strength", (0.15, 0.35))) * shd)[..., None]
            occ_rgb = R * A[..., None] + occ_rgb * (1 - A[..., None])
            occ_a = np.clip(occ_a + A * (1 - occ_a), 0, 1)
            om.update({"centre_canvas": [round(cx, 1), round(cy, 1)], "on_path_frame": f_at, "long_px": long_px,
                       "occluder_blur_sigma": round(blur, 2)})
            occ_meta.append(om)
    has_occ = bool(occ_meta)

    # --- per-sequence constants of the compositing
    cm = _u(rng, cc_cfg.get("color_match", (0, 0)))
    edge_blur = _u(rng, cc_cfg.get("edge_blur_px", (0, 0)))
    shadow = cc_cfg.get("shadow") or {}
    shadow_p = None
    if rng.random() < shadow.get("prob", 0):
        shadow_p = {"strength": _u(rng, shadow.get("strength", (0.2, 0.6))), "blur": _u(rng, shadow.get("blur", (0.03, 0.12))),
                    "offset": _u(rng, shadow.get("offset", (0.0, 0.08))), "angle": float(rng.uniform(0, 2 * math.pi))}
    sprite_aug = dict(cfg.get("sprite_aug") or {})
    sprite_aug.pop("specular", None)                 # a highlight at a random pixel would flicker
    light_modes = cfg.get("lighting") or DEFAULT_MODES
    names = list(light_modes)
    w = np.array([light_modes[n].get("weight", 1.0) for n in names], float)
    mode = names[int(rng.choice(len(names), p=w / w.sum()))]
    lm = light_modes[mode]
    kf_lo, kf_hi = sq["lighting_keyframe_frames"]
    keys, kf = [], 0
    while kf < n_frames:
        keys.append((kf, _light_params(rng, lm)))
        kf += int(rng.integers(int(kf_lo), int(kf_hi) + 1))
    keys.append((n_frames, _light_params(rng, lm)))
    noise = _u(rng, lm.get("noise", 0.0))
    tier = _draw_tier(rng, (cfg.get("camera_effects") or {}).get("tiers"))
    exposure_frac = _u(rng, sq["camera"]["exposure"])
    cm_gain = {}
    clamp = None
    guard = {"mean": (0.12, 0.80), "min_std": 0.05}
    guard.update((cfg.get("camera_effects") or {}).get("guard") or {})

    writer = SequenceWriter(seq_dir, sq["jpeg_quality"])
    order = sim["order"]
    n_obj = sim["n_objects"]
    tracks = {str(k): {"boxes": [], "boxes_full": [], "absence": [], "cover": [], "cut_by_image": [], "visible_share": []}
              for k in range(n_obj)}
    frame_recs = []
    absent_below = float(sq["absent_below"])
    last_box = [None] * n_obj
    t0 = time.time()
    for f in range(n_frames):
        st = sim["frames"][f]
        cam = Camera(**st["camera"])
        z = math.exp(cam.logz)
        canvas = bg.copy()
        layers = [None] * n_obj
        body_px = [0] * n_obj
        obj_meta = [None] * n_obj
        for k in order:
            b = Beetle(**st["beetles"][k])
            long_px = max(4, int(round(b.size)))
            rgb, a = views.view(b.el, b.az, b.roll, b.fov, long_px)
            rgb, a, _ = augment_sprite(rgb, a, np.random.default_rng(b.aug_seed), sprite_aug)
            if b.defocus_sigma > 0.05:
                rgb, a = _gaussian(rgb, b.defocus_sigma), _gaussian(a, b.defocus_sigma)
            if edge_blur > 0.05:
                a = _gaussian(a, edge_blur)
            body_px[k] = int((a > 0.5).sum())
            kern = line_kernel(b.vx * exposure_frac, b.vy * exposure_frac)
            if kern is not None:
                rgb, a = _blur_rgba(rgb, a, kern)
            R, A = _place((Ch, Cw), rgb, a, b.x, b.y)
            if k not in cm_gain:                     # colour match: tint/brightness gain fixed at first sight
                g = 1.0
                if cm > 0 and A.sum() > 1:
                    wsum = float(A.sum())
                    bg_mean = (bg * A[..., None]).sum((0, 1)) / wsum + 1e-3
                    sp_mean = (R * A[..., None]).sum((0, 1)) / wsum + 1e-3
                    gain = bg_mean / sp_mean
                    gain = gain / gain.mean()
                    lum_gain = (bg_mean.mean() / sp_mean.mean()) ** 0.5
                    g = (1 - cm) + cm * gain * lum_gain
                cm_gain[k] = g
            R = np.clip(R * cm_gain[k], 0, 1)
            if shadow_p:
                sz = max(a.shape)
                off = shadow_p["offset"] * sz
                M = np.float32([[1, 0, off * math.cos(shadow_p["angle"])], [0, 1, off * math.sin(shadow_p["angle"])]])
                shd = _gaussian(cv2.warpAffine(A, M, (Cw, Ch)), shadow_p["blur"] * sz)
                canvas *= (1 - shadow_p["strength"] * shd)[..., None]
            canvas = R * A[..., None] + canvas * (1 - A[..., None])
            layers[k] = A
            obj_meta[k] = {"object_id": k, "role": "target" if k == 0 else "distractor",
                           "centre_canvas": [round(b.x, 2), round(b.y, 2)], "speed_px": round(math.hypot(b.vx, b.vy), 3),
                           "heading_deg": round(math.degrees(b.heading), 2), "elevation_deg": round(b.el, 2),
                           "azimuth_deg": round(b.az, 2), "roll_deg": round(b.roll, 2), "fov_deg": round(b.fov, 2),
                           "size_canvas_px": long_px, "paused": b.move < 0.5}
        if has_occ:
            canvas = occ_rgb * occ_a[..., None] + canvas * (1 - occ_a[..., None])
        M = cam.affine(W, H)
        frame = cv2.warpAffine(canvas, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        dxc, dyc = cam.apparent_shift() if f > 0 else (0.0, 0.0)
        kern = line_kernel(dxc * exposure_frac, dyc * exposure_frac)
        if kern is not None:
            frame = cv2.filter2D(frame, -1, kern)
        # lighting: interpolate between the keyframes around f
        j = max(i for i, (kf_, _) in enumerate(keys) if kf_ <= f)
        (ka, pa), (kb, pb) = keys[j], keys[min(j + 1, len(keys) - 1)]
        p = _lerp(pa, pb, (f - ka) / max(kb - ka, 1))
        if clamp is None:
            clamp = _light_clamp(frame, p, lm.get("mean_range"))
            for _try in range(4):                    # legibility guard on the first frame only
                if is_legible(_apply_light(frame, p, clamp, 0.0, rng), tuple(guard["mean"]), float(guard["min_std"])):
                    break
                keys = [(kf_, _light_params(rng, lm)) for kf_, _ in keys]
                p = keys[0][1]
                clamp = _light_clamp(frame, p, lm.get("mean_range"))
        frame = _apply_light(frame, p, clamp, noise, rng)
        frame = _apply_tier(frame, tier, rng)
        fn = writer.frame(f, frame)
        # --- labels from the simulation layers (visible = alpha x not-later-beetles x not-occluder), warped like the frame
        later = np.zeros((Ch, Cw), np.float32)
        pos = {k: i for i, k in enumerate(order)}
        for k in range(n_obj):
            later = np.zeros((Ch, Cw), np.float32)
            for k2 in order[pos[k] + 1:]:
                later = np.maximum(later, layers[k2])
            vis_c = layers[k] * (1 - later)
            if has_occ:
                vis_c = vis_c * (1 - occ_a)
            full_f = cv2.warpAffine(layers[k], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            vis_f = cv2.warpAffine(vis_c, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            body_f = max(body_px[k] * z * z, 1.0)
            in_frame = float((full_f > 0.5).sum()) / body_f
            share = float((vis_f > 0.5).sum()) / body_f
            share = min(share, 1.0)
            bv, bf = _box(vis_f), _box(full_f)
            absent = int(bv is None or share < absent_below)
            box = bv if bv is not None else (bf if bf is not None else last_box[k])
            if box is not None:
                last_box[k] = box
            cx_f = M[0, 0] * st["beetles"][k]["x"] + M[0, 1] * st["beetles"][k]["y"] + M[0, 2]
            cy_f = M[1, 0] * st["beetles"][k]["x"] + M[1, 1] * st["beetles"][k]["y"] + M[1, 2]
            tr = tracks[str(k)]
            tr["boxes"].append(box if box is not None else [0, 0, 0, 0])
            tr["boxes_full"].append(bf)
            tr["absence"].append(absent)
            tr["cover"].append(int(round(share * 8)))
            cut = int(bf is not None and (bf[0] <= 0 or bf[1] <= 0 or bf[2] >= W or bf[3] >= H) and in_frame < 0.999)
            tr["cut_by_image"].append(cut)
            tr["visible_share"].append(round(share, 3))
            obj_meta[k].update({"bbox_visible_xyxy": bv, "bbox_full_xyxy": bf, "visible_share": round(share, 3),
                                "in_frame_frac": round(min(in_frame, 1.0), 3), "absent": absent, "cut_by_image": cut,
                                "centre_frame": [round(float(cx_f), 2), round(float(cy_f), 2)],
                                "size_frame_px": round(st["beetles"][k]["size"] * z, 1)})
        frame_recs.append({"frame": f, "file": fn, "camera": {"pan": [round(cam.cx, 2), round(cam.cy, 2)],
                           "shake": [round(cam.sx, 2), round(cam.sy, 2)], "zoom": round(z, 4), "rot_deg": round(cam.rot, 3),
                           "shift_px": [round(dxc, 2), round(dyc, 2)]},
                           "lighting": {k_: (round(v, 3) if isinstance(v, float) else [round(x, 3) for x in v]) for k_, v in p.items()},
                           "objects": obj_meta})
    # --- motion summary of the target (for matching the real camera later)
    t = tracks["0"]
    disp, scale = [], []
    for a_, b_, aa, ab in zip(t["boxes"][:-1], t["boxes"][1:], t["absence"][:-1], t["absence"][1:]):
        if not aa and not ab:                    # boxes are xyxy
            disp.append(math.hypot((a_[0] + a_[2]) / 2 - (b_[0] + b_[2]) / 2, (a_[1] + a_[3]) / 2 - (b_[1] + b_[3]) / 2))
            la, lb = max(a_[2] - a_[0], a_[3] - a_[1]), max(b_[2] - b_[0], b_[3] - b_[1])
            if la > 0 and lb > 0:
                scale.append(lb / la - 1)
    meta = {"name": name, "split": split, "index": index, "seed": [int(cfg.get("seed", 0)), 23, index],
            "num_frames": n_frames, "fps": sq["fps"], "frame_size": [W, H], "canvas_size": [Cw, Ch],
            **bg_meta, **bam, "num_objects": n_obj, "paste_order": [int(k) for k in order],
            "camera_static": sim["camera_static"], "pan_sigma_px": sim["pan_sigma"], "shake_sigma_px": sim["shake_sigma"],
            "exposure_frac": round(exposure_frac, 3), "lighting": mode, "lighting_noise": round(noise, 3),
            "exposure_clamp": round(clamp or 1.0, 3), **tier, "color_match": round(cm, 3), "edge_blur_px": round(edge_blur, 2),
            "shadow": {k_: round(v, 3) for k_, v in shadow_p.items()} if shadow_p else None,
            "occluders": occ_meta, "num_occluders": len(occ_meta),
            "beetles_start": [{k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in b.items()} for b in sim["frames"][0]["beetles"]],
            "target": {"frames_absent": int(sum(t["absence"])), "frames_cut": int(sum(t["cut_by_image"])),
                       "frames_occluded": int(sum(1 for s, c, a in zip(t["visible_share"], t["cut_by_image"], t["absence"])
                                                  if s < 0.95 and not c and not a)),
                       "first_frame_present": next((i for i, a in enumerate(t["absence"]) if not a), None),
                       "mean_visible_share": round(float(np.mean(t["visible_share"])), 3),
                       "displacement_px": {"mean": round(float(np.mean(disp)), 2), "p50": round(float(np.median(disp)), 2),
                                           "p90": round(float(np.percentile(disp, 90)), 2), "max": round(float(np.max(disp)), 2)} if disp else None,
                       "scale_change_per_frame": {"p50_abs": round(float(np.median(np.abs(scale))), 4),
                                                  "max_abs": round(float(np.max(np.abs(scale))), 4)} if scale else None},
            "seconds": round(time.time() - t0, 1)}
    writer.finish(tracks, frame_recs, meta)
    return meta


# ----------------------------------------------------------------------------- the whole set
REC_KEYS = ("name", "split", "index", "num_frames", "num_objects", "num_occluders", "camera_static", "lighting",
            "quality_tier", "background", "target", "seconds")


def _rec(meta: dict) -> dict:
    return {k: meta.get(k) for k in REC_KEYS}


def _plan(cfg: dict, sq: dict, num: int, out_dir: str) -> tuple[list[tuple[int, str, str | None]], BackgroundPool, BackgroundPool]:
    """(index, split, background photo) for every sequence, independent of the worker count or a resume:
    the split is drawn from the sequence's seed, the photos are handed out in a seeded order per split."""
    seed = int(cfg.get("seed", 0))
    pool = BackgroundPool(cfg["backgrounds"]["dirs"], min_size=int(cfg["backgrounds"].get("min_size", 64)), seed=seed)
    train_pool, val_pool = pool.split(float(sq["val_fraction"]), seed)      # disjoint photos per split
    order = {"train": train_pool.ordered_paths(seed), "val": val_pool.ordered_paths(seed)} if sq.get("own_background") else None
    used = {"train": 0, "val": 0}
    plan = []
    for i in range(num):
        rng = np.random.default_rng([seed, 23, i])
        split = "val" if rng.random() < float(sq["val_fraction"]) else "train"
        bg_path = None
        if order:
            bg_path = order[split][used[split] % len(order[split])]
            used[split] += 1
        plan.append((i, split, bg_path))
    return plan, train_pool, val_pool


def _make_views(cfg: dict, sq: dict, disc: bool):
    return DiscViews() if disc else BeetleViews(cfg, sq.get("min_render_px"))


def _leaf_bank(cfg: dict, sq: dict, div: int = 1) -> OccluderBank | None:
    occ_cfg = cfg["compose"].get("occlusion") or {}
    if float(sq["occlusion"].get("prob", 0)) <= 0:
        return None
    gb = float((cfg.get("resources") or {}).get("occluder_cache_gb", 0.5))
    return OccluderBank(occ_cfg.get("dirs"), cache_gb=min(gb / max(div, 1), 0.5) if div > 1 else gb)


def _one(cfg, sq, item, pools, leaf_bank, views, out_dir) -> dict:
    i, split, bg_path = item
    meta = render_sequence(cfg, sq, i, split, pools[split], leaf_bank, views, out_dir, bg_path=bg_path)
    gif = sq.get("gif") or {}
    if gif and int(gif.get("every", 0)) > 0 and i % int(gif["every"]) == 0:
        from .seqpreview import sequence_gif
        sequence_gif(os.path.join(out_dir, split, meta["name"]), os.path.join(out_dir, "_gifs", split, meta["name"] + ".gif"),
                     max_side=int(gif.get("max_side", 240)), frame_step=int(gif.get("frame_step", 2)))
    return _rec(meta)


def _seq_worker(cfg: dict, sq: dict, out_dir: str, items: list, k: int, n_workers: int, disc: bool, q) -> None:
    """One worker process: its own renderer (GPU), background pools and occluder cache; one message per sequence."""
    try:
        _, train_pool, val_pool = _plan(cfg, sq, 0, out_dir)
        pools = {"train": train_pool, "val": val_pool}
        views = _make_views(cfg, sq, disc)
        leaf_bank = _leaf_bank(cfg, sq, n_workers)
        for item in items:
            q.put(("seq", k, _one(cfg, sq, item, pools, leaf_bank, views, out_dir)))
        q.put(("done", k, {"views": views.count, "seconds": views.seconds}))
    except BaseException as e:   # noqa: BLE001
        q.put(("error", k, repr(e)))
        raise


def generate_sequences(cfg: dict, out_dir: str, num: int | None = None, verbose: bool = True, views=None,
                       overwrite: bool = False, workers: int | None = None, disc: bool = False) -> dict:
    """`workers` > 1 renders the sequences in that many processes (each with the splat on the GPU); every sequence
    is seeded by its index, so the set is the same whatever the worker count. A second call resumes: finished
    sequence folders (meta.json present) are kept unless `overwrite`."""
    from .generate import resolve_workers
    sq = _merge(DEFAULTS, cfg.get("sequences"))
    num = num or int(sq["num_sequences"])
    W, H = int(sq["frame_size"][0]), int(sq["frame_size"][1])
    workers = resolve_workers(workers if workers is not None else (cfg.get("resources") or {}).get("workers", 1))
    plan, train_pool, val_pool = _plan(cfg, sq, num, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    # resume: records of finished sequences (from sequences.jsonl, or rebuilt from their meta.json)
    log_path = os.path.join(out_dir, "sequences.jsonl")
    have = {}
    if os.path.exists(log_path) and not overwrite:
        for line in open(log_path):
            try:
                r = json.loads(line)
                have[r["index"]] = r
            except (json.JSONDecodeError, KeyError):
                continue
    todo, recs = [], []
    for i, split, bg_path in plan:
        meta_path = os.path.join(out_dir, split, f"seq_{i:06d}", "meta.json")
        if not overwrite and os.path.exists(meta_path):
            if i not in have:
                with open(meta_path) as f:
                    have[i] = _rec(json.load(f))
            recs.append(have[i])
        else:
            todo.append((i, split, bg_path))
    if verbose:
        Cw, Ch = canvas_size(W, H, sq["camera"])
        print(f"sequences: {num} x {sq['length'][0]}-{sq['length'][1]} frames of {W}x{H} (canvas {Cw}x{Ch}), "
              f"{len(train_pool)} train / {len(val_pool)} val backgrounds -> {out_dir}"
              + (f"; resume: {len(recs)} done, {len(todo)} to do" if recs else ""))
    t0 = time.time()
    stats = {"views": 0, "seconds": 0.0}
    with open(log_path, "w") as log:
        for r in sorted(recs, key=lambda r: r["index"]):
            log.write(json.dumps(r, default=_json_default) + "\n")
        if workers <= 1 or len(todo) < 2 * workers:
            views = views or _make_views(cfg, sq, disc)
            leaf_bank = _leaf_bank(cfg, sq)
            pools = {"train": train_pool, "val": val_pool}
            if verbose:
                print(f"views: {views.describe()}")
            for item in tqdm(todo, desc="sequences", disable=not verbose):
                rec = _one(cfg, sq, item, pools, leaf_bank, views, out_dir)
                log.write(json.dumps(rec, default=_json_default) + "\n"); log.flush()
                recs.append(rec)
            stats = {"views": views.count, "seconds": views.seconds}
        else:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            q = ctx.Queue()
            shards = [todo[k::workers] for k in range(workers)]
            procs = [ctx.Process(target=_seq_worker, args=(cfg, sq, out_dir, sh, k, workers, disc, q), daemon=True)
                     for k, sh in enumerate(shards)]
            if verbose:
                print(f"{len(todo)} sequences in {workers} worker processes (each with its own renderer on the GPU)")
            for pr in procs:
                pr.start()
            alive, errors = workers, []
            with tqdm(total=len(todo), desc="sequences", disable=not verbose) as bar:
                while alive:
                    msg = q.get()
                    if msg[0] == "seq":
                        log.write(json.dumps(msg[2], default=_json_default) + "\n"); log.flush()
                        recs.append(msg[2]); bar.update(1)
                    elif msg[0] == "done":
                        alive -= 1
                        stats["views"] += msg[2]["views"]; stats["seconds"] += msg[2]["seconds"]
                    else:
                        alive -= 1; errors.append(msg[2])
            for pr in procs:
                pr.join()
            if errors:
                raise RuntimeError(f"{len(errors)} sequence worker(s) failed: {errors[0]}  (re-run to resume)")
    for split in ("train", "val"):
        names = sorted(r["name"] for r in recs if r["split"] == split)
        if names:
            os.makedirs(os.path.join(out_dir, split), exist_ok=True)
            with open(os.path.join(out_dir, split, "list.txt"), "w") as f:
                f.write("\n".join(names) + "\n")
    summary = summarize_sequences(recs, sq)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=_json_default)
    if verbose:
        dt = time.time() - t0
        done_idx = {i for i, _, _ in todo}
        n_new = sum(r["num_frames"] for r in recs if r["index"] in done_idx)
        print(f"done in {dt:.0f}s ({n_new} new frames, {stats['views']} beetle views, "
              f"{stats['seconds'] / max(stats['views'], 1) * 1000:.0f} ms per view)")
        print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=None))
    return summary


def summarize_sequences(recs: list[dict], sq: dict) -> dict:
    def pct(xs, q):
        return round(float(np.percentile(xs, q)), 2) if xs else None
    d50 = [r["target"]["displacement_px"]["p50"] for r in recs if r["target"].get("displacement_px")]
    d90 = [r["target"]["displacement_px"]["p90"] for r in recs if r["target"].get("displacement_px")]
    dmax = [r["target"]["displacement_px"]["max"] for r in recs if r["target"].get("displacement_px")]
    frames = [r["num_frames"] for r in recs]
    return {"num_sequences": len(recs), "num_frames": int(sum(frames)),
            "per_split": {s: sum(1 for r in recs if r["split"] == s) for s in ("train", "val")},
            "frames_per_sequence": {"min": min(frames) if frames else 0, "max": max(frames) if frames else 0,
                                    "mean": round(float(np.mean(frames)), 1) if frames else 0},
            "objects_per_sequence": {str(n): sum(1 for r in recs if r["num_objects"] == n) for n in sorted({r["num_objects"] for r in recs})},
            "sequences_with_occluders": sum(1 for r in recs if r["num_occluders"]),
            "sequences_static_camera": sum(1 for r in recs if r["camera_static"]),
            "target_frames_absent": int(sum(r["target"]["frames_absent"] for r in recs)),
            "target_frames_cut_by_frame": int(sum(r["target"]["frames_cut"] for r in recs)),
            "target_frames_occluded": int(sum(r["target"]["frames_occluded"] for r in recs)),
            "target_displacement_px_per_frame": {"median_of_sequence_medians": pct(d50, 50), "p90_of_sequence_p90": pct(d90, 90),
                                                 "max": max(dmax) if dmax else None,
                                                 "nominal_fps": sq["fps"], "frame_size": sq["frame_size"]},
            "lighting": {m: sum(1 for r in recs if r["lighting"] == m) for m in sorted({r["lighting"] for r in recs})},
            "quality_tier": {m: sum(1 for r in recs if r["quality_tier"] == m) for m in sorted({r["quality_tier"] for r in recs})},
            "config": sq}
