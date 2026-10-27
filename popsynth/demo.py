"""Controlled demonstration slides: each slide varies one factor on the same background.

Writes <out>/slide_NN_<name>.jpg (labelled grid) and <out>/NN_<name>/<label>.png (individual images).
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from .augment import apply_camera_effects, apply_imperfections, apply_lighting
from .backgrounds import load_crop
from .cameras import fixed_camera
from .compose import crop_to_alpha, paste_sprite
from .generate import prepare_splat
from .render import get_renderer
from .render.base import pick_device

DEFAULT_BACKGROUND = "data/backgrounds/inat_hostplants/acer_297975510.jpg"
OTHER_BACKGROUNDS = [
    "data/backgrounds/inat_hostplants/ulmus_31482304.jpg",
    "data/backgrounds/inat_hostplants/acer_46675407.jpg",
]

NO_SHADOW = None
SHADOW = {"prob": 1.0, "strength": (0.4, 0.4), "blur": (0.06, 0.06), "offset": (0.04, 0.04)}
CLEAN = dict(color_match=0.0, shadow=SHADOW, edge_blur_px=0.5, defocus=None)


class Demo:
    def __init__(self, cfg: dict, out_dir: str, image_size: int = 640, sprite_size: int = 1024,
                 background: str | None = None, verbose: bool = True):
        self.cfg, self.out, self.S, self.sprite_size, self.verbose = cfg, out_dir, image_size, sprite_size, verbose
        os.makedirs(out_dir, exist_ok=True)
        self.bg_path = background or DEFAULT_BACKGROUND
        self.bg = load_crop(self.bg_path, image_size, image_size)
        self.splat = prepare_splat(cfg)
        self.radius = self.splat.max_extent()
        self.up = cfg["splat"].get("up", [0, 0, 1])
        device = pick_device(cfg["render"].get("device", "auto"))
        self.backend, self.render = get_renderer(cfg["render"].get("backend", "auto"))
        self.g = self.splat.to_torch(device)
        self._sprites: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        self.rng = np.random.default_rng(0)

    # ------------------------------------------------------------------ helpers
    def sprite(self, elevation=60.0, azimuth=30.0, roll=0.0, fov=40.0):
        key = (round(elevation, 1), round(azimuth, 1), round(roll, 1), round(fov, 1))
        if key not in self._sprites:
            S = self.sprite_size
            cam = fixed_camera(S, S, up=self.up, object_radius=self.radius * 1.05, elevation_deg=elevation,
                               azimuth_deg=azimuth, roll_deg=roll, fov_deg=fov, fill=0.9)
            rgb, a = self.render(self.g, cam.viewmat, cam.K, S, S, antialiased=True)
            self._sprites[key] = crop_to_alpha(rgb.numpy(), a.numpy(), pad=2)
            if self.verbose:
                print(f"  rendered sprite el={elevation} az={azimuth} roll={roll} fov={fov}")
        return self._sprites[key]

    def paste(self, bg, sp, size=0.35, center=(0.5, 0.5), **kw):
        opts = {**CLEAN, **kw}
        img, mask, meta = paste_sprite(bg.copy(), sp[0], sp[1], self.rng, object_size=(size, size),
                                       min_visible=0.0, center=center, **opts)
        return img, mask, meta

    def light(self, img, mode="day", flash=False, seed=1):
        modes = {k: dict(v) for k, v in self.cfg["lighting"].items()}
        for m in modes.values():            # fixed, representative values instead of random ranges
            for k in ("exposure", "contrast", "saturation", "noise", "vignette"):
                if isinstance(m.get(k), list):
                    m[k] = float(np.mean(m[k]))
            m["tint"] = [float(np.mean(t)) for t in m.get("tint", [[1, 1], [1, 1], [1, 1]])]
            m["flash_prob"] = 1.0 if flash else 0.0
        out, _ = apply_lighting(img, np.random.default_rng(seed), modes, mode=mode)
        return out

    def _label(self, img, text):
        u8 = cv2.cvtColor((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.rectangle(u8, (0, 0), (u8.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(u8, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return u8

    def slide(self, idx: int, name: str, items: list[tuple[str, np.ndarray]], cols: int | None = None):
        d = os.path.join(self.out, f"{idx:02d}_{name}")
        os.makedirs(d, exist_ok=True)
        tiles = []
        for i, (label, img) in enumerate(items):
            fn = f"{i + 1:02d}_{label.split(':')[0].strip().replace(' ', '_').replace('/', '-')}.png"
            cv2.imwrite(os.path.join(d, fn), cv2.cvtColor((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR))
            tiles.append(self._label(img, label))
        cols = cols or (3 if len(tiles) <= 6 else 4)
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)]
        sheet = np.concatenate(rows, 0)
        path = os.path.join(self.out, f"slide_{idx:02d}_{name}.jpg")
        cv2.imwrite(path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if self.verbose:
            print(f"slide {idx:02d} {name}: {len(items)} images -> {path}")
        return path

    # ------------------------------------------------------------------ slides
    def build(self, which: list[str] | None = None):
        slides = [
            ("distance", self.slide_distance), ("lighting", self.slide_lighting),
            ("viewpoint", self.slide_viewpoint), ("rotation", self.slide_rotation),
            ("partially_outside", self.slide_partial), ("perspective", self.slide_perspective),
            ("effects", self.slide_effects), ("imperfections", self.slide_imperfections),
            ("backgrounds", self.slide_backgrounds),
            ("multiple_beetles", self.slide_multiple), ("negatives", self.slide_negatives),
        ]
        paths = []
        for i, (name, fn) in enumerate(slides, 1):
            if which and name not in which:
                continue
            paths.append(fn(i))
        return paths

    def slide_distance(self, idx):
        sp = self.sprite()
        items = []
        for size in (0.06, 0.12, 0.2, 0.35, 0.55, 0.85):
            img, _, _ = self.paste(self.bg, sp, size=size)
            items.append((f"size {size:.2f} of frame (distance)", self.light(img)))
        return self.slide(idx, "distance", items)

    def slide_lighting(self, idx):
        sp = self.sprite()
        base, _, _ = self.paste(self.bg, sp, size=0.4)
        items = [("no lighting change (raw composite)", base)]
        for mode in ("day", "overcast", "dusk", "night"):
            items.append((f"{mode}", self.light(base, mode)))
        items.append(("night + camera flash", self.light(base, "night", flash=True)))
        return self.slide(idx, "lighting", items)

    def slide_viewpoint(self, idx):
        items = []
        for el, az in ((90, 0), (60, 0), (30, 0), (0, 0), (60, 90), (60, 180), (60, 270), (30, 225)):
            img, _, _ = self.paste(self.bg, self.sprite(elevation=el, azimuth=az), size=0.4)
            items.append((f"elevation {el} deg, azimuth {az} deg", self.light(img)))
        return self.slide(idx, "viewpoint", items, cols=4)

    def slide_rotation(self, idx):
        items = []
        for roll in (0, 60, 120, 180, 240, 300):
            img, _, _ = self.paste(self.bg, self.sprite(roll=roll), size=0.4)
            items.append((f"in-plane rotation {roll} deg", self.light(img)))
        return self.slide(idx, "rotation", items)

    def slide_partial(self, idx):
        sp = self.sprite()
        items = []
        for label, c in (("fully inside", (0.5, 0.5)), ("half out left", (0.0, 0.5)), ("half out right", (1.0, 0.5)),
                         ("half out top", (0.5, 0.0)), ("half out bottom", (0.5, 1.0)), ("corner, 3/4 out", (0.95, 0.95))):
            img, _, m = self.paste(self.bg, sp, size=0.45, center=c)
            items.append((f"{label} (visible {m['visible_frac']:.0%})", self.light(img)))
        return self.slide(idx, "partially_outside", items)

    def slide_perspective(self, idx):
        items = []
        for fov in (15, 30, 45, 65, 85, 100):
            img, _, _ = self.paste(self.bg, self.sprite(fov=fov), size=0.4)
            items.append((f"camera fov {fov} deg (same size: far+tele vs close+wide)", self.light(img)))
        return self.slide(idx, "perspective", items)

    def slide_effects(self, idx):
        sp = self.sprite()
        base = lambda **kw: self.paste(self.bg, sp, size=0.4, **kw)[0]
        items = [("no shadow", self.light(base(shadow=NO_SHADOW))),
                 ("soft shadow", self.light(base())),
                 ("colour matched to background (0.5)", self.light(base(color_match=0.5))),
                 ("defocus 1.5 px", self.light(base(defocus={"prob": 1.0, "sigma_px": (1.5, 1.5)})))]
        mb, _ = apply_camera_effects(self.light(base()), np.random.default_rng(3),
                                     {"blur": {"prob": 1.0, "motion_frac": 1.0, "motion_max_px": 9}})
        items.append(("motion blur", mb))
        jp, _ = apply_camera_effects(self.light(base()), np.random.default_rng(4), {"jpeg_quality": [35, 35]})
        items.append(("jpeg quality 35", jp))
        return self.slide(idx, "effects", items)

    def slide_imperfections(self, idx):
        """Deliberately bad photos: each effect forced on its own, then a random draw from the 'bad' tier."""
        sp = self.sprite()
        base = self.light(self.paste(self.bg, sp, size=0.4)[0])
        tiers = self.cfg["camera_effects"].get("tiers", {})
        bad = dict(tiers.get("bad", {}))
        def forced(**kw):
            t = {k: 0.0 for k in ("blur_prob", "defocus_prob", "lowres_prob", "haze_prob", "exposure_fault_prob")}
            t.update({"noise": [0, 0], "jpeg_quality": None}); t.update(kw)
            return apply_imperfections(base, np.random.default_rng(7), {"x": t}, tier="x")[0]
        items = [("clean", base),
                 ("motion blur (camera shake)", forced(blur_prob=1.0, motion_px=[17, 17])),
                 ("defocus (out of focus)", forced(defocus_prob=1.0, defocus_sigma=[3.0, 3.0])),
                 ("low resolution (far / cheap sensor)", forced(lowres_prob=1.0, lowres_factor=[4.0, 4.0])),
                 ("haze / dirty lens", forced(haze_prob=1.0, haze=[0.45, 0.45])),
                 ("overexposed", forced(exposure_fault_prob=1.0, overexposure=[2.2, 2.2], underexposure=[2.2, 2.2])),
                 ("underexposed + sensor noise", forced(exposure_fault_prob=1.0, overexposure=[0.3, 0.3], underexposure=[0.3, 0.3], noise=[0.06, 0.06])),
                 ("jpeg quality 25", forced(jpeg_quality=[25, 25]))]
        if bad:
            for i in range(4):
                img, m = apply_imperfections(base, np.random.default_rng(100 + i), {"bad": bad}, tier="bad")
                items.append((f"random 'bad' tier draw {i + 1}", img))
        return self.slide(idx, "imperfections", items, cols=4)

    def slide_backgrounds(self, idx):
        sp = self.sprite()
        items = []
        bgs = [self.bg_path] + [b for b in OTHER_BACKGROUNDS if os.path.exists(b)]
        pool_dirs = [d["path"] if isinstance(d, dict) else d for d in self.cfg["backgrounds"]["dirs"]]
        rng = np.random.default_rng(5)
        for d in pool_dirs:
            if os.path.isdir(d) and len(bgs) < 6:
                files = sorted(f for f in os.listdir(d) if f.lower().endswith((".jpg", ".png")))
                if files:
                    bgs.append(os.path.join(d, files[int(rng.integers(len(files)))]))
        for b in bgs[:6]:
            img, _, _ = self.paste(load_crop(b, self.S, self.S), sp, size=0.4)
            items.append((f"{os.path.basename(os.path.dirname(b))}/{os.path.basename(b)}", self.light(img)))
        return self.slide(idx, "backgrounds", items)

    def slide_multiple(self, idx):
        poses = [(60, 30, 0), (45, 120, 90), (80, 200, 200), (30, 300, 45), (60, 60, 300)]
        centers = [(0.3, 0.35), (0.7, 0.6), (0.25, 0.75), (0.75, 0.25), (0.5, 0.5)]
        sizes = [0.3, 0.25, 0.2, 0.28, 0.18]
        items = []
        for n in (1, 2, 3, 5):
            img = self.bg
            for k in range(n):
                el, az, roll = poses[k]
                img, _, _ = self.paste(img, self.sprite(elevation=el, azimuth=az, roll=roll), size=sizes[k], center=centers[k])
            items.append((f"{n} beetle{'s' if n > 1 else ''}", self.light(img)))
        return self.slide(idx, "multiple_beetles", items, cols=2)

    def slide_negatives(self, idx):
        items = [("background only: day", self.light(self.bg, "day")),
                 ("background only: overcast", self.light(self.bg, "overcast")),
                 ("background only: night", self.light(self.bg, "night"))]
        return self.slide(idx, "negatives", items)


def build_demo(cfg: dict, out_dir: str, image_size: int = 640, sprite_size: int = 1024,
               background: str | None = None, which: list[str] | None = None) -> list[str]:
    return Demo(cfg, out_dir, image_size, sprite_size, background).build(which)
