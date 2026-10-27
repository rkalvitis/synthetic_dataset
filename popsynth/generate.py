"""Two-stage pipeline:  (1) render sprites from the splat,  (2) composite sprites onto backgrounds.

Stage 1 is the expensive, GPU-bound part and depends only on `splat`/`render` config.
Stage 2 is cheap CPU work and can be re-run with new backgrounds / augmentation settings.
Every image is seeded from (seed, index), so a dataset can be regenerated or extended deterministically.
"""
from __future__ import annotations

import json
import os
import time

import cv2
import numpy as np
from tqdm import tqdm

from .augment import apply_camera_effects, apply_lighting
from .backgrounds import BackgroundPool
from .cameras import sample_camera
from .compose import crop_to_alpha, paste_sprite
from .labels import DatasetWriter
from .render import get_renderer
from .render.base import pick_device
from .splat import Splat


# ----------------------------------------------------------------------------- stage 1
def prepare_splat(cfg: dict) -> Splat:
    s = Splat.load(cfg["splat"]["path"])
    c = cfg["splat"].get("clean") or {}
    s = s.clean(opacity_min=c.get("opacity_min", 0.0), max_scale=c.get("max_scale"),
                radius_crop=c.get("radius_crop"), box=c.get("box"))
    return s.normalized()


def render_sprites(cfg: dict, out_dir: str, num: int | None = None, verbose: bool = True) -> str:
    """Render `num` RGBA sprites into <out_dir>/sprites and write sprites.jsonl. Returns the sprites dir."""
    rc = cfg["render"]
    num = num or rc["num_sprites"]
    size = int(rc["size"])
    ss = max(1, int(rc.get("supersample", 1)))      # render at size*ss and area-downsample: sharper, less aliasing
    rsize = size * ss
    sprites_dir = os.path.join(out_dir, "sprites")
    os.makedirs(sprites_dir, exist_ok=True)

    t0 = time.time()
    splat = prepare_splat(cfg)
    device = pick_device(rc.get("device", "auto"))
    name, render = get_renderer(rc.get("backend", "auto"))
    if name == "gsplat" and device != "cuda":
        raise RuntimeError("gsplat backend needs a CUDA device")
    g = splat.to_torch(device)
    if verbose:
        print(f"splat: {len(splat)} gaussians, sh degree {splat.sh_degree}; renderer={name} device={device} "
              f"({time.time() - t0:.1f}s to load); sprites {size}px rendered at {rsize}px")

    up = cfg["splat"].get("up", [0, 0, 1])
    cam_cfg = {k: v for k, v in rc["camera"].items()}
    # frame the true extent (all visible gaussians incl. their size), not the 99.5% radius, plus the
    # look-at jitter, so legs and antennae are never cut by the sprite frame
    radius = splat.max_extent() * (1.0 + 3.0 * float(cam_cfg.get("center_jitter", 0.0)))
    n_val = int(round(num * rc.get("val_fraction", 0.15)))
    meta_path = os.path.join(sprites_dir, "sprites.jsonl")
    n_clipped = 0
    with open(meta_path, "w") as mf:
        for i in tqdm(range(num), desc="render sprites", disable=not verbose):
            rng = np.random.default_rng([int(cfg["seed"]), 7, i])
            for attempt in range(4):
                # should not happen with the true-extent radius; if it does, re-sample with a smaller fill
                kw = dict(cam_cfg)
                if attempt:
                    kw["fill"] = [f * 0.85 ** attempt for f in np.atleast_1d(cam_cfg["fill"])]
                cam = sample_camera(rng, width=rsize, height=rsize, up=up, object_radius=radius, **kw)
                rgb, a = render(g, cam.viewmat, cam.K, rsize, rsize, antialiased=bool(rc.get("antialiased", False)))
                rgb, a = rgb.numpy(), a.numpy()
                if ss > 1:   # downsample premultiplied colour so edge pixels are not darkened
                    prem = cv2.resize(rgb * a[..., None], (size, size), interpolation=cv2.INTER_AREA)
                    a = cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA)
                    rgb = np.where(a[..., None] > 1e-4, prem / np.maximum(a[..., None], 1e-4), 0).astype(np.float32)
                clipped = _touches_border(a)
                if not clipped:
                    break
            if clipped:
                n_clipped += 1
            rgb_c, a_c = crop_to_alpha(rgb, a, pad=2)
            rgba = np.concatenate([rgb_c, a_c[..., None]], 2)
            rgba8 = (np.clip(rgba, 0, 1) * 255 + 0.5).astype(np.uint8)
            fn = f"sprite_{i:05d}.png"
            cv2.imwrite(os.path.join(sprites_dir, fn), cv2.cvtColor(rgba8, cv2.COLOR_RGBA2BGRA))
            rec = {"index": i, "file": fn, "split": "val" if i < n_val else "train",
                   "render_size": size, "supersample": ss, "crop_px": [int(a_c.shape[1]), int(a_c.shape[0])],
                   "coverage": round(float((a > 0.5).mean()), 4), "clipped": bool(clipped), **cam.meta()}
            mf.write(json.dumps(rec) + "\n")
    if verbose:
        print(f"wrote {num} sprites to {sprites_dir} ({time.time() - t0:.1f}s)"
              + (f"; WARNING {n_clipped} sprites still touch the frame border" if n_clipped else "; none clipped"))
    return sprites_dir


def _touches_border(alpha: np.ndarray, thresh: float = 0.03) -> bool:
    return bool(alpha[0].max() > thresh or alpha[-1].max() > thresh or alpha[:, 0].max() > thresh or alpha[:, -1].max() > thresh)


# ----------------------------------------------------------------------------- stage 2
class SpriteBank:
    def __init__(self, sprites_dir: str):
        self.dir = sprites_dir
        self.records = [json.loads(l) for l in open(os.path.join(sprites_dir, "sprites.jsonl"))]
        if not self.records:
            raise RuntimeError(f"no sprites in {sprites_dir}")
        self.by_split = {"train": [r for r in self.records if r["split"] == "train"],
                         "val": [r for r in self.records if r["split"] == "val"]}
        for k, v in self.by_split.items():
            if not v:                       # too few sprites to split: share them
                self.by_split[k] = self.records
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def load(self, rec: dict) -> tuple[np.ndarray, np.ndarray]:
        fn = rec["file"]
        if fn not in self._cache:
            bgra = cv2.imread(os.path.join(self.dir, fn), cv2.IMREAD_UNCHANGED)
            rgba = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA).astype(np.float32) / 255
            self._cache[fn] = (np.ascontiguousarray(rgba[..., :3]), np.ascontiguousarray(rgba[..., 3]))
        return self._cache[fn]

    def sample(self, rng: np.random.Generator, split: str) -> dict:
        lst = self.by_split[split]
        return lst[int(rng.integers(len(lst)))]


def compose_dataset(cfg: dict, sprites_dir: str, out_dir: str, num: int | None = None,
                    verbose: bool = True) -> str:
    cc, oc = cfg["compose"], cfg["output"]
    num = num or cc["num_images"]
    W, H = int(cc["image_size"][0]), int(cc["image_size"][1])
    bank = SpriteBank(sprites_dir)
    pool = BackgroundPool(cfg["backgrounds"]["dirs"], min_size=int(cfg["backgrounds"].get("min_size", 64)))
    writer = DatasetWriter(out_dir, oc.get("image_format", "jpg"), oc.get("jpeg_quality", 95),
                           oc.get("write_masks", True), oc.get("write_yolo", True))
    if verbose:
        print(f"compose: {num} images of {W}x{H}, {len(bank.records)} sprites, {len(pool)} backgrounds "
              f"[{pool.describe()}] -> {out_dir}")
    counts = {"train": 0, "val": 0, "popillia": 0, "background": 0}
    t0 = time.time()
    for i in tqdm(range(num), desc="compose", disable=not verbose):
        rng = np.random.default_rng([int(cfg["seed"]), 11, i])
        split = "val" if rng.random() < cc.get("val_fraction", 0.15) else "train"
        positive = rng.random() < cc.get("positive_fraction", 0.5)
        bg, meta = pool.sample(rng, W, H, tuple(cc.get("crop_scale", cfg["backgrounds"].get("crop_scale", (0.4, 1.0)))),
                               cfg["backgrounds"].get("hflip", 0.5))
        img, mask, boxes, objs, obj_masks = bg, None, [], [], []
        if positive:
            lo, hi = cc.get("objects_per_image", [1, 1])
            mask = np.zeros((H, W), np.float32)
            for _ in range(int(rng.integers(int(lo), int(hi) + 1))):
                rec = bank.sample(rng, split)
                s_rgb, s_a = bank.load(rec)
                img, m, pm = paste_sprite(
                    img, s_rgb, s_a, rng,
                    object_size=tuple(cc["object_size"]), min_visible=cc.get("min_visible", 0.6),
                    color_match=tuple(cc.get("color_match", (0, 0))), shadow=cc.get("shadow"),
                    edge_blur_px=tuple(cc.get("edge_blur_px", (0, 0))), defocus=cc.get("defocus"))
                if pm.get("bbox"):
                    boxes.append(pm["bbox"])
                    mask = np.maximum(mask, m)
                    objs.append({"object_id": len(objs), "sprite": rec["file"], "sprite_index": rec["index"],
                                 "elevation_deg": rec["elevation_deg"], "azimuth_deg": rec["azimuth_deg"],
                                 "roll_deg": rec["roll_deg"], "fov_deg": rec["fov_deg"],
                                 "camera_distance": rec.get("distance"), **pm,
                                 "mask_area_px": int((m > 0.5).sum())})
                    obj_masks.append(m > 0.5)
            # beetles pasted later cover earlier ones: record how much of each beetle is hidden
            for k, o in enumerate(objs):
                later = np.zeros_like(obj_masks[k])
                for mk in obj_masks[k + 1:]:
                    later |= mk
                o["occluded_frac"] = round(float((obj_masks[k] & later).sum() / max(1, obj_masks[k].sum())), 3)
        img, lm = apply_lighting(img, rng, cfg.get("lighting"))
        img, cm = apply_camera_effects(img, rng, cfg.get("camera_effects", {}))
        meta.update(lm); meta.update(cm)
        meta["effects"] = sorted(k for k in cm if k != "quality_tier")
        meta["objects"] = objs
        writer.write(f"img_{i:06d}", split, img, mask, boxes, meta)
        counts[split] += 1
        counts["popillia" if boxes else "background"] += 1
    writer.close()
    if verbose:
        print(f"done in {time.time() - t0:.1f}s: {counts}")
        print(f"labels: {out_dir}/dataset.json (COCO + metadata), images.csv, objects.csv, summary.json")
    return out_dir


def generate(cfg: dict, out_dir: str | None = None, num_sprites: int | None = None,
             num_images: int | None = None, verbose: bool = True) -> str:
    out_dir = out_dir or cfg["output"]["dir"]
    sprites_dir = render_sprites(cfg, out_dir, num_sprites, verbose)
    return compose_dataset(cfg, sprites_dir, out_dir, num_images, verbose)
