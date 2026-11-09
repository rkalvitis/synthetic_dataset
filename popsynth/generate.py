"""Two-stage pipeline:  (1) render sprites from the splat,  (2) composite sprites onto backgrounds.

Stage 1 is the expensive, GPU-bound part and depends only on `splat`/`render` config.
Stage 2 is cheap CPU work and can be re-run with new backgrounds / augmentation settings.
Every image is seeded from (seed, index), so a dataset can be regenerated or extended deterministically.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import time

import cv2
import numpy as np
from tqdm import tqdm

from .augment import apply_photometric, augment_background, augment_sprite
from .backgrounds import BackgroundPool
from .cameras import projected_extent, resized, sample_camera
from .compose import crop_to_alpha, paste_sprite
from .labels import DatasetWriter
from .memory import GB, check_limit, fmt, limit_memory, peak_rss_bytes
from .occlude import OccluderBank, occlude_objects
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
    if verbose:
        print(limit_memory((cfg.get("resources") or {}).get("max_ram_gb")))
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
    # size_mode "object": the beetle's final pixel size is decided here, per sprite, by the camera distance /
    # frame size (uniform in render.object_size x the short side of compose.image_size); compose pastes 1:1.
    # size_mode "fixed": every sprite is rendered at render.size and compose rescales it (compose.object_size).
    size_mode = rc.get("size_mode", "fixed")
    short_side = int(min(cfg["compose"]["image_size"]))
    min_render = int(rc.get("min_render_px", 1024))
    vis = splat.opacities >= 0.05
    means_v, scales_v = splat.means[vis], splat.scales[vis]
    if verbose and size_mode == "object":
        print(f"size_mode=object: beetle longest side uniform in {rc.get('object_size')} x {short_side}px "
              f"(compose.image_size), rendered at >= {min_render}px and area-downsampled")
    meta_path = os.path.join(sprites_dir, "sprites.jsonl")
    n_clipped = 0
    with open(meta_path, "w") as mf:
        for i in tqdm(range(num), desc="render sprites", disable=not verbose):
            rng = np.random.default_rng([int(cfg["seed"]), 7, i])
            extra = {}
            for attempt in range(4):
                # should not happen with the true-extent radius; if it does, re-sample with a smaller fill
                kw = dict(cam_cfg)
                if attempt:
                    kw["fill"] = [f * 0.85 ** attempt for f in np.atleast_1d(cam_cfg["fill"])]
                if size_mode == "object":
                    lo, hi = rc.get("object_size", (0.12, 0.85))
                    size_frac = float(rng.uniform(lo, hi))
                    target_px = max(8, int(round(size_frac * short_side)))
                    cam0 = sample_camera(rng, width=1024, height=1024, up=up, object_radius=radius, **kw)
                    x0, y0, x1, y1 = projected_extent(means_v, scales_v, cam0.viewmat, cam0.K)
                    long0 = max(x1 - x0, y1 - y0, 1.0)
                    out_size = int(np.ceil(1024 * target_px / long0))          # frame where the beetle spans target_px
                    ss_eff = max(ss, int(np.ceil(min_render / out_size)))     # keep enough samples for small beetles
                    rsize_i = out_size * ss_eff
                    cam = resized(cam0, rsize_i, rsize_i)
                    extra = {"object_size_frac_target": round(size_frac, 4), "object_px_target": target_px,
                             "for_short_side": short_side}
                else:
                    out_size, ss_eff, rsize_i = size, ss, rsize
                    cam = sample_camera(rng, width=rsize, height=rsize, up=up, object_radius=radius, **kw)
                for correction in range(3):
                    rgb, a = render(g, cam.viewmat, cam.K, rsize_i, rsize_i, antialiased=bool(rc.get("antialiased", False)))
                    rgb, a = rgb.numpy(), a.numpy()
                    if ss_eff > 1:   # downsample premultiplied colour so edge pixels are not darkened
                        prem = cv2.resize(rgb * a[..., None], (out_size, out_size), interpolation=cv2.INTER_AREA)
                        a = cv2.resize(a, (out_size, out_size), interpolation=cv2.INTER_AREA)
                        rgb = np.where(a[..., None] > 1e-4, prem / np.maximum(a[..., None], 1e-4), 0).astype(np.float32)
                    if size_mode != "object":
                        break
                    # the projected extent over-estimates the visible size a little (gaussian tails): measure the
                    # rendered beetle and re-render with the frame scaled so the longest side hits target_px
                    long_now = max(crop_to_alpha(rgb, a, pad=0)[1].shape)
                    if abs(long_now / target_px - 1) <= 0.02 or long_now < 4:
                        break
                    out_size = max(8, int(round(out_size * target_px / long_now)))
                    ss_eff = max(ss, int(np.ceil(min_render / out_size)))
                    rsize_i = out_size * ss_eff
                    cam = resized(cam0, rsize_i, rsize_i)
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
            rec = {"index": i, "file": fn, "split": "val" if i < n_val else "train", "size_mode": size_mode,
                   "render_size": out_size, "supersample": ss_eff, "crop_px": [int(a_c.shape[1]), int(a_c.shape[0])],
                   "coverage": round(float((a > 0.5).mean()), 4), "clipped": bool(clipped), **extra, **cam.meta()}
            if size_mode == "object":
                rec["object_size_frac"] = round((max(a_c.shape) - 4) / short_side, 4)   # crop has a 2 px pad
            mf.write(json.dumps(rec) + "\n")
    if verbose:
        print(f"peak RAM {fmt(peak_rss_bytes())}")
        print(f"wrote {num} sprites to {sprites_dir} ({time.time() - t0:.1f}s)"
              + (f"; WARNING {n_clipped} sprites still touch the frame border" if n_clipped else "; none clipped"))
    return sprites_dir


CAMERA_EFFECT_KEYS = ("motion_blur_px", "defocus_sigma", "lowres_factor", "haze", "overexposure", "underexposure",
                      "sensor_noise", "jpeg_quality", "gaussian_blur_sigma")


def _u(rng, r):
    return float(rng.uniform(r[0], r[1])) if isinstance(r, (list, tuple)) else float(r)


def _visible_share(alpha: np.ndarray, covered: np.ndarray, body_px: int | None = None) -> float:
    """Share of a pasted beetle not covered by `covered` (> 0.5). Measured against `body_px` (the whole beetle,
    part outside the frame included) when given, else against its pixels inside the frame (alpha > 0.5)."""
    body = alpha > 0.5
    tot = int(body_px) if body_px else int(body.sum())
    return 1.0 if tot == 0 else float((body & ~(covered > 0.5)).sum()) / tot


def _floor_ok(new_alpha: np.ndarray, new_body_px: int, new_floor: float,
              alphas: list, body_px: list, covered: list, floors: list) -> bool:
    """The visibility rule at paste time: the new beetle keeps >= new_floor of its whole body inside the frame,
    and every earlier beetle keeps >= its floor after the new one is on top (`covered` already includes it)."""
    if _visible_share(new_alpha, np.zeros_like(new_alpha), new_body_px) < new_floor:
        return False
    return all(_visible_share(a, c, b) >= v for a, b, c, v in zip(alphas, body_px, covered, floors))


def _occluder_caps(alphas: list, body_px: list, laters: list, floors: list, max_hidden: float, respect: bool) -> list:
    """Per beetle: the largest share of its still-visible pixels an occluder may hide. With `respect` the beetle
    must keep >= floor of its WHOLE body: visible_after >= floor * body  <=>  hidden/remaining <= 1 - floor*body/rem."""
    caps = []
    for a, b, lt, v in zip(alphas, body_px, laters, floors):
        cap = float(max_hidden)
        if respect:
            rem = max(float(((a > 0.5) & ~lt).sum()), 1.0)
            cap = min(cap, max(0.0, 1.0 - v * float(b) / rem))
        caps.append(cap)
    return caps


def _touches_border(alpha: np.ndarray, thresh: float = 0.03) -> bool:
    return bool(alpha[0].max() > thresh or alpha[-1].max() > thresh or alpha[:, 0].max() > thresh or alpha[:, -1].max() > thresh)


# ----------------------------------------------------------------------------- stage 2
class SpriteBank:
    """Rendered sprites on disk + an LRU cache of decoded sprites (uint8 RGBA, `cache_gb` at most).

    A 2048 px sprite is ~4-16 MB decoded, so 2000 sprites would need tens of GB if all were kept: the cap
    keeps the compose stage inside a fixed RAM budget; sprites outside the cache are re-read from PNG."""

    def __init__(self, sprites_dir: str, cache_gb: float = 2.0):
        self.dir = sprites_dir
        self.cache_bytes = int(cache_gb * GB)
        self._used = 0
        self.hits = self.misses = 0
        self.records = [json.loads(l) for l in open(os.path.join(sprites_dir, "sprites.jsonl"))]
        if not self.records:
            raise RuntimeError(f"no sprites in {sprites_dir}")
        self.by_split = {"train": [r for r in self.records if r["split"] == "train"],
                         "val": [r for r in self.records if r["split"] == "val"]}
        for k, v in self.by_split.items():
            if not v:                       # too few sprites to split: share them
                self.by_split[k] = self.records
        from collections import OrderedDict
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def load(self, rec: dict) -> tuple[np.ndarray, np.ndarray]:
        """float32 (rgb, alpha) of a sprite; fresh arrays, safe to modify."""
        fn = rec["file"]
        rgba8 = self._cache.get(fn)
        if rgba8 is None:
            self.misses += 1
            bgra = cv2.imread(os.path.join(self.dir, fn), cv2.IMREAD_UNCHANGED)
            if bgra is None:
                raise FileNotFoundError(os.path.join(self.dir, fn))
            rgba8 = np.ascontiguousarray(cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA))
            if rgba8.nbytes <= self.cache_bytes:
                while self._cache and self._used + rgba8.nbytes > self.cache_bytes:
                    _, old = self._cache.popitem(last=False)
                    self._used -= old.nbytes
                self._cache[fn] = rgba8
                self._used += rgba8.nbytes
        else:
            self.hits += 1
            self._cache.move_to_end(fn)
        rgba = rgba8.astype(np.float32) / 255
        return np.ascontiguousarray(rgba[..., :3]), np.ascontiguousarray(rgba[..., 3])

    def cache_info(self) -> str:
        return f"sprite cache {len(self._cache)} sprites / {fmt(self._used)} (cap {fmt(self.cache_bytes)}), hits {self.hits}, misses {self.misses}"

    def sample(self, rng: np.random.Generator, split: str) -> dict:
        lst = self.by_split[split]
        return lst[int(rng.integers(len(lst)))]


class _Composer:
    """Everything one process needs to compose images: sprites, backgrounds, occluders and a writer.
    `image(i)` composes image i (seeded by (seed, 11, i), independent of every other image)."""

    def __init__(self, cfg: dict, sprites_dir: str, out_dir: str, meta_name: str = "metadata.jsonl", cache_div: int = 1):
        cc, oc = cfg["compose"], cfg["output"]
        res = cfg.get("resources") or {}
        self.cfg, self.cc = cfg, cc
        self.max_ram = res.get("max_ram_gb")
        self.W, self.H = int(cc["image_size"][0]), int(cc["image_size"][1])
        # with several workers each one gets its share of the cache, at most 2 GB sprites / 0.5 GB occluders: a cache
        # miss is one PNG decode (~20 ms) against ~1 s of compositing, so small per-worker caches cost almost nothing
        sprite_gb = float(res.get("sprite_cache_gb", 2.0))
        occ_gb = float(res.get("occluder_cache_gb", 0.5))
        if cache_div > 1:
            sprite_gb, occ_gb = min(sprite_gb / cache_div, 2.0), min(occ_gb / cache_div, 0.5)
        self.bank = SpriteBank(sprites_dir, cache_gb=sprite_gb)
        self.pool = BackgroundPool(cfg["backgrounds"]["dirs"], min_size=int(cfg["backgrounds"].get("min_size", 64)),
                                   seed=int(cfg.get("seed", 0)))
        self.writer = DatasetWriter(out_dir, oc.get("image_format", "jpg"), oc.get("jpeg_quality", 95),
                                    oc.get("write_masks", True), oc.get("write_yolo", True), meta_name=meta_name)
        self.occ_cfg = cc.get("occlusion") or {}
        self.leaf_bank = OccluderBank(self.occ_cfg.get("dirs"), cache_gb=occ_gb) if self.occ_cfg.get("prob", 0) > 0 else None
        self.box_mode = cc.get("box_mode", "full")
        # one visibility floor (share of the WHOLE beetle, part outside the frame included) against everything:
        # the frame edge, beetles pasted later and leaf / background occluders
        self.visible_floor = cc.get("visible_floor", cc.get("overlap_min_visible", 0.33))
        self.counts = {"train": 0, "val": 0, "popillia": 0, "background": 0, "occluded": 0, "skipped_overlap": 0, "redrawn": 0}

    def describe(self) -> str:
        s = f"{len(self.bank.records)} sprites, {len(self.pool)} backgrounds [{self.pool.describe()}]"
        if self.leaf_bank is not None:
            s += f"; occlusion: prob {self.occ_cfg.get('prob')}, {len(self.leaf_bank)} leaf cutouts, sources {self.occ_cfg.get('sources')}"
        return s

    def image(self, i: int) -> None:
        rng = np.random.default_rng([int(self.cfg["seed"]), 11, i])
        split = "val" if rng.random() < self.cc.get("val_fraction", 0.15) else "train"
        positive = rng.random() < self.cc.get("positive_fraction", 0.5)
        bg, meta = self.pool.sample(rng, self.W, self.H, tuple(self.cc.get("crop_scale", self.cfg["backgrounds"].get("crop_scale", (0.4, 1.0)))),
                               self.cfg["backgrounds"].get("hflip", 0.5))
        bg_photo = bg                                   # un-augmented crop: source for "background" occluders
        # augmentation 1/2: background only
        bg, bam = augment_background(bg, rng, self.cfg.get("background_aug"))
        meta.update(bam)
        img, mask, boxes, objs, obj_alphas = bg, None, [], [], []
        covered_by_later, obj_min_vis, obj_body_px = [], [], []   # per pasted beetle: union of later beetles; floor; whole-body px
        if positive:
            lo, hi = self.cc.get("objects_per_image", [1, 1])
            for _ in range(int(rng.integers(int(lo), int(hi) + 1))):
                rec = self.bank.sample(rng, split)
                s_rgb, s_a = self.bank.load(rec)
                # augmentation 2/2: beetle only
                s_rgb, s_a, sam = augment_sprite(s_rgb, s_a, rng, self.cfg.get("sprite_aug"))
                # size decided at render time (render.size_mode: object): paste the sprite 1:1, rescaled only
                # when the output short side differs from the one the sprites were rendered for (preview profile)
                from_sprite = rec.get("size_mode") == "object" or self.cc.get("object_size") in (None, "from_sprite")
                scale = min(self.W, self.H) / rec.get("for_short_side", min(self.W, self.H)) if from_sprite else None
                # the beetle must keep >= its floor of its whole body inside the frame, and must leave every
                # earlier beetle >= its own floor uncovered; otherwise another position is tried and, failing
                # that, this beetle is skipped
                pm = {}
                v_new = _u(rng, self.visible_floor)
                for _try in range(int(self.cc.get("overlap_tries", 12))):
                    cand, m, pm = paste_sprite(
                        img, s_rgb, s_a, rng, scale=scale,
                        object_size=tuple(self.cc.get("object_size") or (0.1, 0.9)), min_visible=self.cc.get("min_visible", 0.6),
                        color_match=tuple(self.cc.get("color_match", (0, 0))), shadow=self.cc.get("shadow"),
                        edge_blur_px=tuple(self.cc.get("edge_blur_px", (0, 0))), defocus=self.cc.get("defocus"),
                        center_sigma=self.cc.get("center_sigma"), placement=self.cc.get("placement", "uniform"))
                    if not pm.get("bbox"):
                        break
                    covered = [np.maximum(c, m) for c in covered_by_later]
                    if _floor_ok(m, pm["body_px"], v_new, obj_alphas, obj_body_px, covered, obj_min_vis):
                        img = cand
                        covered_by_later = covered + [np.zeros((self.H, self.W), np.float32)]
                        obj_min_vis.append(v_new); obj_body_px.append(int(pm["body_px"]))
                        break
                    pm = {}
                if not pm.get("bbox"):
                    self.counts["skipped_overlap"] += 1
                else:
                    objs.append({"object_id": len(objs), "sprite": rec["file"], "sprite_index": rec["index"],
                                 "elevation_deg": rec["elevation_deg"], "azimuth_deg": rec["azimuth_deg"],
                                 "roll_deg": rec["roll_deg"], "fov_deg": rec["fov_deg"],
                                 "camera_distance": rec.get("distance"), **pm, **sam,
                                 "mask_area_px": int((m > 0.5).sum()), "body_area_px": int(pm["body_px"]),
                                 "visible_floor": round(v_new, 3)})
                    obj_alphas.append(m)
            # leaves / foreground patches in front of the beetles. The occluder targets (and max_hidden)
            # refer to the pixels still visible after beetles pasted later have covered earlier ones.
            laters = []
            for k in range(len(obj_alphas)):
                lt = np.zeros((self.H, self.W), bool)
                for ma in obj_alphas[k + 1:]:
                    lt |= ma > 0.5
                laters.append(lt)
            # an occluder may hide at most occlusion.max_hidden of what is still visible and (respect_floor,
            # default on) must leave the beetle >= its floor of its whole body
            caps = _occluder_caps(obj_alphas, obj_body_px, laters, obj_min_vis,
                                  float(self.occ_cfg.get("max_hidden", 0.85)), bool(self.occ_cfg.get("respect_floor", True)))
            img, visible, occ_meta = occlude_objects(img, [a * ~lt for a, lt in zip(obj_alphas, laters)],
                                                     rng, self.occ_cfg, self.leaf_bank, bg_photo, max_hidden=caps)
            meta["occluders"] = occ_meta
            meta["num_occluders"] = len(occ_meta)
            if occ_meta:
                self.counts["occluded"] += 1
            # how much of each beetle is hidden: by beetles pasted later (`occluded_by_beetle_frac`), by
            # occluders (`occluded_by_occluder_frac`), and in total (`occluded_frac`). The full-body box
            # is kept as `bbox`; `bbox_visible_xyxy` is the tight box around the visible pixels.
            mask = np.zeros((self.H, self.W), np.float32)
            keep_objs, keep_boxes = [], []
            for k, o in enumerate(objs):
                full = obj_alphas[k] > 0.5
                tot = max(1, int(full.sum()))
                later = laters[k]
                vis = visible[k] > 0.5
                o["occluded_by_beetle_frac"] = round(float((full & later).sum() / tot), 3)
                o["occluded_by_occluder_frac"] = round(float((full & ~later & ~vis).sum() / tot), 3)
                o["occluded_frac"] = round(float((full & ~vis).sum() / tot), 3)
                o["visible_area_px"] = int(vis.sum())
                o["visible_share"] = round(int(vis.sum()) / max(int(o.get("body_area_px") or tot), 1), 3)   # of the whole beetle
                ys, xs = np.where(vis)
                o["bbox_visible_xyxy"] = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(ys) else None
                o["bbox_full_xyxy"] = list(o["bbox"])
                if self.box_mode == "visible":
                    if o["bbox_visible_xyxy"] is None:
                        continue
                    o["bbox"] = o["bbox_visible_xyxy"]
                mask = np.maximum(mask, visible[k])
                keep_objs.append(o); keep_boxes.append(o["bbox"])
            objs, boxes = keep_objs, keep_boxes
            for k, o in enumerate(objs):
                o["object_id"] = k
        cam_cfg = self.cfg.get("camera_effects", {})
        img, pmeta = apply_photometric(img, rng, self.cfg.get("lighting"), cam_cfg, cam_cfg.get("guard"))
        meta.update(pmeta)
        meta["effects"] = sorted(k for k in pmeta if k in CAMERA_EFFECT_KEYS)
        if pmeta.get("redrawn"):
            self.counts["redrawn"] += 1
        meta["objects"] = objs
        self.writer.write(f"img_{i:06d}", split, img, mask, boxes, meta)
        self.counts[split] += 1
        self.counts["popillia" if boxes else "background"] += 1

    def close(self, export: bool) -> None:
        self.writer.close(export=export)


def _compose_worker(cfg: dict, sprites_dir: str, out_dir: str, indices: list, k: int, n_workers: int, q) -> None:
    """One worker process: its own caches, its own metadata part file; reports progress through `q`."""
    limit_memory((cfg.get("resources") or {}).get("max_ram_gb"))
    comp = _Composer(cfg, sprites_dir, out_dir, meta_name=f"metadata.part{k}.jsonl", cache_div=n_workers)
    try:
        for n, i in enumerate(indices):
            comp.image(i)
            if n % 5 == 4:
                q.put(("progress", k, 5))
                check_limit(comp.max_ram, f"compose worker {k} image {i}")
        q.put(("progress", k, len(indices) % 5))
        comp.close(export=False)
        q.put(("done", k, comp.counts, comp.bank.cache_info()))
    except BaseException as e:   # noqa: BLE001  (the parent must not hang)
        comp.close(export=False)
        q.put(("error", k, repr(e), None))
        raise


def _resume_plan(out_dir: str, num: int, verbose: bool) -> list[int]:
    """Images still to do after a crash / stop: those without a metadata record whose image file exists.
    metadata.jsonl is rewritten with the valid records only (a truncated last line, records of missing files
    and leftover part files of an interrupted parallel run are folded in or dropped)."""
    meta_path = os.path.join(out_dir, "metadata.jsonl")
    lines = []
    for fn in [meta_path] + sorted(glob.glob(os.path.join(out_dir, "metadata.part*.jsonl"))):
        if os.path.exists(fn):
            with open(fn) as f:
                lines += f.read().splitlines()
    kept, done = {}, set()
    for line in lines:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            idx = int(r["name"].split("_")[-1])
        except (KeyError, ValueError):
            continue
        if idx < num and os.path.exists(os.path.join(out_dir, r["image"])):
            kept[idx] = line
            done.add(idx)
    if lines:
        shutil.copy2(meta_path, meta_path + ".resume.bak") if os.path.exists(meta_path) else None
        with open(meta_path, "w") as f:
            for idx in sorted(kept):
                f.write(kept[idx] + "\n")
        for fn in glob.glob(os.path.join(out_dir, "metadata.part*.jsonl")):
            os.remove(fn)
    todo = [i for i in range(num) if i not in done]
    if verbose:
        print(f"resume: {len(done)} of {num} images already there, {len(todo)} to do")
    return todo


def compose_dataset(cfg: dict, sprites_dir: str, out_dir: str, num: int | None = None,
                    verbose: bool = True, workers: int | None = None, resume: bool = False) -> str:
    """Stage 2. `workers` > 1 composes the images in that many processes (each image is seeded by its index, so
    the result is identical to a single-process run); `resume` skips images that already exist (after a stop)."""
    cc, oc = cfg["compose"], cfg["output"]
    res = cfg.get("resources") or {}
    max_ram = res.get("max_ram_gb")
    if verbose:
        print(limit_memory(max_ram))
    num = num or cc["num_images"]
    W, H = int(cc["image_size"][0]), int(cc["image_size"][1])
    workers = resolve_workers(workers if workers is not None else res.get("workers", 1))
    indices = _resume_plan(out_dir, num, verbose) if resume else list(range(num))
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    counts = {"train": 0, "val": 0, "popillia": 0, "background": 0, "occluded": 0, "skipped_overlap": 0, "redrawn": 0}
    if workers <= 1 or len(indices) < 2 * workers:
        comp = _Composer(cfg, sprites_dir, out_dir)
        if verbose:
            print(f"compose: {len(indices)} images of {W}x{H}, {comp.describe()} -> {out_dir}")
        for n, i in enumerate(tqdm(indices, desc="compose", disable=not verbose)):
            comp.image(i)
            if n % 50 == 0:
                check_limit(max_ram, f"compose image {i}")
        counts = comp.counts
        cache_info = comp.bank.cache_info()
        comp.close(export=True)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        shards = [indices[k::workers] for k in range(workers)]
        procs = [ctx.Process(target=_compose_worker, args=(cfg, sprites_dir, out_dir, sh, k, workers, q), daemon=True)
                 for k, sh in enumerate(shards)]
        if verbose:
            print(f"compose: {len(indices)} images of {W}x{H} in {workers} worker processes "
                  f"(per worker <= {min(float(res.get('sprite_cache_gb', 2.0)) / workers, 2.0):.1f} GB sprite cache + "
                  f"{min(float(res.get('occluder_cache_gb', 0.5)) / workers, 0.5):.1f} GB occluders + ~1-2 GB working set) -> {out_dir}")
        for pr in procs:
            pr.start()
        alive, infos, errors = workers, [], []
        with tqdm(total=len(indices), desc="compose", disable=not verbose) as bar:
            while alive:
                msg = q.get()
                if msg[0] == "progress":
                    bar.update(msg[2])
                elif msg[0] == "done":
                    alive -= 1
                    for k_, v in msg[2].items():
                        counts[k_] += v
                    infos.append(msg[3])
                else:
                    alive -= 1
                    errors.append(msg[2])
        for pr in procs:
            pr.join()
        # merge the part files into metadata.jsonl (resume reads parts too, so a crash here loses nothing)
        with open(os.path.join(out_dir, "metadata.jsonl"), "a") as out:
            for k in range(workers):
                part = os.path.join(out_dir, f"metadata.part{k}.jsonl")
                if os.path.exists(part):
                    with open(part) as f:
                        shutil.copyfileobj(f, out)
                    os.remove(part)
        if errors:
            raise RuntimeError(f"{len(errors)} compose worker(s) failed: {errors[0]}  (re-run with --resume)")
        from .labels import export_tables
        export_tables(out_dir)
        cache_info = "; ".join(infos)
    if verbose:
        print(f"done in {time.time() - t0:.1f}s: {counts}")
        print(f"{cache_info}; peak RAM {fmt(peak_rss_bytes())}")
        print(f"labels: {out_dir}/dataset.json (COCO + metadata), images.csv, objects.csv, summary.json")
    return out_dir


def resolve_workers(w) -> int:
    """resources.workers: an int, or 'auto' = half of the cores (at most 16)."""
    if w in (None, 1, "1", ""):
        return 1
    if isinstance(w, str) and w.lower() == "auto":
        return max(1, min((os.cpu_count() or 2) // 2, 16))
    return max(1, int(w))


def generate(cfg: dict, out_dir: str | None = None, num_sprites: int | None = None,
             num_images: int | None = None, verbose: bool = True, workers: int | None = None) -> str:
    out_dir = out_dir or cfg["output"]["dir"]
    sprites_dir = render_sprites(cfg, out_dir, num_sprites, verbose)
    return compose_dataset(cfg, sprites_dir, out_dir, num_images, verbose, workers=workers)
