"""Statistics of the rendered sprites themselves (the beetle views rendered from the gaussian splat), independent
of the backgrounds: how big each rendered beetle is, how it relates to the size it is pasted at, how often each
sprite is re-used, which viewpoints were rendered and how solid the alpha is.

Reads <root>/sprites/sprites.jsonl (written by the render step), the sprite PNGs (alpha only, cached in
<root>/stats/sprite_alpha.npz) and metadata.jsonl (for the sizes the sprites were pasted at)."""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from .labels import read_metadata
from .stats import BLUE, INK_MUTED, _hist, _mpl, _save_singles, _slug

BLUE_LIGHT = "#86aae9"


def _grouped_pct(ax, groups, edges, colors, labels):
    """Side-by-side bars per bin, each series as % of its own count (series of different size stay comparable)."""
    centres = (edges[:-1] + edges[1:]) / 2
    width = (edges[1] - edges[0]) / (len(groups) + 0.6)
    for k, (vals, col, lab) in enumerate(zip(groups, colors, labels)):
        h, _ = np.histogram(vals, bins=edges)
        h = 100.0 * h / max(len(vals), 1)
        ax.bar(centres + (k - (len(groups) - 1) / 2) * width, h, width=width, color=col, edgecolor="white", linewidth=0.5, label=lab)
    ax.set_xlim(edges[0], edges[-1])


def read_sprite_manifest(root: str) -> list[dict]:
    path = os.path.join(root, "sprites", "sprites.jsonl")
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _alpha_stats(path: str) -> tuple[float, float, float]:
    """(solid px = alpha > 0.5, soft px = 0.05 < alpha <= 0.5, mean alpha inside the beetle) of one sprite."""
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None or im.ndim != 3 or im.shape[2] != 4:
        return float("nan"), float("nan"), float("nan")
    a = im[..., 3].astype(np.float32) / 255.0
    inside = a > 0.05
    return float((a > 0.5).sum()), float((inside & (a <= 0.5)).sum()), float(a[inside].mean() if inside.any() else 0)


def sprite_alpha_table(root: str, files: list[str], refresh: bool = False, workers: int | None = None) -> dict:
    cache = os.path.join(root, "stats", "sprite_alpha.npz")
    if not refresh and os.path.exists(cache):
        z = np.load(cache)
        if len(z["solid_px"]) == len(files):
            return {k: z[k] for k in z.files}
    paths = [os.path.join(root, "sprites", f) for f in files]
    with ThreadPoolExecutor(max_workers=workers or min(8, os.cpu_count() or 1)) as ex:
        rows = list(ex.map(_alpha_stats, paths))
    out = {"solid_px": np.array([r[0] for r in rows]), "soft_px": np.array([r[1] for r in rows]),
           "mean_alpha": np.array([r[2] for r in rows])}
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


def sprite_table(root: str, alpha: bool = True, refresh: bool = False) -> dict:
    """Flat arrays over the rendered sprites of a dataset, plus per-use arrays from metadata.jsonl."""
    man = read_sprite_manifest(root)
    t = {
        "index": np.array([m["index"] for m in man]),
        "w_px": np.array([m["crop_px"][0] for m in man], float),
        "h_px": np.array([m["crop_px"][1] for m in man], float),
        "render_size": np.array([m["render_size"] for m in man], float),
        "supersample": np.array([m.get("supersample", 1) for m in man], float),
        "coverage": np.array([m.get("coverage", np.nan) for m in man], float),
        "fill": np.array([m.get("fill", np.nan) for m in man], float),
        "elevation": np.array([m["elevation_deg"] for m in man], float),
        "azimuth": np.array([m["azimuth_deg"] for m in man], float),
        "roll": np.array([m["roll_deg"] for m in man], float),
        "fov": np.array([m["fov_deg"] for m in man], float),
        "distance": np.array([m.get("distance", np.nan) for m in man], float),
        "clipped": np.array([bool(m.get("clipped", False)) for m in man]),
        "split": np.array([m.get("split", "train") for m in man]),
        "size_mode": man[0].get("size_mode", "fixed") if man else "fixed",
    }
    t["long_px"] = np.maximum(t["w_px"], t["h_px"])
    t["aspect"] = t["w_px"] / np.maximum(t["h_px"], 1)
    # samples the renderer produced for the beetle's longest side (before the area downsample to the sprite)
    t["render_samples_px"] = t["long_px"] * t["supersample"]
    if alpha:
        t.update(sprite_alpha_table(root, [m["file"] for m in man], refresh=refresh))
        t["solid_frac"] = t["solid_px"] / np.maximum(t["solid_px"] + t["soft_px"], 1)
    # how the sprites were used
    recs = read_metadata(root)
    uses = np.zeros(len(man), int)
    by_index = {int(i): k for k, i in enumerate(t["index"])}
    pasted_long, sprite_long, sprite_ss, k_of_use = [], [], [], []
    for r in recs:
        for o in r.get("objects", []):
            k = by_index.get(int(o.get("sprite_index", -1)))
            if k is None:
                continue
            uses[k] += 1
            op = o.get("object_px")
            if op:
                pasted_long.append(float(max(op))); sprite_long.append(float(t["long_px"][k]))
                sprite_ss.append(float(t["supersample"][k])); k_of_use.append(k)
    t["uses"] = uses
    t["use_pasted_long_px"] = np.array(pasted_long)
    t["use_sprite_long_px"] = np.array(sprite_long)
    t["use_scale"] = t["use_pasted_long_px"] / np.maximum(t["use_sprite_long_px"], 1)       # pasted / rendered
    t["use_oversampling"] = t["use_sprite_long_px"] * np.array(sprite_ss) / np.maximum(t["use_pasted_long_px"], 1)
    t["num_images"] = len(recs)
    return t


def _q(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if not len(v):
        return {}
    return {"min": float(v.min()), "p5": float(np.percentile(v, 5)), "median": float(np.median(v)),
            "mean": float(v.mean()), "p95": float(np.percentile(v, 95)), "max": float(v.max())}


def sprite_summary(t: dict) -> dict:
    sm = {"num_sprites": int(len(t["index"])), "num_images": int(t["num_images"]),
          "size_mode": t["size_mode"], "clipped_frac": float(t["clipped"].mean()) if len(t["clipped"]) else 0.0,
          "unused_sprites": int((t["uses"] == 0).sum())}
    for k in ("long_px", "aspect", "render_samples_px", "supersample", "elevation", "azimuth", "roll", "fov",
              "distance", "uses", "use_pasted_long_px", "use_scale", "use_oversampling", "solid_px", "solid_frac",
              "mean_alpha"):
        if k in t:
            sm[k] = _q(t[k])
    return sm


def sprite_panels(label: str, t: dict, bins: int = 30):
    """(name, draw) panels; every one is about the rendered sprite, not the background."""
    tag = f"{label}: " if label else ""
    n = len(t["index"])
    ns = f"(n = {n} sprites)"

    def p_size(fig, ax):
        _hist(ax, t["long_px"], bins, f"{tag}rendered sprite size, longest side of the alpha crop {ns}", "px")

    def p_size_vs_pasted(fig, ax):
        hi = max(t["long_px"].max(), t["use_pasted_long_px"].max() if len(t["use_pasted_long_px"]) else 0)
        edges = np.linspace(0, np.ceil(hi / 50) * 50, bins + 1)
        _grouped_pct(ax, [t["long_px"], t["use_pasted_long_px"]], edges, [BLUE, BLUE_LIGHT],
                      [f"as rendered (per sprite, n = {n})", f"as pasted into the image (per use, n = {len(t['use_pasted_long_px'])})"])
        ax.set_xlabel("longest side, px"); ax.set_ylabel("% of sprites / uses")
        ax.set_title(f"{tag}sprite size as rendered vs as pasted", loc="left", fontsize=9.5)
        ax.legend(fontsize=8, frameon=False)

    def p_scale(fig, ax):
        _hist(ax, t["use_scale"], bins, f"{tag}resampling when pasting: pasted size / rendered size (per use)", "scale factor (1 = pasted 1:1)")
        ax.axvline(1.0, color=INK_MUTED, linewidth=0.8)

    def p_oversampling(fig, ax):
        v = t["use_oversampling"]
        _hist(ax, v, bins, f"{tag}render samples per final beetle pixel (longest side)", "samples / pixel")

    def p_reuse(fig, ax):
        u = t["uses"]
        edges = np.arange(-0.5, u.max() + 1.5)
        ax.hist(u, bins=edges, color=BLUE, edgecolor="white", linewidth=0.8)
        ax.set_xlabel("images a sprite appears in"); ax.set_ylabel("sprites")
        ax.set_title(f"{tag}sprite re-use {ns}\nmean {u.mean():.1f} uses, unused {(u == 0).sum()}", loc="left", fontsize=9.5)

    def p_aspect(fig, ax):
        _hist(ax, t["aspect"], bins, f"{tag}sprite crop aspect, width / height {ns}", "w / h")

    def p_elev(fig, ax):
        _hist(ax, t["elevation"], bins, f"{tag}camera elevation {ns}", "degrees (0 = side view, 90 = top-down)")

    def p_azim(fig, ax):
        _hist(ax, t["azimuth"], bins, f"{tag}camera azimuth {ns}", "degrees")

    def p_roll(fig, ax):
        _hist(ax, t["roll"], bins, f"{tag}in-plane roll {ns}", "degrees")

    def p_fov(fig, ax):
        _hist(ax, t["fov"], bins, f"{tag}field of view {ns}", "degrees")

    def p_solid(fig, ax):
        _hist(ax, t["solid_frac"], bins, f"{tag}alpha solidity: share of beetle pixels with alpha > 0.5 {ns}", "solid share")

    def p_mean_alpha(fig, ax):
        _hist(ax, t["mean_alpha"], bins, f"{tag}mean alpha inside the beetle {ns}", "alpha")

    def p_area(fig, ax):
        _hist(ax, t["solid_px"] / 1e3, bins, f"{tag}beetle pixel area in the sprite, alpha > 0.5 {ns}", "thousand px")

    panels = [("size_px", p_size), ("size_vs_pasted", p_size_vs_pasted), ("resample_scale", p_scale),
              ("oversampling", p_oversampling), ("reuse", p_reuse), ("aspect", p_aspect),
              ("elevation", p_elev), ("azimuth", p_azim), ("roll", p_roll), ("fov", p_fov)]
    if "solid_frac" in t:
        panels += [("alpha_area", p_area), ("alpha_solid", p_solid), ("alpha_mean", p_mean_alpha)]
    return panels, sprite_summary(t)


def compare_panels(datasets: list[tuple[str, dict]], bins: int = 30):
    """Panels with all datasets on one axis (grouped bars, % of sprites)."""
    cols = [BLUE, BLUE_LIGHT, "#1d3a79", "#86aae9"][:len(datasets)]
    labels = [l for l, _ in datasets]

    def grouped(key, title, xlabel, lo=None, hi=None):
        def draw(fig, ax):
            vals = [t[key] for _, t in datasets]
            a = min(v.min() for v in vals) if lo is None else lo
            b = max(v.max() for v in vals) if hi is None else hi
            _grouped_pct(ax, vals, np.linspace(a, b, bins + 1), cols, labels)
            ax.set_xlabel(xlabel); ax.set_ylabel("% of sprites"); ax.set_title(title, loc="left", fontsize=9.5)
            ax.legend(fontsize=8, frameon=False)
        return draw

    return [("size_px", grouped("long_px", "rendered sprite size, longest side of the alpha crop", "px", 0)),
            ("elevation", grouped("elevation", "camera elevation", "degrees", 0, 90)),
            ("fov", grouped("fov", "field of view", "degrees")),
            ("aspect", grouped("aspect", "sprite crop aspect, width / height", "w / h"))] + (
            [("alpha_solid", grouped("solid_frac", "alpha solidity: share of beetle pixels with alpha > 0.5", "solid share", 0, 1))]
            if all("solid_frac" in t for _, t in datasets) else [])


def plot_sprite_compare(datasets: list[tuple[str, dict]], path: str, bins: int = 30,
                        separate_dir: str | None = None) -> dict:
    plt, _ = _mpl()
    rows = [(label,) + sprite_panels(label, t, bins) for label, t in datasets]
    ncol = max(len(p) for _, p, _ in rows)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(4.6 * ncol, 3.6 * len(rows)), squeeze=False)
    res = {}
    for axrow, (label, panels, sm) in zip(axes, rows):
        for ax, (_, draw) in zip(axrow, panels):
            draw(fig, ax)
        for ax in axrow[len(panels):]:
            ax.axis("off")
        if separate_dir:
            sm["files"] = _save_singles(panels, separate_dir, f"sprite_{_slug(label)}")
        res[label] = sm
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    if separate_dir and len(datasets) > 1:
        res["_compare_files"] = _save_singles(compare_panels(datasets, bins), separate_dir, "sprite_compare")
    return res
