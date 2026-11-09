"""Dataset statistics plots: beetle size distribution, where beetles sit in the frame, per-pixel density.

    popsynth stats -o outputs/dataset            ->  <out>/stats/*.png + stats.json

Reads metadata.jsonl (and the mask PNGs for the pixel density), nothing is regenerated.
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

from .labels import read_metadata

# single-hue palette: bars / lines in one mid blue, magnitude maps light -> dark of the same hue
INK, INK_MUTED, GRID = "#1f2933", "#6b7280", "#e5e7eb"
BLUE = "#3b6fd4"
BLUE_RAMP = ["#f3f7fe", "#d9e5fa", "#b4cbf3", "#86aae9", "#5a88de", "#3b6fd4", "#2a52a8", "#1d3a79"]


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK,
                         "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "axes.titlecolor": INK,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})
    return plt, LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)


def _hist(ax, values, bins, title, xlabel, quantiles=True):
    ax.hist(values, bins=bins, color=BLUE, edgecolor="white", linewidth=0.8)
    ax.set_xlabel(xlabel); ax.set_ylabel("beetles")
    if quantiles and len(values):
        q = np.percentile(values, [5, 50, 95])
        ax.axvline(q[1], color=INK, linewidth=1.2)
        title = f"{title}\nmedian {q[1]:.3g}   (5-95%: {q[0]:.3g} - {q[2]:.3g})"
    ax.set_title(title, loc="left", fontsize=9.5)


def object_table(recs: list[dict]) -> dict[str, np.ndarray]:
    """Flat arrays over all beetles of a dataset."""
    rows = {"size_frac": [], "long_px": [], "area_frac": [], "cx": [], "cy": [], "occluded": [], "split": [],
            "by_beetle": [], "by_occluder": [], "in_frame": []}
    for r in recs:
        short = max(1, min(r["width"], r["height"]))
        for o in r.get("objects", []):
            x0, y0, x1, y1 = o["bbox_xyxy"]
            occ = float(o.get("occluded_frac", 0.0))
            rows["by_beetle"].append(float(o.get("occluded_by_beetle_frac", occ)))   # older runs: only beetles overlap
            rows["by_occluder"].append(float(o.get("occluded_by_occluder_frac", 0.0)))
            rows["in_frame"].append(float(o.get("body_in_frame_frac", o.get("visible_frac", 1.0))))
            if o.get("body_area_px") and o.get("visible_area_px") is not None:      # whole-body share (frame cut included)
                occ = 1.0 - float(o["visible_area_px"]) / max(int(o["body_area_px"]), 1)
            rows["size_frac"].append(o.get("object_size_frac", max(x1 - x0, y1 - y0) / short))
            rows["long_px"].append(max(x1 - x0, y1 - y0))
            rows["area_frac"].append(o.get("bbox_area_frac", (x1 - x0) * (y1 - y0) / (r["width"] * r["height"])))
            rows["cx"].append((x0 + x1) / 2 / r["width"]); rows["cy"].append((y0 + y1) / 2 / r["height"])
            rows["occluded"].append(o.get("occluded_frac", 0.0)); rows["split"].append(r["split"])
    return {k: np.asarray(v) for k, v in rows.items()}


def plot_size_distribution(t: dict, path: str, bins: int = 30) -> dict:
    plt, _ = _mpl()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    _hist(axes[0], t["size_frac"], bins, "Beetle size (longest side / short image side)", "fraction of image side")
    _hist(axes[1], t["long_px"], bins, "Beetle size in pixels (longest box side)", "px")
    _hist(axes[2], t["area_frac"], bins, "Box area / image area", "fraction")
    fig.suptitle(f"Beetle size distribution, n = {len(t['size_frac'])}", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    out = {}
    for k in ("size_frac", "long_px", "area_frac"):
        v = t[k]
        out[k] = {"min": float(v.min()), "p5": float(np.percentile(v, 5)), "median": float(np.median(v)),
                  "mean": float(v.mean()), "p95": float(np.percentile(v, 95)), "max": float(v.max())} if len(v) else {}
    return out


def plot_center_heatmap(t: dict, path: str, grid: int = 12) -> dict:
    """Where the beetle centres fall: 2-D histogram over the frame (normalised coordinates) and a
    summary of how often the beetle is in the middle of the picture."""
    plt, cmap = _mpl()
    cx, cy = t["cx"], t["cy"]
    H, xe, ye = np.histogram2d(cy, cx, bins=grid, range=[[0, 1], [0, 1]])
    n = max(1, len(cx))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
    im = axes[0].imshow(H / n * 100, cmap=cmap, extent=[0, 1, 1, 0], aspect="equal", interpolation="nearest")
    axes[0].grid(False)
    axes[0].set_title(f"Beetle centre position, {grid}x{grid} cells (% of beetles)", loc="left", fontsize=10)
    axes[0].set_xlabel("x (0 = left, 1 = right)"); axes[0].set_ylabel("y (0 = top, 1 = bottom)")
    for f, ls in ((1 / 3, "-"), (1 / 2, "--")):
        axes[0].add_patch(plt.Rectangle((0.5 - f / 2, 0.5 - f / 2), f, f, fill=False, edgecolor=INK, linewidth=1, linestyle=ls))
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.03, label="% of beetles")
    # radial profile: share of beetles vs distance from centre
    r = np.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2) if len(cx) else np.zeros(0)
    edges = np.linspace(0, 0.75, 16)
    h, _ = np.histogram(r, bins=edges)
    axes[1].bar(edges[:-1], h / n * 100, width=np.diff(edges), align="edge", color=BLUE, edgecolor="white")
    axes[1].set_title("Distance of the centre from the image centre", loc="left", fontsize=10)
    axes[1].set_xlabel("distance (fraction of image side; 0.5 = at an edge midpoint)"); axes[1].set_ylabel("% of beetles")
    in_third = float(((abs(cx - 0.5) < 1 / 6) & (abs(cy - 0.5) < 1 / 6)).mean()) if len(cx) else 0.0
    in_half = float(((abs(cx - 0.5) < 0.25) & (abs(cy - 0.5) < 0.25)).mean()) if len(cx) else 0.0
    in_circle = float((r < 0.25).mean()) if len(cx) else 0.0
    axes[1].text(0.99, 0.97, f"centre in middle third: {in_third:.0%}\ncentre in middle half: {in_half:.0%}\n"
                 f"within r < 0.25: {in_circle:.0%}", transform=axes[1].transAxes, va="top", ha="right", color=INK, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return {"center_in_middle_third": in_third, "center_in_middle_half": in_half, "center_within_r025": in_circle,
            "grid": grid, "cell_percent": (H / n * 100).round(2).tolist()}


def pixel_density(root: str, recs: list[dict], size: int = 256) -> tuple[np.ndarray, np.ndarray, int]:
    """Per-pixel probability that the pixel belongs to a beetle, over all positive images (from the mask
    PNGs; the box rectangles are used for images without a mask). Returns (mask density, box density, n)."""
    acc_m = np.zeros((size, size), np.float64)
    acc_b = np.zeros((size, size), np.float64)
    n = 0
    for r in recs:
        if not r.get("num_objects"):
            continue
        n += 1
        W, H = r["width"], r["height"]
        bm = np.zeros((size, size), np.float64)          # per-image box mask (overlapping boxes count once)
        for x0, y0, x1, y1 in r.get("boxes_xyxy") or []:
            bm[int(y0 / H * size):max(int(y0 / H * size) + 1, int(np.ceil(y1 / H * size))),
               int(x0 / W * size):max(int(x0 / W * size) + 1, int(np.ceil(x1 / W * size)))] = 1
        acc_b += bm
        mf = r.get("mask_file")
        m = cv2.imread(os.path.join(root, mf), cv2.IMREAD_GRAYSCALE) if mf else None
        acc_m += cv2.resize(m, (size, size), interpolation=cv2.INTER_AREA) / 255.0 if m is not None else bm
    n = max(1, n)
    return acc_m / n, acc_b / n, n


def plot_pixel_density(dm: np.ndarray, db: np.ndarray, n: int, path: str) -> dict:
    plt, cmap = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, d, title in ((axes[0], dm, "Beetle pixels (from masks)"), (axes[1], db, "Beetle boxes")):
        im = ax.imshow(d * 100, cmap=cmap, extent=[0, 1, 1, 0], aspect="equal", interpolation="nearest")
        ax.grid(False); ax.set_title(f"{title}: % of positive images covering each pixel", loc="left", fontsize=10)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="% of images")
    fig.suptitle(f"Per-pixel beetle density over {n} positive images", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    S = dm.shape[0]; c = slice(S // 3, 2 * S // 3)
    return {"mask_density_mean": float(dm.mean()), "mask_density_max": float(dm.max()),
            "mask_density_center_third_mean": float(dm[c, c].mean()),
            "box_density_mean": float(db.mean()), "box_density_max": float(db.max()),
            "box_density_center_third_mean": float(db[c, c].mean())}


def plot_occlusion(t: dict, recs: list[dict], path: str) -> dict:
    plt, _ = _mpl()
    occ = t["occluded"]
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    _hist(ax, occ, np.linspace(0, 1, 21), "Hidden fraction of each beetle (leaf / foreground / other beetle)",
          "occluded fraction", quantiles=False)
    frac = float((occ > 0.001).mean()) if len(occ) else 0.0
    ax.text(0.99, 0.97, f"partly hidden: {frac:.0%} of beetles", transform=ax.transAxes, va="top", ha="right", color=INK, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    kinds = {}
    for r in recs:
        for o in r.get("occluders") or []:
            kinds[o.get("occluder")] = kinds.get(o.get("occluder"), 0) + 1
    return {"beetles_partly_hidden_frac": frac, "mean_occluded_frac": float(occ.mean()) if len(occ) else 0.0,
            "occluders_by_source": kinds}


def _grouped_bars(ax, groups, bins, colors, labels):
    """Side-by-side bars per bin (not stacked, so every series is readable)."""
    centres = (bins[:-1] + bins[1:]) / 2
    width = (bins[1] - bins[0]) / (len(groups) + 0.6)
    for k, (vals, col, lab) in enumerate(zip(groups, colors, labels)):
        h, _ = np.histogram(vals, bins=bins)
        ax.bar(centres + (k - (len(groups) - 1) / 2) * width, h, width=width, color=col, edgecolor="white", linewidth=0.5, label=lab)
    ax.set_xlim(bins[0], bins[-1])


def visibility_summary(t: dict, min_visible: float = 0.33) -> dict:
    vis = 1.0 - t["occluded"]                          # visible share (of the whole beetle where recorded)
    covered = (t["by_beetle"] > 0.001) | (t["by_occluder"] > 0.001)
    out = {"min_visible": min_visible, "beetles": int(len(vis)),
           "covered_frac": float(covered.mean()) if len(vis) else 0.0,
           "below_floor_frac": float((vis < min_visible).mean()) if len(vis) else 0.0,
           "invisible_frac": float((vis < 0.05).mean()) if len(vis) else 0.0,
           "covered_by_beetle_frac": float((t["by_beetle"] > 0.001).mean()) if len(vis) else 0.0,
           "covered_by_occluder_frac": float((t["by_occluder"] > 0.001).mean()) if len(vis) else 0.0,
           "cut_by_frame_frac": float((t["in_frame"] < 0.999).mean()) if len(vis) else 0.0}
    if covered.any():
        q = np.percentile(vis[covered], [5, 50, 95])
        out["covered_visible_median"] = float(q[1]); out["covered_visible_p5"] = float(q[0]); out["covered_visible_p95"] = float(q[2])
    return out


def size_panels(label: str, t: dict, bins: int = 30):
    """Sprite-size panels: how big the pasted beetle is in the image (longest box side as a share of the
    short image side, in pixels, and box area / image area). Returns (panels, summary)."""
    tag = f"{label}: " if label else ""
    n = len(t["size_frac"])

    def p_frac(fig, ax):
        _hist(ax, t["size_frac"], bins, f"{tag}beetle size, longest side / short image side (n = {n})", "fraction of image side")

    def p_px(fig, ax):
        _hist(ax, t["long_px"], bins, f"{tag}beetle size in pixels, longest box side (n = {n})", "px")

    def p_area(fig, ax):
        _hist(ax, t["area_frac"], bins, f"{tag}box area / image area (n = {n})", "fraction of the image")

    sm = {}
    for k in ("size_frac", "long_px", "area_frac"):
        v = t[k]
        sm[k] = {"min": float(v.min()), "p5": float(np.percentile(v, 5)), "median": float(np.median(v)),
                 "mean": float(v.mean()), "p95": float(np.percentile(v, 95)), "max": float(v.max())} if len(v) else {}
    return [("size_frac", p_frac), ("size_px", p_px), ("size_area", p_area)], sm


def plot_size_compare(datasets: list[tuple[str, dict]], path: str, bins: int = 30,
                      separate_dir: str | None = None) -> dict:
    """One row of size panels per dataset; with `separate_dir` every panel is also its own PNG
    (<separate_dir>/size_<label>_<panel>.png)."""
    plt, _ = _mpl()
    n = len(datasets)
    fig, axes = plt.subplots(n, 3, figsize=(15, 3.6 * n), squeeze=False)
    res = {}
    for row, (label, t) in zip(axes, datasets):
        panels, sm = size_panels(label, t, bins)
        for ax, (_, draw) in zip(row, panels):
            draw(fig, ax)
        if separate_dir:
            sm["files"] = _save_singles(panels, separate_dir, f"size_{_slug(label)}")
        res[label] = sm
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return res


def _slug(label: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def _save_singles(panels, out_dir: str, prefix: str, figsize=(6.2, 4.2)) -> list[str]:
    """Draw each (name, draw(fig, ax)) panel on its own figure -> <out_dir>/<prefix>_<name>.png."""
    plt, _ = _mpl()
    os.makedirs(out_dir, exist_ok=True)
    files = []
    for name, draw in panels:
        fig, ax = plt.subplots(figsize=figsize)
        draw(fig, ax)
        fig.tight_layout()
        fp = os.path.join(out_dir, f"{prefix}_{name}.png")
        fig.savefig(fp, dpi=150); plt.close(fig); files.append(fp)
    return files


def _floor_line(ax, min_visible):
    ax.axvline(min_visible, color=INK, linewidth=1.2, linestyle="--")
    ax.text(min_visible + 0.01, 0.6, f"{min_visible:.0%} floor", transform=ax.get_xaxis_transform(), fontsize=8, color=INK)


def visibility_panels(label: str, t: dict, min_visible: float = 0.33, percent: bool = True):
    """The three visibility panels of one dataset as (name, draw(fig, ax)) pairs."""
    vis = 1.0 - t["occluded"]; covered = (t["by_beetle"] > 0.001) | (t["by_occluder"] > 0.001)
    bins = np.linspace(0, 1, 21)
    sm = visibility_summary(t, min_visible)
    mb = covered & (t["by_beetle"] >= t["by_occluder"]); mo = covered & ~mb
    w = np.full(len(vis), 100.0 / max(len(vis), 1)) if percent else None
    ylab = "% of beetles" if percent else "beetles"

    def p_all(fig, ax):
        ax.hist(vis, bins=bins, weights=w, color=BLUE, edgecolor="white", linewidth=0.8)
        ax.set_title(f"{label}: visible share of the body, all beetles (n = {len(vis)})\n"
                     f"below {min_visible:.0%}: {sm['below_floor_frac']:.2%}   under 5% visible: {sm['invisible_frac']:.2%}", loc="left", fontsize=9.5)
        ax.set_ylabel(ylab); ax.set_xlabel("visible share of the beetle (1 = whole beetle visible)"); _floor_line(ax, min_visible)

    def p_covered(fig, ax):
        _grouped_bars(ax, [vis[mb], vis[mo]], bins, [BLUE, BLUE_RAMP[3]], ["under another beetle", "under a leaf / patch"])
        sub = (f"median visible {sm['covered_visible_median']:.0%}, 5-95%: {sm['covered_visible_p5']:.0%} - {sm['covered_visible_p95']:.0%}"
               if covered.any() else "none covered")
        ax.set_title(f"{label}: covered beetles only (n = {int(covered.sum())}, {sm['covered_frac']:.1%} of all)\n{sub}", loc="left", fontsize=9.5)
        ax.set_ylabel("beetles"); ax.set_xlabel("visible share"); ax.legend(fontsize=8, frameon=False, loc="upper left"); _floor_line(ax, min_visible)

    def p_frame(fig, ax):
        ax.hist(t["in_frame"], bins=bins, weights=w, color=BLUE, edgecolor="white", linewidth=0.8)
        ax.set_title(f"{label}: part of the box inside the frame\ncut by the frame: {sm['cut_by_frame_frac']:.1%}", loc="left", fontsize=9.5)
        ax.set_ylabel(ylab); ax.set_xlabel("box fraction inside the frame")

    return [("visible_all", p_all), ("visible_covered", p_covered), ("inside_frame", p_frame)], sm


def plot_visibility(t: dict, path: str, min_visible: float = 0.33, title: str | None = None,
                    separate_dir: str | None = None) -> dict:
    """How much of each beetle stays visible when something is placed on top of it (3 panels; with
    `separate_dir` each panel is also written as its own PNG)."""
    plt, _ = _mpl()
    panels, sm = visibility_panels(title or "", t, min_visible, percent=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for ax, (_, draw) in zip(axes, panels):
        draw(fig, ax)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    if separate_dir:
        sm["files"] = _save_singles(panels, separate_dir, "visibility")
    return sm


def plot_visibility_compare(datasets: list[tuple[str, dict]], path: str, min_visible: float = 0.33,
                            separate_dir: str | None = None) -> dict:
    """One row per dataset (label, object_table): the three visibility panels. With `separate_dir`, every panel
    is also saved on its own as visibility_<dataset>_<panel>.png."""
    plt, _ = _mpl()
    n = len(datasets)
    fig, axes = plt.subplots(n, 3, figsize=(15, 3.6 * n), squeeze=False)
    out = {}
    for i, (label, t) in enumerate(datasets):
        panels, sm = visibility_panels(label, t, min_visible)
        for ax, (_, draw) in zip(axes[i], panels):
            draw(fig, ax)
        if separate_dir:
            sm["files"] = _save_singles(panels, separate_dir, f"visibility_{_slug(label)}")
        out[label] = sm
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return out


def coverage_summary(dm: np.ndarray) -> dict:
    """How flat the per-pixel coverage is. centre_edge_ratio = mean coverage in the middle third over the mean in
    the outer ring (1.0 = every pixel equally likely to carry a beetle); cv = std / mean over all pixels."""
    S = dm.shape[0]; c = slice(S // 3, 2 * S // 3)
    inner = float(dm[c, c].mean())
    ring = np.ones_like(dm, bool); ring[c, c] = False
    outer = float(dm[ring].mean())
    border = float(np.concatenate([dm[0], dm[-1], dm[:, 0], dm[:, -1]]).mean())
    return {"mean": float(dm.mean()), "max": float(dm.max()), "center_third_mean": inner, "outer_ring_mean": outer,
            "border_mean": border, "center_edge_ratio": inner / max(outer, 1e-9), "cv": float(dm.std() / max(dm.mean(), 1e-9))}


def coverage_panels(label: str, dm: np.ndarray, npos: int, vmax: float | None = None):
    """The three coverage panels of one dataset as (name, draw(fig, ax)) pairs."""
    _, cmap = _mpl()
    sm = coverage_summary(dm)
    S = dm.shape[0]
    vmax = vmax if vmax is not None else float(dm.max()) * 100

    def p_map(fig, ax):
        im = ax.imshow(dm * 100, cmap=cmap, extent=[0, 1, 1, 0], aspect="equal", interpolation="nearest", vmin=0, vmax=vmax)
        ax.grid(False); ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_title(f"{label}: % of positive images with a beetle\non each pixel (n = {npos})   "
                     f"centre third / outer ring = {sm['center_edge_ratio']:.2f}  (1 = unbiased)", loc="left", fontsize=9.5)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="% of images")

    def p_profiles(fig, ax):
        xs = (np.arange(S) + 0.5) / S
        ax.plot(xs, dm.mean(0) * 100, color=BLUE, linewidth=2, label="along x (averaged over y)")
        ax.plot(xs, dm.mean(1) * 100, color=BLUE_RAMP[3], linewidth=2, label="along y (averaged over x)")
        ax.axhline(sm["mean"] * 100, color=INK, linewidth=1, linestyle="--", label="flat reference (mean)")
        ax.set_ylim(0, vmax * 0.75); ax.set_xlim(0, 1); ax.set_xlabel("position (fraction of the side)"); ax.set_ylabel("% of images")
        ax.set_title(f"{label}: coverage across the frame", loc="left", fontsize=9.5); ax.legend(fontsize=8, frameon=False)

    def p_radial(fig, ax):
        yy, xx = np.mgrid[0:S, 0:S]
        rr = np.sqrt(((xx + 0.5) / S - 0.5) ** 2 + ((yy + 0.5) / S - 0.5) ** 2)
        edges = np.linspace(0, 0.5 * np.sqrt(2), 24)
        prof = [float(dm[(rr >= e0) & (rr < e1)].mean()) * 100 for e0, e1 in zip(edges[:-1], edges[1:])]
        ax.bar(edges[:-1], prof, width=np.diff(edges), align="edge", color=BLUE, edgecolor="white", linewidth=0.8)
        ax.axhline(sm["mean"] * 100, color=INK, linewidth=1, linestyle="--")
        ax.axvline(0.5, color=INK_MUTED, linewidth=0.8); ax.text(0.505, 0.95, "edge midpoint", transform=ax.get_xaxis_transform(), fontsize=8, color=INK_MUTED, va="top")
        ax.set_xlabel("distance from the image centre (fraction of the side)"); ax.set_ylabel("% of images")
        ax.set_title(f"{label}: coverage by distance from the centre\ncv over pixels = {sm['cv']:.2f}   border mean {sm['border_mean'] * 100:.1f}%", loc="left", fontsize=9.5)

    return [("coverage_map", p_map), ("coverage_profiles", p_profiles), ("coverage_radial", p_radial)], sm


def plot_coverage_compare(datasets: list[tuple[str, np.ndarray, np.ndarray, int]], path: str,
                          separate_dir: str | None = None) -> dict:
    """One row per dataset (label, mask density, box density, n positive images): per-pixel coverage map,
    coverage along x and y, radial profile. A flat map / flat profiles mean every pixel is equally likely to
    carry a beetle. With `separate_dir`, every panel is also saved on its own."""
    plt, _ = _mpl()
    n = len(datasets)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4.3 * n), squeeze=False)
    out = {}
    vmax = max(float(d[1].max()) for d in datasets) * 100
    for i, (label, dm, db, npos) in enumerate(datasets):
        panels, sm = coverage_panels(label, dm, npos, vmax)
        for ax, (_, draw) in zip(axes[i], panels):
            draw(fig, ax)
        if separate_dir:
            sm["files"] = _save_singles(panels, separate_dir, _slug(label), figsize=(7.0, 5.2))
        out[label] = sm
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return out


def box_density_by_size(recs: list[dict], size: int = 128, classes=((0.0, 0.3), (0.3, 0.55), (0.55, 2.0)),
                        normalize: bool = True) -> tuple[list[np.ndarray], np.ndarray, list[int]]:
    """Per-pixel coverage from the boxes, one map per beetle-size class plus one over all beetles, where every
    beetle contributes the same total weight (its box is weighted by 1 / its area) when `normalize` is set.
    Each map is scaled to a mean of 1, so 1.0 everywhere = no position bias. Returns (class maps, all map, counts)."""
    maps = [np.zeros((size, size), np.float64) for _ in classes]
    counts = [0] * len(classes)
    for r in recs:
        W, H = r["width"], r["height"]
        short = max(1, min(W, H))
        for o in r.get("objects", []):
            x0, y0, x1, y1 = o["bbox_xyxy"]
            sz = o.get("object_size_frac", max(x1 - x0, y1 - y0) / short)
            k = next((i for i, (lo, hi) in enumerate(classes) if lo <= sz < hi), None)
            if k is None:
                continue
            gx0, gx1 = int(x0 / W * size), max(int(x0 / W * size) + 1, int(np.ceil(x1 / W * size)))
            gy0, gy1 = int(y0 / H * size), max(int(y0 / H * size) + 1, int(np.ceil(y1 / H * size)))
            w = 1.0 / ((gx1 - gx0) * (gy1 - gy0)) if normalize else 1.0
            maps[k][gy0:gy1, gx0:gx1] += w
            counts[k] += 1
    total = sum(maps)
    norm = [m / max(m.mean(), 1e-12) for m in maps]
    return norm, total / max(total.mean(), 1e-12), counts


def plot_coverage_by_size(datasets: list[tuple[str, list[dict]]], path: str, size: int = 128,
                          classes=((0.0, 0.3), (0.3, 0.55), (0.55, 2.0)), separate_dir: str | None = None) -> dict:
    """One row per dataset: coverage with every beetle weighted equally (all sizes), then one map per size class.
    Maps are relative to their own mean (1.0 = flat). With `separate_dir`, every map is also saved on its own."""
    plt, cmap = _mpl()
    n = len(datasets); m = len(classes) + 1
    fig, axes = plt.subplots(n, m, figsize=(4.4 * m, 4.2 * n), squeeze=False)
    out = {}
    names = [f"size {lo:.2f}-{hi:.2f}" if hi < 2 else f"size > {lo:.2f}" for lo, hi in classes]
    slugs = ["all_sizes"] + [f"size_{lo:.2f}_{hi:.2f}".replace(".", "") if hi < 2 else f"size_over_{lo:.2f}".replace(".", "") for lo, hi in classes]
    for i, (label, recs) in enumerate(datasets):
        cls_maps, all_map, counts = box_density_by_size(recs, size, classes)
        row = {}; panels = []
        for j, (d, title, cnt, slug) in enumerate(zip([all_map] + cls_maps, ["all sizes, each beetle weight 1"] + names, [sum(counts)] + counts, slugs)):
            sm = coverage_summary(d); row[title] = {**sm, "beetles": cnt}

            def draw(fig, ax, d=d, title=title, cnt=cnt, sm=sm, cbar=True):
                im = ax.imshow(d, cmap=cmap, extent=[0, 1, 1, 0], aspect="equal", interpolation="nearest", vmin=0, vmax=2.5)
                ax.grid(False); ax.set_xlabel("x"); ax.set_ylabel("y")
                ax.set_title(f"{label}: {title} (n = {cnt})\ncentre third / outer ring = {sm['center_edge_ratio']:.2f}", loc="left", fontsize=9)
                if cbar:
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="coverage / mean (1 = flat)")
            draw(fig, axes[i][j], cbar=(j == m - 1))
            panels.append((f"by_size_{slug}", draw))
        if separate_dir:
            row["files"] = _save_singles(panels, separate_dir, f"coverage_{_slug(label)}", figsize=(6.4, 5.2))
        out[label] = row
    fig.suptitle("Position bias with the beetle size taken out: each beetle counts once, per size class (box coverage, relative to the mean)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return out


def make_stats(root: str, out_dir: str | None = None, grid: int = 12) -> dict:
    out_dir = out_dir or os.path.join(root, "stats")
    os.makedirs(out_dir, exist_ok=True)
    recs = read_metadata(root)
    t = object_table(recs)
    result = {"num_images": len(recs), "num_beetles": int(len(t["size_frac"])), "files": []}
    if len(t["size_frac"]) == 0:
        print("no beetles in the dataset, nothing to plot")
        return result
    p = os.path.join(out_dir, "size_distribution.png"); result["size"] = plot_size_distribution(t, p); result["files"].append(p)
    p = os.path.join(out_dir, "center_heatmap.png"); result["center"] = plot_center_heatmap(t, p, grid); result["files"].append(p)
    dm, db, n = pixel_density(root, recs)
    p = os.path.join(out_dir, "pixel_density.png"); result["pixel_density"] = plot_pixel_density(dm, db, n, p); result["files"].append(p)
    cv2.imwrite(os.path.join(out_dir, "pixel_density_mask.png"), (np.clip(dm / max(dm.max(), 1e-6), 0, 1) * 255).astype(np.uint8))
    p = os.path.join(out_dir, "occlusion.png"); result["occlusion"] = plot_occlusion(t, recs, p); result["files"].append(p)
    p = os.path.join(out_dir, "visibility.png"); result["visibility"] = plot_visibility(t, p); result["files"].append(p)
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result
