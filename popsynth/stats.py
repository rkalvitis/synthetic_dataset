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
    rows = {"size_frac": [], "long_px": [], "area_frac": [], "cx": [], "cy": [], "occluded": [], "split": []}
    for r in recs:
        short = max(1, min(r["width"], r["height"]))
        for o in r.get("objects", []):
            x0, y0, x1, y1 = o["bbox_xyxy"]
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
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result
