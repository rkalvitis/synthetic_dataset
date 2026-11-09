"""Post-hoc cleanup of a generated dataset (`popsynth clean`).

Two defects of earlier runs are repaired in place, without re-rendering:

1. beetles that are (almost) completely hidden behind another beetle or an occluder but still carry a box.
   A beetle whose visible share (visible_area_px / mask_area_px) is below `drop_below` has no usable pixels:
   its box and label line are removed and the image is kept. A beetle between `drop_below` and `min_visible`
   is too hidden to count as a valid example but still shows: the whole image is quarantined.
2. blank frames (burnt-out white, black, or without contrast) produced by stacked exposure effects:
   the image is quarantined, whatever its labels say.
3. (`box_mode`) the label boxes can be switched between the full body (amodal) and the visible pixels only
   (after other beetles, leaves and background patches are on top), from `bbox_visible_xyxy` / `bbox_full_xyxy`
   stored per beetle; no pixels are read.

Quarantined files are MOVED to <root>/_removed/ (same relative paths), never deleted, and listed in
<root>/_removed/removed.jsonl with the reason; metadata.jsonl is backed up to metadata.jsonl.bak and the tables
(dataset.json, images.csv, objects.csv, summary.json) are rebuilt. Running it twice is a no-op.
"""
from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from .labels import export_tables, read_metadata, xyxy_to_cxcywh_norm

LUM = np.array([0.299, 0.587, 0.114], np.float32)


def image_stats(path: str) -> tuple[float, float]:
    """(mean, std) of the luminance in [0,1]; (nan, nan) if the file cannot be read."""
    img = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_4)
    if img is None:
        return float("nan"), float("nan")
    lum = img.astype(np.float32) @ LUM[::-1] / 255.0      # BGR
    return float(lum.mean()), float(lum.std())


def is_blank(mean: float, std: float, mean_range=(0.05, 0.9), min_std: float = 0.03) -> bool:
    return not (mean_range[0] <= mean <= mean_range[1] and std >= min_std)


def visible_share(o: dict) -> float:
    """Visible share of the beetle: of its whole body (frame cut included) when `body_area_px` is recorded
    (runs from 2026-11-09), else of its pixels inside the frame."""
    tot = max(int(o.get("body_area_px") or o.get("mask_area_px") or 0), 1)
    vis = o.get("visible_area_px")
    if vis is None:                                         # older records: fall back to occluded_frac
        return 1.0 - float(o.get("occluded_frac", 0.0))
    return float(vis) / tot


def _rewrite_label(root: str, rec: dict) -> None:
    if not rec.get("label_file"):
        return
    W, H = rec["width"], rec["height"]
    with open(os.path.join(root, rec["label_file"]), "w") as f:
        for b in rec["boxes_xyxy"]:
            cx, cy, w, h = xyxy_to_cxcywh_norm(b, W, H)
            f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def _quarantine(root: str, rec: dict, dry: bool) -> None:
    for key in ("image", "label_file", "mask_file"):
        rel = rec.get(key)
        if not rel:
            continue
        src = os.path.join(root, rel)
        if not os.path.exists(src):
            continue
        dst = os.path.join(root, "_removed", rel)
        if not dry:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)


def _set_box(o: dict, b: list, W: int, H: int) -> bool:
    """Make `b` the label box of beetle `o`; True if it changed."""
    b = [int(v) for v in b]
    if list(o.get("bbox_xyxy", [])) == b:
        return False
    o["bbox"] = o["bbox_xyxy"] = b
    o["bbox_cxcywh_norm"] = xyxy_to_cxcywh_norm(b, W, H)
    o["bbox_w_px"], o["bbox_h_px"] = b[2] - b[0], b[3] - b[1]
    o["bbox_area_frac"] = round((b[2] - b[0]) * (b[3] - b[1]) / (W * H), 6)
    return True


def clean_dataset(root: str, min_visible: float = 0.2, drop_below: float = 0.05, blank: bool = True,
                  mean_range=(0.05, 0.9), min_std: float = 0.03, workers: int | None = None,
                  dry_run: bool = False, verbose: bool = True, box_mode: str = "keep") -> dict:
    """box_mode: keep = leave the label boxes as generated; visible = tight box around the visible pixels
    (`bbox_visible_xyxy`); full = whole body (`bbox_full_xyxy`, or the generated box when that is missing)."""
    if box_mode not in ("keep", "visible", "full"):
        raise ValueError(f"box_mode must be keep, visible or full, not {box_mode!r}")
    recs = read_metadata(root)
    by_name = {}
    for r in recs:
        by_name[r["name"]] = r
    recs = sorted(by_name.values(), key=lambda r: r["name"])
    removed, kept = [], []
    n_dropped_boxes = n_reboxed = 0

    # --- 1. hidden beetles + box mode (metadata only)
    for r in recs:
        reason = None
        keep_objs = []
        reboxed = False
        for o in r.get("objects", []):
            v = visible_share(o)
            if v < drop_below:
                n_dropped_boxes += 1
                continue
            if v < min_visible:
                reason = f"beetle {o.get('object_id')} only {v:.0%} visible (< {min_visible:.0%})"
            if box_mode != "keep":
                o.setdefault("bbox_full_xyxy", list(o["bbox_xyxy"]) if o.get("occluded_frac", 0) == 0 else None)
                if box_mode == "visible":
                    if o.get("bbox_full_xyxy") is None and list(o.get("bbox_visible_xyxy") or []) != list(o["bbox_xyxy"]):
                        o["bbox_full_xyxy"] = list(o["bbox_xyxy"])      # the generated box was the full one
                    target = o.get("bbox_visible_xyxy")
                else:
                    target = o.get("bbox_full_xyxy")
                if target is None and box_mode == "visible":
                    n_dropped_boxes += 1                                  # nothing visible: no box
                    continue
                if target is not None and _set_box(o, target, r["width"], r["height"]):
                    reboxed = True
            keep_objs.append(o)
        if reason is None and r.get("objects") and not keep_objs:
            reason = "all beetles hidden"
        if reason:
            removed.append((r, reason))
            continue
        if reboxed:
            n_reboxed += 1
        if reboxed or len(keep_objs) != len(r.get("objects", [])):
            for k, o in enumerate(keep_objs):
                o["object_id"] = k
            r["objects"] = keep_objs
            r["boxes_xyxy"] = [o["bbox_xyxy"] for o in keep_objs]
            r["boxes_cxcywh_norm"] = [o["bbox_cxcywh_norm"] for o in keep_objs]
            r["num_objects"] = len(keep_objs)
            r["_relabel"] = True
        kept.append(r)

    # --- 2. blank frames (pixels)
    n_blank = 0
    if blank:
        paths = [os.path.join(root, r["image"]) for r in kept]
        if verbose:
            print(f"clean: scanning {len(paths)} images for blank frames ...")
        with ThreadPoolExecutor(max_workers=workers or min(8, os.cpu_count() or 1)) as ex:   # cv2 decodes without the GIL
            stats = list(ex.map(image_stats, paths))
        still = []
        for r, (m, s) in zip(kept, stats):
            if np.isnan(m):
                removed.append((r, "unreadable image")); continue
            if is_blank(m, s, mean_range, min_std):
                n_blank += 1
                removed.append((r, f"blank frame (mean {m:.2f}, std {s:.3f})")); continue
            still.append(r)
        kept = still

    # --- apply
    if not dry_run:
        for r in kept:
            if r.pop("_relabel", False):
                _rewrite_label(root, r)
        for r, reason in removed:
            _quarantine(root, r, dry=False)
        if removed:
            os.makedirs(os.path.join(root, "_removed"), exist_ok=True)
            with open(os.path.join(root, "_removed", "removed.jsonl"), "a") as f:
                for r, reason in removed:
                    f.write(json.dumps({"name": r["name"], "reason": reason, "image": r["image"],
                                        "label_file": r.get("label_file"), "mask_file": r.get("mask_file")}) + "\n")
        meta_path = os.path.join(root, "metadata.jsonl")
        bak = meta_path + ".bak"
        if not os.path.exists(bak):
            shutil.copy2(meta_path, bak)
        with open(meta_path, "w") as f:
            for r in kept:
                r.pop("_relabel", None)
                f.write(json.dumps(r) + "\n")
        summary = export_tables(root)
    else:
        for r in kept:
            r.pop("_relabel", None)
        summary = {}
    report = {"images_before": len(recs), "images_kept": len(kept), "images_removed": len(removed),
              "removed_hidden_beetle": sum(1 for _, why in removed if why.startswith(("beetle", "all"))),
              "removed_blank": n_blank, "boxes_dropped_invisible": n_dropped_boxes, "box_mode": box_mode,
              "images_reboxed": n_reboxed, "dry_run": dry_run,
              "summary": {k: summary.get(k) for k in ("num_images", "num_objects", "per_split", "per_class")} if summary else None}
    if verbose:
        print(json.dumps(report, indent=2))
        if dry_run:
            print("dry run: nothing was changed")
    return report
