"""Label writers: class folders, YOLO boxes, binary masks and dataset-wide metadata.

Per image (streamed while generating):
    <root>/images/<split>/<class>/<name>.<fmt>   classification layout (torchvision ImageFolder)
    <root>/labels/<split>/<name>.txt             YOLO: 'cls cx cy w h' normalised (empty file for negatives)
    <root>/masks/<split>/<name>.png              0/255 beetle mask (positives only)
    <root>/metadata.jsonl                        one JSON record per image with every sampled parameter

Dataset-wide (written on close(), or rebuilt from metadata.jsonl with `export_tables`):
    <root>/dataset.json     COCO detection format (images / annotations / categories) with the full
                            per-image and per-object metadata attached, one file for the whole dataset
    <root>/images.csv       one row per image: labels + metadata (flat columns)
    <root>/objects.csv      one row per beetle: box (pixels + normalised), pose, size, compositing params
    <root>/summary.json     counts and distributions (per split / class / tier / lighting / background ...)
"""
from __future__ import annotations

import csv
import json
import os
from collections import Counter

import cv2
import numpy as np

CLASS_NAMES = ["popillia"]  # YOLO / COCO class ids; negatives simply have no boxes

# columns of images.csv (top-level record keys; missing values are left empty)
IMAGE_COLUMNS = [
    "name", "split", "class", "image", "label_file", "mask_file", "width", "height", "num_objects",
    "background", "background_source", "background_category", "bg_crop_px", "bg_crop_scale", "bg_hflip",
    "lighting", "exposure", "contrast", "saturation", "noise", "flash",
    "quality_tier", "effects", "motion_blur_px", "defocus_sigma", "lowres_factor", "haze",
    "overexposure", "underexposure", "sensor_noise", "jpeg_quality", "gaussian_blur_sigma",
    "boxes_xyxy", "boxes_cxcywh_norm",
]
OBJECT_COLUMNS = [
    "image_name", "split", "object_id", "class", "bbox_xyxy", "bbox_cxcywh_norm", "bbox_w_px", "bbox_h_px",
    "bbox_area_frac", "mask_area_px", "visible_frac", "occluded_frac", "object_size_frac", "object_px",
    "sprite", "sprite_index", "elevation_deg", "azimuth_deg", "roll_deg", "fov_deg", "camera_distance",
    "color_match", "defocus_sigma", "shadow_strength", "shadow_blur_px",
]


def _posix(p: str | None) -> str | None:
    """Relative paths in the label files always use '/', also on Windows."""
    return None if p is None else p.replace(os.sep, "/")


def xyxy_to_cxcywh_norm(box, W: int, H: int) -> list[float]:
    x0, y0, x1, y1 = box
    return [round((x0 + x1) / 2 / W, 6), round((y0 + y1) / 2 / H, 6), round((x1 - x0) / W, 6), round((y1 - y0) / H, 6)]


class DatasetWriter:
    def __init__(self, root: str, image_format: str = "jpg", jpeg_quality: int = 95,
                 write_masks: bool = True, write_yolo: bool = True):
        self.root = root
        self.fmt = image_format
        self.q = jpeg_quality
        self.write_masks = write_masks
        self.write_yolo = write_yolo
        os.makedirs(root, exist_ok=True)
        with open(os.path.join(root, "classes.txt"), "w") as f:
            f.write("\n".join(CLASS_NAMES) + "\n")
        self._meta = open(os.path.join(root, "metadata.jsonl"), "a")

    def write(self, name: str, split: str, img: np.ndarray, mask: np.ndarray | None,
              boxes: list[list[int]], meta: dict) -> str:
        """img float32 RGB [0,1]; mask float [0,1] or None; boxes [[x0,y0,x1,y1], ...] in pixels."""
        cls = "popillia" if boxes else "background"
        H, W = img.shape[:2]
        d = os.path.join(self.root, "images", split, cls)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{name}.{self.fmt}")
        u8 = cv2.cvtColor((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
        if self.fmt in ("jpg", "jpeg"):
            cv2.imwrite(path, u8, [cv2.IMWRITE_JPEG_QUALITY, self.q])
        else:
            cv2.imwrite(path, u8)

        label_file = mask_file = None
        if self.write_yolo:
            ld = os.path.join(self.root, "labels", split)
            os.makedirs(ld, exist_ok=True)
            label_file = f"labels/{split}/{name}.txt"
            with open(os.path.join(self.root, label_file), "w") as f:
                for b in boxes:
                    cx, cy, w, h = xyxy_to_cxcywh_norm(b, W, H)
                    f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        if self.write_masks and mask is not None and boxes:
            md = os.path.join(self.root, "masks", split)
            os.makedirs(md, exist_ok=True)
            mask_file = f"masks/{split}/{name}.png"
            cv2.imwrite(os.path.join(self.root, mask_file), (mask > 0.5).astype(np.uint8) * 255)

        objs = []
        for j, o in enumerate(meta.get("objects", [])):
            b = o.get("bbox")
            if not b:
                continue
            o = dict(o)
            o.setdefault("object_id", j)
            o["class"] = "popillia"
            o["bbox_xyxy"] = b
            o["bbox_cxcywh_norm"] = xyxy_to_cxcywh_norm(b, W, H)
            o["bbox_w_px"], o["bbox_h_px"] = int(b[2] - b[0]), int(b[3] - b[1])
            o["bbox_area_frac"] = round((b[2] - b[0]) * (b[3] - b[1]) / (W * H), 6)
            objs.append(o)
        meta = {k: v for k, v in meta.items() if k != "objects"}
        rec = {"name": name, "split": split, "class": cls, "image": _posix(os.path.relpath(path, self.root)),
               "label_file": label_file, "mask_file": mask_file, "width": W, "height": H,
               "num_objects": len(boxes), "boxes_xyxy": boxes,
               "boxes_cxcywh_norm": [xyxy_to_cxcywh_norm(b, W, H) for b in boxes],
               **meta, "objects": objs}
        self._meta.write(json.dumps(rec) + "\n")
        return path

    def close(self):
        self._meta.close()
        export_tables(self.root)


# ----------------------------------------------------------------------------- dataset-wide exports
def read_metadata(root: str) -> list[dict]:
    with open(os.path.join(root, "metadata.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (list, dict)):
        return json.dumps(v, separators=(",", ":"))
    return v


def export_tables(root: str) -> dict:
    """Rebuild dataset.json, images.csv, objects.csv and summary.json from metadata.jsonl."""
    recs = read_metadata(root)
    # de-duplicate by name (metadata.jsonl is appended to; the last record of a name wins)
    by_name = {}
    for r in recs:
        by_name[r["name"]] = r
    recs = sorted(by_name.values(), key=lambda r: r["name"])

    # --- COCO detection format + metadata
    images, annotations = [], []
    for i, r in enumerate(recs):
        extra = {k: v for k, v in r.items() if k not in ("objects", "image", "width", "height")}
        images.append({"id": i, "file_name": r["image"], "width": r["width"], "height": r["height"],
                       "split": r["split"], "metadata": extra})
        for o in r.get("objects", []):
            x0, y0, x1, y1 = o["bbox_xyxy"]
            annotations.append({"id": len(annotations), "image_id": i, "category_id": 1,
                                "bbox": [x0, y0, x1 - x0, y1 - y0],            # COCO: x, y, w, h in pixels
                                "bbox_cxcywh_norm": o["bbox_cxcywh_norm"],
                                "area": o.get("mask_area_px", (x1 - x0) * (y1 - y0)),
                                "iscrowd": 0, "metadata": {k: v for k, v in o.items() if k not in ("bbox", "bbox_xyxy", "bbox_cxcywh_norm", "class")}})
    summary = summarize(recs)
    coco = {"info": {"description": "popsynth synthetic Popillia japonica dataset",
                     "box_format": "COCO [x, y, w, h] pixels; bbox_cxcywh_norm = YOLO [cx, cy, w, h] normalised",
                     "summary": summary},
            "categories": [{"id": 1, "name": "popillia", "supercategory": "insect"}],
            "images": images, "annotations": annotations}
    with open(os.path.join(root, "dataset.json"), "w") as f:
        json.dump(coco, f)

    # --- flat CSV tables
    with open(os.path.join(root, "images.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=IMAGE_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            w.writerow({k: _cell(r.get(k)) for k in IMAGE_COLUMNS})
    with open(os.path.join(root, "objects.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OBJECT_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            for o in r.get("objects", []):
                row = {"image_name": r["name"], "split": r["split"], **o}
                w.writerow({k: _cell(row.get(k)) for k in OBJECT_COLUMNS})
    with open(os.path.join(root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def summarize(recs: list[dict]) -> dict:
    """Counts and simple distributions over the records of metadata.jsonl."""
    def count(key, items=None):
        return dict(sorted(Counter(str(r.get(key)) for r in (items or recs)).items()))

    objs = [o for r in recs for o in r.get("objects", [])]
    def hist(values, edges):
        values = np.asarray(values, float)
        h, _ = np.histogram(values, bins=edges)
        return {f"{edges[i]}-{edges[i + 1]}": int(h[i]) for i in range(len(h))}

    s = {"num_images": len(recs), "num_objects": len(objs),
         "per_split": count("split"), "per_class": count("class"),
         "per_split_class": dict(sorted(Counter(f'{r["split"]}/{r["class"]}' for r in recs).items())),
         "objects_per_image": count("num_objects"),
         "lighting": count("lighting"), "quality_tier": count("quality_tier"),
         "effects": dict(sorted(Counter(e for r in recs for e in r.get("effects", [])).items())),
         "background_source": count("background_source"),
         "background_category": count("background_category"),
         "unique_backgrounds": len({r.get("background") for r in recs})}
    if objs:
        s["bbox_size_frac"] = hist([max(o["bbox_w_px"], o["bbox_h_px"]) / max(1, min(r["width"], r["height"]))
                                    for r in recs for o in r.get("objects", [])],
                                   [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 10])
        s["elevation_deg"] = hist([o["elevation_deg"] for o in objs], [-90, 0, 15, 30, 45, 60, 75, 90])
        s["roll_deg"] = hist([o["roll_deg"] for o in objs], [-180, -120, -60, 0, 60, 120, 180])
        s["visible_frac"] = hist([o["visible_frac"] for o in objs], [0, 0.25, 0.5, 0.75, 0.999, 1.001])
    return s
