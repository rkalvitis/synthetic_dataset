"""Contact-sheet preview of a generated dataset with boxes drawn."""
from __future__ import annotations

import json
import os

import cv2
import numpy as np


def make_preview(dataset_dir: str, out_path: str | None = None, n: int = 16, cols: int = 4,
                 tile: int | None = None) -> str:
    """tile=None keeps the images at native resolution (no extra downscaling)."""
    recs = [json.loads(l) for l in open(os.path.join(dataset_dir, "metadata.jsonl"))][:n]
    tiles = []
    for r in recs:
        img = cv2.imread(os.path.join(dataset_dir, r["image"]))
        if tile is None:
            tile = max(img.shape[:2])
        for x0, y0, x1, y1 in r.get("boxes_xyxy") or []:
            cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), 2)
        img = cv2.resize(img, (tile, tile), interpolation=cv2.INTER_AREA)
        label = f"{r['class']} | {r.get('lighting', '')} | {r['split']}"
        cv2.rectangle(img, (0, 0), (tile, 18), (0, 0, 0), -1)
        cv2.putText(img, label, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    while len(tiles) % cols:
        tiles.append(np.zeros((tile, tile, 3), np.uint8))
    rows = [np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)]
    sheet = np.concatenate(rows, 0)
    out_path = out_path or os.path.join(dataset_dir, "preview.jpg")
    cv2.imwrite(out_path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return out_path
