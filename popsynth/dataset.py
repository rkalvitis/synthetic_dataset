"""Loading a generated dataset for training: a PyTorch Dataset that yields image, boxes and metadata.

    from popsynth.dataset import PopsynthDataset
    ds = PopsynthDataset("outputs/dataset", split="train")
    img, target = ds[0]      # img uint8 tensor (3,H,W); target: boxes_xyxy (N,4), labels (N,), meta dict
    ds.filter(lambda m: m["quality_tier"] == "bad")   # subsets by any metadata field

Reads metadata.jsonl (same content as dataset.json / images.csv); no dependency beyond torch + cv2.
"""
from __future__ import annotations

import os
from typing import Callable

import cv2
import numpy as np

from .labels import read_metadata


class PopsynthDataset:
    def __init__(self, root: str, split: str | None = "train", transform: Callable | None = None,
                 where: Callable[[dict], bool] | None = None):
        self.root = root
        self.transform = transform
        recs = read_metadata(root)
        if split:
            recs = [r for r in recs if r["split"] == split]
        if where:
            recs = [r for r in recs if where(r)]
        self.records = recs

    def filter(self, where: Callable[[dict], bool]) -> "PopsynthDataset":
        out = PopsynthDataset.__new__(PopsynthDataset)
        out.root, out.transform = self.root, self.transform
        out.records = [r for r in self.records if where(r)]
        return out

    def __len__(self) -> int:
        return len(self.records)

    def load_image(self, i: int) -> np.ndarray:
        img = cv2.imread(os.path.join(self.root, self.records[i]["image"]), cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, i: int):
        import torch
        r = self.records[i]
        img = self.load_image(i)
        boxes = np.array(r["boxes_xyxy"], np.float32).reshape(-1, 4)
        target = {"boxes_xyxy": torch.from_numpy(boxes),
                  "boxes_cxcywh_norm": torch.tensor(r["boxes_cxcywh_norm"], dtype=torch.float32).reshape(-1, 4),
                  "labels": torch.ones(len(boxes), dtype=torch.int64),   # 1 = popillia
                  "has_popillia": torch.tensor(float(r["class"] == "popillia")),
                  "meta": r}
        if self.transform:
            img, target = self.transform(img, target)
        else:
            img = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)
        return img, target
