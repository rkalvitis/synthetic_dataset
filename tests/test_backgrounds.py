import os

import cv2
import numpy as np

from popsynth.backgrounds import BackgroundPool


def _make(d, n, size):
    os.makedirs(d, exist_ok=True)
    for i in range(n):
        cv2.imwrite(os.path.join(d, f"{i}.jpg"), np.full((size, size, 3), 100 + i, np.uint8))


def test_weighted_folders_and_min_size(tmp_path):
    big, small, tiny = str(tmp_path / "big"), str(tmp_path / "small"), str(tmp_path / "tiny")
    _make(big, 50, 128); _make(small, 2, 128); _make(tiny, 5, 32)
    pool = BackgroundPool([{"path": big, "weight": 0.2}, {"path": small, "weight": 0.8}, tiny,
                           str(tmp_path / "missing")], min_size=64)
    assert len(pool) == 52                      # tiny images and the missing folder are dropped
    rng = np.random.default_rng(0)
    picks = [pool.sample(rng, 32, 32)[1]["background"] for _ in range(300)]
    frac_small = np.mean([("small" in p) for p in picks])
    assert 0.7 < frac_small < 0.9               # weight, not file count, decides
