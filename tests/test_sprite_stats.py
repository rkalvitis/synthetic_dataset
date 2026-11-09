"""Sprite-level statistics (rendered beetle views, independent of the backgrounds)."""
import json
import os

import cv2
import numpy as np

from popsynth.sprite_stats import plot_sprite_compare, sprite_table


def _sprite(path, w, h):
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[2:-2, 2:-2] = (120, 80, 40, 255)
    rgba[1, 2:-2, 3] = 90                                      # one soft edge row
    cv2.imwrite(path, rgba)


def _dataset(root):
    os.makedirs(os.path.join(root, "sprites"))
    man = [{"index": 0, "file": "sprite_00000.png", "split": "val", "size_mode": "object", "render_size": 40,
            "supersample": 2, "crop_px": [40, 30], "coverage": 0.3, "clipped": False, "elevation_deg": 10.0,
            "azimuth_deg": 20.0, "roll_deg": 0.0, "fov_deg": 40.0, "distance": 3.0, "fill": 0.9},
           {"index": 1, "file": "sprite_00001.png", "split": "train", "size_mode": "object", "render_size": 80,
            "supersample": 4, "crop_px": [60, 80], "coverage": 0.3, "clipped": False, "elevation_deg": 80.0,
            "azimuth_deg": 200.0, "roll_deg": 90.0, "fov_deg": 60.0, "distance": 5.0, "fill": 0.9}]
    with open(os.path.join(root, "sprites", "sprites.jsonl"), "w") as f:
        for m in man:
            _sprite(os.path.join(root, "sprites", m["file"]), *m["crop_px"])
            f.write(json.dumps(m) + "\n")
    recs = []
    for i, (si, px) in enumerate([(0, [40, 30]), (0, [20, 15]), (1, [60, 80])]):
        recs.append({"name": f"img_{i}", "split": "train", "class": "popillia", "image": f"images/img_{i}.png",
                     "width": 128, "height": 128, "num_objects": 1, "boxes_xyxy": [[0, 0, px[0], px[1]]],
                     "boxes_cxcywh_norm": [[0.1, 0.1, 0.1, 0.1]],
                     "objects": [{"object_id": 0, "sprite": f"sprite_0000{si}.png", "sprite_index": si, "object_px": px,
                                  "bbox_xyxy": [0, 0, px[0], px[1]], "bbox_cxcywh_norm": [0.1, 0.1, 0.1, 0.1],
                                  "bbox_w_px": px[0], "bbox_h_px": px[1], "bbox_area_frac": 0.01,
                                  "mask_area_px": px[0] * px[1], "visible_area_px": px[0] * px[1]}]})
    with open(os.path.join(root, "metadata.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def test_sprite_table_and_plots(tmp_path):
    root = str(tmp_path / "ds"); _dataset(root)
    t = sprite_table(root)
    assert list(t["long_px"]) == [40, 80] and list(t["uses"]) == [2, 1]
    assert np.allclose(t["use_scale"], [1.0, 0.5, 1.0]) and np.allclose(t["use_oversampling"], [2.0, 4.0, 4.0])
    assert np.allclose(t["solid_px"], [36 * 26, 56 * 76]) and (t["solid_frac"] > 0.9).all() and (t["mean_alpha"] > 0.95).all()
    assert os.path.exists(os.path.join(root, "stats", "sprite_alpha.npz"))
    t2 = sprite_table(root)                                                    # from the cache
    assert np.allclose(t2["solid_px"], t["solid_px"])
    sep = str(tmp_path / "plots")
    res = plot_sprite_compare([("a", t), ("b", t2)], str(tmp_path / "cmp.png"), bins=5, separate_dir=sep)
    assert res["a"]["num_sprites"] == 2 and res["a"]["uses"]["max"] == 2
    files = sorted(os.listdir(sep))
    assert "sprite_a_size_vs_pasted.png" in files and "sprite_compare_size_px.png" in files and len(files) >= 2 * 13 + 5
