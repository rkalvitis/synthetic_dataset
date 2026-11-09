"""Exposure guard, beetle-on-beetle visibility rule, and the `clean` command."""
import json
import os

import cv2
import numpy as np

from popsynth.augment import apply_photometric, is_legible, overexpose, underexpose
from popsynth.clean import clean_dataset, is_blank
from popsynth.generate import _visible_share
from popsynth.occlude import occlude_objects


def _photo(seed=0, size=96):
    rng = np.random.default_rng(seed)
    return (rng.random((size, size, 3)).astype(np.float32) * 0.6 + 0.2)


def test_overexpose_keeps_detail_and_caps_mean():
    out, g = overexpose(_photo(), 2.8, max_mean=0.82)
    assert out.max() <= 1.0 and out.mean() <= 0.84 and out.std() > 0.05 and g < 2.8


def test_underexpose_floor():
    out, g = underexpose(_photo() * 0.08, 0.2, min_mean=0.05)
    assert abs(out.mean() - 0.05) < 0.01 and g == 1.0


def test_is_legible():
    assert not is_legible(np.ones((8, 8, 3), np.float32))
    assert not is_legible(np.zeros((8, 8, 3), np.float32))
    assert is_legible(_photo())


def test_photometric_guard_never_blank():
    tiers = {"bad": {"weight": 1, "haze_prob": 1.0, "haze": [0.5, 0.5], "exposure_fault_prob": 1.0,
                     "overexposure": [2.8, 2.8], "underexposure": [0.2, 0.2], "noise": [0, 0], "jpeg_quality": [90, 95]}}
    light = {"day": {"weight": 1, "exposure": [1.3, 1.3]}, "night": {"weight": 1, "exposure": [0.1, 0.1]}}
    for i in range(30):
        out, meta = apply_photometric(_photo(i), np.random.default_rng(i), light, {"tiers": tiers})
        assert is_legible(out), meta
        assert not ("haze" in meta and ("overexposure" in meta or "underexposure" in meta))


def test_visible_share():
    a = np.zeros((10, 10), np.float32); a[2:8, 2:8] = 1
    c = np.zeros((10, 10), np.float32); c[2:5, 2:8] = 1
    assert abs(_visible_share(a, c) - 0.5) < 1e-6
    assert _visible_share(a, np.zeros_like(a)) == 1.0


def test_occluder_per_object_cap():
    H = W = 64
    img = np.full((H, W, 3), 0.4, np.float32)
    m = np.zeros((H, W), np.float32); m[16:48, 16:48] = 1
    cfg = {"prob": 1.0, "count": [3, 3], "cover": [0.9, 0.9], "size": [2.0, 2.0], "sources": {"background": 1.0},
           "max_hidden": 0.9}
    out, vis, metas = occlude_objects(img, [m], np.random.default_rng(0), cfg, None, img, max_hidden=[0.3])
    assert (vis[0] > 0.5).sum() >= 0.7 * (m > 0.5).sum() - 2


def _write_dataset(root, blank_name="img_000003"):
    """3 positives + 1 negative + 1 blank positive; img_000001 has a fully hidden beetle, img_000002 a 20% one."""
    os.makedirs(os.path.join(root, "images/train/popillia"), exist_ok=True)
    os.makedirs(os.path.join(root, "images/train/background"), exist_ok=True)
    os.makedirs(os.path.join(root, "labels/train"), exist_ok=True)
    os.makedirs(os.path.join(root, "masks/train"), exist_ok=True)
    rng = np.random.default_rng(0)
    recs = []

    def obj(i, box, mask_area, visible_area):
        return {"object_id": i, "sprite": "s.png", "sprite_index": 0, "bbox": box, "bbox_xyxy": box,
                "bbox_cxcywh_norm": [0.5, 0.5, 0.2, 0.2], "mask_area_px": mask_area, "visible_area_px": visible_area,
                "occluded_frac": round(1 - visible_area / mask_area, 3), "class": "popillia", "object_size_frac": 0.3,
                "bbox_w_px": box[2] - box[0], "bbox_h_px": box[3] - box[1], "bbox_area_frac": 0.1, "visible_frac": 1.0,
                "elevation_deg": 10.0, "roll_deg": 0.0}

    specs = {"img_000000": [obj(0, [10, 10, 30, 30], 300, 300)],
             "img_000001": [obj(0, [10, 10, 30, 30], 300, 300), obj(1, [12, 12, 28, 28], 200, 0)],
             "img_000002": [obj(0, [10, 10, 30, 30], 300, 300), obj(1, [12, 12, 28, 28], 200, 40)],
             blank_name: [obj(0, [10, 10, 30, 30], 300, 300)],
             "img_000004": []}
    for name, objs in specs.items():
        cls = "popillia" if objs else "background"
        img = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
        if name == blank_name:
            img[:] = 253
        rel = f"images/train/{cls}/{name}.png"
        cv2.imwrite(os.path.join(root, rel), img)
        with open(os.path.join(root, f"labels/train/{name}.txt"), "w") as f:
            for o in objs:
                f.write("0 0.5 0.5 0.2 0.2\n")
        mask_file = None
        if objs:
            mask_file = f"masks/train/{name}.png"
            cv2.imwrite(os.path.join(root, mask_file), np.zeros((64, 64), np.uint8))
        recs.append({"name": name, "split": "train", "class": cls, "image": rel, "label_file": f"labels/train/{name}.txt",
                     "mask_file": mask_file, "width": 64, "height": 64, "num_objects": len(objs),
                     "boxes_xyxy": [o["bbox_xyxy"] for o in objs], "boxes_cxcywh_norm": [o["bbox_cxcywh_norm"] for o in objs],
                     "background": "x.jpg", "background_source": "t", "background_category": "t", "lighting": "day",
                     "quality_tier": "clean", "effects": [], "objects": objs})
    with open(os.path.join(root, "metadata.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(root, "classes.txt"), "w") as f:
        f.write("popillia\n")


def test_is_blank():
    assert is_blank(0.99, 0.001) and is_blank(0.02, 0.01) and not is_blank(0.4, 0.2)


def test_clean_dataset(tmp_path):
    root = str(tmp_path / "ds")
    _write_dataset(root)
    rep = clean_dataset(root, min_visible=0.33, drop_below=0.05, workers=1, verbose=False)
    assert rep["images_before"] == 5 and rep["images_kept"] == 3
    assert rep["boxes_dropped_invisible"] == 1 and rep["removed_blank"] == 1 and rep["removed_hidden_beetle"] == 1
    kept = {json.loads(l)["name"] for l in open(os.path.join(root, "metadata.jsonl"))}
    assert kept == {"img_000000", "img_000001", "img_000004"}
    # the invisible beetle's box is gone from the label file and the record
    assert open(os.path.join(root, "labels/train/img_000001.txt")).read().count("\n") == 1
    r1 = [json.loads(l) for l in open(os.path.join(root, "metadata.jsonl")) if "img_000001" in l][0]
    assert r1["num_objects"] == 1 and len(r1["objects"]) == 1
    # quarantined files moved, not deleted
    assert os.path.exists(os.path.join(root, "_removed/images/train/popillia/img_000002.png"))
    assert os.path.exists(os.path.join(root, "_removed/images/train/popillia/img_000003.png"))
    assert not os.path.exists(os.path.join(root, "images/train/popillia/img_000003.png"))
    assert os.path.exists(os.path.join(root, "metadata.jsonl.bak"))
    coco = json.load(open(os.path.join(root, "dataset.json")))
    assert len(coco["images"]) == 3 and len(coco["annotations"]) == 2
    # idempotent
    rep2 = clean_dataset(root, workers=1, verbose=False)
    assert rep2["images_removed"] == 0 and rep2["boxes_dropped_invisible"] == 0


def test_clean_dry_run(tmp_path):
    root = str(tmp_path / "ds")
    _write_dataset(root)
    rep = clean_dataset(root, min_visible=0.33, workers=1, dry_run=True, verbose=False)
    assert rep["images_removed"] == 2
    assert len(open(os.path.join(root, "metadata.jsonl")).read().splitlines()) == 5
    assert os.path.exists(os.path.join(root, "images/train/popillia/img_000003.png"))


def test_coverage_summary_flat_vs_centered():
    from popsynth.stats import coverage_summary
    flat = np.full((60, 60), 0.3)
    assert abs(coverage_summary(flat)["center_edge_ratio"] - 1.0) < 1e-9 and coverage_summary(flat)["cv"] < 1e-9
    yy, xx = np.mgrid[0:60, 0:60]
    centred = np.exp(-(((xx - 30) ** 2 + (yy - 30) ** 2) / 300.0))
    assert coverage_summary(centred)["center_edge_ratio"] > 3


def test_clean_box_mode_visible_and_back(tmp_path):
    root = str(tmp_path / "ds")
    _write_dataset(root)
    # img_000000: beetle half hidden under a patch on its right side -> visible box is the left half
    recs = [json.loads(l) for l in open(os.path.join(root, "metadata.jsonl"))]
    o = recs[0]["objects"][0]
    o["visible_area_px"] = 150; o["occluded_frac"] = 0.5; o["bbox_visible_xyxy"] = [10, 10, 20, 30]
    with open(os.path.join(root, "metadata.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    rep = clean_dataset(root, min_visible=0.33, workers=1, verbose=False, box_mode="visible")
    assert rep["images_reboxed"] == 1
    r0 = [json.loads(l) for l in open(os.path.join(root, "metadata.jsonl")) if '"img_000000"' in l][0]
    o = r0["objects"][0]
    assert o["bbox_xyxy"] == [10, 10, 20, 30] and r0["boxes_xyxy"] == [[10, 10, 20, 30]]
    assert o["bbox_full_xyxy"] == [10, 10, 30, 30] and o["bbox_w_px"] == 10
    line = open(os.path.join(root, "labels/train/img_000000.txt")).read().split()
    assert np.allclose([float(v) for v in line[1:]], [15 / 64, 20 / 64, 10 / 64, 20 / 64], atol=1e-5)
    assert "bbox_full_xyxy" in open(os.path.join(root, "objects.csv")).readline()
    # idempotent, and reversible
    assert clean_dataset(root, workers=1, verbose=False, box_mode="visible")["images_reboxed"] == 0
    rep = clean_dataset(root, workers=1, verbose=False, box_mode="full")
    r0 = [json.loads(l) for l in open(os.path.join(root, "metadata.jsonl")) if '"img_000000"' in l][0]
    assert rep["images_reboxed"] == 1 and r0["boxes_xyxy"] == [[10, 10, 30, 30]]


def test_floor_rule_frame_beetles_occluders():
    from popsynth.generate import _floor_ok, _occluder_caps
    H = W = 40
    a0 = np.zeros((H, W), np.float32); a0[0:20, 0:20] = 1          # beetle 0: 400 px in frame, body 1000 px (600 cut off)
    body0 = 1000
    z = np.zeros((H, W), np.float32)
    # a new beetle with only 30% of its body inside the frame is refused, 40% is fine
    assert not _floor_ok(a0, 400 / 0.30, 0.33, [], [], [], [])
    assert _floor_ok(a0, 1000, 0.33, [], [], [], [])
    # beetle 1 on top of beetle 0: beetle 0 has 400/1000 visible; covering 100 px leaves 30% -> refused, 50 px -> 35% ok
    a1 = np.zeros((H, W), np.float32); a1[10:20, 10:20] = 1          # 100 px overlap, body 100
    assert not _floor_ok(a1, 100, 0.33, [a0], [body0], [np.maximum(z, a1)], [0.33])
    a1b = np.zeros((H, W), np.float32); a1b[15:20, 10:20] = 1        # 50 px overlap
    assert _floor_ok(a1b, 50, 0.33, [a0], [body0], [np.maximum(z, a1b)], [0.33])
    # occluder cap: beetle 0 has 350 px left (350/1000 visible) -> may lose only 20/350 before 33% of the body
    caps = _occluder_caps([a0], [body0], [a1b > 0.5], [0.33], 0.8, True)
    assert abs(caps[0] - (1 - 0.33 * 1000 / 350)) < 1e-6 and caps[0] < 0.06
    assert _occluder_caps([a0], [body0], [a1b > 0.5], [0.33], 0.8, False) == [0.8]
    # a beetle already at the floor: cap 0 (no occluder may touch it)
    assert _occluder_caps([a0], [1200], [a1b > 0.5], [0.33], 0.8, True) == [0.0]


def test_paste_sprite_reports_whole_body():
    from popsynth.compose import paste_sprite
    rng = np.random.default_rng(0)
    bg = np.zeros((64, 64, 3), np.float32)
    a = np.ones((20, 20), np.float32); rgb = np.ones((20, 20, 3), np.float32) * 0.5
    _, mask, meta = paste_sprite(bg, rgb, a, rng, scale=1.0, min_visible=0.0, center=(0.0, 0.5), color_match=0.0,
                                 shadow=None, edge_blur_px=(0, 0))
    assert meta["body_px"] == 400 and abs(meta["body_in_frame_frac"] - 0.5) < 0.06 and abs(int((mask > 0.5).sum()) - 200) <= 20
