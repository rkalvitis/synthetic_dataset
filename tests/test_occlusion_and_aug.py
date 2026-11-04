import os

import cv2
import numpy as np

from popsynth.augment import augment_background, augment_sprite
from popsynth.compose import paste_sprite
from popsynth.occlude import (OccluderBank, cutout_from_plain_background, leaf_shape_mask, make_occluder,
                              occlude_objects, place_occluder)


def _disc(n=40):
    yy, xx = np.mgrid[0:n, 0:n]
    a = (((xx - n / 2) ** 2 + (yy - n / 2) ** 2) < (n / 2 - 2) ** 2).astype(np.float32)
    rgb = np.zeros((n, n, 3), np.float32); rgb[..., 0] = 1.0
    return rgb, a


def test_leaf_shape_mask_is_soft_blob():
    m = leaf_shape_mask(60, 100, np.random.default_rng(0))
    assert m.shape == (60, 100) and 0 <= m.min() and m.max() <= 1
    assert m[30, 50] > 0.99 and m[0, 0] < 0.01 and 0.3 < m.mean() < 0.8


def test_place_occluder_hits_target_coverage_and_hides_mask():
    rng = np.random.default_rng(0)
    bg = np.full((120, 120, 3), 0.5, np.float32)
    rgb, a = _disc(50)
    img, mask, _ = paste_sprite(bg, rgb, a, rng, object_size=(0.4, 0.4), min_visible=1.0, color_match=0.0,
                                shadow=None, edge_blur_px=0.0, defocus=None, center=(0.5, 0.5))
    occ_rgb = np.zeros((40, 60, 3), np.float32); occ_rgb[..., 1] = 1.0
    occ_a = leaf_shape_mask(40, 60, rng)
    out, occ_full, hidden = place_occluder(img, mask, occ_rgb, occ_a, rng, cover=0.4, tries=30)
    assert 0.25 < hidden < 0.55
    vis = mask * (1 - occ_full)
    assert vis.sum() < mask.sum() * 0.8                       # part of the beetle is now hidden
    assert out[occ_full > 0.99][:, 1].mean() > 0.95           # occluder pixels are on top


def test_occlude_objects_background_source_and_max_hidden():
    rng = np.random.default_rng(1)
    bg = rng.random((100, 100, 3)).astype(np.float32)
    rgb, a = _disc(30)
    img, mask, _ = paste_sprite(bg, rgb, a, rng, object_size=(0.3, 0.3), min_visible=1.0, color_match=0.0,
                                shadow=None, edge_blur_px=0.0, defocus=None, center=(0.5, 0.5))
    cfg = {"prob": 1.0, "count": [2, 2], "cover": [0.3, 0.5], "size": [1.0, 1.0], "sources": {"background": 1.0},
           "max_hidden": 0.8, "blur_sigma_px": [0.0, 1.0]}
    out, visible, metas = occlude_objects(img, [mask], rng, cfg, bank=None, bg_photo=bg)
    assert 1 <= len(metas) <= 2 and all(m["occluder"] == "background" for m in metas)
    hidden = 1 - (visible[0] > 0.5).sum() / (mask > 0.5).sum()
    assert 0.1 < hidden <= 0.8
    # prob 0 -> untouched
    out2, vis2, m2 = occlude_objects(img, [mask], rng, {**cfg, "prob": 0.0}, None, bg)
    assert m2 == [] and np.array_equal(vis2[0], mask)


def test_cutout_and_leaf_bank(tmp_path):
    # a dark leaf on white paper -> alpha mask; saved as RGBA and loaded by the bank
    img = np.full((80, 120, 3), 250, np.uint8)
    cv2.ellipse(img, (60, 40), (45, 20), 20, 0, 360, (30, 120, 40), -1)
    alpha = cutout_from_plain_background(img)
    assert alpha is not None and alpha[40, 60] == 255 and alpha[2, 2] == 0
    assert 0.2 < (alpha > 127).mean() < 0.5
    d = tmp_path / "leaves"; os.makedirs(d)
    cv2.imwrite(str(d / "leaf.png"), cv2.merge([img[..., 0], img[..., 1], img[..., 2], alpha]))
    bank = OccluderBank([str(d), str(tmp_path / "missing")])
    assert len(bank) == 1
    rng = np.random.default_rng(0)
    rgb, a, meta = make_occluder(rng, {"sources": {"leaf": 1.0}, "rotate_deg": [30, 30]}, 50, bank, None)
    assert meta["occluder"] == "leaf" and 38 <= max(a.shape) <= 60 and a.max() > 0.9   # ellipse bbox shrinks when tilted


def test_background_and_sprite_augmentation_are_independent_and_safe():
    rng = np.random.default_rng(0)
    img = rng.random((64, 80, 3)).astype(np.float32)
    cfg = {"prob": 1.0, "brightness": [0.7, 1.3], "contrast": [0.8, 1.2], "saturation": [0.6, 1.4], "hue_deg": [-15, 15],
           "gamma": [0.8, 1.25], "rotate_deg": [-20, 20], "vflip": 0.5, "blur_sigma_px": [0.0, 1.5], "grayscale_prob": 0.2}
    for _ in range(10):
        out, m = augment_background(img, rng, cfg)
        assert out.shape == img.shape and out.dtype == np.float32 and 0 <= out.min() and out.max() <= 1
        assert "bg_brightness" in m and set(m["bg_aug"]) >= {"brightness", "contrast"}
    assert augment_background(img, rng, {"prob": 0.0})[1] == {"bg_aug": []}

    rgb, a = _disc(40)
    rgb0, a0 = rgb.copy(), a.copy()
    scfg = {"prob": 1.0, "brightness": [0.8, 1.2], "hue_deg": [-8, 8], "saturation": [0.8, 1.2], "hflip": 1.0,
            "aspect": [0.9, 1.1], "specular": {"prob": 1.0, "strength": [0.3, 0.5]}, "blur_sigma_px": [0.0, 0.5]}
    r, a2, m = augment_sprite(rgb, a, rng, scfg)
    assert np.array_equal(rgb, rgb0) and np.array_equal(a, a0)       # inputs (sprite cache) untouched
    assert r.shape[:2] == a2.shape and m["aug_hflip"] is True and "aug_specular" in m
    assert abs(a2.shape[0] - 40) <= 5 and abs(a2.shape[1] - 40) <= 5
    r, a3, m = augment_sprite(rgb, a, rng, None)
    assert m == {"sprite_aug": []} and a3 is a


def test_center_sigma_clusters_in_the_middle():
    rgb, a = _disc(20)
    bg = np.zeros((200, 200, 3), np.float32)
    cs = []
    for i in range(60):
        rng = np.random.default_rng(i)
        _, _, meta = paste_sprite(bg, rgb, a, rng, object_size=(0.1, 0.1), min_visible=1.0, color_match=0.0,
                                  shadow=None, edge_blur_px=0.0, defocus=None, center_sigma=0.05)
        x0, y0, x1, y1 = meta["bbox"]
        cs.append(((x0 + x1) / 2 / 200, (y0 + y1) / 2 / 200))
    cs = np.array(cs)
    assert (np.abs(cs - 0.5) < 1 / 6).all(axis=1).mean() > 0.9


def test_sprite_bank_cache_is_capped(tmp_path):
    import json
    from popsynth.generate import SpriteBank
    d = tmp_path / "sprites"; os.makedirs(d)
    with open(d / "sprites.jsonl", "w") as f:
        for i in range(6):
            rgba = np.full((100, 100, 4), 128, np.uint8)
            cv2.imwrite(str(d / f"sprite_{i:05d}.png"), rgba)
            f.write(json.dumps({"index": i, "file": f"sprite_{i:05d}.png", "split": "train"}) + "\n")
    bank = SpriteBank(str(d), cache_gb=3 * 40000 / 1024 ** 3)      # room for 3 sprites of 40 kB
    for i in range(6):
        rgb, a = bank.load(bank.records[i])
        assert rgb.dtype == np.float32 and rgb.shape == (100, 100, 3) and abs(a[0, 0] - 128 / 255) < 1e-3
    assert len(bank._cache) == 3 and bank._used == 3 * 40000 and bank.misses == 6
    bank.load(bank.records[5]); assert bank.hits == 1                     # most recent stays cached
    bank.load(bank.records[0]); assert bank.misses == 7                   # oldest was evicted
    rgb, _ = bank.load(bank.records[5]); rgb[:] = 0                       # returned arrays are private copies
    assert bank.load(bank.records[5])[0].max() > 0


def test_memory_helpers():
    from popsynth.memory import check_limit, limit_memory, peak_rss_bytes, rss_bytes
    assert rss_bytes() > 0 and peak_rss_bytes() > 0
    assert "no RAM limit" in limit_memory(None)
    check_limit(None); check_limit(1e6, "x")                             # far above usage: no error
    try:
        check_limit(1e-9, "test"); raise AssertionError("should have raised")
    except MemoryError as e:
        assert "max_ram_gb" in str(e)


def test_paste_scale_one_keeps_sprite_pixels():
    rgb, a = _disc(30)
    bg = np.zeros((100, 100, 3), np.float32)
    rng = np.random.default_rng(0)
    img, mask, meta = paste_sprite(bg, rgb, a, rng, scale=1.0, min_visible=1.0, color_match=0.0, shadow=None,
                                   edge_blur_px=0.0, defocus=None, object_size=(0.9, 0.9))
    x0, y0, x1, y1 = meta["bbox"]
    ys, xs = np.where(a > 0.5)
    assert (x1 - x0, y1 - y0) == (xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)   # no resampling
    assert abs(meta["object_size_frac"] - (xs.max() - xs.min() + 3) / 100) < 0.02
    assert (mask > 0.5).sum() == (a > 0.5).sum()


def test_projected_extent_scales_with_frame():
    from popsynth.cameras import fixed_camera, projected_extent, resized
    rng = np.random.default_rng(0)
    means = rng.normal(0, 0.4, (500, 3)).astype(np.float32)
    scales = np.full((500, 3), 0.01, np.float32)
    cam = fixed_camera(1024, 1024, object_radius=1.5, fov_deg=40, fill=0.8)
    x0, y0, x1, y1 = projected_extent(means, scales, cam.viewmat, cam.K)
    assert 0 < x0 < x1 < 1024 and 0 < y0 < y1 < 1024
    cam2 = resized(cam, 512, 512)
    a0, b0, a1, b1 = projected_extent(means, scales, cam2.viewmat, cam2.K)
    assert abs((a1 - a0) - (x1 - x0) / 2) < 1.0 and abs((b1 - b0) - (y1 - y0) / 2) < 1.0
    assert np.allclose(cam2.viewmat, cam.viewmat) and cam2.fov_deg == cam.fov_deg
