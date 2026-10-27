import numpy as np

from popsynth.augment import apply_camera_effects, apply_lighting
from popsynth.compose import paste_sprite


def _sprite(n=40):
    yy, xx = np.mgrid[0:n, 0:n]
    a = (((xx - n / 2) ** 2 + (yy - n / 2) ** 2) < (n / 2 - 2) ** 2).astype(np.float32)
    rgb = np.zeros((n, n, 3), np.float32); rgb[..., 0] = 1.0
    return rgb, a


def test_paste_bbox_and_mask_consistent():
    rng = np.random.default_rng(1)
    bg = np.full((128, 160, 3), 0.5, np.float32)
    rgb, a = _sprite()
    img, mask, meta = paste_sprite(bg, rgb, a, rng, object_size=(0.3, 0.3), min_visible=1.0,
                                   color_match=0.0, shadow=None, edge_blur_px=0.0, defocus=None)
    x0, y0, x1, y1 = meta["bbox"]
    assert 0 <= x0 < x1 <= 160 and 0 <= y0 < y1 <= 128
    assert abs(max(x1 - x0, y1 - y0) - 0.3 * 128) <= 3
    ys, xs = np.where(mask > 0.5)
    assert xs.min() == x0 and ys.min() == y0 and xs.max() + 1 == x1 and ys.max() + 1 == y1
    inside = img[mask > 0.5]
    assert inside[:, 0].mean() > 0.95 and inside[:, 1].mean() < 0.05
    assert np.allclose(img[mask == 0], 0.5)


def test_paste_partial_visibility_and_shadow():
    rng = np.random.default_rng(3)
    bg = np.full((64, 64, 3), 0.8, np.float32)
    rgb, a = _sprite()
    img, mask, meta = paste_sprite(bg, rgb, a, rng, object_size=(0.9, 0.9), min_visible=0.5, color_match=0.0,
                                   shadow={"prob": 1.0, "strength": (0.5, 0.5), "blur": (0.05, 0.05), "offset": (0.05, 0.05)},
                                   edge_blur_px=0.0, defocus=None)
    assert meta["visible_frac"] >= 0.45
    assert (img[mask == 0] < 0.8 - 1e-4).any()        # some shadowed background pixels


def test_lighting_and_effects_keep_range():
    rng = np.random.default_rng(0)
    img = rng.random((48, 64, 3)).astype(np.float32)
    for _ in range(20):
        out, meta = apply_lighting(img, rng)
        assert out.shape == img.shape and out.min() >= 0 and out.max() <= 1 and meta["lighting"]
        out, m2 = apply_camera_effects(out, rng, {"blur": {"prob": 1.0}, "jpeg_quality": [50, 90]})
        assert out.shape == img.shape and out.dtype == np.float32 and "jpeg_quality" in m2
