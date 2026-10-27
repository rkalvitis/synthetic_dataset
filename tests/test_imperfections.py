import numpy as np

from popsynth.augment import apply_camera_effects, apply_imperfections

TIERS = {"clean": {"weight": 0.5, "jpeg_quality": [90, 95]},
         "bad": {"weight": 0.5, "blur_prob": 1.0, "motion_px": [9, 15], "defocus_prob": 1.0, "defocus_sigma": [2, 3],
                 "lowres_prob": 1.0, "lowres_factor": [3, 4], "haze_prob": 1.0, "haze": [0.3, 0.5],
                 "exposure_fault_prob": 1.0, "noise": [0.05, 0.05], "jpeg_quality": [20, 30]}}


def test_tiers_apply_and_keep_range():
    rng = np.random.default_rng(0)
    img = rng.random((64, 80, 3)).astype(np.float32)
    out, meta = apply_imperfections(img, rng, TIERS, tier="bad")
    assert out.shape == img.shape and out.dtype == np.float32 and out.min() >= 0 and out.max() <= 1
    assert meta["quality_tier"] == "bad" and "motion_blur_px" in meta and "lowres_factor" in meta and "haze" in meta
    # a bad image is much smoother than the random input
    assert np.abs(np.diff(out, axis=1)).mean() < 0.3 * np.abs(np.diff(img, axis=1)).mean()
    smooth = np.linspace(0, 1, 80, dtype=np.float32)[None, :, None].repeat(64, 0).repeat(3, 2)
    clean, m2 = apply_imperfections(smooth, rng, TIERS, tier="clean")
    assert np.abs(clean - smooth).mean() < 0.02 and m2["quality_tier"] == "clean"


def test_camera_effects_dispatches_to_tiers():
    rng = np.random.default_rng(1)
    img = rng.random((32, 32, 3)).astype(np.float32)
    seen = {apply_camera_effects(img, rng, {"tiers": TIERS})[1]["quality_tier"] for _ in range(40)}
    assert seen == {"clean", "bad"}
