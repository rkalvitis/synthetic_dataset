import torch

from popsynth.cameras import fixed_camera
from popsynth.render.torch_raster import render_torch


def _scene(n=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    c0 = 0.28209479177387814
    sh = torch.zeros(n, 16, 3)
    sh[:, 0] = (torch.rand(n, 3, generator=g) - 0.5) / c0
    return {"means": torch.randn(n, 3, generator=g) * 0.4, "quats": torch.randn(n, 4, generator=g),
            "scales": torch.rand(n, 3, generator=g) * 0.15 + 0.02, "opacities": torch.rand(n, generator=g), "sh": sh}


def test_band_rendering_matches_single_pass():
    g = _scene()
    cam = fixed_camera(96, 80, elevation_deg=25, azimuth_deg=10, fov_deg=50, fill=0.6)
    rgb1, a1 = render_torch(g, cam.viewmat, cam.K, 96, 80)
    rgb2, a2 = render_torch(g, cam.viewmat, cam.K, 96, 80, max_band_pixels=96 * 17)   # 5 uneven bands
    assert torch.allclose(a1, a2, atol=1e-5) and torch.allclose(rgb1, rgb2, atol=1e-4)


def test_antialiased_reduces_opacity_of_tiny_gaussians():
    g = _scene()
    g["scales"] = torch.full_like(g["scales"], 0.002)   # far below one pixel
    cam = fixed_camera(64, 64, elevation_deg=25, azimuth_deg=10, fov_deg=50, fill=0.6)
    _, a_plain = render_torch(g, cam.viewmat, cam.K, 64, 64)
    _, a_aa = render_torch(g, cam.viewmat, cam.K, 64, 64, antialiased=True)
    assert a_aa.sum() < 0.5 * a_plain.sum()
