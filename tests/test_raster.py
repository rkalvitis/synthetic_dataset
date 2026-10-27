import numpy as np
import torch

from popsynth.cameras import fixed_camera
from popsynth.render.torch_raster import render_torch, quat_to_rotmat


def _one_gaussian(color=(1.0, 0.0, 0.0), opacity=0.9, scale=0.1, pos=(0, 0, 0)):
    c0 = 0.28209479177387814
    sh = torch.zeros(1, 16, 3)
    sh[0, 0] = (torch.tensor(color) - 0.5) / c0
    return {"means": torch.tensor([pos], dtype=torch.float32), "quats": torch.tensor([[1.0, 0, 0, 0]]),
            "scales": torch.full((1, 3), scale), "opacities": torch.tensor([opacity]), "sh": sh}


def test_quat_identity():
    assert torch.allclose(quat_to_rotmat(torch.tensor([[1.0, 0, 0, 0]]))[0], torch.eye(3))


def test_single_gaussian_centre_colour_and_alpha():
    g = _one_gaussian(scale=0.3)   # projected sigma ~5 px, so the half-pixel offset of the centre barely matters
    cam = fixed_camera(64, 64, elevation_deg=30, azimuth_deg=40, fov_deg=40, fill=0.5)
    rgb, a = render_torch(g, cam.viewmat, cam.K, 64, 64)
    assert a.shape == (64, 64)
    # the gaussian sits at the look-at target: peak alpha at the image centre, equal to its opacity
    yx = np.unravel_index(int(a.argmax()), a.shape)
    assert abs(yx[0] - 31.5) < 1.5 and abs(yx[1] - 31.5) < 1.5
    assert abs(a.max().item() - 0.9) < 0.03
    assert torch.allclose(rgb[yx], torch.tensor([1.0, 0, 0]), atol=1e-3)
    assert a[0, 0] == 0


def test_front_to_back_compositing():
    # opaque-ish red in front of green: centre pixel is mostly red, alpha accumulates
    c0 = 0.28209479177387814
    sh = torch.zeros(2, 16, 3)
    sh[0, 0] = (torch.tensor([1.0, 0, 0]) - 0.5) / c0
    sh[1, 0] = (torch.tensor([0, 1.0, 0]) - 0.5) / c0
    g = {"means": torch.tensor([[0, 0, -0.3], [0, 0, 0.3]]), "quats": torch.tensor([[1.0, 0, 0, 0]] * 2),
         "scales": torch.full((2, 3), 0.5), "opacities": torch.tensor([0.6, 0.9]), "sh": sh}   # wide: ~4 px sigma
    # with up=(0,1,0) azimuth 0 puts the camera on +z looking at the origin: z=+0.3 is closer (green, opacity .9)
    cam = fixed_camera(32, 32, up=(0, 1, 0), elevation_deg=0, azimuth_deg=0, fov_deg=40, fill=0.5)
    cam_pos = -cam.viewmat[:3, :3].T @ cam.viewmat[:3, 3]
    assert cam_pos[2] > 0 and abs(cam_pos[0]) < 1e-6
    rgb, a = render_torch(g, cam.viewmat, cam.K, 32, 32)
    c = rgb[16, 16]
    expected_alpha = 0.9 + 0.1 * 0.6          # front-to-back: a_g + (1 - a_g) * a_r
    assert abs(a[16, 16].item() - expected_alpha) < 0.03
    assert c[1] > 0.85 and c[0] < 0.15   # green in front
