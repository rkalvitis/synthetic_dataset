import numpy as np

from popsynth.cameras import fixed_camera, look_at, sample_camera


def test_look_at_puts_target_on_optical_axis():
    cam = np.array([3.0, -2.0, 1.5]); target = np.array([0.1, 0.2, -0.3])
    M = look_at(cam, target, up=(0, 0, 1))
    t_cam = M[:3, :3] @ target + M[:3, 3]
    assert abs(t_cam[0]) < 1e-5 and abs(t_cam[1]) < 1e-5 and t_cam[2] > 0
    c_cam = M[:3, :3] @ cam + M[:3, 3]
    assert np.allclose(c_cam, 0, atol=1e-6)
    # rotation part is orthonormal, right handed
    R = M[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6) and np.linalg.det(R) > 0


def test_up_maps_to_image_up():
    # world up should project to -y (upwards) in an OpenCV camera looking horizontally
    M = look_at([5, 0, 0], [0, 0, 0], up=(0, 0, 1))
    up_cam = M[:3, :3] @ np.array([0, 0, 1.0])
    assert up_cam[1] < -0.99


def test_fill_controls_projected_radius():
    cam = fixed_camera(400, 400, fov_deg=50, fill=0.8, elevation_deg=20, azimuth_deg=70)
    f = cam.K[0, 0]
    # silhouette of a unit sphere at distance d has angular radius asin(1/d)
    proj_r = f * np.tan(np.arcsin(1.0 / cam.distance))
    assert abs(proj_r / 200 - 0.8) < 1e-6


def test_sphere_distance_increases_with_perspective_correction():
    from popsynth.cameras import sphere_distance
    assert sphere_distance(1.0, 60, 0.9) > 1.0 / (0.9 * np.tan(np.deg2rad(30)))


def test_sample_camera_ranges():
    rng = np.random.default_rng(0)
    for _ in range(100):
        c = sample_camera(rng, width=64, height=64, up=(0, 1, 0), elevation_deg=(10, 20), fov_deg=(30, 40))
        assert 10 <= c.elevation_deg <= 20 and 30 <= c.fov_deg <= 40
        cam_pos = -c.viewmat[:3, :3].T @ c.viewmat[:3, 3]
        assert abs(np.linalg.norm(cam_pos) - c.distance) < 0.2  # target jitter
