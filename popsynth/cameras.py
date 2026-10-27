"""Random camera sampling around a normalised object (OpenCV convention: x right, y down, z forward)."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class CameraSample:
    viewmat: np.ndarray  # (4,4) world -> camera
    K: np.ndarray        # (3,3) intrinsics
    width: int
    height: int
    elevation_deg: float
    azimuth_deg: float
    roll_deg: float
    fov_deg: float
    distance: float
    fill: float

    def meta(self) -> dict:
        d = asdict(self)
        d.pop("viewmat"); d.pop("K")
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / max(np.linalg.norm(v), 1e-12)


def orthonormal_basis(up) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (u1, u2, up) right-handed basis with the given up vector."""
    up = _unit(up)
    helper = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
    u1 = _unit(np.cross(helper, up))
    u2 = np.cross(up, u1)
    return u1, u2, up


def look_at(cam_pos, target, up, roll_deg: float = 0.0) -> np.ndarray:
    """World->camera 4x4 for an OpenCV camera at cam_pos looking at target."""
    cam_pos = np.asarray(cam_pos, np.float64); target = np.asarray(target, np.float64)
    z = _unit(target - cam_pos)                     # forward
    up = _unit(up)
    if abs(np.dot(z, up)) > 0.999:                  # looking straight along up: pick any perpendicular
        up = orthonormal_basis(z)[0]
    x = _unit(np.cross(z, up))                      # right  (y_c = -up  ->  x = y x z = (-up) x z = z x up)
    y = np.cross(z, x)                              # down
    if roll_deg:
        r = np.deg2rad(roll_deg); c, s = np.cos(r), np.sin(r)
        x, y = c * x + s * y, -s * x + c * y
    R_c2w = np.stack([x, y, z], 1)                  # columns = camera axes in world coords
    R_w2c = R_c2w.T
    t = -R_w2c @ cam_pos
    M = np.eye(4)
    M[:3, :3] = R_w2c
    M[:3, 3] = t
    return M.astype(np.float32)


def sphere_distance(radius: float, fov_deg: float, fill: float) -> float:
    """Camera distance at which a sphere of `radius` covers `fill` of the shorter image side."""
    theta = np.arctan(fill * np.tan(np.deg2rad(fov_deg) / 2))
    return float(radius / np.sin(theta))


def intrinsics(width: int, height: int, fov_deg: float) -> np.ndarray:
    """Pinhole intrinsics; fov_deg is measured over the shorter image side."""
    f = 0.5 * min(width, height) / np.tan(np.deg2rad(fov_deg) / 2)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], np.float32)


def sample_camera(
    rng: np.random.Generator,
    *,
    width: int,
    height: int,
    up=(0, 0, 1),
    object_radius: float = 1.0,
    elevation_deg=(-30, 90),
    azimuth_deg=(0, 360),
    roll_deg=(-180, 180),
    fov_deg=(25, 70),
    fill=(0.6, 0.9),
    center_jitter: float = 0.05,
) -> CameraSample:
    """Sample a camera on a sphere around the origin.

    elevation: angle above the plane perpendicular to `up` (90 = straight down onto the back).
    fill: fraction of the shorter image side the object's bounding sphere should cover.
    Distance follows from fov and fill, so a small fov + far camera gives a flat telephoto
    look and a wide fov + close camera gives strong perspective, at the same image size.
    The sphere's silhouette subtends asin(r/d), so d = r / sin(theta) with tan(theta) = fill*tan(fov/2);
    object_radius should be the true maximum extent (not a quantile) for nothing to be clipped.
    """
    u1, u2, upv = orthonormal_basis(up)
    el = np.deg2rad(rng.uniform(*elevation_deg))
    az = np.deg2rad(rng.uniform(*azimuth_deg))
    roll = float(rng.uniform(*roll_deg))
    fov = float(rng.uniform(*fov_deg))
    fl = float(rng.uniform(*fill))
    direction = np.cos(el) * np.cos(az) * u1 + np.cos(el) * np.sin(az) * u2 + np.sin(el) * upv
    dist = sphere_distance(object_radius, fov, fl)
    target = rng.normal(0, center_jitter * object_radius, 3) if center_jitter > 0 else np.zeros(3)
    cam_pos = target + dist * direction
    return CameraSample(
        viewmat=look_at(cam_pos, target, upv, roll),
        K=intrinsics(width, height, fov),
        width=width, height=height,
        elevation_deg=float(np.rad2deg(el)), azimuth_deg=float(np.rad2deg(az)), roll_deg=roll,
        fov_deg=fov, distance=float(dist), fill=fl,
    )


def fixed_camera(width, height, *, up=(0, 0, 1), object_radius=1.0, elevation_deg=45.0,
                 azimuth_deg=0.0, roll_deg=0.0, fov_deg=40.0, fill=0.8) -> CameraSample:
    """Deterministic camera, handy for previews and tests."""
    u1, u2, upv = orthonormal_basis(up)
    el, az = np.deg2rad(elevation_deg), np.deg2rad(azimuth_deg)
    direction = np.cos(el) * np.cos(az) * u1 + np.cos(el) * np.sin(az) * u2 + np.sin(el) * upv
    dist = sphere_distance(object_radius, fov_deg, fill)
    return CameraSample(look_at(dist * direction, np.zeros(3), upv, roll_deg), intrinsics(width, height, fov_deg),
                        width, height, float(elevation_deg), float(azimuth_deg), float(roll_deg), float(fov_deg),
                        float(dist), float(fill))
