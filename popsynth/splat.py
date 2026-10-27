"""Gaussian splat container: loading, cleaning, normalisation."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .io_ply import read_gaussian_ply


@dataclass
class Splat:
    """Activated gaussian parameters in numpy.

    means      (N,3) float32
    quats      (N,4) float32, wxyz, unit norm
    scales     (N,3) float32, already exp()-activated (world units)
    opacities  (N,)  float32, already sigmoid-activated in [0,1]
    sh         (N,K,3) float32 spherical-harmonic colour coefficients, K=(deg+1)^2
    """

    means: np.ndarray
    quats: np.ndarray
    scales: np.ndarray
    opacities: np.ndarray
    sh: np.ndarray

    # ------------------------------------------------------------------ I/O
    @classmethod
    def load(cls, path: str) -> "Splat":
        p = read_gaussian_ply(path)
        n = len(p["x"])
        means = np.stack([p["x"], p["y"], p["z"]], 1).astype(np.float32)
        quats = np.stack([p["rot_0"], p["rot_1"], p["rot_2"], p["rot_3"]], 1).astype(np.float32)
        quats /= np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-8)
        scales = np.exp(np.stack([p["scale_0"], p["scale_1"], p["scale_2"]], 1)).astype(np.float32)
        opac = (1.0 / (1.0 + np.exp(-p["opacity"]))).astype(np.float32)

        dc = np.stack([p["f_dc_0"], p["f_dc_1"], p["f_dc_2"]], 1)[:, None, :]  # (N,1,3)
        rest_keys = sorted((k for k in p if k.startswith("f_rest_")), key=lambda k: int(k.split("_")[-1]))
        if rest_keys:
            rest = np.stack([p[k] for k in rest_keys], 1)  # (N, 3*(K-1)) channel-major
            k_rest = rest.shape[1] // 3
            rest = rest.reshape(n, 3, k_rest).transpose(0, 2, 1)  # (N, K-1, 3)
            sh = np.concatenate([dc, rest], 1)
        else:
            sh = dc
        return cls(means, quats, scales, opac, sh.astype(np.float32))

    # ------------------------------------------------------------ properties
    def __len__(self) -> int:
        return len(self.means)

    @property
    def sh_degree(self) -> int:
        return int(round(np.sqrt(self.sh.shape[1]))) - 1

    def subset(self, mask: np.ndarray) -> "Splat":
        return Splat(self.means[mask], self.quats[mask], self.scales[mask], self.opacities[mask], self.sh[mask])

    # -------------------------------------------------------------- cleaning
    def clean(
        self,
        opacity_min: float = 0.0,
        max_scale: float | None = None,
        radius_crop: float | None = None,
        box: list[list[float]] | None = None,
        center_opacity: float = 0.5,
    ) -> "Splat":
        """Drop gaussians that are (almost) transparent, oversized, or far from the object.

        radius_crop is relative to the object's bounding radius (see bounding_radius()).
        box is [[xmin,ymin,zmin],[xmax,ymax,zmax]] in the splat's own coordinates.
        """
        keep = self.opacities >= opacity_min
        if max_scale is not None:
            keep &= self.scales.max(1) <= max_scale * self.bounding_radius()
        if radius_crop is not None:
            c = self.center(center_opacity)
            keep &= np.linalg.norm(self.means - c, axis=1) <= radius_crop * self.bounding_radius()
        if box is not None:
            lo, hi = np.asarray(box[0], np.float32), np.asarray(box[1], np.float32)
            keep &= np.all((self.means >= lo) & (self.means <= hi), axis=1)
        return self.subset(keep)

    def center(self, opacity_min: float = 0.5) -> np.ndarray:
        """Robust object centre: median of the reasonably opaque gaussians."""
        m = self.opacities >= opacity_min
        if m.sum() < 10:
            m = np.ones(len(self), bool)
        return np.median(self.means[m], axis=0).astype(np.float32)

    def bounding_radius(self, quantile: float = 0.995, opacity_min: float = 0.5) -> float:
        m = self.opacities >= opacity_min
        if m.sum() < 10:
            m = np.ones(len(self), bool)
        d = np.linalg.norm(self.means[m] - self.center(opacity_min), axis=1)
        return float(np.quantile(d, quantile))

    def max_extent(self, opacity_min: float = 0.05, sigmas: float = 2.0) -> float:
        """Radius of the smallest origin-centred sphere containing every visible gaussian (+ `sigmas` of its size)."""
        m = self.opacities >= opacity_min
        if not m.any():
            m = np.ones(len(self), bool)
        d = np.linalg.norm(self.means[m], axis=1) + sigmas * self.scales[m].max(1)
        return float(d.max())

    def principal_axes(self, opacity_min: float = 0.5) -> np.ndarray:
        """Columns = principal axes sorted by decreasing variance (3,3). Useful to guess 'up'."""
        m = self.opacities >= opacity_min
        pts = self.means[m] - self.center(opacity_min)
        _, _, vt = np.linalg.svd(pts, full_matrices=False)
        return vt.T.astype(np.float32)

    # --------------------------------------------------------- normalisation
    def normalized(self) -> "Splat":
        """Translate the centre to the origin and scale so the bounding radius is 1.

        Only translation + uniform scale are applied, which leaves the SH colours valid.
        Orientation is handled on the camera side (see cameras.py, `up` vector).
        """
        c = self.center()
        r = self.bounding_radius()
        s = np.float32(1.0 / max(r, 1e-8))
        return replace(self, means=(self.means - c) * s, scales=self.scales * s)

    # ------------------------------------------------------------- to torch
    def to_torch(self, device="cpu", dtype=None):
        import torch

        dtype = dtype or torch.float32
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device=device, dtype=dtype)
        return {
            "means": t(self.means),
            "quats": t(self.quats),
            "scales": t(self.scales),
            "opacities": t(self.opacities),
            "sh": t(self.sh),
        }
