"""gsplat (CUDA) backend. Install with `pip install gsplat` on a machine with an NVIDIA GPU."""
from __future__ import annotations

import math

import torch


@torch.no_grad()
def render_gsplat(g: dict, viewmat, K, width: int, height: int, *, sh_degree: int | None = None,
                  near: float = 0.01, eps2d: float = 0.3, antialiased: bool = False):
    """Render one view with gsplat.rasterization.  Returns (rgb (H,W,3) straight, alpha (H,W)) on CPU."""
    import gsplat  # noqa: WPS433  (optional dependency)

    dev = g["means"].device
    sh = g["sh"]
    if sh_degree is None:
        sh_degree = int(round(math.sqrt(sh.shape[1]))) - 1
    vm = torch.as_tensor(viewmat, dtype=torch.float32, device=dev)[None]
    Km = torch.as_tensor(K, dtype=torch.float32, device=dev)[None]
    colors, alphas, _ = gsplat.rasterization(
        g["means"], g["quats"], g["scales"], g["opacities"], sh,
        vm, Km, width, height,
        sh_degree=sh_degree, near_plane=near, eps2d=eps2d, packed=False,
        render_mode="RGB", backgrounds=None,
        rasterize_mode="antialiased" if antialiased else "classic",
    )
    rgb_premul = colors[0, ..., :3]          # black background => premultiplied colour
    a = alphas[0, ..., 0].clamp(0, 1)
    rgb = (rgb_premul / a.clamp_min(1e-6)[..., None]).clamp(0, 1)
    return rgb.cpu(), a.cpu()
