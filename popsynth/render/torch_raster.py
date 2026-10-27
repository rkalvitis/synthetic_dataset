"""Pure-PyTorch Gaussian splat rasteriser (runs on CPU, Apple MPS or CUDA; no custom kernels).

Same maths as the 3DGS reference (EWA projection, 3-sigma footprint, front-to-back alpha
compositing) but implemented as a global (pixel, depth)-sorted list of gaussian/pixel pairs
with a segmented prefix-scan for the transmittance.  Exact rather than tile-approximate,
slower than gsplat: a few seconds per 512x512 frame for ~1M gaussians on an M-series Mac.
"""
from __future__ import annotations

import math

import torch

from .sh import eval_sh


def _release(device) -> None:
    """Return cached allocator memory between bands (large frames)."""
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """(N,4) wxyz -> (N,3,3)."""
    q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q.unbind(1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], 1).reshape(-1, 3, 3)


@torch.no_grad()
def render_torch(
    g: dict,
    viewmat,
    K,
    width: int,
    height: int,
    *,
    sh_degree: int | None = None,
    near: float = 0.01,
    eps2d: float = 0.3,
    sigma_cut: float = 3.0,
    alpha_min: float = 1.0 / 255,
    pair_budget: int = 6_000_000,
    scan_device: str = "cpu",
    antialiased: bool = False,
    max_band_pixels: int = 512 * 512,
    _clamp_hw: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render one view.  Returns (rgb (H,W,3) straight colour in [0,1], alpha (H,W)) on CPU.

    g: dict with means (N,3), quats (N,4), scales (N,3), opacities (N,), sh (N,K,3) torch tensors
       (already activated), all on the same device.  Projection and footprint generation run on
       that device; the sort/scan/accumulate stage runs on scan_device in float64 for stability.
    antialiased: Mip-Splatting style opacity compensation sqrt(det(cov)/det(cov+eps)).
    Frames larger than max_band_pixels are rendered in horizontal bands to bound memory
    (each band is an independent, exact render with a shifted principal point).
    """
    if width * height > max_band_pixels:
        rows = max(1, max_band_pixels // width)
        rgb_parts, a_parts = [], []
        Kb = torch.as_tensor(K, dtype=torch.float32).clone()
        for y0 in range(0, height, rows):
            h = min(rows, height - y0)
            Kb[1, 2] = float(K[1][2]) - y0
            r, a = render_torch(g, viewmat, Kb, width, h, sh_degree=sh_degree, near=near, eps2d=eps2d,
                                sigma_cut=sigma_cut, alpha_min=alpha_min, pair_budget=pair_budget,
                                scan_device=scan_device, antialiased=antialiased, max_band_pixels=max_band_pixels,
                                _clamp_hw=(width, height))
            rgb_parts.append(r); a_parts.append(a)
            _release(g["means"].device)
        return torch.cat(rgb_parts, 0), torch.cat(a_parts, 0)

    dev = g["means"].device
    means, quats, scales, opac, sh = g["means"], g["quats"], g["scales"], g["opacities"], g["sh"]
    N = means.shape[0]
    if sh_degree is None:
        sh_degree = int(round(math.sqrt(sh.shape[1]))) - 1
    vm = torch.as_tensor(viewmat, dtype=torch.float32, device=dev)
    Km = torch.as_tensor(K, dtype=torch.float32, device=dev)
    R, t = vm[:3, :3], vm[:3, 3]
    fx, fy, cx, cy = Km[0, 0], Km[1, 1], Km[0, 2], Km[1, 2]

    # ---- project centres
    p = means @ R.T + t
    z = p[:, 2]
    valid = z > near
    # ---- 2D covariance (EWA)
    tx = p[:, 0] / z.clamp_min(near)
    ty = p[:, 1] / z.clamp_min(near)
    clamp_w, clamp_h = _clamp_hw or (width, height)      # frustum clamp of the EWA Jacobian (full frame, not band)
    lim_x = 1.3 * (clamp_w / 2) / fx
    lim_y = 1.3 * (clamp_h / 2) / fy
    txc = tx.clamp(-lim_x, lim_x) * z
    tyc = ty.clamp(-lim_y, lim_y) * z
    zz = z.clamp_min(near)
    J = torch.zeros(N, 2, 3, device=dev)
    J[:, 0, 0] = fx / zz
    J[:, 0, 2] = -fx * txc / (zz * zz)
    J[:, 1, 1] = fy / zz
    J[:, 1, 2] = -fy * tyc / (zz * zz)
    Rg = quat_to_rotmat(quats)
    M = Rg * scales[:, None, :]                      # R S
    cov3d = M @ M.transpose(1, 2)
    T = J @ R[None]                                  # (N,2,3)
    cov2d = T @ cov3d @ T.transpose(1, 2)            # (N,2,2)
    a = cov2d[:, 0, 0] + eps2d
    b = cov2d[:, 0, 1]
    c = cov2d[:, 1, 1] + eps2d
    det = a * c - b * b
    valid &= det > 0
    if antialiased:
        det0 = cov2d[:, 0, 0] * cov2d[:, 1, 1] - b * b
        opac = opac * torch.sqrt((det0 / det.clamp_min(1e-12)).clamp(0, 1))
    inv_det = 1.0 / det.clamp_min(1e-12)
    con_a, con_b, con_c = c * inv_det, -b * inv_det, a * inv_det
    mid = 0.5 * (a + c)
    lam1 = mid + torch.sqrt((mid * mid - det).clamp_min(0.1))
    radius = torch.ceil(sigma_cut * torch.sqrt(lam1))
    mx = fx * tx + cx
    my = fy * ty + cy
    x0 = (mx - radius).floor().clamp(0, width - 1)
    x1 = (mx + radius).ceil().clamp(0, width - 1)
    y0 = (my - radius).floor().clamp(0, height - 1)
    y1 = (my + radius).ceil().clamp(0, height - 1)
    valid &= (mx + radius >= 0) & (mx - radius <= width) & (my + radius >= 0) & (my - radius <= height)
    valid &= (x1 >= x0) & (y1 >= y0) & (opac >= alpha_min)

    idx = valid.nonzero(as_tuple=True)[0]
    out_rgb = torch.zeros(height * width, 3, dtype=torch.float64, device=scan_device)
    out_a = torch.zeros(height * width, dtype=torch.float64, device=scan_device)
    if idx.numel() == 0:
        return out_rgb.view(height, width, 3).float(), out_a.view(height, width).float()

    # ---- colours from SH, view direction per gaussian
    cam_pos = -R.T @ t
    dirs = means[idx] - cam_pos
    dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-8)
    colors = (eval_sh(sh_degree, sh[idx], dirs) + 0.5).clamp_min(0.0)   # (V,3)

    # ---- depth rank for the sort key
    V = idx.numel()
    order = torch.argsort(z[idx])
    rank = torch.empty(V, dtype=torch.int64, device=dev)
    rank[order] = torch.arange(V, device=dev)

    x0v, x1v, y0v, y1v = x0[idx].long(), x1[idx].long(), y0[idx].long(), y1[idx].long()
    wv = x1v - x0v + 1
    area = wv * (y1v - y0v + 1)
    mxv, myv, cav, cbv, ccv, opv = mx[idx], my[idx], con_a[idx], con_b[idx], con_c[idx], opac[idx]

    # ---- generate (pixel, gaussian) pairs in chunks bounded by pair_budget
    cum = torch.cumsum(area, 0)
    keys, alphas, gids = [], [], []
    start = 0
    total = int(cum[-1])
    while start < V:
        target = int(cum[start]) - int(area[start]) + pair_budget
        end = int(torch.searchsorted(cum, torch.tensor(target, device=dev), right=True))
        end = max(end, start + 1)
        sl = slice(start, end)
        ar = area[sl]
        G = end - start
        rep = torch.repeat_interleave(torch.arange(G, device=dev), ar)
        offs = torch.cumsum(ar, 0) - ar
        local = torch.arange(rep.numel(), device=dev) - offs[rep]
        w_rep = wv[sl][rep]
        px = x0v[sl][rep] + local % w_rep
        py = y0v[sl][rep] + local // w_rep
        dx = px.float() + 0.5 - mxv[sl][rep]
        dy = py.float() + 0.5 - myv[sl][rep]
        power = -0.5 * (cav[sl][rep] * dx * dx + ccv[sl][rep] * dy * dy) - cbv[sl][rep] * dx * dy
        alpha = (opv[sl][rep] * torch.exp(power)).clamp_max(0.99)
        keep = (power <= 0) & (alpha >= alpha_min)
        if keep.any():
            g_local = rep[keep] + start
            pix = py[keep] * width + px[keep]
            keys.append(pix * V + rank[g_local])
            alphas.append(alpha[keep])
            gids.append(g_local)
        start = end
    if not keys:
        return out_rgb.view(height, width, 3).float(), out_a.view(height, width).float()

    key = torch.cat(keys).to(scan_device)
    alpha = torch.cat(alphas).to(scan_device).double()   # .to(dev, float64) in one step silently zeroes MPS tensors
    gid = torch.cat(gids).to(scan_device)
    del keys, alphas, gids

    # ---- sort by (pixel, depth) and segmented prefix product of (1 - alpha)
    key, perm = torch.sort(key)
    alpha = alpha[perm]
    gid = gid[perm]
    pix = key // V
    log1m = torch.log1p(-alpha)
    cs = torch.cumsum(log1m, 0)
    excl = cs - log1m
    is_start = torch.ones_like(pix, dtype=torch.bool)
    is_start[1:] = pix[1:] != pix[:-1]
    seg = torch.cumsum(is_start.long(), 0) - 1
    start_excl = excl[is_start][seg]
    trans = torch.exp(excl - start_excl)
    wgt = alpha * trans
    col = colors.to(scan_device).double()[gid]
    out_rgb.index_add_(0, pix, wgt[:, None] * col)
    out_a.index_add_(0, pix, wgt)

    out_a = out_a.clamp(0, 1)
    rgb = out_rgb / out_a.clamp_min(1e-6)[:, None]
    return rgb.view(height, width, 3).float().clamp(0, 1), out_a.view(height, width).float()
