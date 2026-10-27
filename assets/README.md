# assets/

Put the Gaussian-splat PLY file(s) here. They are git-ignored because of their size.

Expected format: the standard 3DGS `point_cloud.ply` layout (x y z, nx ny nz,
f_dc_0..2, f_rest_0..44, opacity, scale_0..2, rot_0..3), as written by the
official Gaussian Splatting trainer, gsplat, nerfstudio/splatfacto, Postshot etc.

Current model: `popillia_ex15.ply` (928k gaussians, Oct 2026 reconstruction).
