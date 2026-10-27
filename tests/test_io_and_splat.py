import numpy as np

from popsynth.io_ply import read_gaussian_ply
from popsynth.splat import Splat


def write_fake_ply(path, n=50, seed=0):
    rng = np.random.default_rng(seed)
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{i}" for i in range(45)] + \
            ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    data = np.zeros(n, dtype=np.dtype([(k, "<f4") for k in names]))
    for k in names:
        data[k] = rng.normal(size=n)
    data["rot_0"] = 1.0
    with open(path, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n".encode()
                + b"".join(f"property float {k}\n".encode() for k in names) + b"end_header\n")
        f.write(data.tobytes())
    return data


def test_read_ply_roundtrip(tmp_path):
    p = tmp_path / "a.ply"
    data = write_fake_ply(str(p))
    d = read_gaussian_ply(str(p))
    assert np.allclose(d["x"], data["x"]) and np.allclose(d["f_rest_44"], data["f_rest_44"])


def test_splat_load_activations_and_sh_layout(tmp_path):
    p = tmp_path / "a.ply"
    data = write_fake_ply(str(p))
    s = Splat.load(str(p))
    assert s.sh.shape == (50, 16, 3) and s.sh_degree == 3
    assert np.allclose(s.opacities, 1 / (1 + np.exp(-data["opacity"])))
    assert np.allclose(s.scales[:, 1], np.exp(data["scale_1"]))
    # f_rest is channel-major in the file: coefficient k of channel c is f_rest_{c*15 + k-1}
    assert np.allclose(s.sh[:, 3, 2], data["f_rest_32"])
    assert np.allclose(np.linalg.norm(s.quats, axis=1), 1)


def test_normalized_and_clean(tmp_path):
    p = tmp_path / "a.ply"
    write_fake_ply(str(p), n=400)
    s = Splat.load(str(p)).normalized()
    assert abs(s.bounding_radius() - 1.0) < 1e-4
    assert np.linalg.norm(s.center()) < 1e-4
    c = s.clean(opacity_min=0.5)
    assert len(c) < len(s) and (c.opacities >= 0.5).all()
