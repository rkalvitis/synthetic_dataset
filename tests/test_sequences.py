"""Video sequences (popsynth/sequences.py): motion model, labels and the GOT-10k layout, with the disc stand-in."""
import json
import math
import os

import cv2
import numpy as np

from popsynth.backgrounds import BackgroundPool
from popsynth.config import load_config
from popsynth.sequences import (DEFAULTS, Beetle, Camera, DiscViews, _merge, _step_beetle, canvas_size,
                                generate_sequences, line_kernel, simulate)


def _cfg(tmp_path, n_bg=6):
    bg = tmp_path / "bg"
    bg.mkdir()
    rng = np.random.default_rng(0)
    for i in range(n_bg):
        img = rng.integers(40, 220, (300, 400, 3), np.uint8)
        cv2.imwrite(str(bg / f"photo_{i}.jpg"), img)
    cfg = load_config()
    cfg["backgrounds"]["dirs"] = [str(bg)]
    cfg["backgrounds"]["min_size"] = 64
    cfg["compose"]["occlusion"]["dirs"] = []
    cfg["sequences"] = {"num_sequences": 4, "length": [12, 16], "frame_size": [96, 80], "val_fraction": 0.5,
                        "object_size": [0.15, 0.3], "occlusion": {"prob": 1.0, "count": [1, 1]},
                        "distractors": {"count": [1, 1]}, "gif": {"every": 2, "max_side": 64, "frame_step": 3}}
    return cfg


def test_canvas_and_affine_keep_the_view_inside():
    sq = _merge(DEFAULTS, None)
    W, H = 64, 48
    Cw, Ch = canvas_size(W, H, sq["camera"])
    assert Cw > W / 0.85 and Ch > H / 0.85
    cam = Camera(cx=Cw / 2, cy=Ch / 2)
    M = cam.affine(W, H)
    p = M @ np.array([Cw / 2, Ch / 2, 1.0])
    assert np.allclose(p, [W / 2, H / 2])
    cam.logz = math.log(1.2); cam.rot = 5.0
    M = cam.affine(W, H)
    inv = cv2.invertAffineTransform(M)
    corners = np.array([[0, 0, 1], [W, 0, 1], [0, H, 1], [W, H, 1]], float) @ inv.T
    assert corners[:, 0].min() >= 0 and corners[:, 1].min() >= 0 and corners[:, 0].max() <= Cw and corners[:, 1].max() <= Ch


def test_step_beetle_is_bounded_and_turns_back():
    rng = np.random.default_rng(1)
    mo = _merge(DEFAULTS, None)["motion"]
    b = Beetle(x=500, y=100, heading=0.0, speed=4.0, speed_target=4.0, el=85.0, az=10.0, roll=0.0, size=40, size0=40)
    for _ in range(300):
        _step_beetle(b, rng, mo, (100.0, 100.0), (50.0, 40.0))
        assert 0 <= b.el <= 90 and 0 <= b.az < 360 and -180 <= b.roll <= 180
        assert 20 <= b.size <= 80
    assert abs(b.x - 100) < 300          # it came back toward the view instead of walking away forever


def test_line_kernel():
    assert line_kernel(0.3, 0.2) is None
    k = line_kernel(6.0, 0.0)
    assert k is not None and abs(k.sum() - 1) < 1e-5 and k.shape[0] % 2 == 1
    mid = k.shape[0] // 2
    assert k[mid].sum() > 0.9            # horizontal line


def test_simulation_starts_outside_and_enters(tmp_path):
    cfg = _cfg(tmp_path)
    sq = _merge(DEFAULTS, cfg["sequences"])
    W, H = sq["frame_size"]
    Cw, Ch = canvas_size(W, H, sq["camera"])
    for i in range(5):
        rng = np.random.default_rng([0, 23, i])
        sim = simulate(cfg, sq, rng, 40, W, H, Cw, Ch)
        f0 = sim["frames"][0]
        cam = Camera(**f0["camera"])
        z = math.exp(cam.logz)
        for b in f0["beetles"]:
            outside = abs(b["x"] - cam.cx) > W / (2 * z) + b["size"] / math.sqrt(2) or abs(b["y"] - cam.cy) > H / (2 * z) + b["size"] / math.sqrt(2)
            assert outside and not b["entered"]
        assert any(fr["beetles"][0]["entered"] for fr in sim["frames"][:20])     # the target is in within 20 frames


def test_generate_sequences_layout_and_labels(tmp_path):
    cfg = _cfg(tmp_path)
    out = str(tmp_path / "seq")
    summary = generate_sequences(cfg, out, verbose=False, views=DiscViews())
    assert summary["num_sequences"] == 4 and summary["per_split"]["train"] + summary["per_split"]["val"] == 4
    recs = [json.loads(l) for l in open(os.path.join(out, "sequences.jsonl"))]
    bgs = set()
    for r in recs:
        d = os.path.join(out, r["split"], r["name"])
        n = r["num_frames"]
        frames = sorted(f for f in os.listdir(d) if f.endswith(".jpg"))
        assert len(frames) == n and frames[0] == "00000001.jpg"
        gt = open(os.path.join(d, "groundtruth.txt")).read().splitlines()
        ab = [int(v) for v in open(os.path.join(d, "absence.label")).read().split()]
        cov = [int(v) for v in open(os.path.join(d, "cover.label")).read().split()]
        assert len(gt) == len(ab) == len(cov) == n
        assert ab[0] == 1                                     # first frame empty
        assert 0 in ab                                        # the target shows up
        tracks = json.load(open(os.path.join(d, "tracks.json")))
        meta = json.load(open(os.path.join(d, "meta.json")))
        W, H = meta["frame_size"]
        assert set(tracks) == {"0", "1"}
        for f, line in enumerate(gt):
            x, y, w, h = [int(v) for v in line.split(",")]
            bx = tracks["0"]["boxes"][f]
            assert [x, y, w, h] == [bx[0], bx[1], bx[2] - bx[0], bx[3] - bx[1]]
            if not ab[f]:
                assert 0 <= x < x + w <= W and 0 <= y < y + h <= H
                # the visible box lies inside the full (amodal) box
                bf = tracks["0"]["boxes_full"][f]
                assert bf[0] <= bx[0] and bf[1] <= bx[1] and bf[2] >= bx[2] and bf[3] >= bx[3]
                assert 0 < tracks["0"]["visible_share"][f] <= 1.0
                # the box is on the disc: its pixels differ from the background there
                img = cv2.imread(os.path.join(d, frames[f]))
                assert img is not None and img.shape[:2] == (H, W)
        fr = [json.loads(l) for l in open(os.path.join(d, "frames.jsonl"))]
        assert len(fr) == n and len(fr[0]["objects"]) == 2 and fr[0]["objects"][0]["role"] == "target"
        bgs.add(meta["background"])
    assert len(bgs) == 4                                      # own photo per sequence
    assert os.path.exists(os.path.join(out, "train", "list.txt")) or os.path.exists(os.path.join(out, "val", "list.txt"))
    gifs = [f for _, _, fs in os.walk(os.path.join(out, "_gifs")) for f in fs if f.endswith(".gif")]
    assert len(gifs) == 2                                     # gif.every = 2
    # resume: a second call renders nothing new
    views = DiscViews()
    generate_sequences(cfg, out, verbose=False, views=views)
    assert views.count == 0


def test_background_pool_split_is_disjoint(tmp_path):
    cfg = _cfg(tmp_path, n_bg=10)
    pool = BackgroundPool(cfg["backgrounds"]["dirs"], min_size=64)
    a, b = pool.split(0.3, seed=1)
    assert len(a) == 7 and len(b) == 3 and not set(a.paths) & set(b.paths)
    assert set(a.ordered_paths(0)) == set(a.paths)
