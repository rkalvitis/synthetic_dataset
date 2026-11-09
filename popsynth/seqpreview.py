"""Visual checks of generated sequences: an animated GIF per sequence with the ground-truth boxes drawn
(target green, distractors grey, full/amodal box thin yellow, frame index and ABSENT flag), and a contact sheet.

    popsynth sequence-gifs -o <root> -n 12            # GIFs of the first 12 sequences -> <root>/_gifs/
    popsynth sequence-gifs -o <root> --sheet           # + one contact sheet per sequence
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

GREEN, GREY, YELLOW, RED, WHITE = (40, 220, 60), (170, 170, 170), (240, 210, 40), (230, 50, 50), (255, 255, 255)


def _load(seq_dir: str) -> tuple[dict, dict, list[dict]]:
    with open(os.path.join(seq_dir, "meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(seq_dir, "tracks.json")) as f:
        tracks = json.load(f)
    frames = [json.loads(l) for l in open(os.path.join(seq_dir, "frames.jsonl"))]
    return meta, tracks, frames


def draw_frame(img: np.ndarray, rec: dict, tracks: dict, f: int, scale: float = 1.0) -> np.ndarray:
    """Boxes of frame `f` on an RGB uint8 image already resized by `scale`."""
    out = np.ascontiguousarray(img)
    t = max(1, int(round(2 * scale)))
    for o in rec["objects"]:
        k = str(o["object_id"])
        col = GREEN if k == "0" else GREY
        bf = o.get("bbox_full_xyxy")
        if bf and k == "0":
            x0, y0, x1, y1 = [int(round(v * scale)) for v in bf]
            cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), YELLOW, 1)
        b = tracks[k]["boxes"][f]                       # xyxy (groundtruth.txt holds the same box as x,y,w,h)
        if b[2] > b[0] and not tracks[k]["absence"][f]:
            x0, y0, x1, y1 = [int(round(v * scale)) for v in b]
            cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), col, t)
    label = f"{f + 1}/{len(tracks['0']['boxes'])}"
    if tracks["0"]["absence"][f]:
        label += "  ABSENT"
    cv2.putText(out, label, (4, int(14 * max(scale, 0.6)) + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * max(scale, 0.6),
                RED if tracks["0"]["absence"][f] else WHITE, 1, cv2.LINE_AA)
    return out


def sequence_gif(seq_dir: str, out_path: str, max_side: int = 240, frame_step: int = 2, fps: float | None = None) -> str:
    """Animated GIF of one sequence folder with the boxes drawn. Every `frame_step`-th frame, longest side `max_side`."""
    from PIL import Image
    meta, tracks, frames = _load(seq_dir)
    W, H = meta["frame_size"]
    scale = min(1.0, max_side / max(W, H))
    size = (max(1, int(round(W * scale))), max(1, int(round(H * scale))))
    fps = fps or float(meta.get("fps", 30)) / max(frame_step, 1)
    pil = []
    for f in range(0, len(frames), max(1, frame_step)):
        img = cv2.imread(os.path.join(seq_dir, frames[f]["file"]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(cv2.resize(img, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        pil.append(Image.fromarray(draw_frame(img, frames[f], tracks, f, scale)))
    if not pil:
        raise RuntimeError(f"no frames in {seq_dir}")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    pil[0].save(out_path, save_all=True, append_images=pil[1:], duration=int(round(1000 / fps)), loop=0, optimize=True)
    return out_path


def sequence_sheet(seq_dir: str, out_path: str, n: int = 12, tile: int = 256) -> str:
    """Contact sheet: `n` frames spread over the sequence, boxes drawn."""
    meta, tracks, frames = _load(seq_dir)
    W, H = meta["frame_size"]
    scale = tile / max(W, H)
    idx = np.linspace(0, len(frames) - 1, min(n, len(frames))).round().astype(int)
    tiles = []
    for f in idx:
        img = cv2.imread(os.path.join(seq_dir, frames[f]["file"]), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (int(round(W * scale)), int(round(H * scale))), interpolation=cv2.INTER_AREA)
        tiles.append(draw_frame(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), frames[f], tracks, int(f), scale))
    cols = min(6, len(tiles))
    rows = int(np.ceil(len(tiles) / cols))
    th, tw = tiles[0].shape[:2]
    sheet = np.full((rows * th + 24, cols * tw, 3), 255, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[24 + r * th:24 + (r + 1) * th, c * tw:(c + 1) * tw] = t
    txt = (f"{meta['name']} ({meta['split']}): {meta['num_frames']} frames, {meta['num_objects']} beetles, "
           f"{meta['num_occluders']} occluders, camera {'static' if meta['camera_static'] else 'moving'}, "
           f"{meta['lighting']} / {meta.get('quality_tier')}")
    cv2.putText(sheet, txt, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cv2.imwrite(out_path, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out_path


def make_sequence_gifs(root: str, out_dir: str | None = None, n: int | None = 12, max_side: int = 240,
                       frame_step: int = 2, sheet: bool = False) -> list[str]:
    """GIFs (and optionally sheets) of the first `n` sequences listed in <root>/sequences.jsonl (None = all)."""
    recs = [json.loads(l) for l in open(os.path.join(root, "sequences.jsonl"))]
    recs = sorted(recs, key=lambda r: r["index"])[: n or len(recs)]
    out_dir = out_dir or os.path.join(root, "_gifs")
    paths = []
    for r in recs:
        seq_dir = os.path.join(root, r["split"], r["name"])
        paths.append(sequence_gif(seq_dir, os.path.join(out_dir, r["split"], r["name"] + ".gif"), max_side, frame_step))
        if sheet:
            paths.append(sequence_sheet(seq_dir, os.path.join(out_dir, r["split"], r["name"] + "_sheet.jpg")))
    return paths
