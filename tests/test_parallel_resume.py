"""Worker count resolution and the resume plan of the compose stage (no rendering)."""
import json
import os

from popsynth.generate import _resume_plan, resolve_workers


def test_resolve_workers():
    assert resolve_workers(None) == 1 and resolve_workers(1) == 1 and resolve_workers("4") == 4 and resolve_workers(6) == 6
    assert 1 <= resolve_workers("auto") <= 16


def test_resume_plan_keeps_valid_records_and_merges_parts(tmp_path):
    root = str(tmp_path)
    os.makedirs(os.path.join(root, "images/train/popillia"))
    def rec(i, exists=True):
        img = f"images/train/popillia/img_{i:06d}.jpg"
        if exists:
            open(os.path.join(root, img), "wb").write(b"x")
        return json.dumps({"name": f"img_{i:06d}", "image": img})
    with open(os.path.join(root, "metadata.jsonl"), "w") as f:
        f.write(rec(0) + "\n" + rec(1) + "\n" + rec(2, exists=False) + "\n" + rec(3)[:20])   # 2: file missing, 3: truncated
    with open(os.path.join(root, "metadata.part1.jsonl"), "w") as f:                          # leftover of a killed parallel run
        f.write(rec(5) + "\n")
    todo = _resume_plan(root, 8, verbose=False)
    assert todo == [2, 3, 4, 6, 7]
    names = [json.loads(l)["name"] for l in open(os.path.join(root, "metadata.jsonl"))]
    assert names == ["img_000000", "img_000001", "img_000005"]
    assert not os.path.exists(os.path.join(root, "metadata.part1.jsonl"))
    assert os.path.exists(os.path.join(root, "metadata.jsonl.resume.bak"))
    assert _resume_plan(root, 8, verbose=False) == todo                                        # idempotent
