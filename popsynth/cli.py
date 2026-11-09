"""Command line interface: popsynth <command> [options]."""
from __future__ import annotations

import argparse
import os
import sys

from .config import load_config


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", "-c", default=None, help="YAML config (default: configs/default.yaml)")
    p.add_argument("--set", "-s", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. --set compose.num_images=10000")
    p.add_argument("--out", "-o", default=None, help="output directory (default: output.dir in config)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="popsynth", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("info", help="print splat statistics and available render backends"); _common(p)
    p = sub.add_parser("render-sprites", help="stage 1: render RGBA sprites from the splat"); _common(p)
    p.add_argument("--num-sprites", "-n", type=int, default=None)
    p = sub.add_parser("compose", help="stage 2: composite existing sprites onto backgrounds"); _common(p)
    p.add_argument("--num-images", "-n", type=int, default=None)
    p.add_argument("--sprites", default=None, help="sprites dir (default: <out>/sprites)")
    p.add_argument("--workers", "-j", default=None, help="worker processes for stage 2 (int or auto; default resources.workers)")
    p.add_argument("--resume", action="store_true", help="continue a stopped / crashed run: keep the images already written")
    p = sub.add_parser("generate", help="stage 1 + stage 2"); _common(p)
    p.add_argument("--num-sprites", type=int, default=None)
    p.add_argument("--num-images", "-n", type=int, default=None)
    p.add_argument("--workers", "-j", default=None, help="worker processes for stage 2 (int or auto; default resources.workers)")
    p = sub.add_parser("sequences", help="video sequences for tracker training: the beetle re-rendered every frame "
                                         "while it walks across the same background (GOT-10k layout, see popsynth/sequences.py)"); _common(p)
    p.add_argument("--num-sequences", "-n", type=int, default=None)
    p.add_argument("--overwrite", action="store_true", help="redo sequences that already exist (default: resume)")
    p.add_argument("--disc", action="store_true", help="stand-in disc instead of the splat (dry run of the motion, no GPU)")
    p.add_argument("--workers", "-j", default=None, help="worker processes, each with its own renderer (int or auto; default resources.workers)")
    p = sub.add_parser("sequence-gifs", help="animated GIFs (+ sheets) of generated sequences with the boxes drawn -> <out>/_gifs"); _common(p)
    p.add_argument("-n", type=int, default=12, help="first n sequences (0 = all)")
    p.add_argument("--gif-out", default=None)
    p.add_argument("--max-side", type=int, default=240)
    p.add_argument("--frame-step", type=int, default=2)
    p.add_argument("--sheet", action="store_true", help="also a contact sheet per sequence")
    p = sub.add_parser("preview", help="write a contact sheet of a generated dataset"); _common(p)
    p.add_argument("-n", type=int, default=16)
    p.add_argument("--preview-out", default=None)
    p = sub.add_parser("demo", help="comparison slides: one factor varied per slide on the same background"); _common(p)
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--sprite-size", type=int, default=1024)
    p.add_argument("--background", default=None, help="background photo used on every slide")
    p.add_argument("--only", default=None, help="comma-separated slide names (distance,lighting,viewpoint,...)")
    p = sub.add_parser("export", help="rebuild dataset.json / images.csv / objects.csv / summary.json from metadata.jsonl"); _common(p)
    p = sub.add_parser("stats", help="plots: beetle size distribution, centre heatmap, per-pixel density, occlusion -> <out>/stats"); _common(p)
    p.add_argument("--stats-out", default=None, help="output folder (default: <out>/stats)")
    p.add_argument("--grid", type=int, default=12, help="cells per side of the centre heatmap")

    p = sub.add_parser("clean", help="repair a generated dataset in place: drop boxes of invisible beetles, quarantine images "
                                     "with too-hidden beetles or blank frames (-> <out>/_removed), rebuild the tables"); _common(p)
    p.add_argument("--min-visible", type=float, default=0.2, help="a beetle visible less than this -> image quarantined "
                                                                    "(0.2 = the v2 leaf cap, so only beetle-on-beetle cases are hit)")
    p.add_argument("--drop-below", type=float, default=0.05, help="a beetle visible less than this -> its box is dropped, image kept")
    p.add_argument("--no-blank", action="store_true", help="skip the pixel scan for blank frames")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--boxes", default="keep", choices=["keep", "visible", "full"],
                   help="rewrite the label boxes: visible = tight around the pixels still visible under leaves / patches / "
                        "other beetles (bbox_visible_xyxy), full = whole body; keep = leave as generated")
    p.add_argument("--dry-run", action="store_true", help="report only, change nothing")
    p = sub.add_parser("download-backgrounds", help="fetch Places365 outdoor categories (see scripts/)")
    p.add_argument("--size", default="256", choices=["256", "large"])
    p.add_argument("--out", default="data/backgrounds/places365")

    a = ap.parse_args(argv)
    if a.cmd == "download-backgrounds":
        import runpy
        sys.argv = ["download_backgrounds.py", "--size", a.size, "--out", a.out]
        import os
        script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "download_backgrounds.py")
        runpy.run_path(script, run_name="__main__")
        return 0

    cfg = load_config(a.config, a.set)
    out = a.out or cfg["output"]["dir"]

    if a.cmd == "info":
        from .render import available_backends
        from .render.base import pick_device
        from .generate import prepare_splat
        print("render backends:", available_backends(), "| device:", pick_device(cfg["render"].get("device", "auto")))
        s = prepare_splat(cfg)
        print(f"splat {cfg['splat']['path']}: {len(s)} gaussians after cleaning, SH degree {s.sh_degree}")
        print("principal axes (columns, decreasing variance):\n", s.principal_axes().round(3))
        print("configured up:", cfg["splat"].get("up"))
        return 0
    if a.cmd == "render-sprites":
        from .generate import render_sprites
        render_sprites(cfg, out, a.num_sprites)
        return 0
    if a.cmd == "compose":
        import os
        from .generate import compose_dataset
        compose_dataset(cfg, a.sprites or os.path.join(out, "sprites"), out, a.num_images, workers=a.workers, resume=a.resume)
        return 0
    if a.cmd == "generate":
        from .generate import generate
        generate(cfg, out, a.num_sprites, a.num_images, workers=a.workers)
        return 0
    if a.cmd == "demo":
        from .demo import build_demo
        paths = build_demo(cfg, a.out or "outputs/demo", a.image_size, a.sprite_size, a.background,
                           a.only.split(",") if a.only else None)
        print("\n".join(paths))
        return 0
    if a.cmd == "export":
        import json
        from .labels import export_tables
        print(json.dumps(export_tables(out), indent=2))
        return 0
    if a.cmd == "stats":
        import json
        from .stats import make_stats
        r = make_stats(out, a.stats_out, a.grid)
        print(json.dumps({k: v for k, v in r.items() if k != "center"}, indent=2))
        print("centre summary:", {k: v for k, v in r.get("center", {}).items() if k != "cell_percent"})
        print("wrote", *r["files"], sep="\n  ")
        return 0
    if a.cmd == "clean":
        from .clean import clean_dataset
        clean_dataset(out, a.min_visible, a.drop_below, not a.no_blank, workers=a.workers, dry_run=a.dry_run, box_mode=a.boxes)
        return 0
    if a.cmd == "sequences":
        from .sequences import generate_sequences
        generate_sequences(cfg, out, a.num_sequences, overwrite=a.overwrite, workers=a.workers, disc=a.disc)
        return 0
    if a.cmd == "sequence-gifs":
        from .seqpreview import make_sequence_gifs
        paths = make_sequence_gifs(out, a.gif_out, a.n or None, a.max_side, a.frame_step, a.sheet)
        print(f"wrote {len(paths)} files to {a.gif_out or os.path.join(out, '_gifs')}")
        return 0
    if a.cmd == "preview":
        from .preview import make_preview
        print("wrote", make_preview(out, a.preview_out, a.n))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
