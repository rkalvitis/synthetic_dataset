"""Command line interface: popsynth <command> [options]."""
from __future__ import annotations

import argparse
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
    p = sub.add_parser("generate", help="stage 1 + stage 2"); _common(p)
    p.add_argument("--num-sprites", type=int, default=None)
    p.add_argument("--num-images", "-n", type=int, default=None)
    p = sub.add_parser("preview", help="write a contact sheet of a generated dataset"); _common(p)
    p.add_argument("-n", type=int, default=16)
    p.add_argument("--preview-out", default=None)
    p = sub.add_parser("demo", help="comparison slides: one factor varied per slide on the same background"); _common(p)
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--sprite-size", type=int, default=1024)
    p.add_argument("--background", default=None, help="background photo used on every slide")
    p.add_argument("--only", default=None, help="comma-separated slide names (distance,lighting,viewpoint,...)")
    p = sub.add_parser("export", help="rebuild dataset.json / images.csv / objects.csv / summary.json from metadata.jsonl"); _common(p)

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
        compose_dataset(cfg, a.sprites or os.path.join(out, "sprites"), out, a.num_images)
        return 0
    if a.cmd == "generate":
        from .generate import generate
        generate(cfg, out, a.num_sprites, a.num_images)
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
    if a.cmd == "preview":
        from .preview import make_preview
        print("wrote", make_preview(out, a.preview_out, a.n))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
