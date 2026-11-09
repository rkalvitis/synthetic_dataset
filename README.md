# popsynth: synthetic Popillia japonica images from a Gaussian splat

Code to generate a synthetic image dataset of *Popillia japonica* (Japanese beetle) from a 3D
Gaussian Splatting (3DGS) reconstruction, for training a lightweight CNN classifier
(`popillia` vs `background`). The generator varies viewpoint, rotation, scale/distance, focal
length, background, lighting (day / overcast / dusk / night), shadows, blur, noise and JPEG
compression, and writes classification folders plus bounding boxes and masks.

The dataset size is a config value (`compose.num_images`); the default is deliberately small (10)
for validating the output. The full run (for example 10 000 images from ~2 000 rendered views) is
one command-line override away.

## Pipeline

```
assets/popillia_ex15.ply  --(1) render sprites-->  <out>/sprites/*.png (RGBA, random cameras)
                                                        |
data/backgrounds/**      --(2) compose + augment-->  <out>/images/{train,val}/{popillia,background}/*.jpg
                                                     <out>/labels/{train,val}/*.txt   (YOLO boxes)
                                                     <out>/masks/{train,val}/*.png
                                                     <out>/metadata.jsonl            (every sampled parameter)
```

1. **Render sprites** (`popsynth/generate.py::render_sprites`). The splat is loaded
   (`popsynth/splat.py`), near-transparent gaussians dropped, and the object centred and scaled to a
   unit bounding sphere. Cameras are sampled on a sphere around it (`popsynth/cameras.py`):
   elevation, azimuth, in-plane roll, field of view and frame fill. Distance follows from fov and
   fill, so a narrow fov + far camera gives a flat telephoto look and a wide fov + close camera gives
   macro perspective. Each view is rendered with transparent background and saved as an RGBA sprite.
   **Size.** With `render.size_mode: object` (default) the beetle's final pixel size is decided here, per
   sprite: its longest side is drawn uniformly from `render.object_size` (fraction of the short side of
   `compose.image_size`), the projected extent of the gaussians is measured, and the frame size / camera
   distance are set so the render has exactly that many pixels (small beetles are rendered at >=
   `min_render_px` and area-downsampled). The compose stage then pastes the sprite 1:1, so distance,
   perspective and resolution all come from the render and never from resampling. `size_mode: fixed` is the
   old behaviour (every sprite at `render.size`, rescaled by compose to `compose.object_size`).
   Two renderers share one interface:
   - `gsplat` (CUDA, fast, ~ms per frame) - use on the GPU machine.
   - `torch` (`popsynth/render/torch_raster.py`, pure PyTorch, CPU / Apple MPS / CUDA, exact but
     ~4-7 s per 512 px frame for 0.9 M gaussians) - for development on a laptop.
2. **Compose** (`popsynth/compose.py`, `popsynth/augment.py`). For each output image a random
   background crop is taken (Places365 outdoor categories by default), and for positives a random
   sprite from the matching split is pasted with random size (= distance), position (fully inside the
   frame by default, `compose.min_visible` < 1 allows cut-off), colour matching to the background, a soft shadow, edge softening and optional
   defocus. Then a lighting mode is sampled (day / overcast / dusk / night with optional flash) and
   applied to the whole image, followed by camera imperfections drawn from a quality tier (clean, mild
   or clearly bad: motion blur, defocus, low resolution, haze, bad exposure, noise, heavy JPEG), because
   real photos are never perfect. Negatives go through the same steps without a sprite.

   Three more sources of variation sit inside this stage:
   - **Two separate augmentations** (`background_aug`, `sprite_aug`): before compositing, the background
     crop and each beetle sprite get their *own* independent photometric/geometric jitter (brightness,
     contrast, saturation, hue, gamma; background: rotation, flip, blur, grayscale; beetle: mirror,
     aspect, specular glint, blur), so the beetle does not always share the background's tint, exposure
     and sharpness. The whole-image lighting and camera effects come on top.
   - **Beetle behind a leaf** (`compose.occlusion`): after the beetles are pasted, one or two foreground
     occluders are placed over a random beetle so that a target fraction of it is hidden. Occluders are
     either RGBA leaf cutouts (`scripts/download_leaf_occluders.py` converts the Flavia leaf scans) or a
     leaf-shaped patch cut from the same background photo at the same texture scale (needs no extra data,
     used alone if no cutouts exist, works on any background type). Occluders are defocused (closer than the
     focus plane), slightly darkened and flattened, colour-matched to the spot they cover, and cast a soft
     contact shadow, so they do not look like pasted stickers. The hidden fraction, source
     and the visible box are recorded per beetle; `compose.box_mode` chooses whether the label box covers
     only the visible pixels (default since 2026-11-06: a beetle under a leaf or a background patch is boxed
     by what is left of it) or the whole body (`full`, amodal). Both boxes are always stored.
   - **Placement** (`compose.placement`): `uniform` places the box uniformly over every position where the
     whole beetle fits (big beetles then necessarily sit near the middle); `uniform_center` draws the centre
     uniformly over the whole frame and shifts the beetle inside only as far as needed, which gives a flat
     centre distribution. `compose.center_sigma` instead clusters beetles around the centre like hand-held
     photos. `popsynth stats` plots where beetles actually land.

All randomness derives from `seed` plus the image index, so a dataset can be regenerated or
extended exactly. Validation sprites (`render.val_fraction`) are never used in training images, so
the val split contains unseen viewpoints.

## Setup

```bash
# macOS / CPU (development)
conda env create -f environment-mac.yml && conda activate popsynth

# Linux + NVIDIA (production)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-cuda.txt
```

### Windows + NVIDIA (e.g. RTX 4060 Ti)

gsplat ships pre-built Windows wheels only for **Python 3.10 + PyTorch 2.4 + CUDA 12.4** (as of Oct 2026,
see https://docs.gsplat.studio/whl/gsplat/). Using exactly that combination avoids compiling CUDA code:

```powershell
conda create -n popsynth python=3.10 && conda activate popsynth
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
pip install ninja numpy jaxtyping rich
pip install gsplat --index-url https://docs.gsplat.studio/whl/pt24cu124
pip install -r requirements-cuda.txt --no-deps
pip install numpy pillow opencv-python-headless pyyaml tqdm pytest
python -c "import torch, gsplat; print(torch.cuda.get_device_name(0), gsplat.__version__)"
```

Any other Python/torch combination makes `pip install gsplat` compile on first import, which needs the
CUDA Toolkit matching your torch build plus Visual Studio Build Tools (MSVC 143), run from a shell where
`vcvars64.bat` has been executed (see gsplat's `docs/INSTALL_WIN.md`). Everything else in this repo is
plain numpy/OpenCV and runs unchanged on Windows; the generated label files use `/` path separators on all
platforms.

**First run on the GPU machine** (the gsplat backend was written on a Mac and is untested on CUDA):

```powershell
python -m popsynth.cli info                                        # must print "render backends: ['gsplat', 'torch']"
python -m popsynth.cli render-sprites --num-sprites 4 -o outputs/smoke --set render.size=512 --set render.supersample=1
python -m popsynth.cli generate --num-sprites 24 --num-images 10 -o outputs/check   # full quality, 10 images
python -m popsynth.cli preview -o outputs/check
```

If gsplat fails, `--set render.backend=torch` uses the fallback rasteriser on the GPU (slower, ~5 s per 2048 px
sprite on a 4060 Ti, still fine for a few hundred sprites). The full run, about 1-1.5 h, dominated by compositing:

```powershell
python -m popsynth.cli generate --set render.num_sprites=2000 --set compose.num_images=20000 -o outputs/dataset_20k
```

**RAM budget.** `resources.max_ram_gb` (16 in `default.yaml` and `e_synth_20k.yaml`) is a hard cap on Windows
(a job object: an allocation beyond it raises `MemoryError` instead of swapping) and a watchdog on every platform
(the compose loop aborts with a message if the process grows past it). What actually grows with the sprite count
is the decoded-sprite cache, now an LRU of uint8 sprites limited to `resources.sprite_cache_gb` (4 GB default, about
700 sprites of 2048 px; the rest are re-read from PNG, ~20 ms each) plus `occluder_cache_gb` for leaf cutouts.
Everything else is per image and small: a 1024 px compose run peaks well under 2 GB, and the render stage keeps
only one frame on the CPU. Peak RAM and cache hit rate are printed at the end of each stage.
`popsynth info` has no effect on it; the limit is applied when `render-sprites`, `compose` or `generate` start.

Put the splat PLY in `assets/` (git-ignored) and fetch backgrounds:

```bash
python scripts/download_inat_backgrounds.py --per-taxon 40    # iNaturalist CC photos of Popillia host plants, up to 2048 px (best)
python scripts/download_backgrounds.py --source flowers102    # Oxford Flowers102: 8 189 close-up plant photos, ~500 px (~330 MB)
python scripts/download_backgrounds.py --size large           # Places365 val, 52 outdoor categories, 5 200 images, 512 px short side (~2.3 GB)
```

Quality and realism ranking: the iNaturalist set (linden, rose, grapevine, raspberry, maple ... see
`configs/inat_hostplants.txt`) is both the sharpest and the most realistic, since those are the
surfaces the beetle actually sits on; attribution and licences are written to `attribution.csv`
next to the photos; only June-September observations are fetched (adult flight season), and photos may
occasionally contain other insects (hard negatives) or even a Popillia. Flowers102 matches the scale of a 1 cm insect but is only ~500 px. Places365
adds scene variety (lawns, streets, orchards) at human scale, so a pasted beetle can look oversized.
`backgrounds.min_size` drops files whose short side is below the output size, and each folder has a
`weight` (probability of being picked, independent of how many files it holds). Any folder of jpg/png
files can be used instead or in addition (`backgrounds.dirs`), e.g. your own photos of rose bushes,
grapevine leaves, soil and beetle traps. For the full run, fetch more host-plant photos
(`--per-taxon 200` gives ~3 000) so backgrounds repeat less often.

## Usage

```bash
popsynth info                                   # splat stats, principal axes, render backend
popsynth generate                               # 24 sprites + 10 images at full quality (2048 px sprites, 1024 px PNGs)
popsynth generate -c configs/mac_preview.yaml   # same pipeline, laptop-sized (1024 px sprites, 640 px JPGs, a few minutes)
popsynth preview -n 10                          # contact sheet with boxes -> outputs/dataset/preview.jpg
popsynth stats                                  # size distribution, centre heatmap, per-pixel density -> outputs/dataset/stats/
python scripts/download_leaf_occluders.py       # Flavia leaf scans (~1 GB) -> data/occluders/flavia/*.png cutouts

# full dataset on the GPU machine (40k images, 5000 sprites, every variation, all backgrounds, 16 GB RAM cap)
popsynth generate -c configs/e_synth_40k.yaml
popsynth stats    -c configs/e_synth_40k.yaml
popsynth clean    -o E:/synth_v2                # repair an older run: drop invisible beetles, quarantine blank frames (see below)
python scripts/compare_visibility.py "v1=E:/synth/outputs/dataset_20k" "v2=E:/synth_v2"   # visibility graphs, several datasets side by side
python scripts/compare_coverage.py   "v1=E:/synth/outputs/dataset_20k" "v2=E:/synth_v2"   # per-pixel coverage (position bias) side by side
python scripts/compare_size.py       "v1=E:/synth/outputs/dataset_20k" "v2=E:/synth_v2"   # beetle size in the images, side by side
python scripts/compare_sprites.py    "v1=E:/synth/outputs/dataset_20k" "v2=E:/synth_v2"   # the rendered sprites themselves: size, rendered vs pasted, re-use, viewpoints, alpha
#   all four scripts take --separate <folder> to write every panel as its own PNG (one plot per file, to pick from),
#   compare_coverage.py also --by-size <png> for the size-normalised maps

# re-composite with different backgrounds / augmentation without re-rendering
popsynth compose -o outputs/popillia_10k --set compose.image_size=[224,224] --set lighting.night.weight=0.3
```

`python -m popsynth.cli ...` works without installing the package.

## Configuration

Everything lives in `configs/default.yaml`; every key can be overridden with `--set key.sub=value`.
Key knobs:

| key | meaning |
|---|---|
| `splat.up` | dorsal direction of the beetle in splat coordinates (so elevation 90 = top-down) |
| `render.camera.elevation_deg / fov_deg / fill` | viewpoint distribution, perspective, framing |
| `render.num_sprites` | number of distinct rendered views (re-used across backgrounds); 5000 in `configs/e_synth_40k.yaml` |
| `render.size_mode`, `render.object_size`, `render.min_render_px` | `object` (default): beetle pixel size drawn uniformly per sprite and produced by the camera, pasted 1:1; `fixed`: constant sprite size, rescaled at compose time |
| `render.size`, `render.supersample`, `render.antialiased` | sprite resolution, supersampling factor and Mip-Splatting opacity compensation (2048 / 2 / on by default = highest quality; use `configs/mac_preview.yaml` on a laptop) |
| `compose.image_size` | output image size; keep it above the CNN input size and resize at training time |
| `compose.object_size` | beetle size as fraction of the shorter image side; only used with `render.size_mode: fixed` |
| `compose.positive_fraction` | share of images containing a beetle |
| `compose.objects_per_image` | `[1, 3]` to paste several beetles (detection) |
| `lighting.*` | per-mode exposure, white balance, contrast, saturation, noise, vignette, flash; `mean_range` bounds the mean luminance of the finished frame for that light (night 0.14-0.34 ... day 0.30-0.75), so background jitter and camera faults cannot stack into a black or white frame |
| `camera_effects.tiers` | quality tiers (clean / mild / bad) with per-effect probabilities: motion blur, defocus, low resolution, haze, over/under-exposure, noise, JPEG |
| `background_aug`, `sprite_aug` | the two independent pre-compositing augmentations (`prob` + a range per parameter; a parameter is skipped if its range is absent) |
| `compose.visible_floor`, `compose.overlap_tries` | the 33% rule: every beetle keeps at least this share of its *whole* body visible (default `0.33`; a range `[lo, hi]` draws it per beetle) against the frame edge (the part outside counts as hidden), beetles pasted later and, with `occlusion.respect_floor: true` (default), leaf / background occluders; a placement that would break it is re-drawn `overlap_tries` times, then that beetle is skipped. `overlap_min_visible` is the old name and still read |
| `camera_effects.guard` | `{mean: [lo, hi], min_std, tries}`: lighting + effects are re-drawn when the frame comes out blank (white / black / no contrast); over/under-exposure cap the gain so the mean stays within the tier's `min_mean` / `max_mean` and highlights roll off softly; an exposure fault never stacks on haze |
| `compose.occlusion` | beetle behind a leaf: `prob`, `count`, `cover` (target hidden fraction), `max_hidden`, `size`, `blur_sigma_px`, `brightness`, `contrast`, `match_color` (blend toward the covered spot), `shadow`, `sources {leaf, background}`, `dirs` (cutout folders) |
| `compose.placement`, `compose.center_sigma` | `uniform` (box uniform where it fits) or `uniform_center` (flat centre distribution); `center_sigma` = std of a normal distribution around the image centre instead |
| `compose.box_mode` | `visible` (default: tight around the pixels still visible under leaves / patches / other beetles) or `full` (whole beetle, hidden parts included). `popsynth clean --boxes visible|full` switches an existing dataset |
| `resources.max_ram_gb`, `sprite_cache_gb`, `occluder_cache_gb` | RAM cap of the process (hard on Windows, watchdog elsewhere) and the sizes of the decoded sprite / leaf-cutout caches |

## Output format

```
<out>/images/<split>/<class>/img_XXXXXX.png   classification layout (torchvision ImageFolder)
<out>/labels/<split>/img_XXXXXX.txt           YOLO boxes: "0 cx cy w h" normalised; empty file for negatives
<out>/masks/<split>/img_XXXXXX.png            0/255 beetle mask (positives)
<out>/metadata.jsonl                          one JSON record per image, written while generating
<out>/dataset.json                            whole dataset in COCO detection format + all metadata
<out>/images.csv                              one row per image (flat table, pandas/Excel friendly)
<out>/objects.csv                             one row per beetle: box, pose, size, compositing parameters
<out>/summary.json                            counts and distributions
<out>/sprites/sprites.jsonl                   camera of every rendered sprite
<out>/stats/*.png, stats.json                 `popsynth stats`: size_distribution, center_heatmap, pixel_density, occlusion
```

`dataset.json` is loadable by any COCO tool (pycocotools, torchvision `CocoDetection`, Ultralytics, FiftyOne);
`images` and `annotations` entries carry an extra `metadata` dict with the fields below. `images.csv` / `objects.csv`
hold the same information as flat columns. `python -m popsynth.cli export -o <out>` rebuilds all of them from
`metadata.jsonl` (e.g. after merging runs).

Per image (`images.csv`, `metadata.jsonl`, `dataset.json[images][i].metadata`):

| Field | Meaning |
|---|---|
| `name`, `split`, `class`, `image`, `label_file`, `mask_file` | id, train/val, `popillia`/`background`, relative paths |
| `width`, `height`, `num_objects` | image size in px, number of beetles (0 for negatives) |
| `boxes_xyxy`, `boxes_cxcywh_norm` | all beetle boxes in pixels (x0,y0,x1,y1) and YOLO-normalised (cx,cy,w,h) |
| `background`, `background_source`, `background_category` | photo path, source folder (`inat_hostplants`/`flowers102`/`places365`), taxon or scene category |
| `bg_crop_px`, `bg_crop_scale`, `bg_hflip` | crop window in the source photo, crop size fraction, flip |
| `lighting`, `exposure`, `contrast`, `saturation`, `noise`, `flash` | lighting mode (day/overcast/dusk/night) and its parameters |
| `quality_tier`, `effects` | clean/mild/bad, list of imperfections applied |
| `bg_aug`, `bg_brightness`, `bg_contrast`, `bg_saturation`, `bg_hue_deg`, `bg_gamma`, `bg_rotate_deg`, `bg_vflip`, `bg_blur_sigma_px`, `bg_grayscale` | background augmentation: list of applied parameters and their values |
| `num_occluders`, `occluders` | foreground occluders in this image: per occluder `occluder` (leaf/background), `occluder_file`, `target_object`, `cover_target`, `cover_achieved`, `occluder_px`, `occluder_rotate_deg`, `occluder_blur_sigma`, `occluder_brightness`, `occluder_contrast`, `occluder_match_color` |
| `motion_blur_px`, `defocus_sigma`, `lowres_factor`, `haze`, `overexposure`, `underexposure`, `sensor_noise`, `jpeg_quality` | parameter of each applied imperfection (absent if not applied) |

Per beetle (`objects.csv`, `metadata.jsonl[objects]`, `dataset.json[annotations][i]`):

| Field | Meaning |
|---|---|
| `image_name`, `object_id` | which image, index within it |
| `bbox_xyxy`, `bbox_cxcywh_norm`, `bbox_w_px`, `bbox_h_px`, `bbox_area_frac` | the label box (`box_mode`: visible pixels by default), in pixels and normalised |
| `mask_area_px`, `visible_frac`, `occluded_frac` | beetle pixel count, fraction of the sprite inside the frame (1.0 = whole beetle visible), fraction of its pixels hidden (by a beetle pasted later or by an occluder; with `box_mode: full` the box still covers the full body) |
| `occluded_by_beetle_frac`, `occluded_by_occluder_frac`, `visible_area_px`, `bbox_visible_xyxy`, `bbox_full_xyxy` | split of the hidden fraction, visible pixel count, the tight box around the visible pixels and the full-body box (both always present, whichever is the label) |
| `body_area_px`, `body_in_frame_frac`, `visible_share`, `visible_floor` | pixels of the whole beetle (part outside the frame included), share of them inside the frame, visible pixels / whole body (the quantity the 33% rule bounds) and the floor drawn for this beetle |
| `sprite_aug`, `aug_brightness`, `aug_contrast`, `aug_saturation`, `aug_hue_deg`, `aug_gamma`, `aug_hflip`, `aug_aspect`, `aug_specular`, `aug_blur_sigma_px` | beetle augmentation: list of applied parameters and their values |
| `object_size_frac`, `object_px` | beetle longest side as fraction of the short image side, and in px |
| `elevation_deg`, `azimuth_deg`, `roll_deg`, `fov_deg`, `camera_distance` | 3D view: elevation above the beetle's horizon, heading, in-plane rotation, lens field of view, camera distance (model units) |
| `sprite`, `sprite_index` | which rendered sprite was pasted (details in `sprites/sprites.jsonl`) |
| `color_match`, `defocus_sigma`, `shadow_strength`, `shadow_blur_px` | compositing parameters |

Boxes are tight around the composited beetle after lighting; imperfections (blur, JPEG, low resolution) do not
move pixels, so the boxes stay valid for the degraded image. For the classifier/detector use `boxes_cxcywh_norm`
or the YOLO txt files; `popsynth.dataset.PopsynthDataset` is a minimal PyTorch loader that yields
`(image, {boxes_xyxy, boxes_cxcywh_norm, labels, has_popillia, meta})` and can be filtered on any field, e.g.
`ds.filter(lambda m: m["quality_tier"] == "bad")` for per-tier evaluation.

Config files can inherit with `_base: other.yaml` (see `configs/mac_preview.yaml`).

## Quality settings

The defaults are the highest-quality settings: 2048 px sprites rendered at 4096 px and area-downsampled,
antialiased, composited onto 1024 px crops of >= 1024 px backgrounds, saved as lossless PNG, with only
mild blur / JPEG simulation. This is meant for the gsplat/CUDA machine (milliseconds per frame). The
pure-PyTorch renderer handles the same settings (large frames are rendered in bands to bound memory)
but needs minutes per sprite, so use `configs/mac_preview.yaml` for local checks.

## Demo slides

`popsynth demo -c configs/mac_preview.yaml` writes `outputs/demo/slide_NN_*.jpg`, one factor varied per
slide on the same background (distance, lighting, viewpoint, rotation, partially outside, perspective,
compositing effects, imperfections, backgrounds, multiple beetles, negatives, occlusion, background
augmentation, sprite augmentation), plus the individual images
as PNG in `outputs/demo/NN_*/`.

## Dataset statistics

`popsynth stats -o <out>` reads `metadata.jsonl` and the masks and writes to `<out>/stats/`:

| File | What it shows |
|---|---|
| `size_distribution.png` | histograms of the beetle size: longest side / short image side, in pixels, box area fraction (median and 5-95 % in the titles) |
| `center_heatmap.png` | where the beetle box centres fall (12x12 cells, % of beetles; middle third / middle half outlined) and a radial profile: how often the beetle is in the centre vs. near the edges |
| `pixel_density.png` | per pixel, the % of positive images in which that pixel is beetle (from the masks) and inside a box |
| `occlusion.png` | hidden fraction per beetle |
| `stats.json` | the numbers behind the plots (quantiles, centre fractions, density means) |

`summary.json` also carries `occluded_frac`, `center_in_middle_third`, `center_in_middle_half`, `center_x/y`,
`images_with_occluder`, `occluder_source`, `background_aug` and `sprite_aug` counts.

## Tests

```bash
pytest            # or: python scripts/run_tests_no_pytest.py
```

## Repairing a generated dataset (`popsynth clean`)

Runs before 2026-11-05 could label a beetle that was completely hidden behind another beetle, and the
`bad` camera tier could burn a frame out to pure white (haze + over-exposure) or black. Both are fixed
in the generator (`overlap_min_visible`, `camera_effects.guard`); an existing dataset is repaired in place:

```bash
popsynth clean -o E:/synth_v2 --dry-run      # report only
popsynth clean -o E:/synth_v2                # apply; options: --min-visible 0.2 --drop-below 0.05 --workers 8 --no-blank
popsynth clean -o E:/synth_v2 --boxes visible   # also rewrite the label boxes to the visible pixels (from bbox_visible_xyxy)
```

- a beetle visible less than `--drop-below` (5%) of its body has no usable pixels: its box / label line is
  removed, the image stays;
- `--boxes visible` (or `full`) rewrites every label box, label file and table from the boxes stored per beetle,
  without reading pixels; the previous box is kept as `bbox_full_xyxy`, so the switch is reversible;
- a beetle visible less than `--min-visible` (20%, the v2 leaf cap) still shows but is too hidden: the image is quarantined;
- a blank frame (mean luminance outside 0.05-0.9 or std < 0.03, measured on the pixels) is quarantined.

Quarantined images / labels / masks are *moved* to `<out>/_removed/` (same paths, reasons in
`_removed/removed.jsonl`), `metadata.jsonl` is backed up to `metadata.jsonl.bak`, and `dataset.json`,
`images.csv`, `objects.csv`, `summary.json` are rebuilt. The command is idempotent. Expect roughly 1 000
hidden-beetle images and 850 blank frames (~4.5%) to be removed from the 40k run of 2026-11-04.

## Notes / known limitations

- The splat is rendered as-is: no relighting. Lighting variation is photometric (whole image), so
  specular highlights stay where the capture rig put them. Shadows are approximated.
- The current model (`popillia_ex15.ply`) has a few white streaks from the mounting pin and curled
  hind legs (frozen specimen); `splat.clean.box` can crop residual floaters if needed.
- Backgrounds: Places365 (Zhou et al. 2017, MIT CSAIL) is licensed for research use.
