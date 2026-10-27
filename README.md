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

# full dataset on the GPU machine
popsynth generate --set render.num_sprites=2000 --set compose.num_images=10000 -o outputs/popillia_10k

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
| `render.num_sprites` | number of distinct rendered views (re-used across backgrounds) |
| `render.size`, `render.supersample`, `render.antialiased` | sprite resolution, supersampling factor and Mip-Splatting opacity compensation (2048 / 2 / on by default = highest quality; use `configs/mac_preview.yaml` on a laptop) |
| `compose.image_size` | output image size; keep it above the CNN input size and resize at training time |
| `compose.object_size` | beetle size as fraction of the shorter image side (= distance) |
| `compose.positive_fraction` | share of images containing a beetle |
| `compose.objects_per_image` | `[1, 3]` to paste several beetles (detection) |
| `lighting.*` | per-mode exposure, white balance, contrast, saturation, noise, vignette, flash |
| `camera_effects.tiers` | quality tiers (clean / mild / bad) with per-effect probabilities: motion blur, defocus, low resolution, haze, over/under-exposure, noise, JPEG |

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
| `motion_blur_px`, `defocus_sigma`, `lowres_factor`, `haze`, `overexposure`, `underexposure`, `sensor_noise`, `jpeg_quality` | parameter of each applied imperfection (absent if not applied) |

Per beetle (`objects.csv`, `metadata.jsonl[objects]`, `dataset.json[annotations][i]`):

| Field | Meaning |
|---|---|
| `image_name`, `object_id` | which image, index within it |
| `bbox_xyxy`, `bbox_cxcywh_norm`, `bbox_w_px`, `bbox_h_px`, `bbox_area_frac` | tight box around visible beetle pixels (mask > 0.5), in pixels and normalised |
| `mask_area_px`, `visible_frac`, `occluded_frac` | beetle pixel count, fraction of the sprite inside the frame (1.0 = whole beetle visible), fraction of its pixels hidden behind a beetle pasted later (box still covers the full body) |
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
compositing effects, imperfections, backgrounds, multiple beetles, negatives), plus the individual images
as PNG in `outputs/demo/NN_*/`.

## Tests

```bash
pytest            # or: python scripts/run_tests_no_pytest.py
```

## Notes / known limitations

- The splat is rendered as-is: no relighting. Lighting variation is photometric (whole image), so
  specular highlights stay where the capture rig put them. Shadows are approximated.
- The current model (`popillia_ex15.ply`) has a few white streaks from the mounting pin and curled
  hind legs (frozen specimen); `splat.clean.box` can crop residual floaters if needed.
- Backgrounds: Places365 (Zhou et al. 2017, MIT CSAIL) is licensed for research use.
