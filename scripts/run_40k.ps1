# Dataset v3: 40k images -> E:\synth_v3   (run from the repo root, PowerShell, env popsynth)
#   v3 = 33% whole-body visibility floor against frame / beetles / occluders, visible-pixel boxes, bounded exposure,
#   no RAM cap (resources.max_ram_gb: null), all sprites and leaf cutouts cached in RAM.
#   conda activate popsynth
#   .\scripts\run_40k.ps1            # or copy the lines one by one

# 1. one-time: plotting dependency for `stats`
pip install matplotlib

# 2. one-time: Flavia leaf cutouts for the occlusion step (~1 GB download, then conversion)
python scripts/download_leaf_occluders.py --out E:/synth/data/occluders/flavia --raw E:/synth/data/_raw/flavia

# 3. the full run: 5000 sprites, then 40 000 images
python -m popsynth.cli generate -c configs/e_synth_40k.yaml

# 4. afterwards: plots and a contact sheet
python -m popsynth.cli stats   -c configs/e_synth_40k.yaml
python -m popsynth.cli preview -c configs/e_synth_40k.yaml -n 24

# if stage 2 is stopped (Ctrl+C) or dies partway, continue it on 10 cores (resources.workers in the config; -j N overrides):
#   python -m popsynth.cli compose -c configs/e_synth_40k.yaml --resume
# a run stopped early and left as is: rebuild dataset.json / csv tables from what exists
#   python -m popsynth.cli export -c configs/e_synth_40k.yaml

# 5. repair the v2 run of 2026-11-04 in place (hidden beetles, blank frames); files go to E:\synth_v2\_removed
#   python -m popsynth.cli clean -o E:/synth_v2 --dry-run
#   python -m popsynth.cli clean -o E:/synth_v2 --boxes visible     # also switch the labels to visible-pixel boxes (2026-11-06)
