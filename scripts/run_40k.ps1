# Dataset v2: 40k images -> E:\synth_v2   (run from the repo root, PowerShell, env popsynth)
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

# if stage 2 dies partway, re-run only compose (re-uses the rendered sprites):
#   python -m popsynth.cli compose -c configs/e_synth_40k.yaml
