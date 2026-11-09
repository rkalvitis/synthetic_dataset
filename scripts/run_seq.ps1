# Sequence dataset: 10 000 clips -> E:\synth_seq   (run from the repo root, PowerShell, env popsynth)
#   conda activate popsynth
#   .\scripts\run_seq.ps1

# 1. the run on 10 worker processes (resources.workers from e_synth_40k.yaml; -j N overrides), resumable:
#    a second call skips finished sequences; add --overwrite to redo them
python -m popsynth.cli sequences -c configs/e_synth_seq.yaml

# 2. afterwards: contact sheets next to the GIFs (the GIFs themselves are written during the run)
python -m popsynth.cli sequence-gifs -c configs/e_synth_seq.yaml -n 24 --sheet

# dry run of the motion only (no GPU, a disc instead of the beetle), 20 short sequences:
#   python -m popsynth.cli sequences -c configs/e_synth_seq.yaml -o E:/synth_seq_dry --disc -n 20 --set "sequences.length=[30,40]"
