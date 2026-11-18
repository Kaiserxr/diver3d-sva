# Hybrid CNN + attention middle stages (experiments 0-2)

## Question

Does adding attention to the sparse-conv middle stage (Run A) improve diver detection, beyond what a
larger receptive field alone gives? Motivation: Run A beat DSVT, PTv3 and SST (all untuned, from
scratch), and all models fail in the same places: within 2 m of the sonar, off-centre in the beam,
and on box fit. Convolution is good at local detail; attention gives scene-wide context.

## Rules (same as docs/DSVT_PLAN.md)

- Only the middle stage changes. VFE, BEV backbone, head, loss, augmentation, schedule and splits
  stay identical. Every config is generated from `centerpoint_sparseconv.yaml` and differs only in
  its marked middle-stage block (experiments 1-2) or in one marked optimisation line (experiment 0).
- Every middle stage hands the same sparse tensor to `HeightCompression`, so the BEV map is
  identical: 128 channels at 100 x 120 (checked by `scripts/check_gpu_stack.py`).
- 3 seeds per model (666, 1234, 2024).

## Experiments

| # | Config | What | Status |
|---|---|---|---|
| 0 | `centerpoint_sparseconv_lr5e-4`, `_lr2e-3`, `_ep12` | Tune Run A: LR 5e-4 and 2e-3 at 20 epochs; 12 epochs at LR 1e-3 (Run A = LR 1e-3, 20 epochs). Budget: 4 runs; the hybrids get the same budget once a position encoding is chosen | ready |
| 1 | `hybrid_global_rope`, `hybrid_global_cpe` | **Hybrid-global** (LitePT recipe): Run A's sparse conv unchanged, then 2 global self-attention layers over all voxels of a frame (~1,500-2,000) before the BEV map. Position encoding: 3D rotary (`rope`, no parameters) or a SubM 3x3x3 conv before each layer (`cpe`) | ready |
| 2 | `centerpoint_sparseconv_rf` | **Control-RF**: Run A plus 2 SubM (1,5,5) layers at exactly the place where the hybrids have attention: larger receptive field, no attention. 0.88M parameters, within 20% of both hybrids (RoPE 0.78M, CPE 1.01M; Run A 0.68M) | ready |
| F | `fshnet_diver3d` | **FSHNet** (CVPR 2025) as a whole detector, ported to Diver3D: a newer hybrid as a reference point (not a middle-stage-only comparison) | porting |
| 3-5 | (later) | FSHNet's SlotFormer as a middle stage; SphereFormer radial windows; ConDaFormer block | later |

### Hybrid-global details (`SonarHybridMiddle`, `voxel_global_attention.py`)

- Attention runs after Run A's last height-downsampling layer (64 channels, height 2). Our network
  never downsamples x/y, so this is the stage with the fewest voxels, and full attention is cheap.
- Pre-norm transformer layers: 4 heads, FFN 256, no dropout. Frames are padded and masked, so
  attention never crosses frames (tested: batched = per-frame to ~1e-6).
- Output projections (and the CPE branch) start at zero: at initialisation the hybrid computes
  exactly Run A, and attention is learned on top. This keeps the start of training identical.
- RoPE: rotation pairs per head shared out over x, y and z (z gets a quarter; it is only 2 cells
  tall here); base 100. Scores depend only on relative voxel offsets (tested).
- spconv has no depthwise conv, so `cpe` uses a full SubM 3x3x3 conv (+ linear + LayerNorm), as in
  PTv3 (~110k parameters per layer).

Deviations from the original plan: AP3D is reported at IoU 0.35 (challenge default) and 0.50, not
0.25 (the official scorer has no 0.25); the control uses larger kernels, not an extra downsampling
stage, which would change the BEV shape.

## How to run (lab PC)

```bash
cd ~/projects/diver3d-sva && git pull && conda activate sva
bash scripts/setup_openpcdet.sh ~/data/diver3d_v2          # registers SonarHybridMiddle
python scripts/check_gpu_stack.py                         # 'middle stages A-D + hybrids' must PASS
```

Sanity check first (each must memorise 105 frames, a few minutes each):

```bash
for c in hybrid_global_rope hybrid_global_cpe; do RESUME=1 KEEP_CKPTS="last" bash scripts/train.sh ${c}_overfit sanity --num_epochs_to_eval 5 --max_ckpt_save_num 5; done
```

Then everything else runs from one queue (in tmux; rerun-safe):

```bash
bash scripts/queue_until_presentation.sh
```

Order: tuning (3 runs) -> hybrids RoPE + CPE x 3 seeds -> Control-RF (sanity check, then 3 seeds)
-> Run A third seed -> FSHNet (the queue runs `git pull` + setup first; if `fshnet_diver3d.yaml`
exists: sanity check, then 3 seeds; otherwise skipped) -> third seed for B, C, D -> FSHNet again if
it was skipped. A sanity check must reach APBEV@0.35 >= 0.8 on its 105 frames, otherwise that
model's runs are skipped. Note the queue pulls new code from GitHub unattended at the FSHNet step.

Progress and results, at any time (rewritten after every run):

```bash
cat ~/projects/diver3d-sva/logs/REPORT.md
```

`scripts/report.py` writes it: queue state per run (done / running with time left / interrupted /
FAILED / pending), free disk, per-model mean+-std over seeds (epochs 6-12 average: AP3D@0.35 and
@0.50, SDS, < 2 m vs >= 2 m, centre vs off-centre, recall, precision, hours per run, difference
to Run A and whether it exceeds the seed spread), and per-run numbers.

### Disk space

The lab PC's main disk is shared and nearly full. Each checkpoint is ~68 MB, so keeping one per
epoch costs ~1.4 GB per 20-epoch run. Runs therefore keep only what the analysis uses:

- `KEEP_CKPTS="10 last"` (set by the queue): once a run has finished and been scored, only the
  epoch-10 checkpoint (for `rescore_threshold.sh`) and the final one are kept, ~0.15 GB per run.
  All per-epoch scores, predictions, logs and TensorBoard curves are always kept.
- Sanity runs: `--max_ckpt_save_num 5` while training (their evaluation needs the last 5), then
  `KEEP_CKPTS="last"`.
- Older runs: `bash scripts/prune_checkpoints.sh` shows what it would delete (dry run);
  add `--yes` to delete. It skips runs whose last epoch has not been scored.
- A deleted checkpoint can be recreated by rerunning the training (fixed seeds).

## Evaluation (same scripts for every model)

```bash
python scripts/eval_by_range.py --epochs 6-12                 # overall + by distance
python scripts/eval_by_range.py --epochs 6-12 --bins 0,2      # < 2 m vs >= 2 m
python scripts/eval_by_range.py --epochs 6-12 --by angle      # centre (< 15 deg) vs off-centre
python scripts/failure_stats.py --epochs 6-12                 # false alarms: real divers vs background
bash scripts/rescore_threshold.sh 0.1 10                      # also at score cut-off 0.1
```

Reported per model: AP3D@0.35, AP3D@0.50, SDS (mean +- std over 3 seeds, epochs 6-12 average),
the range / angle splits, the false-alarm breakdown, parameter count (from `check_gpu_stack.py`)
and training time per epoch (from the logs). A difference counts only if it is larger than the
spread between seeds.
