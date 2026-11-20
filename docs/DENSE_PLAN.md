# Is dense attention OK on Diver3D?

## Question

Sparse attention middle stages (DSVT, PTv3, SST) lost to sparse conv (Run A), and a sparse
conv + attention hybrid tied it. Those comparisons changed many things at once (operator, block
design, windows, training). This experiment asks one thing at a time, on a **dense** grid where every
voxel of 120 x 100 x 10 is a token, empty ones included.

## Design: one skeleton, swappable token mixer

`overlay/pcdet/models/backbones_3d/dense_mixer_middle.py` (`DenseMixerMiddle`). The skeleton is
MetaFormer's (Yu et al., *MetaFormer Baselines*, TPAMI 2023): `x + mixer(norm(x))`, `x + mlp(norm(x))`,
2 blocks at height 10, height 10 -> 5, 2 blocks at height 5, height 5 -> 2, fold to a 128-channel BEV
map. Everything outside the middle stage is Run A's (same VFE, BEV backbone, head, schedule).

| Config | Mixer | Params | Answers |
|---|---|---|---|
| `dense_densea` | Run A's exact layers as dense 3D convs | 0.68M (= Run A) | sparse vs dense, nothing else changed |
| `dense_identity` | none (IdentityFormer) | 0.17M | what the skeleton alone gives (the floor) |
| `dense_conv` | depthwise 3x3x3 conv (+ 1x1x1 in/out) | 0.25M | dense conv in the skeleton |
| `dense_na` | 3x3x3 neighborhood attention (NAT, CVPR 2023), learned relative position bias | 0.23M | same footprint as conv: fixed vs content-dependent weights |
| `dense_hybrid` | conv blocks + 2 global attention layers on the (1,4,4)-pooled grid (3,750 tokens, 3D RoPE) | 0.35M | attention where context matters (LitePT / CAFormer / Primus); starts identical to `dense_conv` |

A fair shot for attention (seed 666): `*_lsa` (locality self-attention: learned temperature, no
self-attention; Lee et al., *ViT for Small-Size Datasets*, 2021) and LR 5e-4 / 2e-3 for NA and hybrid.

The Primus test (Wald et al., *Primus*, 2025): `scripts/attn_off_test.sh` re-scores trained NA /
hybrid checkpoints with `ATTN_OFF True` (attention removed, all other weights kept). No drop means the
model was not using its attention.

Neighborhood attention is implemented in plain PyTorch (27 shifted views), no NATTEN needed; it was
checked against a brute-force loop. Caveat for reading the results: ~98.6% of the voxels are empty
(zero input), unlike medical volumes where every voxel carries signal.

## Protocol

Same as every earlier run: AP3D@0.35 averaged over epochs 6-12, mean +- std over seeds 666 / 1234 /
2024, val = every 4th frame, official scorer. Compared directly with Run A and DSVT / PTv3 / SST.

## Speed (changed after the first runs)

The first dense runs (tag runE: fp32, 20 epochs) were far slower than the sparse models: ~4 h per
run for dense Run A / identity, ~11 h for conv, and NA was on course for ~29 h (every step processes
120,000 voxels x 4 frames; sparse Run A processes ~1,700 per frame and took 0.5 h). From tag runG on:

- **Stop after epoch 12** (`PCDET_STOP_EPOCH=12`, patch 3e in setup_openpcdet.sh): the learning-rate
  schedule is still the 20-epoch one, training just stops, so epochs 1-12 are exactly those of a full
  run with the same seed. The report averages epochs 6-12, so the protocol is unchanged.
- **bf16 autocast in the dense middle stage** (`BF16: True` in the configs): norms, softmax and the
  hybrid's low-resolution attention stay fp32; the rest of the detector is fp32.
- **torch.compile** (`COMPILE: True`): same maths; on the RTX 5080 it made NA ~8x faster (533 -> 67 ms per
  step of the middle stage), conv and hybrid ~15-20%.
- `scripts/bench_dense.py` times one step of each variant in fp32, bf16 and bf16 + compile and checks that
  bf16 stays finite.

The runE runs are ignored by the report.

## How to run (lab PC)

```bash
cd ~/projects/diver3d-sva && git fetch && git checkout exp/dense-mixer
conda activate sva && bash scripts/setup_openpcdet.sh ~/data/diver3d_v2
python scripts/check_gpu_stack.py           # builds every middle stage on the GPU, dense ones included
tmux new -s dense
conda activate sva && cd ~/projects/diver3d-sva && bash scripts/queue_dense.sh
```
