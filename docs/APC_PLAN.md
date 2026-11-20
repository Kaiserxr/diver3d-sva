# Attention on top of APC_TDC

## Why move to APC_TDC

The OpenPCDet test bench (sparse conv, DSVT, PTv3, SST, hybrids, FSHNet) all used yaw-only boxes.
Backbone changes there moved AP3D@0.35 by at most ~0.05, while the SonarVoxNet paper's SO(3) rotation
head adds ~0.33. So the attention question has to be asked on a baseline that already has the right box.

APC_TDC (team APC_TDC, Diver3D @ MaCVi 2027) is the strongest open-source entry:
https://github.com/seantangth/Diver3D-MaCVi2027-APC_TDC (code MIT, weights CC BY-NC 4.0,
copyright 2026 Tze-Hsiang Tang). It is a **dense** 3D CNN stem + 2D BEV U-Net (~15.3M params),
CenterPoint-style heads with a 6D (SO(3)) rotation, an IoU branch, beam-space copy-paste and a
5-frame variant. Their full system (6-checkpoint ensemble + rescoring + tracking + calibration)
reports test SDS 0.760. We use their code unchanged, pinned to commit 7d444da, in third_party/APC_TDC.

Research question: **does attention (temporal, or object-level refinement) improve a strong dense
SO(3) sonar detector, compared with a same-size non-attention control?**

## Protocol

- Raw single models only: no ensemble, tracking, rescoring or calibration (they would hide the effect
  of the model change; APC's rescoring alone is worth +0.032 SDS).
- Full official val split, official scorer, every epoch (`metrics.csv` per run).
- Run summary = mean over the last 6 validated epochs; model = mean +- std over 3 seeds.
- Every attention model gets a same-size control without attention (like Control-RF).
- APC trained one seed per model; we first measure their seed spread ourselves.

## Stages

| Stage | Runs | Status |
|---|---|---|
| 0 | Smoke test (48 frames, 2 epochs) | queue_apc.sh |
| 1 | Single-frame baseline, APC exp002 recipe (24 epochs), seeds 0/1/2 | queue_apc.sh |
| 2 | 5-frame baseline, APC exp010/011 recipe (12 epochs, t-2..t+2), seeds 0/1/2 | queue_apc.sh |
| 3 | Model 1: temporal attention fusion of per-frame BEV maps (zero-initialised) + control | to write |
| 4 | Model 2: centre-query refinement head (CenterFormer / V-DETR style) + control | later |

## How to run (lab PC)

```bash
cd ~/projects/diver3d-sva && git pull && git checkout exp/apc-attention
bash scripts/apc/setup_apc.sh ~/data/diver3d_v2        # once: code, conda env "apc", packed data
tmux new -s apc
conda activate apc && cd ~/projects/diver3d-sva && bash scripts/apc/queue_apc.sh
cat logs/REPORT_APC.md                                  # any time; curves in logs/apc_curves.png
```

Outputs go to `output/apc/<name>_s<seed>/` (ignored by git); logs to `logs/apc__*.log`.
