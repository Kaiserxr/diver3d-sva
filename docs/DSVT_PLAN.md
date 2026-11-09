# Plan: sparse conv vs DSVT-V on Diver3D

**Status: approved and implemented (configs `centerpoint_sparseconv.yaml`, `dsvt_voxel.yaml`).
Not trained yet; run it with Part 4 of [LAB_RUNBOOK.md](../LAB_RUNBOOK.md).**

## 1. The question

Is **sparse voxel attention** (DSVT-V) a better 3D middle stage than **sparse 3D
convolution** for detecting divers in 3D sonar?

To answer that fairly, we train two models that are **identical except for the middle
stage**: same voxel grid, same voxel encoder, same BEV encoder, same head, same
training settings, same seed and splits. Any difference in score then comes from the
middle stage alone.

```
                       ┌── Run A: sparse 3D conv (SonarSparseMiddle) ──┐
points ─► voxel grid ─►│                                               ├─► BEV 128 ch, 120×100 ─► BEV encoder ─► CenterHead ─► boxes
         0.1×0.1×0.5 m └── Run B: DSVT-V (attention, pools Z only) ────┘     (0.1 m cells)        768 ch, 60×50     (yaw, for now)
          + VFE 128 ch
```

Both middle stages squeeze the **height** dimension into channels and keep the full
120 × 100 top-down grid, so everything after them sees exactly the same shape.

## 2. What already exists, and what is new

| Piece | Status |
|---|---|
| DSVT backbone (`DSVT`), its 3D→BEV scatter (`PointPillarScatter3d`), its CUDA op (`ingroup_inds`) | ✅ already in our pinned OpenPCDet; **no porting from the DSVT repo needed** |
| Voxel encoder `DynamicVoxelVFE`, `CenterHead`, `BaseBEVBackbone`, `HeightCompression` | ✅ in OpenPCDet |
| **Sparse conv middle stage for our grid** | ❌ **new, about 70 lines.** OpenPCDet's `VoxelBackBone8x` can't be used: it shrinks the top-down grid 8× (0.8 m cells, too coarse for a 0.9 m diver), and its height layers break on our 10-voxel-tall grid (they assume 41) |
| `torch_scatter` package | ❌ needed by `DynamicVoxelVFE`; install step to add to the runbook |

### Files I would add or change

| File | Change |
|---|---|
| `overlay/pcdet/models/backbones_3d/sonar_sparse_middle.py` | **new** `SonarSparseMiddle`: the Run A middle stage (paper: "sparse 3D conv middle stage that collapses height 10→2") |
| `scripts/setup_openpcdet.sh` | link that file and register `SonarSparseMiddle` (same mechanism as `Diver3DDataset`) |
| `overlay/tools/cfgs/diver3d_models/centerpoint_sparseconv.yaml` | **new**, Run A |
| `overlay/tools/cfgs/diver3d_models/dsvt_voxel.yaml` | **new**, Run B |
| `scripts/check_gpu_stack.py` | add GPU tests for `torch_scatter` and one DSVT forward/backward pass |
| `LAB_RUNBOOK.md` | new Part 5: install `torch_scatter`, sanity checks, Run A and Run B |

No existing files change behaviour: the PointPillars configs and the dataset loader
stay as they are.

## 3. Shared settings (identical in Run A and Run B)

| Setting | Value | Why |
|---|---|---|
| Crop | x 0–12, y −5–5, z −2.5–2.5 m | paper |
| Voxel size | 0.1 × 0.1 × 0.5 m → grid **120 × 100 × 10** | paper |
| Voxel encoder | `DynamicVoxelVFE`, 128 channels | paper: VoxelNet-style VFE, 128 ch. DSVT needs this kind of encoder (DSVT's own choice) |
| BEV input | 128 channels at 120 × 100 | paper: "128-channel BEV feature" |
| BEV encoder | `BaseBEVBackbone`, 2 levels (strides 2, 2; 5 + 5 layers; 128 / 256 ch), upsampled and concatenated to **768 ch at 60 × 50** (0.2 m cells) | matches the paper's "768-channel, 60×50 feature map". Two levels, not three, because 100 doesn't divide by 8 (three levels give 52 ≠ 50 and crash) |
| Head | `CenterHead`, 1 class: heatmap, offset, height, log-size, **yaw (sin, cos)** | paper's head minus the 6D rotation, which comes later and is added to both runs equally |
| Head targets | Gaussian min overlap 0.7, min radius 2, unit loss weights | paper |
| Decoding | ≤ 100 peaks, score ≥ 0.30, rotated BEV NMS at IoU 0.10 | paper |
| Optimiser | `adam_onecycle`, peak LR 1e-3, weight decay 1e-2, grad clip 35, batch 4, **20 epochs**, FP32 | the paper's AdamW 1e-3 / 1e-2 / clip 35 / 20 epochs; one-cycle approximates its two-phase cosine |
| Augmentation | none | paper's ablation protocol (Table III is without augmentation) |
| Selection | score every epoch on the val subset, pick best AP3D@0.35 after the first 20% of epochs, then score that one on full val | paper |

Run A is essentially the paper's **"Sparse, Center + yaw"** row (Table III: AP3D@0.35
0.319, SDS 0.448, on test), so it doubles as a check that our setup lands near the paper.

## 4. Run A: `SonarSparseMiddle` (sparse 3D conv)

| Layer | Output (Z × Y × X), channels |
|---|---|
| input voxels | 10 × 100 × 120, 128 |
| 2 × SubM conv 3³ | 10 × 100 × 120, 64 |
| sparse conv 3³, stride (2,1,1) | **5** × 100 × 120, 64 |
| 2 × SubM conv 3³ | 5 × 100 × 120, 64 |
| sparse conv (3,1,1), stride (2,1,1), no padding | **2** × 100 × 120, 64 |
| `HeightCompression` (2 × 64) | BEV 100 × 120, **128** |

It only shrinks height (10 → 5 → 2), exactly like the paper's "collapses height 10→2".
About 0.8 M parameters.

## 5. Run B: DSVT-V, scaled from Waymo to Diver3D

### Measured on our data
(400 random training frames, at the 0.1 × 0.1 × 0.5 m grid)

| | median | 95th pct | max |
|---|---|---|---|
| non-empty voxels per frame | 1,700 | 2,551 | 2,880 |
| voxels inside one diver box | 89 | 147 | 226 |
| voxels per non-empty 1.6 m window (16 × 16 voxels, full height) | 26 | 187 | 470 |
| voxels per non-empty 3.2 m window (32 × 32 voxels, full height) | 70 | 495 | 1,072 |

For comparison, Waymo frames have on the order of 100× more voxels. DSVT's defaults
assume dense, wide scenes; ours is small and sparse, so compute is no concern and the
choices are about **what context each attention set sees**.

### Settings

| Setting | Waymo default | **Diver3D (proposed)** | Reasoning |
|---|---|---|---|
| Voxel size | 0.32 × 0.32 × 0.1875 m | **0.1 × 0.1 × 0.5 m** | same grid as Run A (fairness) |
| `sparse_shape` (x, y, z) | 468 × 468 × 32 | **120 × 100 × 10** | follows from the grid |
| Window (x × y voxels) | 12 × 12 (3.8 m) | **16 × 16 (1.6 m)** | about one diver's footprint; a typical diver window holds ~1–4 sets |
| Hybrid factor (2nd window) | × 2 → 24 × 24 (7.7 m) | **× 2 → 32 × 32 (3.2 m)** | the larger, alternating window covers a whole diver (1.74 m) plus surroundings |
| Window height per stage | 32, 8, 2, 1 (full height) | **10, 5, 3, 1** (full height) | windows always span the whole column, as in DSVT-V |
| Shift (2nd pass) | 6, 6, 0 | **8, 8, 0** (half a window) | same rule as DSVT |
| Set size (voxels per attention group) | 48 | **48** | a diver (~89 voxels) splits over ~2 sets; alternating x/y sorting and the hybrid window mix information between them. **First ablation: 96**, so a whole diver fits in one set |
| Height pooling between stages | stride 4, 4, 2 (32 → 8 → 2 → 1) | **stride 2, 2, 3 (10 → 5 → 3 → 1)** | pools only along Z, like DSVT-V; x/y resolution never drops |
| Pooling type | attention | attention | DSVT default |
| Stages × blocks | 4 × 1 | **4 × 1** (8 attention layers) | DSVT default |
| Width `d_model` / heads / FFN | 192 / 8 / 384 | **128 / 8 / 256** | 8× less data than Waymo, one class; 128 also matches Run A's BEV width. About 1 M parameters, similar to Run A |
| Output | 468 × 468 × 1 → 192 ch | **120 × 100 × 1 → 128 ch** via `PointPillarScatter3d` | same BEV shape and channels as Run A |

The last row is the compatibility guarantee: Run B's BEV map has the same shape (128
channels, 100 × 120) as Run A's, so the BEV encoder and head need no change.

## 6. Order of work, and what you run

1. **I implement** the files above on `exp/sparse-voxel-attention`. Here on the laptop
   I check that the configs load and that both middle stages output the same BEV shape
   (a CPU shape test where possible).
2. **Lab, setup:** `git pull`, rerun the setup script, rebuild, install `torch_scatter`,
   run `check_gpu_stack.py` (new DSVT test included).
3. **Lab, sanity checks** (about 15 min each): overfit Run A and Run B on the same 105
   training frames, exactly like the PointPillars sanity check. Both must memorise them.
4. **Lab, real runs:** Run A, then Run B, 20 epochs each, in tmux.
5. **Second seed for both** (needs the small seed patch to `train.sh`), because the
   paper itself finds backbone differences "within seed noise", so one seed can't
   settle the question.
6. **Report:** one table, Run A vs Run B vs PointPillars: AP3D@0.35/0.50, mAP3D, SDS,
   ATE, ASE, AOE3D, TILT, plus parameters and speed.

## 7. Risks, said up front

- **DSVT may not win.** Transformers usually need more data than convolutions, and
  Diver3D is small. "Attention doesn't help on 20k sonar frames" is still a valid,
  reportable result. The fair setup is what makes either answer believable.
- **Learning rate.** DSVT's authors trained it with LR 3e-3 and weight decay 0.05. We
  start with the paper's settings for both runs (fairness). If DSVT fails the overfit
  check or trains unstably, a second, labelled DSVT config with its own optimiser
  settings is the next step.
- **`torch_scatter` on the RTX 5080.** The prebuilt wheel may lack Blackwell kernels.
  The fallback is building it from source with the CUDA 12.8 toolkit we're installing
  anyway (about 10 minutes).
- **Yaw-only for now.** Both runs share the yaw-only ceiling (perfect upright boxes:
  AP3D@0.35 ≈ 0.94). The 6D rotation head, the paper's main gain, is the next change
  after this comparison, again applied to both.

## 8. Decisions I need from you

1. Approve the shared settings in §3, especially **using the paper's optimiser for both runs**.
2. Approve the DSVT settings in §5: **1.6 m / 3.2 m windows, set size 48, `d_model` 128**.
3. Approve adding the new `SonarSparseMiddle` (§4) as the sparse-conv control, instead
   of OpenPCDet's `VoxelBackBone8x`, which can't handle our grid.
