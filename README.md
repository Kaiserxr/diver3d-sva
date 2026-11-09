# diver3d-sva

Sparse voxel attention backbones for **diver detection from 3D sonar**, evaluated
on the [Diver3D](#data) dataset and built on
[OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

The goal is a controlled comparison: keep the voxelizer, BEV encoder and detection
head fixed, and swap only the 3D middle stage — a sparse 3D convolution
backbone versus a sparse voxel attention backbone, starting with
[DSVT-V](https://arxiv.org/abs/2301.06051).

> **Status: work in progress.** In place: the Diver3D dataset loader for OpenPCDet,
> scoring with the official Diver3D evaluator, a PointPillars baseline, a zero-shot
> evaluation script, and configs comparing a sparse-conv middle stage with DSVT-V, PTv3 and SST
> ([plan](docs/DSVT_PLAN.md)). None of the new configs has been trained yet.

## What is in this repo

This repo holds only the code that is specific to Diver3D. OpenPCDet and the data
are fetched separately.

```
overlay/                     files linked into OpenPCDet at the same paths
  pcdet/datasets/diver3d/    Diver3DDataset: loads frames, scores with the official evaluator
  pcdet/models/backbones_3d/sonar_sparse_middle.py   SonarSparseMiddle: sparse conv middle stage (Run A)
  pcdet/models/backbones_3d/ptv3_backbone.py, ptv3/  PTv3 adapter + vendored Pointcept PTv3 (Run C, MIT)
  pcdet/models/backbones_3d/sst_backbone.py          SST-style window attention (Run D)
  pcdet/models/backbones_2d/map_to_bev/voxel_column_max.py   per-voxel features -> BEV (Runs C, D)
  tools/cfgs/dataset_configs/diver3d_dataset.yaml   crop, voxel size, splits
  tools/cfgs/diver3d_models/ one config per model
zeroshot/                    run pretrained LiDAR checkpoints on Diver3D, with results
scripts/setup_openpcdet.sh   installs OpenPCDet, links overlay/, links the data
```

## Data

Diver3D is the dataset of the 3D sonar diver detection challenge at the
[MaCVi workshop](https://macvi.org/), WACV 2027: sparse 3D sonar point clouds from
Blue Grotto, Florida, with full-SO(3) (9-DoF) diver boxes. Get it through the
challenge; it is **not** redistributed here. Its own `FORMAT.md` and `METRICS.md`
describe the point and box format and the metrics.

Keep the dataset folder (`diver3d_v2/`, the one containing `splits.json`) as
downloaded; the setup script only links to it.

## Setup

Requires Linux, an NVIDIA GPU, and a Python environment with PyTorch and
[spconv](https://github.com/traveller59/spconv) matching your CUDA version
(see OpenPCDet's [install guide](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md)).

```bash
git clone https://github.com/Kaiserxr/diver3d-sva.git
cd diver3d-sva
bash scripts/setup_openpcdet.sh /path/to/diver3d_v2
cd third_party/OpenPCDet && python setup.py develop
```

The setup script:

1. clones OpenPCDet into `third_party/OpenPCDet`, pinned to commit `233f849`;
2. links the files under `overlay/` into it;
3. registers `Diver3DDataset` in OpenPCDet's dataset list;
4. replaces `np.int` / `np.float`, which numpy ≥ 1.24 removed;
5. links the dataset to `third_party/OpenPCDet/data/diver3d`.

It is safe to run again.

For a complete walkthrough on a fresh Ubuntu machine without admin rights
(conda, CUDA toolkit, an RTX 50-series / Blackwell GPU, tmux), see
[LAB_RUNBOOK.md](LAB_RUNBOOK.md). `python scripts/check_gpu_stack.py` checks
the GPU software stack, and `python scripts/check_dataset.py` checks data loading.

## Training and evaluation

```bash
bash scripts/train.sh pointpillar my_run --num_epochs_to_eval 20      # train, then score every epoch
```

`scripts/train.sh` wraps OpenPCDet's `train.py`. It fixes the seed, refuses to reuse
a run tag, and writes a log in `logs/` that records the git commit and exact command.
To score a single checkpoint, run from `third_party/OpenPCDet/tools`:

```bash
python test.py --cfg_file cfgs/diver3d_models/pointpillar.yaml --ckpt <checkpoint.pth>
```

Configs in `overlay/tools/cfgs/diver3d_models/`:

| Config | What |
|---|---|
| `pointpillar` | PointPillars baseline (yaw-only anchors) |
| `centerpoint_sparseconv` | Run A: sparse 3D conv middle stage (`SonarSparseMiddle`) + CenterHead |
| `dsvt_voxel` | Run B: DSVT-V sparse voxel attention middle stage + the same CenterHead |
| `ptv3_voxel` | Run C: Point Transformer V3 (official Pointcept code, vendored) + the same CenterHead |
| `sst_voxel` | Run D: SST-style single-stride window attention (re-implementation) + the same CenterHead |
| `*_overfit` | sanity checks: train and evaluate on the same 105 training frames |

Runs A–D are identical except for the 3D middle stage; see [docs/DSVT_PLAN.md](docs/DSVT_PLAN.md).

Evaluation writes predictions in the Diver3D submission format and scores them
with the dataset's official `tools/diver3d_eval.py` (SDS, mAP3D, AP3D, AOE3D, …).
By default the validation set is subsampled to every 4th frame for fast
per-epoch checkpoint selection; set `SUBSAMPLE.test: 1` for the full split.

## Known limitations

- **Yaw-only boxes.** `Diver3DDataset` converts each SO(3) ground-truth box to
  its tightest yaw-only box, as standard LiDAR detectors expect. Pitch and roll are
  therefore not learned, which caps AP3D and AOE3D. A full-rotation (6D) head is
  planned.
- **Yaw-only ceiling.** Even the tightest upright box placed perfectly on every
  diver scores only AP3D@0.35 = 0.936, AP3D@0.50 = 0.658 and AOE3D = 65.0° on val,
  measured with the official evaluator on those oracle boxes.
- **Empty frames skipped in training.** OpenPCDet's `DatasetTemplate` skips training
  frames without boxes; about 27% of Diver3D frames contain no diver.
- **Seeds.** OpenPCDet hard-codes seed 666; the setup script patches `train.py` to read
  `$PCDET_SEED` instead, and `SEED=1234 bash scripts/train.sh ...` sets it.
- **Validation only.** Diver3D test labels are withheld, so all results here
  are on the validation split.

## Branches

`main` holds stable code. Experiments are developed on `exp/*` branches and merged
when they work.

## Acknowledgements

- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) (Apache-2.0), which provides the detectors, including DSVT.
- [Pointcept](https://github.com/Pointcept/Pointcept) (MIT): Point Transformer V3, vendored in
  `overlay/pcdet/models/backbones_3d/ptv3/` with its licence.
- SST ([Fan et al., CVPR 2022](https://arxiv.org/abs/2112.06375)): `sst_backbone.py` re-implements its
  core mechanism; differences are listed at the top of that file.
- Diver3D, derived from [uScenes](https://github.com/era-research-lab/uScenes).
  If you use the data, cite the dataset authors as described in its README.
