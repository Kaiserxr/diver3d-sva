# diver3d-sva

Sparse voxel attention backbones for **diver detection from 3D sonar**, evaluated
on the [Diver3D](#data) dataset and built on
[OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

The goal is a controlled comparison: keep the voxelizer, BEV encoder and detection
head fixed, and swap only the 3D middle stage — a sparse 3D convolution
backbone versus a sparse voxel attention backbone, starting with
[DSVT-V](https://arxiv.org/abs/2301.06051).

> **Status: work in progress.** What works today: the Diver3D dataset loader for
> OpenPCDet, scoring with the official Diver3D evaluator, a PointPillars config and
> a zero-shot evaluation script. The sparse-conv and DSVT-V configs are next.

## What is in this repo

This repo holds only the code that is specific to Diver3D. OpenPCDet and the data
are fetched separately.

```
overlay/                     files linked into OpenPCDet at the same paths
  pcdet/datasets/diver3d/    Diver3DDataset: loads frames, scores with the official evaluator
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

## Training and evaluation

From `third_party/OpenPCDet/tools`:

```bash
python train.py --cfg_file cfgs/diver3d_models/pointpillar.yaml --fix_random_seed
python test.py  --cfg_file cfgs/diver3d_models/pointpillar.yaml --ckpt <checkpoint.pth>
```

Evaluation writes predictions in the Diver3D submission format and scores them
with the dataset's official `tools/diver3d_eval.py` (SDS, mAP3D, AP3D, AOE3D, …).
By default the validation set is subsampled to every 4th frame for fast
per-epoch checkpoint selection; set `SUBSAMPLE.test: 1` for the full split.

## Known limitations

- **Yaw-only boxes.** `Diver3DDataset` converts each SO(3) ground-truth box to
  its tightest yaw-only box, as standard LiDAR detectors expect. Pitch and roll are
  therefore not learned, which caps AP3D and AOE3D. A full-rotation (6D) head is
  planned.
- **One seed.** OpenPCDet's `--fix_random_seed` always uses seed 666.
- **Validation only.** Diver3D test labels are withheld, so all results here
  are on the validation split.

## Branches

`main` holds stable code. Experiments are developed on `exp/*` branches and merged
when they work.

## Acknowledgements

- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) (Apache-2.0), which provides the detectors, including DSVT.
- Diver3D, derived from [uScenes](https://github.com/era-research-lab/uScenes).
  If you use the data, cite the dataset authors as described in its README.
