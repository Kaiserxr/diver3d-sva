# diver3d-sva — sparse voxel attention on Diver3D

Code for testing sparse voxel attention backbones (starting with DSVT-V) for
diver detection on **Diver3D** 3D sonar data, built on
[OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

This repo holds **only our own code**. It does not contain:

- **OpenPCDet.** `scripts/setup_openpcdet.sh` downloads it into
  `third_party/OpenPCDet`, pinned to one commit. git ignores that folder.
- **The Diver3D data.** It stays in its own `diver3d_v2/` folder exactly as
  downloaded from the challenge. It is linked, never copied or edited.

## Layout

```
overlay/                       files that get linked into OpenPCDet at the same path
  pcdet/datasets/diver3d/      Diver3DDataset: loads frames, scores with the official scorer
  tools/cfgs/dataset_configs/diver3d_dataset.yaml   crop, voxel size, split
  tools/cfgs/diver3d_models/   one model config per experiment
zeroshot/                      pretrained LiDAR checkpoints run on Diver3D (no training)
scripts/setup_openpcdet.sh     installs OpenPCDet + links overlay + links data
third_party/OpenPCDet/         (created by the setup script, not in git)
```

## Branches

| branch | what |
|---|---|
| `main` | the Colab-era files, copied unchanged from `diver3d_v2/finetune` and `diver3d_v2/zeroshot` |
| `exp/sparse-voxel-attention` | setup script, path fixes, then the sparse-conv vs DSVT-V experiments |

## Setup (lab machine)

```bash
git clone <this repo> diver3d-sva && cd diver3d-sva
git checkout exp/sparse-voxel-attention
bash scripts/setup_openpcdet.sh /path/to/diver3d_v2
# then build OpenPCDet inside the conda env, see LAB_RUNBOOK.md
```

## Known caveats

- `Diver3DDataset` currently converts each full-SO(3) ground-truth box to its
  tightest **yaw-only** box (`so3_to_yaw_box`), so models train and predict
  yaw only. That matches the paper's LiDAR baselines, but not SonarVoxNet's 6D head.
- OpenPCDet's `--fix_random_seed` always uses seed 666. A second seed needs a
  small patch (planned).
- Results are on **val**. Test labels are withheld, so paper numbers (test split)
  are a reference, not a target to match exactly.
