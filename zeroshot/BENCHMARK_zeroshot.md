# Diver3D — zero-shot benchmark (val split)

Pretrained OpenPCDet checkpoints run directly on Diver3D sonar point clouds,
**no training / fine-tuning**. Scored with the official `tools/diver3d_eval.py`
on `val` (7 scenes, 8037 frames, 7509 GT divers).

## Setup

- Codebase: OpenPCDet (master), Colab T4, torch 2.5.1+cu124, spconv-cu124.
- Input: raw `(x, y, z, intensity)`; sonar frame has the same axes as KITTI LiDAR.
  nuScenes models: intensity ×255, timestamp column = 0.
- Default config of each checkpoint, including point-cloud range and `SCORE_THRESH`.
- Only the `Pedestrian` class is kept (closest to a diver); max 100 boxes/frame.
- Yaw box `(x,y,z,dx,dy,dz,θ)` → Diver3D: `dimensions=[dx,dy,dz]`,
  `quaternion=[cos θ/2, 0, 0, sin θ/2]` (ez = sonar z-up).

## Results

| Model | Pretrained on | Checkpoint | #Det | TP / GT (IoU≥0.35) | mAP3D | AP3D@0.35 | AP3D@0.50 | APBEV@0.35 | ATE (m) | ASE | AOE3D (°) | TILT (°) | SDS* |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| VoxelNet | — | no public checkpoint | — | — | — | — | — | — | — | — | — | — | — |
| PointPillars | KITTI | `pointpillar_7728` | 15,067 | 24 / 7509 (0.3%) | 0.0000 | 0.0000 | 0.0000 | 0.0005 | 0.333 | 0.418 | 66.5 | 13.0 | 0.252 |
| SECOND | KITTI | `second_7862` | 11,947 | 15 / 7509 (0.2%) | 0.0000 | 0.0000 | 0.0000 | 0.0006 | 0.255 | 0.432 | 49.0 | 14.0 | 0.295 |
| CenterPoint† | nuScenes | `cbgs_pp_centerpoint_nds6070` | 45,272 | 112 / 7509 (1.5%) | 0.0001 | 0.0000 | 0.0000 | 0.0010 | 0.242 | 0.442 | 47.8 | 12.6 | 0.298 |
| VoxelNeXt | nuScenes | `cbgs_voxel0075_voxelnext` | 43,975 | 136 / 7509 (1.8%) | 0.0001 | 0.0001 | 0.0000 | 0.0001 | 0.201 | 0.477 | 49.2 | 8.2 | 0.296 |

For reference, the SonarVoxNet paper's **trained** baselines (on its own test split, not val):
AP3D@0.35 = 0.126 (VoxelNet), 0.190 (PointPillars), 0.315 (SECOND), 0.324 (CenterPoint),
0.374 (VoxelNeXt), 0.703 (SonarVoxNet).

## Notes

1. **Zero-shot detectors essentially fail on sonar** — mAP3D ≈ 0 for all four models. Sanity check on
   PointPillars (200 frames): predictions have human-like sizes and correct quaternion form, but their
   median distance to the nearest diver is 8.9 m and none are within 1 m. They fire on cave walls,
   rocks and sonar clutter. This is a LiDAR → sonar domain gap, not a conversion bug.
2. **\*SDS is misleading here.** With mAP3D ≈ 0 the detection half contributes nothing, but ATE/ASE/AOE3D
   are averaged over only 15–136 chance matches, which alone adds ~0.25–0.30. Use mAP3D/AP3D as the
   headline metric for zero-shot; report SDS only with the TP count.
3. Low TILT (8–14°) is selection bias: yaw-only boxes are always upright, so the few TPs are upright divers.
4. **† CenterPoint**: the voxel-0.075 and voxel-0.1 nuScenes checkpoints have been removed from the
   OpenPCDet Google Drive; the PointPillar-backbone variant (`cbgs_dyn_pp_centerpoint.yaml`) is the only
   one still available.
5. KITTI configs crop `z ∈ [-3, 1] m`; divers above 1 m are cut. Kept as-is (zero-shot = default config).
6. OpenPCDet patches needed: `np.int/np.float` → builtins (numpy ≥1.24); VoxelNeXt `_topk_1d` /
   `decode_bbox_from_voxels_nuscenes` assumed ≥ K=500 voxels per frame (crashes and mislabels classes
   on sparse sonar frames) — fixed to use the actual count.
