"""Voxel features -> BEV map by max-pooling each vertical column.

Used by middle stages that return one feature per voxel at full resolution (PTv3, SST-style).
Every non-empty (x, y) column becomes one BEV cell holding the element-wise max over the voxels
stacked in it; empty cells are zero. Output: spatial_features (B, C, ny, nx), the same layout
HeightCompression and PointPillarScatter3d produce, so the BEV encoder and head are shared.

Config keys:
    NUM_BEV_FEATURES: C, must equal the middle stage's output channels
"""
import torch
import torch.nn as nn


class VoxelColumnMaxScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.nx, self.ny = int(grid_size[0]), int(grid_size[1])
        self.num_bev_features = model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict, **kwargs):
        feats = batch_dict['voxel_features']                 # (N, C)
        coords = batch_dict['voxel_coords'].long()           # (batch, z, y, x)
        assert feats.shape[1] == self.num_bev_features, (feats.shape, self.num_bev_features)
        batch_size = batch_dict.get('batch_size', int(coords[:, 0].max()) + 1)

        cell = (coords[:, 0] * self.ny + coords[:, 2]) * self.nx + coords[:, 3]
        bev = feats.new_zeros(batch_size * self.ny * self.nx, feats.shape[1])
        bev = bev.scatter_reduce(0, cell[:, None].expand_as(feats), feats, reduce='amax', include_self=False)
        bev = bev.view(batch_size, self.ny, self.nx, -1).permute(0, 3, 1, 2).contiguous()
        batch_dict['spatial_features'] = bev
        batch_dict['spatial_features_stride'] = 1
        return batch_dict
