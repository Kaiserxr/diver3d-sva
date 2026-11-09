"""Sparse 3D convolution middle stage for Diver3D (the "Run A" control in docs/DSVT_PLAN.md).

Follows the SonarVoxNet paper: "a sparse 3D convolutional middle stage that collapses height
10 -> 2", reshaped into a 128-channel BEV feature. Unlike OpenPCDet's VoxelBackBone8x it never
shrinks the top-down (x, y) grid, only the height (z), so the BEV map keeps 0.1 m cells.

    input voxels      Z=10, 128 ch  (from the VFE)
    2 x SubM 3^3      Z=10,  64 ch
    SpConv  stride z2 Z=5,   64 ch
    2 x SubM 3^3      Z=5,   64 ch
    SpConv  (3,1,1) stride z2, no z padding   Z=2, 64 ch
    -> HeightCompression: 2 x 64 = 128 BEV channels (set NUM_BEV_FEATURES: 128)

Config keys (all optional):
    NUM_FILTERS: [64, 64]   channels of the first and second block
    NUM_SUBM:    2          SubM layers per block
"""
from functools import partial

import torch.nn as nn

from ...utils.spconv_utils import spconv
from .spconv_backbone import post_act_block


class SonarSparseMiddle(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        c1, c2 = model_cfg.get('NUM_FILTERS', [64, 64])
        n_subm = model_cfg.get('NUM_SUBM', 2)
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        # spconv wants (z, y, x); OpenPCDet's grid_size is (x, y, z)
        self.sparse_shape = [int(v) for v in grid_size[::-1]]
        nz = self.sparse_shape[0]
        z_after_1 = (nz + 2 * 1 - 3) // 2 + 1          # kernel 3, stride 2, padding 1
        z_after_2 = (z_after_1 - 3) // 2 + 1           # kernel 3, stride 2, no padding
        assert z_after_2 >= 1, f'grid is only {nz} voxels tall; SonarSparseMiddle needs at least 4'

        def subm_stack(cin, cout, key):
            layers = [post_act_block(cin, cout, 3, norm_fn=norm_fn, indice_key=key)]
            layers += [post_act_block(cout, cout, 3, norm_fn=norm_fn, indice_key=key) for _ in range(n_subm - 1)]
            return spconv.SparseSequential(*layers)

        self.block1 = subm_stack(input_channels, c1, 'subm1')
        self.down1 = post_act_block(c1, c2, 3, norm_fn=norm_fn, stride=(2, 1, 1), padding=1,
                                    indice_key='down1', conv_type='spconv')
        self.block2 = subm_stack(c2, c2, 'subm2')
        self.down2 = post_act_block(c2, c2, (3, 1, 1), norm_fn=norm_fn, stride=(2, 1, 1), padding=0,
                                    indice_key='down2', conv_type='spconv')

        self.out_height = z_after_2                    # 2 for the 10-voxel Diver3D grid
        self.num_point_features = c2
        self.backbone_channels = {'x_conv1': c1, 'x_conv2': c2}

    def forward(self, batch_dict):
        x = spconv.SparseConvTensor(
            features=batch_dict['voxel_features'],
            indices=batch_dict['voxel_coords'].int(),   # (batch, z, y, x)
            spatial_shape=self.sparse_shape,
            batch_size=batch_dict['batch_size'],
        )
        x1 = self.block1(x)
        x2 = self.block2(self.down1(x1))
        out = self.down2(x2)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 1,          # x/y never downsampled
            'multi_scale_3d_features': {'x_conv1': x1, 'x_conv2': x2},
            'multi_scale_3d_strides': {'x_conv1': 1, 'x_conv2': 1},
        })
        return batch_dict
