"""Point Transformer V3 as an OpenPCDet 3D backbone (RUN C in docs/DSVT_PLAN.md).

Wraps the official PTv3 model (vendored in ./ptv3 from Pointcept) so it takes the VFE's voxels
and returns one feature per voxel, in the same order. VoxelColumnMaxScatter then turns those
into the BEV map, so everything after the middle stage is shared with Runs A and B.

PTv3 groups voxels by sorting them along space-filling curves (z-order and Hilbert, in several
orientations) and running attention inside equal-size "patches" of that ordering; an
encoder-decoder (U-Net) gives it multi-scale context and brings features back to every voxel.

Config keys (see dsvt_voxel / ptv3_voxel.yaml): ORDER, STRIDE, ENC_DEPTHS, ENC_CHANNELS,
ENC_NUM_HEAD, ENC_PATCH_SIZE, DEC_DEPTHS, DEC_CHANNELS, DEC_NUM_HEAD, DEC_PATCH_SIZE,
MLP_RATIO, DROP_PATH. FlashAttention is off (no Blackwell build needed); PTv3 then uses its
plain-PyTorch attention path.
"""
import torch
import torch.nn as nn


class PTv3Backbone(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        # imported here, not at the top: OpenPCDet imports every registered backbone at startup,
        # and PTv3 needs addict + torch_scatter, which other models (e.g. PointPillars) don't
        from .ptv3 import PointTransformerV3
        c = model_cfg
        self.model_cfg = model_cfg
        self.register_buffer('voxel_size', torch.tensor(voxel_size, dtype=torch.float32), persistent=False)
        self.register_buffer('pc_min', torch.tensor(point_cloud_range[:3], dtype=torch.float32), persistent=False)

        self.ptv3 = PointTransformerV3(
            in_channels=input_channels,
            order=tuple(c.get('ORDER', ['z', 'z-trans', 'hilbert', 'hilbert-trans'])),
            stride=tuple(c.STRIDE),
            enc_depths=tuple(c.ENC_DEPTHS),
            enc_channels=tuple(c.ENC_CHANNELS),
            enc_num_head=tuple(c.ENC_NUM_HEAD),
            enc_patch_size=tuple(c.ENC_PATCH_SIZE),
            dec_depths=tuple(c.DEC_DEPTHS),
            dec_channels=tuple(c.DEC_CHANNELS),
            dec_num_head=tuple(c.DEC_NUM_HEAD),
            dec_patch_size=tuple(c.DEC_PATCH_SIZE),
            mlp_ratio=c.get('MLP_RATIO', 4),
            drop_path=c.get('DROP_PATH', 0.1),
            enable_flash=False,
            upcast_attention=True,
            upcast_softmax=True,
            enc_mode=False,
        )
        self.num_point_features = c.DEC_CHANNELS[0]

    def forward(self, batch_dict):
        feats = batch_dict['voxel_features']
        coords = batch_dict['voxel_coords'].long()          # (batch, z, y, x)

        # PTv3 needs voxels grouped by frame (its "offset" bookkeeping); keep the permutation
        # so the output can be returned in the caller's order
        order = torch.argsort(coords[:, 0], stable=True)
        coords_sorted = coords[order]
        grid_coord = coords_sorted[:, [3, 2, 1]].int()       # (x, y, z) voxel indices
        point = self.ptv3({
            'feat': feats[order],
            'grid_coord': grid_coord,
            'coord': (grid_coord.float() + 0.5) * self.voxel_size + self.pc_min,   # voxel centres, metres
            'batch': coords_sorted[:, 0],
        })

        out = torch.empty_like(point.feat)
        out[order] = point.feat
        batch_dict['voxel_features'] = out                   # same voxels, same order as the input
        return batch_dict
