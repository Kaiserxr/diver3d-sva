"""FSHNet backbone for Diver3D (from FSHNet_base, FSHNet's Waymo model).

Source: FSHNet pcdet/models/backbones_3d/fshnet_base.py (https://github.com/Say2L/FSHNet, commit
8e8f23e, Apache-2.0). Pipeline (voxel grid 120 x 100 x 10 at 0.1 x 0.1 x 0.5 m):

    stem (cpe)   SubM conv + 3 sparse residual blocks, then height stride 2      z 11 -> 5
    stage 1      SlotFormer (2 layers), then sparse conv stride 2 in x, y, z     -> 60 x 50 x 3
    stage 2      SlotFormer (2 layers)
    APP          attention pooling of each vertical column -> 2D sparse BEV (0.2 m cells)
    AFD          "adaptive feature diffusion": a small classifier predicts foreground cells
                 (extra loss, get_loss) and empty cells around them are added, to fill the
                 "missing centre" of objects; then a 2D sparse encoder-decoder (SEDLayer)
    upsampling   back to 0.1 m cells (FSHNet's sparse upsampling), then a shared SubM conv
    -> batch_dict['spatial_features_2d']: sparse 2D tensor for SparseDynamicHead (stride 1)

Changes from the original:
  * class names come from the config (CLASS_NAMES), because this OpenPCDet version does not pass
    them to backbones
  * the column-pooling height (AttnPillarPool) is computed from the actual height after stage 1;
    the original used grid_height // 4, which only matches nuScenes/Waymo (41-cell-tall grids)
    and would silently drop voxels on our 10-cell grid
  * AttnPillarPool matches each pooled column to its voxels by (frame, y, x) explicitly; the
    original assumed spconv lists pooled columns in torch.unique order
  * each SlotFormer stage has its own spconv indice_key prefix; pure-PyTorch SlotFormer
  * imports point to this package (fshnet_utils, slotformer)
"""
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils.spconv_utils import replace_feature, spconv
from ..backbones_3d.spconv_backbone import post_act_block, SparseBasicBlock
from .fshnet_utils import SparseBasicBlock2D, SparseBasicBlock3D, focal_loss_sparse, norm_fn_1d, \
    post_act_block_sparse_2d, post_act_block_sparse_3d
from .slotformer import SlotFormer


def scatter_nd(indices, updates, shape):
    """PyTorch version of tf.scatter_nd (no repeated indices)."""
    ret = torch.zeros(*shape, dtype=updates.dtype, device=updates.device)
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + shape[indices.shape[-1]:]
    flat = indices.view(-1, ndim)
    ret[[flat[:, i] for i in range(ndim)] + [Ellipsis]] = updates.view(*output_shape)
    return ret


class SEDLayer(spconv.SparseModule):
    """Sparse encoder-decoder (from HEDNet): downsample with strides, then inverse convs back."""

    def __init__(self, dim, down_kernel_size, down_stride, num_SBB, indice_key, xy_only=False, bias=False):
        super().__init__()
        block = SparseBasicBlock2D if xy_only else SparseBasicBlock3D
        act_block = post_act_block_sparse_2d if xy_only else post_act_block_sparse_3d
        self.encoder = nn.ModuleList([spconv.SparseSequential(
            *[block(dim, indice_key=f'{indice_key}_0') for _ in range(num_SBB[0])])])
        for idx in range(1, len(down_stride)):
            self.encoder.append(spconv.SparseSequential(
                act_block(dim, dim, down_kernel_size[idx], down_stride[idx], down_kernel_size[idx] // 2,
                          conv_type='spconv', indice_key=f'spconv_{indice_key}_{idx}'),
                *[block(dim, indice_key=f'{indice_key}_{idx}', bias=bias) for _ in range(num_SBB[idx])]))
        self.decoder, self.decoder_norm = nn.ModuleList(), nn.ModuleList()
        for idx in range(len(down_stride) - 1, 0, -1):
            self.decoder.append(act_block(dim, dim, down_kernel_size[idx], conv_type='inverseconv',
                                          indice_key=f'spconv_{indice_key}_{idx}'))
            self.decoder_norm.append(norm_fn_1d(dim))

    def forward(self, x):
        feats = []
        for conv in self.encoder:
            x = conv(x)
            feats.append(x)
        x = feats[-1]
        for deconv, norm, up_x in zip(self.decoder, self.decoder_norm, feats[:-1][::-1]):
            x = deconv(x)
            x = replace_feature(x, norm(x.features + up_x.features))
        return x


class AttnPillarPool(nn.Module):
    """Collapse each vertical column to one BEV cell: its max-pooled feature attends to the
    column's voxels (with a learned height embedding)."""

    def __init__(self, dim, pillar_size):
        super().__init__()
        self.dim, self.pillar_size = dim, pillar_size
        self.query_func = spconv.SparseMaxPool3d((pillar_size, 1, 1), stride=(pillar_size, 1, 1), padding=0)
        self.norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.pos_embedding = nn.Embedding(pillar_size, dim)
        nn.init.normal_(self.pos_embedding.weight, std=.01)

    def forward(self, x):
        src = self.query_func(x)                     # one voxel per column (the max-pooled query)
        ny, nx = int(x.spatial_shape[1]), int(x.spatial_shape[2])
        idx = x.indices.long()
        col_key = (idx[:, 0] * ny + idx[:, 2]) * nx + idx[:, 3]
        uniq, batch_win_inds = torch.unique(col_key, return_inverse=True)
        scatter_indices = torch.stack([batch_win_inds, idx[:, 1]], -1)
        num_pillars = len(uniq)
        value = scatter_nd(scatter_indices, x.features, [num_pillars, self.pillar_size, self.dim])
        key_padding_mask = ~scatter_nd(scatter_indices, torch.ones_like(x.features[:, 0]),
                                       [num_pillars, self.pillar_size]).bool()
        # match each pooled query to its own column explicitly (the original assumed spconv lists
        # the pooled columns in torch.unique order)
        s = src.indices.long()
        pos = torch.searchsorted(uniq, (s[:, 0] * ny + s[:, 2]) * nx + s[:, 3])
        value, key_padding_mask = value[pos], key_padding_mask[pos]
        key = value + self.pos_embedding.weight.unsqueeze(0)
        output = self.self_attn(src.features.unsqueeze(1), key, value, key_padding_mask)[0].squeeze(1)
        return replace_feature(src, self.norm(output + src.features))


class FSHNetDiver3D(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, class_names=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.sparse_shape = list(np.array(grid_size[::-1]) + [1, 0, 0])
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        dim, win_size = model_cfg.FEATURE_DIM, model_cfg.WIN_SIZE

        self.cpe = spconv.SparseSequential(
            post_act_block(input_channels, dim, 3, norm_fn=norm_fn, padding=1, indice_key='subm1', conv_type='subm'),
            SparseBasicBlock(dim, dim, norm_fn=norm_fn, indice_key='stem'),
            SparseBasicBlock(dim, dim, norm_fn=norm_fn, indice_key='stem'),
            SparseBasicBlock(dim, dim, norm_fn=norm_fn, indice_key='stem'),
            post_act_block(dim, dim, (3, 1, 1), norm_fn=norm_fn, stride=(2, 1, 1), padding=0, indice_key='spconv1',
                           conv_type='spconv'),
        )
        self.stage1 = spconv.SparseSequential(
            SlotFormer(dim, 4, num_layers=2, win_size=win_size, shift=True, indice_key='sf1'),
            nn.BatchNorm1d(dim, eps=1e-3, momentum=0.01),
            spconv.SparseConv3d(dim, dim, 3, stride=2, padding=1, bias=False, indice_key='down'),
        )
        self.stage2 = spconv.SparseSequential(
            SlotFormer(dim, 4, num_layers=2, win_size=win_size, shift=True, indice_key='sf2'),
            nn.BatchNorm1d(dim, eps=1e-3, momentum=0.01),
        )
        z0 = self.sparse_shape[0]
        z1 = (z0 - 3) // 2 + 1                       # stem: kernel 3, stride 2, no padding
        z2 = (z1 + 2 - 3) // 2 + 1                   # stage 1 down: kernel 3, stride 2, padding 1
        self.app = AttnPillarPool(dim, z2)           # original: z0 // 4 (only right for 41-cell grids)

        self.adaptive_feature_diffusion = model_cfg.get('AFD', False)
        if self.adaptive_feature_diffusion:
            self.class_names = list(class_names or model_cfg.CLASS_NAMES)
            self.voxel_size, self.point_cloud_range = voxel_size, point_cloud_range
            self.fg_thr = model_cfg['FG_THRESHOLD']
            self.featmap_stride = model_cfg['FEATMAP_STRIDE']
            self.group_pooling_kernel_size = model_cfg['GREOUP_POOLING_KERNEL_SIZE']
            self.detach_feature = model_cfg['DETACH_FEATURE']
            self.group_class_names = [[x for x in names if x in self.class_names] for names in model_cfg['GROUP_CLASS_NAMES']]
            self.cls_conv = spconv.SparseSequential(
                spconv.SubMConv2d(dim, dim, 3, stride=1, padding=1, bias=False, indice_key='conv_cls'),
                norm_fn(dim), nn.ReLU(),
                spconv.SubMConv2d(dim, len(self.group_class_names), 1, bias=True, indice_key='cls_out'))
            self.forward_ret_dict = {}

        afd_dim = model_cfg.AFD_FEATURE_DIM
        assert model_cfg.AFD_DOWN_STRIDE[0] == 1 and len(model_cfg.AFD_NUM_SBB) == len(model_cfg.AFD_DOWN_STRIDE)
        self.afd_layers = nn.ModuleList([
            SEDLayer(afd_dim, model_cfg.AFD_DOWN_KERNEL_SIZE, model_cfg.AFD_DOWN_STRIDE, model_cfg.AFD_NUM_SBB,
                     indice_key=f'afdlayer{idx}', xy_only=True) for idx in range(model_cfg.AFD_NUM_LAYERS)])
        self.up_block = spconv.SparseSequential(
            spconv.SparseConv2d(afd_dim, afd_dim, 3, stride=1, padding=1, bias=False, indice_key='up_conv'),
            norm_fn_1d(afd_dim), nn.ReLU())
        self.shared_conv = spconv.SparseSequential(
            spconv.SubMConv2d(afd_dim, afd_dim, 3, stride=1, padding=1, bias=False),
            norm_fn_1d(afd_dim), nn.ReLU())

        self.num_point_features = dim
        self.num_bev_features = afd_dim
        self.init_weights()

    def init_weights(self):
        for _, m in self.named_modules():
            if isinstance(m, (spconv.SubMConv2d, spconv.SubMConv3d)):
                nn.init.kaiming_normal_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0)
        if self.adaptive_feature_diffusion:
            self.cls_conv[-1].bias.data.fill_(-2.19)

    # ---- adaptive feature diffusion: foreground targets and loss
    def assign_target(self, batch_spatial_indices, batch_gt_boxes):
        all_names = np.array(['bg', *self.class_names])
        inside_box_target = batch_spatial_indices.new_zeros((len(self.group_class_names), batch_spatial_indices.shape[0]))
        for gidx, names in enumerate(self.group_class_names):
            batch_inside_box_mask = []
            for bidx in range(len(batch_gt_boxes)):
                spatial_indices = batch_spatial_indices[batch_spatial_indices[:, 0] == bidx][:, [2, 1]]
                points = spatial_indices.clone().float() + 0.5
                points[:, 0] = points[:, 0] * self.featmap_stride * self.voxel_size[0] + self.point_cloud_range[0]
                points[:, 1] = points[:, 1] * self.featmap_stride * self.voxel_size[1] + self.point_cloud_range[1]
                points = torch.cat([points, points.new_zeros((points.shape[0], 1))], dim=-1)
                gt_boxes = batch_gt_boxes[bidx].clone()
                gt_boxes = gt_boxes[(gt_boxes[:, 3] > 0) & (gt_boxes[:, 4] > 0)]
                gt_class_names = all_names[gt_boxes[:, -1].cpu().long().numpy()]
                sel = [gt_boxes[i] for i, name in enumerate(gt_class_names) if name in names]
                inside_box_mask = points.new_zeros((points.shape[0]))
                if len(sel) > 0 and len(points) > 0:
                    boxes = torch.stack(sel)[:, :7].contiguous()
                    boxes[:, 2] = 0
                    inside_box_mask[roiaware_pool3d_utils.points_in_boxes_gpu(points[None].contiguous(), boxes[None])[0] > -1] = 1
                batch_inside_box_mask.append(inside_box_mask)
            inside_box_target[gidx] = torch.cat(batch_inside_box_mask)
        return inside_box_target

    def get_loss(self):
        spatial_indices = self.forward_ret_dict['spatial_indices']
        batch_size = self.forward_ret_dict['batch_size']
        batch_index = spatial_indices[:, 0]
        inside_box_pred = self.forward_ret_dict['inside_box_pred']
        inside_box_target = self.forward_ret_dict['inside_box_target']
        inside_box_pred = torch.cat([inside_box_pred[:, batch_index == bidx] for bidx in range(batch_size)], dim=1)
        inside_box_pred = torch.clamp(inside_box_pred.sigmoid(), min=1e-4, max=1 - 1e-4)
        cls_loss, recall_dict = 0.0, {}
        for gidx in range(len(self.group_class_names)):
            group_cls_loss = focal_loss_sparse(inside_box_pred[gidx], inside_box_target[gidx].float())
            cls_loss += group_cls_loss
            fg_mask = inside_box_target[gidx] > 0
            pred_mask = inside_box_pred[gidx][fg_mask] > self.fg_thr
            recall_dict[f'afd_recall_{gidx}'] = (pred_mask.sum() / fg_mask.sum().clamp(min=1.0)).item()
            recall_dict[f'afd_cls_loss_{gidx}'] = group_cls_loss.item()
        return cls_loss, recall_dict

    def feature_diffusion(self, x, batch_dict):
        if not self.adaptive_feature_diffusion:
            return x
        detached_x = x
        if self.detach_feature:
            detached_x = spconv.SparseConvTensor(features=x.features.detach(), indices=x.indices,
                                                 spatial_shape=x.spatial_shape, batch_size=x.batch_size)
        inside_box_pred = self.cls_conv(detached_x).features.permute(1, 0)
        if self.training:
            # targets are built frame by frame (frame order); get_loss puts predictions in the same order
            self.forward_ret_dict['batch_size'] = x.batch_size
            self.forward_ret_dict['spatial_indices'] = x.indices
            self.forward_ret_dict['inside_box_pred'] = inside_box_pred
            self.forward_ret_dict['inside_box_target'] = self.assign_target(x.indices, batch_dict['gt_boxes'])

        group_inside_mask = inside_box_pred.sigmoid() > self.fg_thr
        bg_mask = ~group_inside_mask.max(dim=0, keepdim=True)[0]
        group_inside_mask = torch.cat([group_inside_mask, bg_mask], dim=0)

        one_mask = x.features.new_zeros((x.batch_size, 1, x.spatial_shape[0], x.spatial_shape[1]))
        for gidx, inside_mask in enumerate(group_inside_mask):
            selected_indices = x.indices[inside_mask]
            single_one_mask = spconv.SparseConvTensor(
                features=x.features.new_ones(selected_indices.shape[0], 1), indices=selected_indices,
                spatial_shape=x.spatial_shape, batch_size=x.batch_size).dense()
            k = self.group_pooling_kernel_size[gidx]
            one_mask = torch.maximum(one_mask, F.max_pool2d(single_one_mask, kernel_size=k, stride=1, padding=k // 2))

        zero_indices = (one_mask[:, 0] > 0).nonzero().int()
        zero_features = x.features.new_zeros((len(zero_indices), x.features.shape[1]))
        cat_indices = torch.cat([x.indices, zero_indices], dim=0)
        cat_features = torch.cat([x.features, zero_features], dim=0)
        indices_unique, _inv = torch.unique(cat_indices, dim=0, return_inverse=True)
        features_unique = x.features.new_zeros((indices_unique.shape[0], x.features.shape[1]))
        features_unique.index_add_(0, _inv, cat_features)
        return spconv.SparseConvTensor(features=features_unique, indices=indices_unique,
                                       spatial_shape=x.spatial_shape, batch_size=x.batch_size)

    def to_bev(self, x):
        return spconv.SparseConvTensor(features=x.features, indices=x.indices[:, [0, 2, 3]],
                                       spatial_shape=x.spatial_shape[1:], batch_size=x.batch_size)

    def upsampling(self, x, up_stride=2):
        x = spconv.SparseConvTensor(features=x.features, indices=x.indices.clone(),
                                    spatial_shape=[s * up_stride for s in x.spatial_shape], batch_size=x.batch_size)
        x.indices[:, 1:] *= up_stride
        return self.up_block(x) if x.indices.shape[0] > 0 else x

    def forward(self, batch_dict):
        x = spconv.SparseConvTensor(features=batch_dict['voxel_features'], indices=batch_dict['voxel_coords'].int(),
                                    spatial_shape=self.sparse_shape, batch_size=batch_dict['batch_size'])
        x = self.cpe(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.app(x)
        x = self.to_bev(x)
        x = self.feature_diffusion(x, batch_dict)
        for layer in self.afd_layers:
            x = layer(x)
        x = self.upsampling(x)
        x = self.shared_conv(x)
        batch_dict['spatial_features_2d'] = x
        return batch_dict
