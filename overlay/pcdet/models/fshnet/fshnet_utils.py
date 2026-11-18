"""Helpers for the FSHNet port that our OpenPCDet version does not have.

Copied from FSHNet (https://github.com/Say2L/FSHNet, commit 8e8f23e, Apache-2.0):
  - pcdet/models/model_utils/hednet_utils.py   post_act_block_sparse_2d/3d, SparseBasicBlock2D/3D
  - pcdet/utils/loss_utils.py                    focal_loss_sparse, RWIoULoss, SlotFormerIoULoss,
                                                 DynamicPositiveMask
  - pcdet/utils/box_utils.py                     get_rwiou
  - pcdet/models/model_utils/centernet_utils.py  _topk_sparse, _gather_feat_sparse,
                                                 decode_bbox_from_sparse_heatmap
Changes: _topk_sparse pads frames that have fewer than K candidate cells (the original stacks
frames of different lengths and crashes); imports adjusted. Everything else is unchanged.
"""
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.spconv_utils import spconv

norm_fn_1d = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)


# ---------------------------------------------------------------- sparse conv blocks (HEDNet)
def post_act_block_sparse_2d(input_dim, output_dim, kernel_size, stride=1, padding=0, norm_fn=norm_fn_1d,
                             conv_type='subm', indice_key=None):
    if conv_type == 'subm':
        conv = spconv.SubMConv2d(input_dim, output_dim, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv2d(input_dim, output_dim, kernel_size, stride=stride, padding=padding, bias=False,
                                   indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv2d(input_dim, output_dim, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError
    return spconv.SparseSequential(conv, norm_fn(output_dim), nn.ReLU())


def post_act_block_sparse_3d(input_dim, output_dim, kernel_size, stride=1, padding=0, norm_fn=norm_fn_1d,
                             conv_type='subm', indice_key=None):
    if conv_type == 'subm':
        conv = spconv.SubMConv3d(input_dim, output_dim, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(input_dim, output_dim, kernel_size, stride=stride, padding=padding, bias=False,
                                   indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(input_dim, output_dim, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError
    return spconv.SparseSequential(conv, norm_fn(output_dim), nn.ReLU())


class SparseBasicBlock2D(spconv.SparseModule):
    def __init__(self, dim, indice_key, norm_fn=norm_fn_1d, bias=False):
        super().__init__()
        self.conv1 = spconv.SubMConv2d(dim, dim, 3, 1, 1, bias=bias, indice_key=indice_key)
        self.bn1 = norm_fn(dim)
        self.conv2 = spconv.SubMConv2d(dim, dim, 3, 1, 1, bias=bias, indice_key=indice_key)
        self.bn2 = norm_fn(dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = out.replace_feature(self.relu(self.bn1(out.features)))
        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))
        return out.replace_feature(self.relu(out.features + x.features))


class SparseBasicBlock3D(spconv.SparseModule):
    def __init__(self, dim, indice_key, norm_fn=norm_fn_1d, bias=False):
        super().__init__()
        self.conv1 = spconv.SubMConv3d(dim, dim, 3, 1, 1, bias=bias, indice_key=indice_key)
        self.bn1 = norm_fn(dim)
        self.conv2 = spconv.SubMConv3d(dim, dim, 3, 1, 1, bias=bias, indice_key=indice_key)
        self.bn2 = norm_fn(dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = out.replace_feature(self.relu(self.bn1(out.features)))
        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))
        return out.replace_feature(self.relu(out.features + x.features))


# ---------------------------------------------------------------- losses
def focal_loss_sparse(pred, target, eps=1e-4):
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    neg_weights = torch.pow(1 - target, 4)
    pos_loss = (torch.log(pred + eps) * torch.pow(1 - pred, 2) * pos_inds).sum()
    neg_loss = (torch.log(1 - pred + eps) * torch.pow(pred, 2) * neg_weights * neg_inds).sum()
    num_pos = pos_inds.float().sum()
    if num_pos == 0:
        return -neg_loss
    return -(pos_loss + neg_loss) / num_pos


class FocalLossSparse(nn.Module):
    def forward(self, out, target):
        return focal_loss_sparse(out, target)


def get_rwiou(bboxes1, bboxes2, r_factor=1, voxel_size=(0.8, 0.8)):
    x1u, y1u, z1u = bboxes1[..., 0], bboxes1[..., 1], bboxes1[..., 2]
    l1, w1, h1 = torch.exp(bboxes1[..., 3]), torch.exp(bboxes1[..., 4]), torch.exp(bboxes1[..., 5])
    s1, c1 = bboxes1[..., 6], bboxes1[..., 7]
    x2u, y2u, z2u = bboxes2[..., 0], bboxes2[..., 1], bboxes2[..., 2]
    l2, w2, h2 = torch.exp(bboxes2[..., 3]), torch.exp(bboxes2[..., 4]), torch.exp(bboxes2[..., 5])
    s2, c2 = bboxes2[..., 6], bboxes2[..., 7]

    x1, y1, z1 = x1u * voxel_size[0], y1u * voxel_size[1], z1u
    x2, y2, z2 = x2u * voxel_size[0], y2u * voxel_size[1], z2u

    eps = 1e-4                                       # clamp is necessary to avoid inf
    l1, w1, h1 = torch.clamp(l1, min=eps, max=30), torch.clamp(w1, min=eps, max=10), torch.clamp(h1, min=eps, max=10)
    s1, c1 = torch.clamp(s1, min=-1, max=1), torch.clamp(c1, min=-1, max=1)
    volume_1, volume_2 = l1 * w1 * h1, l2 * w2 * h2

    inter_l, inter_r = torch.max(x1 - l1 / 2, x2 - l2 / 2), torch.min(x1 + l1 / 2, x2 + l2 / 2)
    inter_t, inter_b = torch.max(y1 - w1 / 2, y2 - w2 / 2), torch.min(y1 + w1 / 2, y2 + w2 / 2)
    inter_u, inter_d = torch.max(z1 - h1 / 2, z2 - h2 / 2), torch.min(z1 + h1 / 2, z2 + h2 / 2)
    inter_volume = torch.clamp(inter_r - inter_l, min=0) * torch.clamp(inter_b - inter_t, min=0) \
        * torch.clamp(inter_d - inter_u, min=0)
    inter_volume *= (1 - r_factor * torch.abs(s1 - s2) / 2) * (1 - r_factor * torch.abs(c1 - c2) / 2)

    c_l, c_r = torch.min(x1 - l1 / 2, x2 - l2 / 2), torch.max(x1 + l1 / 2, x2 + l2 / 2)
    c_t, c_b = torch.min(y1 - w1 / 2, y2 - w2 / 2), torch.max(y1 + w1 / 2, y2 + w2 / 2)
    c_u, c_d = torch.min(z1 - h1 / 2, z2 - h2 / 2), torch.max(z1 + h1 / 2, z2 + h2 / 2)

    inter_diag = (x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2
    c_diag = torch.clamp(c_r - c_l, min=0) ** 2 + torch.clamp(c_b - c_t, min=0) ** 2 + torch.clamp(c_d - c_u, min=0) ** 2
    union = volume_1 + volume_2 - inter_volume
    return inter_diag / c_diag, inter_volume / union


class RWIoULoss(nn.Module):
    def __init__(self, voxel_size):
        super().__init__()
        self.voxel_size = voxel_size

    def forward(self, pred_reg, gt_reg, mask, r_factor=0.5):
        isnotnan = (~torch.isnan(gt_reg)).float().all(dim=-1)
        u, rwiou = get_rwiou(pred_reg, gt_reg, r_factor, self.voxel_size)
        loss = (1 - torch.clamp(rwiou, min=0, max=1.0) + u) * mask * isnotnan
        num = (mask * isnotnan).float().sum()
        return loss.sum() / torch.clamp(num, min=1.0)


class SlotFormerIoULoss(nn.Module):
    def forward(self, pred_ious, gt_ious, mask):
        pred_ious, gt_ious = pred_ious[mask.bool()], gt_ious[mask.bool()]
        return F.l1_loss(pred_ious, gt_ious, reduction='sum') / torch.clamp_min(mask.sum(), min=1.0)


class DynamicPositiveMask(nn.Module):
    """Dynamic sparse label assignment: per object, pick the candidate cells with the lowest
    classification + box-regression cost as positives (count = round(sum of IoUs), at least 1)."""

    def __init__(self, cls_weight=1, reg_weight=2, voxel_size=(0.8, 0.8)):
        super().__init__()
        self.cls_weight, self.reg_weight, self.voxel_size = cls_weight, reg_weight, voxel_size

    def cls_cost(self, pred_cls, pos_mask):
        return 1 - torch.max(pred_cls * pos_mask, dim=-1)[0]

    def rwiou_cost(self, pred_reg, gt_reg, mask, r_factor=0.2):
        isnotnan = (~torch.isnan(gt_reg)).float().all(dim=-1)
        u, rdiou = get_rwiou(pred_reg, gt_reg, r_factor, self.voxel_size)
        return (1 - torch.clamp(rdiou, min=0, max=1.0) + u) * mask * isnotnan, u

    def forward(self, pred_cls, target_cls, pred_reg, gt_reg, masks, iou_target, r_factor=0.5):
        with torch.no_grad():
            cls_cost = self.cls_cost(pred_cls, target_cls)
            reg_cost, _ = self.rwiou_cost(pred_reg, gt_reg, masks, r_factor)
        positive_masks = target_cls.new_zeros(*masks.shape)
        all_cost = self.cls_weight * cls_cost * masks + self.reg_weight * reg_cost + (1 - masks.float()) * 100
        _, local_sort_inds = torch.sort(all_cost, dim=-1)
        obj_positive_nums = torch.sum(iou_target, dim=-1).clamp(1).int()
        for batch_id in range(pred_cls.shape[0]):
            box_num = (torch.sum(masks[batch_id], -1) > 0).sum()
            for box_id in range(box_num):
                local_pos_inds = local_sort_inds[batch_id][box_id][:obj_positive_nums[batch_id][box_id]]
                positive_masks[batch_id][box_id] = iou_target[batch_id][box_id]
                positive_masks[batch_id][box_id][local_pos_inds] = 1
        return positive_masks * masks


# ---------------------------------------------------------------- decoding sparse heatmaps
def _topk_sparse(scores, batch_size, batch_idx, K=500):
    """Top-K cells per frame over all classes. Frames with fewer than K cells are padded with
    score 0 (removed later by the score threshold) so every frame returns exactly K entries."""
    s_list, i_list, c_list = [], [], []
    for bidx in range(batch_size):
        score = scores[batch_idx == bidx].permute(1, 0)            # (classes, cells)
        topk_per_cls = min(K, score.shape[-1])
        topk_scores, topk_inds = torch.topk(score, topk_per_cls)
        k = min(K, topk_scores.numel())
        topk_score, topk_ind = torch.topk(topk_scores.reshape(-1), k)
        topk_classes = (topk_ind // max(topk_per_cls, 1)).int()
        topk_inds = topk_inds.reshape(-1).gather(0, topk_ind)
        if k < K:
            pad = K - k
            topk_score = F.pad(topk_score, (0, pad), value=0.0)
            topk_inds = F.pad(topk_inds, (0, pad), value=0)
            topk_classes = F.pad(topk_classes, (0, pad), value=0)
        s_list.append(topk_score); i_list.append(topk_inds); c_list.append(topk_classes)
    return torch.stack(s_list), torch.stack(i_list), torch.stack(c_list)


def _gather_feat_sparse(feats, inds, batch_size, batch_idx):
    _inds = inds.unsqueeze(-1).expand(inds.size(0), inds.size(1), feats.size(-1))
    out = []
    for bidx in range(batch_size):
        f = feats[batch_idx == bidx]
        if f.shape[0] == 0:                                        # empty frame: padded entries only
            f = feats.new_zeros((1, feats.shape[-1]))
        out.append(f.gather(0, _inds[bidx]))
    return torch.stack(out)


def decode_bbox_from_sparse_heatmap(heatmap, rot_cos, rot_sin, center, center_z, dim, vel, iou,
                                    point_cloud_range, voxel_size, sparse_indices, feature_map_stride,
                                    K=500, score_thresh=None, post_center_limit_range=None, batch_size=None):
    if batch_size is None:
        batch_size = int(sparse_indices[:, 0].max()) + 1
    batch_idx = sparse_indices[:, 0]
    spatial_indices = sparse_indices[:, 1:]
    scores, inds, class_ids = _topk_sparse(heatmap, batch_size, batch_idx, K)

    center = _gather_feat_sparse(center, inds, batch_size, batch_idx)
    rot_sin = _gather_feat_sparse(rot_sin, inds, batch_size, batch_idx)
    rot_cos = _gather_feat_sparse(rot_cos, inds, batch_size, batch_idx)
    center_z = _gather_feat_sparse(center_z, inds, batch_size, batch_idx)
    dim = _gather_feat_sparse(dim, inds, batch_size, batch_idx)
    spatial_indices = _gather_feat_sparse(spatial_indices, inds, batch_size, batch_idx)

    angle = torch.atan2(rot_sin, rot_cos)
    xs = (spatial_indices[:, :, -1:] + center[:, :, 0:1] + 0.5) * feature_map_stride * voxel_size[0] + point_cloud_range[0]
    ys = (spatial_indices[:, :, -2:-1] + center[:, :, 1:2] + 0.5) * feature_map_stride * voxel_size[1] + point_cloud_range[1]
    box_part_list = [xs, ys, center_z, dim, angle]
    if vel is not None:
        box_part_list.append(_gather_feat_sparse(vel, inds, batch_size, batch_idx))
    if iou is not None:
        iou = torch.clamp(_gather_feat_sparse(iou, inds, batch_size, batch_idx).view(batch_size, K), min=0, max=1.)

    final_box_preds = torch.cat(box_part_list, dim=-1)
    final_scores = scores.view(batch_size, K)
    final_class_ids = class_ids.view(batch_size, K)

    assert post_center_limit_range is not None
    mask = (final_box_preds[..., :3] >= post_center_limit_range[:3]).all(2)
    mask &= (final_box_preds[..., :3] <= post_center_limit_range[3:]).all(2)
    if score_thresh is not None:
        mask &= (final_scores > score_thresh)

    ret = []
    for k in range(batch_size):
        m = mask[k]
        ret.append({'pred_boxes': final_box_preds[k, m], 'pred_scores': final_scores[k, m],
                    'pred_labels': final_class_ids[k, m]})
        if iou is not None:
            ret[-1]['pred_ious'] = iou[k, m]
    return ret
