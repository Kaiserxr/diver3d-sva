"""FSHNet's sparse detection head with dynamic sparse label assignment (SparseDynamicHead).

Source: FSHNet pcdet/models/dense_heads/sparse_dynamic_head.py (https://github.com/Say2L/FSHNet,
commit 8e8f23e, Apache-2.0). The head predicts on the sparse BEV cells directly (no dense map).
For each object, the DYNAMIC_POS_NUM cells nearest its centre are candidates; DynamicPositiveMask
picks the positives among them by classification + rotation-weighted-IoU cost. Box loss: RWIoU;
an IoU branch rectifies scores at test time.

Changes from the original:
  * imports point to this package; unused functions (generate_predicted_boxes v1, distance)
    removed; batch size passed to the decoder (frames without cells no longer break decoding)
  * a frame with no target objects no longer crashes target assignment (the original indexed
    mask[:0][0]); "nearest candidate is always positive" is applied per object
    (mask[:n, 0] = True), which the original wrote as mask[:n][0] (first object only). For
    divers both give the same positives (the 5 nearest cells are well inside the box radius).
  * the feature-map size is passed as (x, y): the original passed spconv's (y, x) but clamps
    object centres as if it were (x, y), which only works on square grids (Waymo); ours is
    120 x 100.
"""
import copy

import numpy as np
import torch
import torch.nn as nn

from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils.spconv_utils import spconv
from ..model_utils import model_nms_utils
from .fshnet_utils import DynamicPositiveMask, FocalLossSparse, RWIoULoss, SlotFormerIoULoss, \
    decode_bbox_from_sparse_heatmap


class SeparateHead(nn.Module):
    def __init__(self, input_channels, sep_head_dict, kernel_size, init_bias=-2.19, use_bias=False):
        super().__init__()
        self.sep_head_dict = sep_head_dict
        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']
            fc_list = []
            for _ in range(num_conv - 1):
                fc_list.append(spconv.SparseSequential(
                    spconv.SubMConv2d(input_channels, input_channels, kernel_size, padding=int(kernel_size // 2),
                                      bias=use_bias, indice_key=cur_name),
                    nn.BatchNorm1d(input_channels), nn.ReLU()))
            fc_list.append(spconv.SubMConv2d(input_channels, output_channels, 1, bias=True, indice_key=cur_name + 'out'))
            fc = nn.Sequential(*fc_list)
            if 'heatmap' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, spconv.SubMConv2d):
                        nn.init.normal_(m.weight, mean=0, std=0.001)
            self.__setattr__(cur_name, fc)

    def forward(self, x):
        return {name: self.__getattr__(name)(x) for name in self.sep_head_dict}


class SparseDynamicHead(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size,
                 predict_boxes_when_training=False):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.grid_size = grid_size
        self.point_cloud_range = torch.Tensor(point_cloud_range).cuda()
        self.voxel_size = torch.Tensor(voxel_size).cuda()
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)
        self.bn_momentum = model_cfg.get('BN_MOM', 0.1)
        self.bn_eps = model_cfg.get('BN_EPS', 1e-5)
        self.class_names = class_names
        self.dynamic_pos_num = self.model_cfg.get('DYNAMIC_POS_NUM', 5)
        self.r_factor = self.model_cfg.get('R_FACTOR', 0.5)

        self.class_names_each_head, self.class_id_mapping_each_head = [], []
        for cur_class_names in self.model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            self.class_id_mapping_each_head.append(torch.from_numpy(np.array(
                [self.class_names.index(x) for x in cur_class_names if x in class_names])).cuda())
        assert sum(len(x) for x in self.class_names_each_head) == len(self.class_names)

        kernel_size_head = self.model_cfg.get('KERNEL_SIZE_HEAD', 3)
        self.heads_list = nn.ModuleList()
        self.separate_head_cfg = model_cfg.SEPARATE_HEAD_CFG
        for cur_class_names in self.class_names_each_head:
            cur_head_dict = copy.deepcopy(self.separate_head_cfg.HEAD_DICT)
            cur_head_dict['heatmap'] = dict(out_channels=len(cur_class_names), num_conv=model_cfg.NUM_HM_CONV)
            self.heads_list.append(SeparateHead(
                input_channels=input_channels, sep_head_dict=cur_head_dict, kernel_size=kernel_size_head,
                init_bias=-np.log((1 - 0.01) / 0.01), use_bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)))

        self.get_dynamic_masks = DynamicPositiveMask(1, self.model_cfg.get('DCLA_REG_WEIGHT', 3),
                                                     self.voxel_size[:2] * self.feature_map_stride)
        self.predict_boxes_when_training = predict_boxes_when_training
        self.forward_ret_dict = {}
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.momentum, m.eps = self.bn_momentum, self.bn_eps
        self.add_module('hm_loss_func', FocalLossSparse())
        self.add_module('reg_loss_func', RWIoULoss(self.voxel_size[:2] * self.feature_map_stride))
        self.add_module('crit_iou', SlotFormerIoULoss())

    @staticmethod
    def sigmoid(x):
        return torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)

    @staticmethod
    def manhattan_distance(voxel_indices, centers):
        return torch.abs(voxel_indices[None, :, :] - centers[:, None, :]).sum(-1)       # (M, N)

    def assign_target_of_single_head(self, num_classes, gt_boxes, feature_map_size, feature_map_stride,
                                     spatial_indices, num_max_objs=500):
        heatmap = gt_boxes.new_zeros(num_classes, len(spatial_indices))
        ret_boxes = gt_boxes.new_zeros((num_max_objs, self.dynamic_pos_num, gt_boxes.shape[-1] - 1 + 1))
        inds = gt_boxes.new_zeros(num_max_objs, self.dynamic_pos_num).long()
        mask = gt_boxes.new_zeros(num_max_objs, self.dynamic_pos_num).long()
        center_distances = gt_boxes.new_zeros(num_max_objs, self.dynamic_pos_num)

        pcr = self.point_cloud_range
        box_masks = ((gt_boxes[:, 3] > 0) & (gt_boxes[:, 4] > 0) & (gt_boxes[:, 5] > 0)
                     & (gt_boxes[:, 0] >= pcr[0]) & (gt_boxes[:, 1] >= pcr[1])
                     & (gt_boxes[:, 0] < pcr[3]) & (gt_boxes[:, 1] < pcr[4]))
        gt_boxes = gt_boxes[box_masks][:num_max_objs]
        box_num = len(gt_boxes)
        if box_num == 0 or len(spatial_indices) == 0:
            return heatmap, ret_boxes, inds, mask, center_distances

        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = torch.clamp((x - pcr[0]) / self.voxel_size[0] / feature_map_stride, min=0, max=feature_map_size[0] - 0.5)
        coord_y = torch.clamp((y - pcr[1]) / self.voxel_size[1] / feature_map_stride, min=0, max=feature_map_size[1] - 0.5)
        center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        dx = gt_boxes[:, 3] / self.voxel_size[0] / feature_map_stride
        dy = gt_boxes[:, 4] / self.voxel_size[1] / feature_map_stride
        box_radius = torch.sqrt((dx / 2) ** 2 + (dy / 2) ** 2)

        k = min(self.dynamic_pos_num, len(spatial_indices))
        distances = self.manhattan_distance(spatial_indices.float() + 0.5, center)
        sort_distance, sort_inds = torch.sort(distances, dim=-1)
        sort_distance, sort_inds = sort_distance[:, :k], sort_inds[:, :k]

        inds[:box_num, :k] = sort_inds
        mask[:box_num, :k] = (sort_distance <= box_radius[:, None]).long()
        mask[:box_num, 0] = 1                                  # nearest candidate always positive
        center_distances[:box_num, :k] = sort_distance

        ret_boxes[:box_num, :, 0:2] = center[:, None, :] - spatial_indices[inds[:box_num]][..., :2] - 0.5
        ret_boxes[:box_num, :, 2:3] = z[:, None, None]
        ret_boxes[:box_num, :, 3:6] = gt_boxes[:, None, 3:6].log()
        ret_boxes[:box_num, :, 6:7] = torch.cos(gt_boxes[:, None, 6:7])
        ret_boxes[:box_num, :, 7:] = torch.sin(gt_boxes[:, None, 6:7])
        if gt_boxes.shape[1] > 8:
            ret_boxes[:box_num, :, 8:] = gt_boxes[:, None, 7:-1]

        cur_class_id = (gt_boxes[:, -1] - 1).long()
        heatmap[cur_class_id[:, None], sort_inds] = 1
        return heatmap, ret_boxes, inds, mask, center_distances

    def assign_targets(self, gt_boxes, feature_map_size, spatial_indices, **kwargs):
        cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        batch_size = gt_boxes.shape[0]
        ret_dict = {'heatmaps': [], 'target_boxes': [], 'inds': [], 'masks': [], 'distances': [], 'gt_boxes': []}
        all_names = np.array(['bg', *self.class_names])
        for cur_class_names in self.class_names_each_head:
            hm_l, box_l, ind_l, mask_l, gt_l, dist_l = [], [], [], [], [], []
            for bs_idx in range(batch_size):
                cur_gt_boxes = gt_boxes[bs_idx]
                gt_class_names = all_names[cur_gt_boxes[:, -1].cpu().long().numpy()]
                sel = []
                for i, name in enumerate(gt_class_names):
                    if name in cur_class_names:
                        temp_box = cur_gt_boxes[i].clone()
                        temp_box[-1] = cur_class_names.index(name) + 1
                        sel.append(temp_box[None, :])
                gt_single = torch.cat(sel, dim=0) if sel else cur_gt_boxes[:0, :]
                heatmap, ret_boxes, inds, mask, distances = self.assign_target_of_single_head(
                    num_classes=len(cur_class_names), gt_boxes=gt_single, feature_map_size=feature_map_size,
                    feature_map_stride=cfg.FEATURE_MAP_STRIDE,
                    spatial_indices=spatial_indices[spatial_indices[:, 0] == bs_idx][:, [2, 1]],
                    num_max_objs=cfg.NUM_MAX_OBJS)
                hm_l.append(heatmap); box_l.append(ret_boxes); ind_l.append(inds); mask_l.append(mask)
                gt_l.append(gt_single); dist_l.append(distances)
            ret_dict['heatmaps'].append(torch.cat(hm_l, dim=1).permute(1, 0))
            ret_dict['target_boxes'].append(torch.stack(box_l, dim=0))
            ret_dict['inds'].append(torch.stack(ind_l, dim=0))
            ret_dict['masks'].append(torch.stack(mask_l, dim=0))
            ret_dict['gt_boxes'].append(gt_l)
            ret_dict['distances'].append(torch.stack(dist_l, dim=0))
        return ret_dict

    def get_loss(self):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']
        batch_size = self.forward_ret_dict['batch_size']
        tb_dict, loss = {}, 0
        batch_indices = self.forward_ret_dict['spatial_indices'][:, 0]
        spatial_indices = self.forward_ret_dict['spatial_indices'][:, 1:]

        for idx, pred_dict in enumerate(pred_dicts):
            pred_hm = self.sigmoid(pred_dict['heatmap'])
            pred_reg = torch.cat([pred_dict[name] for name in self.separate_head_cfg.HEAD_ORDER], dim=1)
            pred_reg[..., -2:] = pred_reg[..., -2:].sigmoid() * 2 - 1
            target_boxes = target_dicts['target_boxes'][idx]
            target_hm = target_dicts['heatmaps'][idx]
            masks = target_dicts['masks'][idx]
            inds = target_dicts['inds'][idx]

            pred_boxes, pred_cls, target_cls, batch_spatial_indices = [], [], [], []
            for bs_idx in range(batch_size):
                b = batch_indices == bs_idx
                pred_boxes.append(pred_reg[b][inds[bs_idx]])
                pred_cls.append(pred_hm[b][inds[bs_idx]])
                target_cls.append(target_hm[b][inds[bs_idx]])
                batch_spatial_indices.append(spatial_indices[b][inds[bs_idx]])
            batch_spatial_indices = torch.stack(batch_spatial_indices)
            target_cls, pred_boxes, pred_cls = torch.stack(target_cls), torch.stack(pred_boxes), torch.stack(pred_cls)

            iou_targets = self.get_iou_targets(pred_boxes.float(), target_boxes, masks, batch_spatial_indices)
            pos_masks = self.get_dynamic_masks(pred_cls, target_cls, pred_boxes, target_boxes, masks, iou_targets)

            for bs_idx in range(batch_size):
                b = (batch_indices == bs_idx).nonzero().reshape(-1)
                cur_target_hm = target_hm[b]
                cur_target_hm[inds[bs_idx]] *= pos_masks[bs_idx][..., None]
                target_hm[b] = cur_target_hm

            hm_loss = self.hm_loss_func(pred_hm, target_hm) * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
            reg_loss = self.reg_loss_func(pred_boxes.view(-1, pred_boxes.shape[-1]), target_boxes.view(-1, target_boxes.shape[-1]),
                                          pos_masks.eq(1).view(-1), self.r_factor)
            loc_loss = reg_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
            tb_dict['hm_loss_head_%d' % idx] = hm_loss.item()
            tb_dict['loc_loss_head_%d' % idx] = loc_loss.item()

            if 'iou' in pred_dict:
                batch_pred_ious = torch.stack([pred_dict['iou'][batch_indices == bs][inds[bs]] for bs in range(batch_size)])
                iou_loss = self.crit_iou(batch_pred_ious.view(-1), iou_targets.view(-1) * 2 - 1, pos_masks.eq(1).view(-1))
                loss += hm_loss + loc_loss + iou_loss
                tb_dict['iou_loss_head_%d' % idx] = iou_loss.item()
            else:
                loss += hm_loss + loc_loss
        tb_dict['rpn_loss'] = loss.item()
        return loss, tb_dict

    def get_iou_targets(self, box_preds, box_targets, masks, spatial_indices):
        iou_targets = torch.zeros_like(masks).float()
        box_preds = box_preds.reshape(-1, box_preds.shape[-1])
        box_targets = box_targets.reshape(-1, box_targets.shape[-1])
        spatial_indices = spatial_indices.reshape(-1, spatial_indices.shape[-1])
        box_inds = masks.view(-1).nonzero().squeeze(-1)
        if len(box_inds) == 0:
            return iou_targets
        qboxes = self._get_predicted_boxes(box_preds[box_inds], spatial_indices[box_inds])
        gboxes = self._get_predicted_boxes(box_targets[box_inds], spatial_indices[box_inds])
        iou_pos = iou3d_nms_utils.boxes_aligned_iou3d_gpu(qboxes, gboxes).detach()
        iou_targets.view(-1)[box_inds] = iou_pos.squeeze(-1)
        return torch.clamp(iou_targets, 0, 1)

    def _get_predicted_boxes(self, pred_boxes, spatial_indices):
        center, center_z, dim = pred_boxes[..., :2], pred_boxes[..., 2:3], pred_boxes[..., 3:6]
        rot_cos, rot_sin = pred_boxes[..., 6:7], pred_boxes[..., 7:8]
        dim = torch.exp(torch.clamp(dim, min=-5, max=5))
        angle = torch.atan2(rot_sin, rot_cos)
        xs = (spatial_indices[:, 1:2] + center[:, 0:1]) * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
        ys = (spatial_indices[:, 0:1] + center[:, 1:2]) * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]
        return torch.cat([xs, ys, center_z, dim, angle], dim=-1)

    def generate_predicted_boxes(self, batch_size, pred_dicts, sparse_indices):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        post_center_limit_range = torch.tensor(post_process_cfg.POST_CENTER_LIMIT_RANGE).cuda().float()
        ret_dict = [{'pred_boxes': [], 'pred_scores': [], 'pred_labels': []} for _ in range(batch_size)]
        for idx, pred_dict in enumerate(pred_dicts):
            batch_hm = pred_dict['heatmap'].sigmoid()
            bbox = pred_dict['bbox']
            batch_iou = (pred_dict['iou'] + 1) * 0.5 if 'iou' in pred_dict else None
            final_pred_dicts = decode_bbox_from_sparse_heatmap(
                heatmap=batch_hm, rot_cos=bbox[:, 6:7].sigmoid() * 2 - 1, rot_sin=bbox[:, 7:8].sigmoid() * 2 - 1,
                center=bbox[:, :2], center_z=bbox[:, 2:3], dim=bbox[:, 3:6].exp(), vel=None, iou=batch_iou,
                point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size,
                feature_map_stride=self.feature_map_stride, sparse_indices=sparse_indices,
                K=post_process_cfg.MAX_OBJ_PER_SAMPLE, score_thresh=post_process_cfg.SCORE_THRESH,
                post_center_limit_range=post_center_limit_range, batch_size=batch_size)
            for bidx, final_dict in enumerate(final_pred_dicts):
                final_dict['pred_labels'] = self.class_id_mapping_each_head[idx][final_dict['pred_labels'].long()]
                if post_process_cfg.get('USE_IOU_TO_RECTIFY_SCORE', False) and 'pred_ious' in final_dict:
                    scores, labels, ious = final_dict['pred_scores'], final_dict['pred_labels'], final_dict['pred_ious']
                    rect = scores.new_tensor(post_process_cfg.IOU_RECTIFIER)
                    final_dict['pred_scores'] = torch.pow(scores, 1 - rect[labels]) * torch.pow(ious, rect[labels])
                if post_process_cfg.NMS_CONFIG.NMS_TYPE == 'class_specific_nms':
                    selected, selected_scores = model_nms_utils.class_specific_nms(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        box_labels=final_dict['pred_labels'], nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.NMS_CONFIG.get('SCORE_THRESH', None))
                else:
                    selected, selected_scores = model_nms_utils.class_agnostic_nms(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        nms_config=post_process_cfg.NMS_CONFIG, score_thresh=None)
                ret_dict[bidx]['pred_boxes'].append(final_dict['pred_boxes'][selected])
                ret_dict[bidx]['pred_scores'].append(selected_scores)
                ret_dict[bidx]['pred_labels'].append(final_dict['pred_labels'][selected])
        for bidx in range(batch_size):
            for k in ('pred_boxes', 'pred_scores', 'pred_labels'):
                ret_dict[bidx][k] = torch.cat(ret_dict[bidx][k], dim=0)
            ret_dict[bidx]['pred_labels'] = ret_dict[bidx]['pred_labels'] + 1
        return ret_dict

    def forward(self, data_dict):
        x = data_dict['spatial_features_2d']
        spatial_indices, pred_dicts = None, []
        for head in self.heads_list:
            pred_dict = head(x)
            for k, v in pred_dict.items():
                spatial_indices = v.indices
                pred_dict[k] = v.features
            pred_dicts.append(pred_dict)
        self.forward_ret_dict.update(pred_dicts=pred_dicts, spatial_indices=spatial_indices,
                                     batch_size=data_dict['batch_size'], feature_map_size=x.spatial_shape)
        if self.training:
            self.forward_ret_dict['target_dicts'] = self.assign_targets(
                data_dict['gt_boxes'], x.spatial_shape[-2:][::-1], spatial_indices)
        else:
            data_dict['final_box_dicts'] = self.generate_predicted_boxes(data_dict['batch_size'], pred_dicts, spatial_indices)
        return data_dict
