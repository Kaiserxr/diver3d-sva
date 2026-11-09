import json
import sys
from pathlib import Path

import numpy as np

from ..dataset import DatasetTemplate
from .diver3d_utils import so3_to_yaw_box, yaw_box_to_record


class Diver3DDataset(DatasetTemplate):
    """Diver3D sonar point clouds. GT SO(3) boxes are converted to enclosing yaw-only boxes,
    predictions are scored with the official tools/diver3d_eval.py."""

    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training,
                         root_path=root_path, logger=logger)
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.src_features = self.dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list
        self.intensity_scale = self.dataset_cfg.get('INTENSITY_SCALE', 1.0)

        self.gt_path = self.root_path / 'annotations' / f'gt_{self.split}.json'
        if self.gt_path.exists():
            with open(self.gt_path, encoding='utf-8') as fh:
                self.gt = json.load(fh)['results']
            self.tokens = sorted(self.gt)
        else:
            self.gt = None
            with open(self.root_path / 'annotations' / 'test_tokens.txt', encoding='utf-8') as fh:
                self.tokens = [ln.strip() for ln in fh if ln.strip()]

        # every k-th frame, for fast per-epoch checkpoint selection on val
        step = self.dataset_cfg.get('SUBSAMPLE', {}).get(self.mode, 1)
        self.tokens = self.tokens[::step]
        if self.logger is not None:
            self.logger.info(f'Diver3D {self.split} ({self.mode}): {len(self.tokens)} frames')

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.tokens) * self.total_epochs
        return len(self.tokens)

    def get_points(self, token):
        scene, frame = token.split('/')
        path = self.root_path / 'scenes' / scene / 'sonar' / f'frame_{frame}.bin'
        pts = np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)
        pts[:, 3] *= self.intensity_scale
        if 'timestamp' in self.src_features:          # nuScenes-pretrained models: single sweep
            pts = np.hstack([pts, np.zeros((len(pts), 1), dtype=np.float32)])
        return pts

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.tokens)
        token = self.tokens[index]
        input_dict = {'points': self.get_points(token), 'frame_id': token}
        if self.gt is not None:
            boxes = [so3_to_yaw_box(b['center'], b['dimensions'], b['quaternion']) for b in self.gt[token]]
            input_dict['gt_boxes'] = np.array(boxes, dtype=np.float32).reshape(-1, 7)
            input_dict['gt_names'] = np.array([self.class_names[0]] * len(boxes))
        return self.prepare_data(data_dict=input_dict)

    def evaluation(self, det_annos, class_names, **kwargs):
        results = {}
        for anno in det_annos:
            order = np.argsort(-anno['score'])[:100]
            results[anno['frame_id']] = [yaw_box_to_record(anno['boxes_lidar'][k], anno['score'][k])
                                         for k in order]
        output_path = Path(kwargs.get('output_path', '.'))
        output_path.mkdir(parents=True, exist_ok=True)
        pred_file = output_path / f'diver3d_{self.split}_pred.json'
        with open(pred_file, 'w', encoding='utf-8') as fh:
            json.dump({'results': results}, fh)
        if self.gt is None:
            return f'Saved test predictions to {pred_file}', {}

        sys.path.insert(0, str(self.root_path / 'tools'))
        import diver3d_eval
        gt = diver3d_eval.load_boxes(str(self.gt_path), require_score=False, max_file_mb=0)
        gt = {t: gt[t] for t in results}              # score only the evaluated (sub)set
        pred = diver3d_eval.load_boxes(str(pred_file), require_score=True, max_file_mb=0)
        res = diver3d_eval.evaluate(gt, pred, verbose=False)

        keys = ['SDS', 'mAP3D', 'AP3D@0.30', 'AP3D@0.35', 'AP3D@0.40', 'AP3D@0.50', 'APBEV@0.35',
                'TP_IoU_3D', 'ATE_m', 'ASE', 'AOE_deg', 'AOE3D_deg', 'TILT_deg']
        result_str = '\n' + '\n'.join(f'  {k:<11} {res[k]:.4f}' for k in keys)
        result_str += (f"\n  ({res['_meta']['num_true_positives']} TPs / "
                       f"{res['_meta']['num_gt_boxes']} GT, {len(results)} frames)")
        return result_str, {f'diver3d/{k}': float(res[k]) for k in keys}
