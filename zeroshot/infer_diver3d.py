#!/usr/bin/env python3
"""
Zero-shot inference of an OpenPCDet pretrained checkpoint on Diver3D.
Run from OpenPCDet/tools (configs use relative _BASE_CONFIG_ paths):

    python infer_diver3d.py --cfg cfgs/kitti_models/pointpillar.yaml \
        --ckpt ckpts/pointpillar_7728.pth --root /content/diver3d_v2 \
        --split val --out /content/preds/pointpillar_val.json

Keeps only the pedestrian class (closest to a diver) and writes the Diver3D
submission JSON. Yaw-only box (x,y,z,dx,dy,dz,yaw) -> quaternion Rz(yaw).
"""

import argparse
import json
import os

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class Diver3DDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, root, tokens, logger):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names,
                         training=False, root_path=None, logger=logger)
        self.diver_root = root
        self.tokens = tokens
        # nuScenes models expect (x, y, z, intensity 0-255, timestamp)
        self.n_feat = len(dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        scene, frame = self.tokens[i].split("/")
        path = os.path.join(self.diver_root, "scenes", scene, "sonar", f"frame_{frame}.bin")
        pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        if self.n_feat == 5:
            pts = np.hstack([pts[:, :3], pts[:, 3:4] * 255.0,
                             np.zeros((len(pts), 1), dtype=np.float32)])
        return self.prepare_data(data_dict={"points": pts, "frame_id": i})


def load_tokens(root, split):
    if split == "test":
        with open(os.path.join(root, "annotations", "test_tokens.txt"), encoding="utf-8") as fh:
            return [ln.strip() for ln in fh if ln.strip()]
    with open(os.path.join(root, "annotations", f"gt_{split}.json"), encoding="utf-8") as fh:
        return sorted(json.load(fh)["results"])


def yaw_box_to_diver3d(box, score):
    x, y, z, dx, dy, dz, yaw = [float(v) for v in box[:7]]
    return {
        "center": [x, y, z],
        "dimensions": [dx, dy, dz],                 # length, width, height along ex, ey, ez
        "quaternion": [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))],
        "score": float(score),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", required=True, help="diver3d_v2 folder")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="only first N frames (quick test)")
    args = ap.parse_args()

    cfg_from_yaml_file(args.cfg, cfg)
    logger = common_utils.create_logger()

    tokens = load_tokens(args.root, args.split)
    if args.limit:
        tokens = tokens[:args.limit]
    ds = Diver3DDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, args.root, tokens, logger)

    names = [n.lower() for n in cfg.CLASS_NAMES]
    ped_label = names.index("pedestrian") + 1       # pred_labels are 1-indexed
    logger.info(f"classes={cfg.CLASS_NAMES}, keeping label {ped_label}, frames={len(ds)}")

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=ds)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda().eval()

    results = {}
    with torch.no_grad():
        for i, tok in enumerate(tokens):
            batch = ds.collate_batch([ds[i]])
            load_data_to_gpu(batch)
            pred = model(batch)[0][0]
            boxes = pred["pred_boxes"].cpu().numpy()
            scores = pred["pred_scores"].cpu().numpy()
            labels = pred["pred_labels"].cpu().numpy()
            keep = np.where(labels == ped_label)[0]
            keep = keep[np.argsort(-scores[keep])][:100]
            results[tok] = [yaw_box_to_diver3d(boxes[k], scores[k]) for k in keep]
            if i % 500 == 0:
                logger.info(f"{i}/{len(tokens)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"results": results}, fh)
    n_det = sum(len(v) for v in results.values())
    logger.info(f"wrote {args.out}: {len(results)} frames, {n_det} detections")


if __name__ == "__main__":
    main()
