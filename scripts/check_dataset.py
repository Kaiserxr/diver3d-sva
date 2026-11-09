#!/usr/bin/env python
"""Load a few Diver3D samples through OpenPCDet, exactly as training will.

    python scripts/check_dataset.py [config name, default: pointpillar]

No GPU or training involved. Paste the output back when asked.
"""
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / 'third_party' / 'OpenPCDet' / 'tools'
os.chdir(TOOLS)                                 # configs use paths relative to tools/
sys.path.insert(0, str(TOOLS.parent))

from pcdet.config import cfg, cfg_from_yaml_file   # noqa: E402
from pcdet.datasets import build_dataloader        # noqa: E402
from pcdet.utils import common_utils               # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else 'pointpillar'
cfg_from_yaml_file(f'cfgs/diver3d_models/{name}.yaml', cfg)
logger = common_utils.create_logger()

for training in (True, False):
    ds, _, _ = build_dataloader(cfg.DATA_CONFIG, cfg.CLASS_NAMES, batch_size=1, dist=False,
                                workers=0, logger=logger, training=training)
    mode = 'train' if training else 'test'
    print(f'\n== {mode}: split={ds.split}, {len(ds)} frames, data={ds.root_path.resolve()}')
    n_boxes, shown = 0, 0
    for i in range(0, len(ds), max(1, len(ds) // 20)):     # ~20 samples spread over the split
        s = ds[i]
        gt = s.get('gt_boxes', np.zeros((0, 8)))
        n_boxes += len(gt)
        if shown < 3 and len(gt):
            shown += 1
            print(f"  {s['frame_id']}: points {s['points'].shape}, voxels {s['voxels'].shape}, "
                  f'gt_boxes {gt.shape}')
            print(f'    first box (x y z dx dy dz yaw class): {np.round(gt[0], 2).tolist()}')
    print(f'  sampled ~20 frames: {n_boxes} boxes total')

print('\nDATASET CHECK OK')
