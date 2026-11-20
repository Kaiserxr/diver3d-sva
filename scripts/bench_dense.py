#!/usr/bin/env python3
"""Time one training step of each dense middle stage on the GPU: fp32, bf16, bf16 + torch.compile
(~5-10 minutes; compiling takes a minute or two per model).

    conda activate sva && python scripts/bench_dense.py [model ...]     (default: all 5 base models)

Builds each DenseMixerMiddle variant from its real config, feeds a batch of 4 random frames
(~1,700 occupied voxels each, like Diver3D), and times forward + backward of the middle stage only.
Also checks that the bf16 outputs and gradients are finite. Run it while nothing else uses the GPU.
Full training steps are slower (VFE, BEV backbone, head, data loading), but the ratios carry over.
"""
import os
import sys
import time
from pathlib import Path

import torch
from easydict import EasyDict

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / 'third_party' / 'OpenPCDet' / 'tools'
sys.path.insert(0, str(TOOLS.parent))
from pcdet.config import cfg_from_yaml_file                          # noqa: E402
from pcdet.models.backbones_3d import __all__ as backbones_3d        # noqa: E402

NAMES = ['dense_densea', 'dense_identity', 'dense_conv', 'dense_na', 'dense_hybrid']
GRID = [120, 100, 10]


def batch(n_frames=4, n_vox=1700, ch=128):
    coords = []
    for b in range(n_frames):
        zyx = torch.stack([torch.randint(0, n, (n_vox * 2,)) for n in (10, 100, 120)], 1)
        zyx = torch.unique(zyx, dim=0)[:n_vox]
        coords.append(torch.cat([torch.full((len(zyx), 1), b), zyx], 1))
    coords = torch.cat(coords).int().cuda()
    return torch.randn(len(coords), ch, device='cuda'), coords, n_frames


def step_ms(m, feats, coords, bs, n=10):
    for i in range(n + 3):
        if i == 3:
            torch.cuda.synchronize(); t = time.time()
        f = feats.clone().requires_grad_(True)
        out = m({'voxel_features': f, 'voxel_coords': coords, 'batch_size': bs})['spatial_features']
        out.float().square().mean().backward()
    torch.cuda.synchronize()
    ok = bool(torch.isfinite(out).all()) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in m.parameters())
    return (time.time() - t) / n * 1000, ok


def main():
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    old = os.getcwd(); os.chdir(TOOLS)
    try:
        cfgs = {n: cfg_from_yaml_file(f'cfgs/diver3d_models/{n}.yaml', EasyDict()) for n in (sys.argv[1:] or NAMES)}
    finally:
        os.chdir(old)
    feats, coords, bs = batch()
    print(f'{"model":16}{"fp32 ms":>10}{"bf16 ms":>10}{"bf16+compile":>14}{"speed-up":>10}  finite   peak GB (bf16)')
    for name, c in cfgs.items():
        row = []
        for bf16, comp in ((False, False), (True, False), (True, True)):
            mc = EasyDict(dict(c.MODEL.BACKBONE_3D)); mc.BF16 = bf16; mc.COMPILE = comp
            m = backbones_3d[mc.NAME](model_cfg=mc, input_channels=128, grid_size=GRID).cuda().train()
            torch.cuda.reset_peak_memory_stats()
            ms, ok = step_ms(m, feats, coords, bs)
            row.append((ms, ok, torch.cuda.max_memory_allocated() / 1e9))
            del m; torch.cuda.empty_cache()
        (a, _, _), (b, ok, gb), (cc, ok2, _) = row
        print(f'{name:16}{a:10.0f}{b:10.0f}{cc:14.0f}{a / min(b, cc):9.1f}x  {str(ok and ok2):8} {gb:6.1f}', flush=True)
    print('\nOne training epoch is 5,216 steps; a run trained to epoch 12 is ~62,600 steps.')


if __name__ == '__main__':
    main()
