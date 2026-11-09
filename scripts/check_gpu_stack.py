#!/usr/bin/env python
"""Check that the GPU software stack works, one piece at a time.

    python scripts/check_gpu_stack.py

Prints PASS / FAIL / SKIP for each piece. Paste the whole output back when asked.
A piece that is not installed yet shows SKIP, so this is safe to run at any stage.
"""
import importlib
import sys
import traceback

results = []


def check(name, fn):
    try:
        info = fn()
        results.append(('PASS', name, info or ''))
    except ModuleNotFoundError as e:
        results.append(('SKIP', name, f'not installed ({e.name})'))
    except Exception as e:                       # noqa: BLE001 - we want every failure reported
        results.append(('FAIL', name, f'{type(e).__name__}: {e}'))
        traceback.print_exc()


def torch_cuda():
    import torch
    assert torch.cuda.is_available(), 'torch.cuda.is_available() is False'
    cap = torch.cuda.get_device_capability(0)
    arch = f'sm_{cap[0]}{cap[1]}'
    archs = torch.cuda.get_arch_list()
    assert arch in archs, f'GPU is {arch} but this torch build only has {archs}'
    x = torch.randn(1024, 1024, device='cuda')
    float((x @ x).sum())                         # forces a real kernel launch
    return (f'torch {torch.__version__}, built for CUDA {torch.version.cuda}, '
            f'GPU {torch.cuda.get_device_name(0)} ({arch})')


def spconv_conv():
    import torch
    import spconv
    import spconv.pytorch as spconv_t
    # 200 random voxels in a 41x40x40 grid, 4 features each
    coords = torch.unique(torch.randint(0, 40, (200, 3), dtype=torch.int32), dim=0)
    coords = torch.cat([torch.zeros(len(coords), 1, dtype=torch.int32), coords], 1).cuda()
    feats = torch.randn(len(coords), 4, device='cuda')
    x = spconv_t.SparseConvTensor(feats, coords, [41, 40, 40], batch_size=1)
    net = spconv_t.SparseSequential(
        spconv_t.SubMConv3d(4, 16, 3, indice_key='s1'),
        spconv_t.SparseConv3d(16, 32, 3, stride=2),
    ).cuda()
    y = net(x)
    y.features.sum().backward()                  # backward pass too: training needs it
    return f'spconv {spconv.__version__}, output {tuple(y.features.shape)}'


def pcdet_ops():
    import torch
    from pcdet.ops.iou3d_nms import iou3d_nms_utils
    boxes = torch.tensor([[0., 0., 0., 1., 1., 1., 0.],
                          [0.2, 0., 0., 1., 1., 1., 0.3]], device='cuda')
    iou = iou3d_nms_utils.boxes_iou3d_gpu(boxes, boxes)
    keep, _ = iou3d_nms_utils.nms_gpu(boxes, torch.tensor([0.9, 0.8], device='cuda'), thresh=0.1)
    return f'iou3d_nms CUDA ops run, IoU[0,1]={iou[0, 1].item():.3f}, nms kept {len(keep)}'


def torch_scatter_gpu():
    import torch
    import torch_scatter
    src = torch.tensor([1., 5., 2., 7.], device='cuda', requires_grad=True)
    idx = torch.tensor([0, 0, 1, 1], device='cuda')
    out, _ = torch_scatter.scatter_max(src, idx)
    out.sum().backward()
    assert out.tolist() == [5., 7.], out.tolist()
    return f'torch_scatter {torch_scatter.__version__}, scatter_max on GPU ok'


def middle_stages():
    """Build every run's middle stage from the real configs and check they give the same BEV shape."""
    import os
    from pathlib import Path
    import numpy as np
    import torch
    from pcdet.config import cfg_from_yaml_file
    from easydict import EasyDict
    from pcdet.models.backbones_3d import __all__ as backbones_3d
    from pcdet.models.backbones_2d.map_to_bev import __all__ as map_to_bev

    tools = Path(__file__).resolve().parents[1] / 'third_party' / 'OpenPCDet' / 'tools'
    old = os.getcwd()
    os.chdir(tools)
    try:
        cfgs = {n: cfg_from_yaml_file(f'cfgs/diver3d_models/{n}.yaml', EasyDict())
                for n in ('centerpoint_sparseconv', 'dsvt_voxel', 'ptv3_voxel', 'sst_voxel')}
    finally:
        os.chdir(old)

    grid = np.array([120, 100, 10])                    # x, y, z
    torch.manual_seed(0)
    coords = []
    for b in range(2):                                  # 2 frames, ~1700 voxels each (typical Diver3D)
        xyz = torch.stack([torch.randint(0, n, (2500,)) for n in (10, 100, 120)], 1)   # z, y, x
        xyz = torch.unique(xyz, dim=0)[:1700]
        coords.append(torch.cat([torch.full((len(xyz), 1), b), xyz], 1))
    coords = torch.cat(coords).int().cuda()

    shapes, info = {}, []
    for name, c in cfgs.items():
        m3d = backbones_3d[c.MODEL.BACKBONE_3D.NAME](
            model_cfg=c.MODEL.BACKBONE_3D, input_channels=c.MODEL.VFE.NUM_FILTERS[-1], grid_size=grid,
            voxel_size=[0.1, 0.1, 0.5], point_cloud_range=[0, -5, -2.5, 12, 5, 2.5]).cuda()
        bev = map_to_bev[c.MODEL.MAP_TO_BEV.NAME](model_cfg=c.MODEL.MAP_TO_BEV, grid_size=grid).cuda()
        feats = torch.randn(len(coords), c.MODEL.VFE.NUM_FILTERS[-1], device='cuda', requires_grad=True)
        d = bev(m3d({'voxel_features': feats, 'voxel_coords': coords, 'batch_size': 2}))
        out = d['spatial_features']
        out.mean().backward()
        n_params = sum(p.numel() for p in m3d.parameters())
        shapes[name] = tuple(out.shape)
        info.append(f'{c.MODEL.BACKBONE_3D.NAME} {n_params / 1e6:.2f}M params -> {tuple(out.shape)}')
    assert len(set(shapes.values())) == 1, f'BEV shapes differ: {shapes}'
    assert shapes['dsvt_voxel'] == (2, 128, 100, 120), shapes
    return 'same BEV shape; ' + '; '.join(info)


def module(name):
    def fn():
        m = importlib.import_module(name)
        return getattr(m, '__version__', 'imported')
    return fn


print(f'python {sys.version.split()[0]}')
check('torch + CUDA on GPU', torch_cuda)
check('spconv sparse conv on GPU', spconv_conv)
check('OpenPCDet CUDA ops', pcdet_ops)
check('torch_scatter on GPU', torch_scatter_gpu)
check('middle stages A/B/C/D', middle_stages)
for name in ('numpy', 'numba', 'scipy', 'tensorboardX'):
    check(name, module(name))

print('\n==== summary ====')
for status, name, info in results:
    print(f'{status:4}  {name:28} {info}')
sys.exit(1 if any(r[0] == 'FAIL' for r in results) else 0)
