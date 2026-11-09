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


def module(name):
    def fn():
        m = importlib.import_module(name)
        return getattr(m, '__version__', 'imported')
    return fn


print(f'python {sys.version.split()[0]}')
check('torch + CUDA on GPU', torch_cuda)
check('spconv sparse conv on GPU', spconv_conv)
check('OpenPCDet CUDA ops', pcdet_ops)
for name in ('numpy', 'numba', 'scipy', 'tensorboardX', 'torch_scatter'):
    check(name, module(name))

print('\n==== summary ====')
for status, name, info in results:
    print(f'{status:4}  {name:28} {info}')
sys.exit(1 if any(r[0] == 'FAIL' for r in results) else 0)
