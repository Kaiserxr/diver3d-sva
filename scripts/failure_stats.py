#!/usr/bin/env python
"""Why do detections fail? Recall by diver posture, and what the false alarms really are.

Uses the val predictions every run saved after each epoch (CPU only, no retraining). Each frame
is matched like the official scorer does: 3D IoU >= 0.35, highest score first.

    python scripts/failure_stats.py                    # all runs, epochs 6-12
    python scripts/failure_stats.py --epochs 16-20

1) Recall by posture. Posture = angle between the diver's head-to-toe axis (box axis ez) and
   vertical: upright < 30 deg, tilted 30-60, flat (swimming) > 60. For each posture:
     recall3D   found with 3D IoU >= 0.35 (what AP3D@0.35 counts)
     recallBEV  found with bird's-eye-view IoU >= 0.35 (right place seen from above, 3D box may be off)
     touched    some detection overlaps the diver at all from above (BEV IoU > 0)
     bestIoU    mean 3D IoU of the best-overlapping detection (0 if none)
   Also recall3D per posture x distance band, to separate posture from range.
   Also the posture mix of the train and val labels, for reference.

2) False alarms (detections with no 3D IoU >= 0.35 match), split into:
     on missed diver   overlaps (BEV IoU > 0) or is within 1 m of a diver that nothing matched:
                       the model saw the diver, but the box is wrong
     extra on found    overlaps or is near a diver that another detection already matched
     background        no diver nearby: the model fired on something else
   Reported for all false alarms and for those within 2 m of the sonar.

Results per run, then per model (mean over seeds); CSV in logs/failure_stats_epochs_<epochs>.csv.
"""
import argparse
import csv
import math
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PRED_FILE = 'eval/eval_with_train/epoch_{}/val/final_result/data/diver3d_val_pred.json'
MODEL_NAMES = {'centerpoint_sparseconv': 'A sparse conv', 'dsvt_voxel': 'B DSVT',
               'ptv3_voxel': 'C PTv3', 'sst_voxel': 'D SST'}
POSTURES = ['upright', 'tilted', 'flat']
BANDS = [('0-2 m', 0, 2), ('2-4 m', 2, 4), ('4-6 m', 4, 6), ('6+ m', 6, math.inf)]
FA_KINDS = ['on missed diver', 'extra on found', 'background']
MATCH_IOU, NEAR_M = 0.35, 1.0

_scorer = _gt = None


def _init(data_root):
    global _scorer, _gt
    sys.path.insert(0, str(Path(data_root) / 'tools'))
    import diver3d_eval
    _scorer = diver3d_eval
    _gt = diver3d_eval.load_boxes(str(Path(data_root) / 'annotations' / 'gt_val.json'),
                                  require_score=False, max_file_mb=0)


def posture(box):
    tilt = math.degrees(math.acos(min(1.0, abs(float(box.R[2, 2])))))   # ez vs vertical
    return 'upright' if tilt < 30 else 'tilted' if tilt < 60 else 'flat'


def band(c):
    d = math.hypot(c[0], c[1])
    return next(n for n, lo, hi in BANDS if lo <= d < hi)


def analyse_file(pred_path):
    s = _scorer
    pred = s.load_boxes(pred_path, require_score=True, max_file_mb=0)
    cnt = {'post': {p: dict(n=0, tp3d=0, tpbev=0, touched=0, iou=0.0) for p in POSTURES},
           'pr': {(p, b): dict(n=0, tp3d=0) for p in POSTURES for b, _, _ in BANDS},
           'fa': {sc: {k: 0 for k in FA_KINDS} for sc in ('all', 'within 2 m')}}
    for t, preds in pred.items():
        gts = _gt.get(t)
        if gts is None:
            continue
        preds = sorted(preds, key=lambda b: -b.score)
        if preds and gts:
            i3 = s._iou_matrix(preds, gts, s.iou_3d)
            ib = s._iou_matrix(preds, gts, s.iou_bev)
            m3, mb = s._greedy_match(i3, MATCH_IOU), s._greedy_match(ib, MATCH_IOU)
        else:
            i3 = ib = np.zeros((len(preds), len(gts)))
            m3 = mb = np.full(len(preds), -1)
        hit3 = set(int(j) for j in m3 if j >= 0)
        hitb = set(int(j) for j in mb if j >= 0)
        for j, g in enumerate(gts):
            p, b = posture(g), band(g.center)
            c = cnt['post'][p]
            c['n'] += 1
            c['tp3d'] += j in hit3
            c['tpbev'] += j in hitb
            c['touched'] += bool(len(preds) and ib[:, j].max() > 0)
            c['iou'] += float(i3[:, j].max()) if len(preds) else 0.0
            cnt['pr'][(p, b)]['n'] += 1
            cnt['pr'][(p, b)]['tp3d'] += j in hit3
        for i, pb in enumerate(preds):
            if m3[i] >= 0:
                continue
            near = [j for j, g in enumerate(gts)
                    if ib[i, j] > 0 or np.linalg.norm(pb.center - g.center) < NEAR_M]
            kind = ('background' if not near else
                    'on missed diver' if any(j not in hit3 for j in near) else 'extra on found')
            cnt['fa']['all'][kind] += 1
            if math.hypot(pb.center[0], pb.center[1]) < 2:
                cnt['fa']['within 2 m'][kind] += 1
    return cnt


def add(a, b):
    """Sum nested count dicts."""
    if isinstance(a, dict):
        return {k: add(a[k], b[k]) for k in a}
    return a + b


def summarise(c):
    """Counts -> ratios."""
    out = {}
    for p in POSTURES:
        x = c['post'][p]
        n = max(x['n'], 1)
        out[('posture', p)] = dict(n=x['n'], recall3D=x['tp3d'] / n, recallBEV=x['tpbev'] / n,
                                   touched=x['touched'] / n, bestIoU=x['iou'] / n)
    for (p, b), x in c['pr'].items():
        out[('posture x range', f'{p}, {b}')] = dict(n=x['n'], recall3D=x['tp3d'] / x['n'] if x['n'] else float('nan'))
    for sc, x in c['fa'].items():
        tot = sum(x.values())
        out[('false alarms', sc)] = dict(n=tot, **{k: x[k] / tot if tot else float('nan') for k in FA_KINDS})
    return out


def mean_dicts(ds):
    keys = ds[0].keys()
    return {k: {m: float(np.nanmean([d[k][m] for d in ds])) for m in ds[0][k]} for k in keys}


def label_mix(data_root):
    sys.path.insert(0, str(Path(data_root) / 'tools'))
    import diver3d_eval
    out = {}
    for split in ('train', 'val'):
        boxes = diver3d_eval.load_boxes(str(Path(data_root) / 'annotations' / f'gt_{split}.json'),
                                        require_score=False, max_file_mb=0)
        ps = [posture(b) for v in boxes.values() for b in v]
        out[split] = {p: ps.count(p) / len(ps) for p in POSTURES}
    return out


def print_section(title, rows, section):
    print(f'\n=== {title}: {section}')
    keys = None
    for label, res in rows:
        items = [(k[1], v) for k, v in res.items() if k[0] == section]
        if keys is None:
            keys = list(items[0][1].keys())
            print(f'  {"":20}{"":16}' + ''.join(f'{k:>16}' for k in keys))
        for name, v in items:
            cells = ''.join(f'{v[k]:>16.0f}' if k == 'n' else f'{v[k]:>16.3f}' for k in keys)
            print(f'  {label:20}{name:16}{cells}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default=str(REPO / 'third_party' / 'OpenPCDet' / 'data' / 'diver3d'))
    ap.add_argument('--output-root', default=str(REPO / 'third_party' / 'OpenPCDet' / 'output' / 'diver3d_models'))
    ap.add_argument('--runs', default=r'^run[A-Z]_s\d+$')
    ap.add_argument('--epochs', default='6-12')
    ap.add_argument('--workers', type=int, default=min(16, os.cpu_count() or 1))
    args = ap.parse_args()

    eps = set()
    for part in args.epochs.split(','):
        a, _, b = part.partition('-')
        eps.update(range(int(a), int(b or a) + 1))

    jobs, index = [], []
    for cfg_dir in sorted(p for p in Path(args.output_root).iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in cfg_dir.iterdir() if re.match(args.runs, p.name)):
            for e in sorted(eps):
                f = run_dir / PRED_FILE.format(e)
                if f.is_file():
                    jobs.append(str(f))
                    index.append((cfg_dir.name, run_dir.name))
    if not jobs:
        sys.exit(f'no saved predictions under {args.output_root} for epochs {args.epochs}')

    mix = label_mix(args.data)
    print('posture mix of the labels:  ' + '   '.join(
        f'{s}: ' + ', '.join(f'{p} {100 * mix[s][p]:.0f}%' for p in POSTURES) for s in mix))
    print(f'analysing {len(jobs)} prediction files with {args.workers} processes ...')
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.data,)) as pool:
        counts = list(pool.map(analyse_file, jobs))

    per_run = {}
    for key, c in zip(index, counts):
        per_run[key] = add(per_run[key], c) if key in per_run else c
    run_rows = [(tag, summarise(c)) for (cfg, tag), c in per_run.items()]
    by_model = {}
    for (cfg, tag), c in per_run.items():
        by_model.setdefault(cfg, []).append(summarise(c))
    model_rows = [(f'{MODEL_NAMES.get(cfg, cfg)} ({len(v)}s)', mean_dicts(v)) for cfg, v in by_model.items()]

    for section in ('posture', 'posture x range', 'false alarms'):
        print_section(f'per run, epochs {args.epochs}', run_rows, section)
    for section in ('posture', 'posture x range', 'false alarms'):
        print_section(f'per model, mean over seeds, epochs {args.epochs}', model_rows, section)

    out_csv = REPO / 'logs' / f'failure_stats_epochs_{args.epochs.replace(",", "_")}.csv'
    out_csv.parent.mkdir(exist_ok=True)
    with open(out_csv, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['level', 'name', 'section', 'group', 'metric', 'value'])
        for level, rows in (('run', run_rows), ('model', model_rows)):
            for label, res in rows:
                for (section, group), v in res.items():
                    for m, x in v.items():
                        w.writerow([level, label, section, group, m, f'{x:.4f}'])
    print(f'\nwritten: {out_csv}')


if __name__ == '__main__':
    main()
