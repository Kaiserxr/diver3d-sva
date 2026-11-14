#!/usr/bin/env python
"""Score saved predictions separately for each distance (or angle) from the sonar.

Every training run already saved its val predictions after each epoch. This script re-scores
them with the official Diver3D scorer (tools/diver3d_eval.py in the data folder), once for all
boxes and once per distance band. No GPU and no retraining needed.

    python scripts/eval_by_range.py                     # all runs, averaged over epochs 6-12
    python scripts/eval_by_range.py --epochs 20         # only the last epoch
    python scripts/eval_by_range.py --epochs 16-20      # average of the last 5 epochs
    python scripts/eval_by_range.py --bins 0,3,6        # other bands: 0-3 m, 3-6 m, 6+ m
    python scripts/eval_by_range.py --pred some.json    # score one prediction file
    python scripts/eval_by_range.py --by angle          # by bearing: 0-15, 15-30, 30+ deg off-centre
    python scripts/eval_by_range.py --by grid           # every distance band x every angle band

How a band is scored: distance = horizontal distance of the box centre from the sonar,
sqrt(x^2 + y^2). Ground-truth boxes are kept by their own distance, predicted boxes by theirs,
then the official scorer runs on what is left. So a detection near a band edge can fall into
the neighbouring band from its ground truth (it then counts as a miss in one band and a false
alarm in the other). That is the usual way range breakdowns are computed.
Angle = bearing of the box centre, |atan2(y, x)| in degrees: 0 = straight ahead of the sonar,
larger = towards the edge of the beam (Diver3D divers are all within about 45 deg). --by grid
crosses both, which separates "close" from "off-centre" (close divers also tend to be off-centre).

Per band it reports: number of ground-truth divers, SDS, AP3D@0.35, AP3D@0.50, APBEV@0.35,
precision and recall (3D IoU >= 0.35, all detections the model outputs, i.e. score >= 0.3), and the
error terms of the true positives: ATE (m), ASE, AOE3D (deg).
A full table is also written to logs/range_eval_<by>_epochs_<epochs>.csv.
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
METRICS = ['SDS', 'AP3D@0.35', 'AP3D@0.50', 'APBEV@0.35', 'precision', 'recall', 'ATE_m', 'ASE', 'AOE3D_deg']
MODEL_NAMES = {'centerpoint_sparseconv': 'A sparse conv', 'dsvt_voxel': 'B DSVT',
               'ptv3_voxel': 'C PTv3', 'sst_voxel': 'D SST'}

_scorer = _gt = None


def _init(data_root):
    """Runs once in each worker process: import the official scorer and load the val labels."""
    global _scorer, _gt
    sys.path.insert(0, str(Path(data_root) / 'tools'))
    import diver3d_eval
    _scorer = diver3d_eval
    _gt = diver3d_eval.load_boxes(str(Path(data_root) / 'annotations' / 'gt_val.json'),
                                  require_score=False, max_file_mb=0)


def _dist(box):
    return math.hypot(float(box.center[0]), float(box.center[1]))


def _bearing(box):
    return abs(math.degrees(math.atan2(float(box.center[1]), float(box.center[0]))))


def _intervals(edges, unit):
    return [(f'{lo:g}-{hi:g} {unit}' if hi != math.inf else f'{lo:g}+ {unit}', lo, hi)
            for lo, hi in zip(edges, edges[1:] + [math.inf])]


def make_bands(by, range_edges, angle_edges):
    """List of (name, dist_lo, dist_hi, angle_lo, angle_hi); the first one is everything."""
    inf = math.inf
    out = [('all', -inf, inf, -inf, inf)]
    if by == 'range':
        out += [(n, lo, hi, -inf, inf) for n, lo, hi in _intervals(range_edges, 'm')]
    elif by == 'angle':
        out += [(n, -inf, inf, lo, hi) for n, lo, hi in _intervals(angle_edges, 'deg')]
    else:
        out += [(f'{rn}, {an}', rlo, rhi, alo, ahi)
                for rn, rlo, rhi in _intervals(range_edges, 'm')
                for an, alo, ahi in _intervals(angle_edges, 'deg')]
    return out


def _in(box, band):
    _, rlo, rhi, alo, ahi = band
    return rlo <= _dist(box) < rhi and alo <= _bearing(box) < ahi


def score_file(job):
    """Score one prediction file overall and per band. Returns {band: {metric: value} or None}."""
    pred_path, band_list = job
    pred = _scorer.load_boxes(pred_path, require_score=True, max_file_mb=0)
    pred = {t: v for t, v in pred.items() if t in _gt}
    gt = {t: _gt[t] for t in pred}                      # only the frames that were evaluated
    out = {}
    for band in band_list:
        name = band[0]
        g = {t: [b for b in v if _in(b, band)] for t, v in gt.items()}
        p = {t: [b for b in v if _in(b, band)] for t, v in pred.items()}
        n_gt = sum(len(v) for v in g.values())
        if n_gt == 0:
            out[name] = None
            continue
        r = _scorer.evaluate(g, p, verbose=False)
        tp, n_pred = r['_meta']['num_true_positives'], r['_meta']['num_pred_boxes']
        out[name] = {'n_gt': n_gt, 'n_pred': n_pred,
                     **{k: float(r[k]) for k in METRICS if k in r},
                     'precision': tp / n_pred if n_pred else float('nan'),
                     'recall': tp / n_gt}
    return out


def parse_epochs(s):
    eps = set()
    for part in s.split(','):
        a, _, b = part.partition('-')
        eps.update(range(int(a), int(b or a) + 1))
    return sorted(eps)


def mean_of(results, band, key):
    vals = [r[band][key] for r in results if r.get(band) and not math.isnan(r[band][key])]
    return float(np.mean(vals)) if vals else float('nan')


def print_table(title, rows, band_names):
    """rows: list of (label, {band: {metric: value}})"""
    print(f'\n=== {title}')
    cols = ['n_gt'] + METRICS
    for band in band_names:
        print(f'\n  [{band}]')
        print('  ' + f'{"":20}' + ''.join(f'{c:>11}' for c in cols))
        for label, res in rows:
            if not res.get(band):
                print(f'  {label:20}' + f'{"no divers":>11}')
                continue
            cells = []
            for c in cols:
                v = res[band][c]
                cells.append(f'{int(round(v)):>11}' if c == 'n_gt' else f'{v:>11.3f}' if not math.isnan(v) else f'{"n/a":>11}')
            print(f'  {label:20}' + ''.join(cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default=str(REPO / 'third_party' / 'OpenPCDet' / 'data' / 'diver3d'),
                    help='Diver3D folder (contains annotations/ and tools/)')
    ap.add_argument('--output-root', default=str(REPO / 'third_party' / 'OpenPCDet' / 'output' / 'diver3d_models'))
    ap.add_argument('--runs', default=r'^run[A-Z]_s\d+$', help='regex for the run tags to include')
    ap.add_argument('--epochs', default='6-12', help='epochs to average, e.g. 6-12, 20, 16-20, 6,9,12')
    ap.add_argument('--by', choices=['range', 'angle', 'grid'], default='range',
                    help='split by distance, by bearing angle, or by both (grid)')
    ap.add_argument('--bins', default='0,2,4,6', help='distance band edges in metres; the last band is open-ended')
    ap.add_argument('--angle-bins', default='0,15,30', help='bearing band edges in degrees; the last is open-ended')
    ap.add_argument('--pred', help='score this one prediction file instead of the training runs')
    ap.add_argument('--workers', type=int, default=min(16, os.cpu_count() or 1))
    args = ap.parse_args()

    band_list = make_bands(args.by, [float(x) for x in args.bins.split(',')],
                           [float(x) for x in args.angle_bins.split(',')])
    band_names = [b[0] for b in band_list]
    if not (Path(args.data) / 'annotations' / 'gt_val.json').is_file():
        sys.exit(f'no annotations/gt_val.json under {args.data}; pass --data /path/to/diver3d_v2')

    if args.pred:
        _init(args.data)
        print_table(f'{args.pred}', [('file', score_file((args.pred, band_list)))], band_names)
        return

    epochs = parse_epochs(args.epochs)
    jobs, index = [], []                                  # index: (cfg, tag, epoch) per job
    for cfg_dir in sorted(Path(args.output_root).iterdir()):
        if not cfg_dir.is_dir():
            continue
        for run_dir in sorted(cfg_dir.iterdir()):
            if not re.match(args.runs, run_dir.name):
                continue
            for e in epochs:
                f = run_dir / PRED_FILE.format(e)
                if f.is_file():
                    jobs.append((str(f), band_list))
                    index.append((cfg_dir.name, run_dir.name, e))
                else:
                    print(f'  (missing: {cfg_dir.name}/{run_dir.name} epoch {e})')
    if not jobs:
        sys.exit(f'no saved predictions found under {args.output_root} for epochs {args.epochs}')
    print(f'scoring {len(jobs)} prediction files ({len(set(i[:2] for i in index))} runs, '
          f'epochs {args.epochs}) with {args.workers} processes ...')

    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.data,)) as pool:
        results = list(pool.map(score_file, jobs))

    # average over epochs per run, then over seeds per model
    per_run = {}
    for (cfg, tag, _), res in zip(index, results):
        per_run.setdefault((cfg, tag), []).append(res)
    keys = ['n_gt', 'n_pred'] + METRICS
    run_rows, model_runs = [], {}
    for (cfg, tag), rs in per_run.items():
        avg = {b: ({k: mean_of(rs, b, k) for k in keys} if any(r.get(b) for r in rs) else None) for b in band_names}
        run_rows.append((f'{tag} ({len(rs)} ep)', avg))
        model_runs.setdefault(cfg, []).append((tag, avg))
    model_rows = []
    for cfg, runs in model_runs.items():
        avgs = [a for _, a in runs]
        model_rows.append((f'{MODEL_NAMES.get(cfg, cfg)} ({len(runs)}s)',
                           {b: ({k: mean_of(avgs, b, k) for k in keys} if any(a.get(b) for a in avgs) else None)
                            for b in band_names}))

    print_table(f'per run, mean over epochs {args.epochs}', run_rows, band_names)
    print_table(f'per model, mean over seeds (epochs {args.epochs})', model_rows, band_names)

    out_csv = REPO / 'logs' / f'range_eval_{args.by}_epochs_{args.epochs.replace(",", "_")}.csv'
    out_csv.parent.mkdir(exist_ok=True)
    with open(out_csv, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['level', 'name', 'band'] + keys)
        for level, rows in (('run', run_rows), ('model', model_rows)):
            for label, res in rows:
                for b in band_names:
                    if res.get(b):
                        w.writerow([level, label, b] + [f'{res[b][k]:.4f}' for k in keys])
    print(f'\nwritten: {out_csv}')


if __name__ == '__main__':
    main()
