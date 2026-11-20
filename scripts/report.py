#!/usr/bin/env python
"""One-page progress + results report: logs/REPORT.md. Rewritten by the queue after every run.

    python scripts/report.py              # rewrite logs/REPORT.md and print it
    cat logs/REPORT.md                    # just read the last one

Sections:
  1. Queue: every planned run (from logs/queue_plan.tsv, written by the queue script) with its
     state (done / running + time left / interrupted / failed / pending), and free disk space.
  2. Results per model: mean +- std over finished seeds of the per-run average over epochs 6-12
     (val, every 4th frame, score >= 0.3, official scorer), overall and split by distance
     (< 2 m / >= 2 m) and by angle off-centre (< 15 / >= 15 deg); the difference to Run A and
     whether it is larger than the seed spread.
  3. Results per run (each seed), for checking.

Scores are cached in logs/report_cache.json (keyed by file + modification time), so a refresh
only scores prediction files it has not seen. Uses the same scoring code as eval_by_range.py.
"""
import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
import eval_by_range as ebr                                 # noqa: E402  (shared, tested scoring code)

OUT_ROOT = REPO / 'third_party' / 'OpenPCDet' / 'output' / 'diver3d_models'
LOGS = REPO / 'logs'
INF = math.inf
BANDS = [('all', -INF, INF, -INF, INF), ('<2 m', 0, 2, -INF, INF), ('>=2 m', 2, INF, -INF, INF),
         ('centre<15', -INF, INF, 0, 15), ('off>=15', -INF, INF, 15, INF)]
LABELS = {
    'centerpoint_sparseconv': 'A sparse conv (baseline)',
    'centerpoint_sparseconv_lr5e-4': 'A, LR 5e-4 (exp 0)',
    'centerpoint_sparseconv_lr2e-3': 'A, LR 2e-3 (exp 0)',
    'centerpoint_sparseconv_ep12': 'A, 12 epochs (exp 0)',
    'centerpoint_sparseconv_rf': 'Control-RF (exp 2)',
    'hybrid_global_rope': 'Hybrid RoPE (exp 1)',
    'hybrid_global_cpe': 'Hybrid CPE (exp 1)',
    'dsvt_voxel': 'B DSVT', 'ptv3_voxel': 'C PTv3', 'sst_voxel': 'D SST',
    'fshnet_diver3d': 'FSHNet (CVPR 2025)',
    'dense_densea': 'Dense Run A (dense conv 3^3)',
    'dense_identity': 'Dense: identity (floor)',
    'dense_conv': 'Dense: conv mixer',
    'dense_na': 'Dense: neighborhood attention',
    'dense_hybrid': 'Dense: conv + global attention',
    'dense_na_lsa': 'Dense: NA + LSA',
    'dense_hybrid_lsa': 'Dense: hybrid + LSA',
    'dense_na_lr5e-4': 'Dense: NA, LR 5e-4', 'dense_na_lr2e-3': 'Dense: NA, LR 2e-3',
    'dense_hybrid_lr5e-4': 'Dense: hybrid, LR 5e-4', 'dense_hybrid_lr2e-3': 'Dense: hybrid, LR 2e-3',
}
BASE = 'centerpoint_sparseconv'


def parse_epochs(s):
    eps = set()
    for part in s.split(','):
        a, _, b = part.partition('-')
        eps.update(range(int(a), int(b or a) + 1))
    return sorted(eps)


def run_logs(cfg, tag):
    return sorted(LOGS.glob(f'{cfg}__{tag}__2*.log'))


def wall_hours(logs):
    """Wall-clock hours of a run (training + scoring), summed over resumed sessions."""
    total, found = 0.0, False
    for f in logs:
        txt = f.read_text(errors='replace')
        a = re.search(r'^date:\s+(\S+)', txt, re.M)
        b = re.search(r'^finished (\S+) with exit code', txt, re.M)
        if a and b:
            try:
                total += (datetime.fromisoformat(b.group(1)) - datetime.fromisoformat(a.group(1))).total_seconds()
                found = True
            except ValueError:
                pass
    return total / 3600 if found else float('nan')


def run_state(cfg, tag, n_epochs):
    out = OUT_ROOT / cfg / tag
    if (out / 'FINISHED').is_file() or list(out.glob(f'eval/eval_with_train/epoch_{n_epochs}/*/final_result/data/*.json')):
        return 'done', ''
    logs = run_logs(cfg, tag)
    if not logs:
        return 'pending', ''
    last = logs[-1]
    txt = last.read_text(errors='replace')
    m = re.search(r'finished \S+ with exit code (\d+)', txt)
    if m and m.group(1) != '0':
        return 'FAILED', f'see {last.name}'
    progress = ''
    trains = re.findall(r'Train:\s+(\d+)/(\d+).*?\[(\d[\d:]*)/(\d[\d:]*)\]', txt)
    if trains:
        ep, tot, el, rem = trains[-1]
        progress = f'epoch {ep}/{tot}, training time left {rem}'
    elif 'EVALUATION' in txt:
        progress = 'scoring epochs'
    if time.time() - last.stat().st_mtime > 15 * 60:
        return 'interrupted', progress + ' (log idle >15 min; rerun the queue to resume)'
    return 'running', progress


def score_jobs(jobs, data, workers):
    cache_file = LOGS / 'report_cache.json'
    try:
        cache = json.loads(cache_file.read_text())
    except (OSError, ValueError):
        cache = {}
    sig = json.dumps(BANDS)
    keys = [f'{p}|{os.path.getmtime(p):.0f}|{sig}' for p in jobs]
    todo = [(p, k) for p, k in zip(jobs, keys) if k not in cache]
    if todo:
        with ProcessPoolExecutor(workers, initializer=ebr._init, initargs=(data,)) as pool:
            for (p, k), res in zip(todo, pool.map(ebr.score_file, [(p, BANDS) for p, _ in todo])):
                cache[k] = res
        cache_file.write_text(json.dumps(cache))
    return [cache[k] for k in keys]


def fmt(v, std=None):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return 'n/a'
    if std is None or math.isnan(std):
        return f'{v:.3f}'
    return f'{v:.3f}+-{std:.3f}'


def table(header, rows):
    w = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    line = lambda r: '  '.join(str(c).ljust(w[i]) if i == 0 else str(c).rjust(w[i]) for i, c in enumerate(r))
    return '\n'.join([line(header), line(['-' * x for x in w])] + [line(r) for r in rows])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default=str(REPO / 'third_party' / 'OpenPCDet' / 'data' / 'diver3d'))
    ap.add_argument('--epochs', default='6-12')
    ap.add_argument('--out', default=str(LOGS / 'REPORT.md'))
    ap.add_argument('--workers', type=int, default=min(16, os.cpu_count() or 1))
    args = ap.parse_args()
    LOGS.mkdir(exist_ok=True)
    epochs = parse_epochs(args.epochs)
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    parts = [f'# Experiment report  (updated {now})', '']

    # ---- 1. queue
    plan_file = LOGS / 'queue_plan.tsv'
    plan = []
    if plan_file.is_file():
        for line in plan_file.read_text().splitlines():
            f = line.split('\t')
            if len(f) >= 4 and not line.startswith('#'):
                plan.append((f[0], f[1], f[2], int(f[3]), f[4] if len(f) > 4 else ''))
    disk = shutil.disk_usage(str(REPO))
    parts += ['## 1. Queue', '', f'Free disk: {disk.free / 1e9:.0f} GB', '']
    if plan:
        rows, counts = [], {}
        for cfg, tag, seed, n_ep, stage in plan:
            state, info = run_state(cfg, tag, n_ep)
            counts[state] = counts.get(state, 0) + 1
            rows.append([f'{LABELS.get(cfg, cfg)} [{tag}]', stage, state, info])
        parts += [', '.join(f'{v} {k}' for k, v in counts.items()), '', '```',
                  table(['run', 'stage', 'state', 'progress'], rows), '```', '']
    else:
        parts += ['(no logs/queue_plan.tsv yet: start scripts/queue_until_presentation.sh)', '']
    qlogs = sorted(LOGS.glob('queue__*.log'))
    if qlogs:
        tail = qlogs[-1].read_text(errors='replace').splitlines()[-6:]
        parts += [f'Last lines of {qlogs[-1].name}:', '```', *tail, '```', '']

    # ---- 2./3. results
    runs = []                                               # (cfg, tag) of finished runs
    if OUT_ROOT.is_dir():
        for cfg_dir in sorted(p for p in OUT_ROOT.iterdir() if p.is_dir() and not p.name.endswith('_overfit')):
            for run_dir in sorted(p for p in cfg_dir.iterdir() if re.match(r'^run[A-Z]_s\d+$', p.name)):
                if run_dir.joinpath('FINISHED').is_file() or list(run_dir.glob('eval/eval_with_train/epoch_*/*/final_result/data/*.json')):
                    runs.append((cfg_dir.name, run_dir.name))
    jobs, index = [], []
    for cfg, tag in runs:
        for e in epochs:
            f = OUT_ROOT / cfg / tag / ebr.PRED_FILE.format(e)
            if f.is_file():
                jobs.append(str(f)); index.append((cfg, tag))
    per_run = {}
    if jobs:
        for key, res in zip(index, score_jobs(jobs, args.data, args.workers)):
            per_run.setdefault(key, []).append(res)

    def avg(results, band, metric):
        v = [r[band][metric] for r in results if r.get(band) and not math.isnan(r[band][metric])]
        return float(np.mean(v)) if v else float('nan')

    run_vals = {}
    for (cfg, tag), rs in per_run.items():
        run_vals[(cfg, tag)] = {
            'AP35': avg(rs, 'all', 'AP3D@0.35'), 'AP50': avg(rs, 'all', 'AP3D@0.50'), 'SDS': avg(rs, 'all', 'SDS'),
            'recall': avg(rs, 'all', 'recall'), 'precision': avg(rs, 'all', 'precision'),
            'near': avg(rs, '<2 m', 'AP3D@0.35'), 'far': avg(rs, '>=2 m', 'AP3D@0.35'),
            'centre': avg(rs, 'centre<15', 'AP3D@0.35'), 'off': avg(rs, 'off>=15', 'AP3D@0.35'),
            'n_ep': len(rs), 'hours': wall_hours(run_logs(cfg, tag)),
        }
    models = {}
    for (cfg, tag), v in run_vals.items():
        models.setdefault(cfg, []).append(v)

    def ms(vals, k):
        x = np.array([v[k] for v in vals if not math.isnan(v[k])])
        if not len(x):
            return float('nan'), float('nan')
        return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else float('nan')

    parts += ['## 2. Results per model', '',
              f'AP3D@0.35 unless named; each run averaged over epochs {args.epochs}, then mean+-std over seeds.',
              'near/far = diver < 2 m / >= 2 m from the sonar; centre/off = < 15 / >= 15 deg off the beam axis.',
              f'vs A = difference in AP3D@0.35 to "{LABELS[BASE]}"; "*" = larger than both models\' seed std.', '']
    order = sorted(models, key=lambda c: (c != BASE, LABELS.get(c, c)))
    base_m, base_s = ms(models.get(BASE, []), 'AP35') if BASE in models else (float('nan'), float('nan'))
    rows = []
    for cfg in order:
        vals = models[cfg]
        m35, s35 = ms(vals, 'AP35')
        diff = ''
        if cfg != BASE and not math.isnan(base_m):
            d = m35 - base_m
            big = (not math.isnan(s35) and not math.isnan(base_s) and abs(d) > max(s35, base_s))
            diff = f'{d:+.3f}' + ('*' if big else '')
        rows.append([LABELS.get(cfg, cfg), len(vals), fmt(m35, s35), diff, fmt(*ms(vals, 'AP50')), fmt(*ms(vals, 'SDS')),
                     fmt(ms(vals, 'near')[0]), fmt(ms(vals, 'far')[0]), fmt(ms(vals, 'centre')[0]), fmt(ms(vals, 'off')[0]),
                     fmt(ms(vals, 'recall')[0]), fmt(ms(vals, 'precision')[0]), f'{ms(vals, "hours")[0]:.1f}'])
    parts += ['```', table(['model', 'seeds', 'AP3D@0.35', 'vs A', 'AP3D@0.50', 'SDS', 'near', 'far', 'centre', 'off',
                            'recall', 'precision', 'h/run'], rows) if rows else '(no finished runs yet)', '```', '']

    parts += ['## 3. Results per run', '']
    rows = [[f'{LABELS.get(c, c)} [{t}]', v['n_ep'], fmt(v['AP35']), fmt(v['AP50']), fmt(v['SDS']), fmt(v['near']),
             fmt(v['off']), f'{v["hours"]:.1f}'] for (c, t), v in sorted(run_vals.items(), key=lambda x: (LABELS.get(x[0][0], x[0][0]), x[0][1]))]
    parts += ['```', table(['run', 'epochs', 'AP3D@0.35', 'AP3D@0.50', 'SDS', 'near', 'off', 'hours'], rows)
              if rows else '(none)', '```', '']

    if plot_curves(runs):
        parts += ['Loss / val curves of the dense runs (Run A for reference): logs/dense_curves.png', '']
    text = '\n'.join(parts)
    Path(args.out).write_text(text, encoding='utf-8')
    print(text)


def log_curves(cfg, tag):
    """Per-epoch train loss (OpenPCDet's running epoch mean at the epoch's last log line) and the
    per-epoch val AP3D@0.35 printed after each epoch's evaluation, read from the run's console logs."""
    loss, ap = {}, []
    for f in run_logs(cfg, tag):
        txt = f.read_text(errors='replace')
        for ep, avg in re.findall(r'Train:\s+(\d+)/\d+.*?Loss: [\d.e+-]+ \(([\d.e+-]+)\)', txt):
            loss[int(ep)] = float(avg)
        ap += [float(v) for v in re.findall(r'^\s+AP3D@0\.35\s+([\d.]+)', txt, re.M)]
    return loss, ap


def plot_curves(runs):
    """logs/dense_curves.png: train loss and val AP3D@0.35 per epoch for every dense run + Run A."""
    sel = [(c, t) for c, t in runs if c.startswith('dense_') or c == BASE]
    if not any(c.startswith('dense_') for c, _ in sel):
        return False
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    cmap = plt.get_cmap('tab10')
    cfgs = sorted({c for c, _ in sel})
    for c, t in sorted(sel):
        loss, ap = log_curves(c, t)
        kw = dict(color=cmap(cfgs.index(c) % 10), lw=1.3, alpha=0.85)
        if loss:
            ax[0].plot(sorted(loss), [loss[e] for e in sorted(loss)], label=f'{LABELS.get(c, c)} [{t}]', **kw)
        if ap:
            ax[1].plot(range(1, len(ap) + 1), ap, **kw)
    for a, title in zip(ax, ['Train loss (epoch mean)', 'Val AP3D@0.35']):
        a.set_title(title); a.set_xlabel('epoch'); a.grid(alpha=0.3)
    ax[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(LOGS / 'dense_curves.png', dpi=130)
    plt.close(fig)
    return True


if __name__ == '__main__':
    main()
