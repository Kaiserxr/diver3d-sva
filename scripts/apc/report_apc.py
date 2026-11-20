#!/usr/bin/env python3
"""Summarise the APC_TDC runs: writes logs/REPORT_APC.md and logs/apc_curves.png.

  python scripts/apc/report_apc.py [--window 6]

Reads output/apc/<name>_s<seed>/metrics.csv (one row per validated epoch, official scorer, full val,
raw single model). Each run is summarised by the mean over its last --window validated epochs
(the learning rate is lowest there and the scores are steadiest; one epoch alone is noisy on Diver3D),
then each model by mean +- std over seeds. Best epoch is shown for reference only.
"""
import argparse, csv, math, os, re, shutil, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'output' / 'apc'
LOGS = REPO / 'logs'
LABELS = {'smoke': 'Smoke test (48 frames)', 'single': 'APC single-frame', 'five': 'APC 5-frame'}
METRICS = ['SDS', 'AP3D@0.35', 'AP3D@0.50', 'ATE_m', 'AOE3D_deg']


def read_csv(p):
    with open(p, newline='') as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for r in rows:                         # a resumed run can repeat an epoch: keep the last row
        out[int(r['epoch'])] = {k: float(v) for k, v in r.items() if v not in ('', None)}
    return [out[e] for e in sorted(out)]


def mean_std(xs):
    xs = [x for x in xs if not math.isnan(x)]
    if not xs:
        return float('nan'), float('nan')
    m = sum(xs) / len(xs)
    s = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else float('nan')
    return m, s


def fmt(m, s=float('nan'), nd=3):
    if math.isnan(m):
        return 'n/a'
    return f'{m:.{nd}f}' if math.isnan(s) else f'{m:.{nd}f}+-{s:.{nd}f}'


def table(header, rows):
    w = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    line = lambda r: '  '.join(str(c).ljust(w[i]) if i == 0 else str(c).rjust(w[i]) for i, c in enumerate(r))
    return '\n'.join([line(header), line(['-' * x for x in w])] + [line(r) for r in rows])


def state(run_dir, n_epochs):
    log = run_dir / 'train_log.txt'
    if not log.is_file():
        return 'pending', ''
    txt = log.read_text(errors='replace')
    if '=== done' in txt:
        return 'done', ''
    m = re.findall(r'^ep (\d+) it (\d+)/(\d+)', txt, re.M)
    prog = f'epoch {int(m[-1][0]) + 1}/{n_epochs}, it {m[-1][1]}/{m[-1][2]}' if m else 'starting'
    if time.time() - log.stat().st_mtime > 20 * 60:
        return 'interrupted', prog + ' (idle >20 min; rerun the queue to resume)'
    return 'running', prog


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--window', type=int, default=6)
    args = ap.parse_args()
    LOGS.mkdir(exist_ok=True)
    parts = [f'# APC_TDC report  (updated {time.strftime("%Y-%m-%d %H:%M")})', '',
             f'Free disk: {shutil.disk_usage(REPO).free / 1e9:.0f} GB', '']

    plan = []
    pf = LOGS / 'apc_plan.tsv'
    if pf.is_file():
        for line in pf.read_text().splitlines():
            f = line.split('\t')
            if len(f) >= 4 and not line.startswith('#'):
                plan.append((f[0], f[1], int(f[2]), f[3]))
    if plan:
        rows = []
        for name, seed, ep, stage in plan:
            st, info = state(OUT / f'{name}_s{seed}', ep)
            rows.append([f'{LABELS.get(name, name)} [s{seed}]', stage, st, info])
        parts += ['## 1. Queue', '', '```', table(['run', 'stage', 'state', 'progress'], rows), '```', '']

    runs = {}
    for d in sorted(OUT.glob('*_s*')) if OUT.is_dir() else []:
        m = re.match(r'^(.+)_s(\d+)$', d.name)
        if m and (d / 'metrics.csv').is_file():
            rows = read_csv(d / 'metrics.csv')
            if rows:
                runs[(m.group(1), int(m.group(2)))] = rows

    parts += ['## 2. Results per model', '',
              f'Raw single model (no ensemble / tracking / rescoring), full official val, official scorer.',
              f'Each run = mean of its last {args.window} validated epochs; model = mean+-std over finished seeds.', '']
    models = {}
    for (name, seed), rows in runs.items():
        if name == 'smoke' or state(OUT / f'{name}_s{seed}', 0)[0] != 'done':
            continue                       # unfinished runs appear only in the per-run table
        last = rows[-args.window:]
        models.setdefault(name, []).append({k: mean_std([r.get(k, float('nan')) for r in last])[0] for k in METRICS})
    mrows = []
    for name, vals in models.items():
        mrows.append([LABELS.get(name, name), len(vals)] +
                     [fmt(*mean_std([v[k] for v in vals]), nd=2 if k == 'AOE3D_deg' else 3) for k in METRICS])
    parts += ['```', table(['model', 'seeds'] + METRICS, mrows) if mrows else '(no finished epochs yet)', '```', '']

    rrows = []
    for (name, seed), rows in sorted(runs.items()):
        last = rows[-args.window:]
        best = max(rows, key=lambda r: r.get('SDS', -1))
        rrows.append([f'{LABELS.get(name, name)} [s{seed}]', len(rows),
                      fmt(mean_std([r['SDS'] for r in last])[0]), fmt(mean_std([r['AP3D@0.35'] for r in last])[0]),
                      f"{best['SDS']:.3f} (ep {int(best['epoch'])})", f"{rows[-1]['train_loss']:.3f}"])
    parts += ['## 3. Results per run', '', '```',
              table(['run', 'epochs', f'SDS last{args.window}', f'AP3D@0.35 last{args.window}', 'best SDS', 'train loss'], rrows)
              if rrows else '(none yet)', '```', '', 'Curves: logs/apc_curves.png', '']
    (LOGS / 'REPORT_APC.md').write_text('\n'.join(parts))

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    real = {k: v for k, v in runs.items() if k[0] != 'smoke'}
    if not real:
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    colors = {'single': 'tab:blue', 'five': 'tab:orange'}
    for (name, seed), rows in sorted(real.items()):
        e = [r['epoch'] + 1 for r in rows]
        kw = dict(color=colors.get(name), alpha=0.85, lw=1.4, label=f'{LABELS.get(name, name)} s{seed}')
        ax[0].plot(e, [r['train_loss'] for r in rows], **kw)
        ax[1].plot(e, [r['AP3D@0.35'] for r in rows], **kw)
        ax[2].plot(e, [r['SDS'] for r in rows], **kw)
    for a, t in zip(ax, ['Train loss', 'Val AP3D@0.35', 'Val SDS']):
        a.set_title(t); a.set_xlabel('epoch'); a.grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(LOGS / 'apc_curves.png', dpi=130)


if __name__ == '__main__':
    main()
