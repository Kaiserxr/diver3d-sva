#!/usr/bin/env python
"""Draw example frames: sonar points with ground-truth and predicted boxes, to see WHY a model fails.

Uses the val predictions a training run saved after each epoch (no GPU, no retraining). Each
frame is matched like the official scorer does (3D IoU >= 0.35, highest score first), then
frames are sorted into categories and a sample of each is drawn:

  near_false_alarm   a prediction within 2 m of the sonar that matches no diver
  offcentre_miss     a diver at 2-4 m, >= 15 deg off-centre, that no prediction matches
  good               every diver found and no false alarms (for comparison)

    python scripts/show_examples.py                                  # Run A seed 666, epoch 10
    python scripts/show_examples.py --run ptv3_voxel/runC_s1234 --epoch 6
    python scripts/show_examples.py --tokens scene_0009/000120,scene_0019/000400   # specific frames
    python scripts/show_examples.py --per-category 30

Output: logs/examples/<config>__<tag>__ep<N>/<category>/<scene>_<frame>.png and index.csv
(every frame with its counts, so you can find more examples later).

Each picture has two views, sonar at the origin:
  top:    seen from above (x forward = up, y left = left), rings every 2 m, beam lines at 15/30/45 deg
  bottom: seen from the side (x forward = right, z up = up)
Colours: green = diver found, magenta = diver missed, blue = correct detection, red = false alarm
(with its score). Grey dots = sonar points (brighter = stronger echo).
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
PRED_FILE = 'eval/eval_with_train/epoch_{}/val/final_result/data/diver3d_val_pred.json'
MATCH_IOU = 0.35

PX = 60                              # pixels per metre
X_MAX, Y_HALF, Z_HALF = 10.0, 6.0, 2.5
TOP_W, TOP_H = int(2 * Y_HALF * PX), int(X_MAX * PX)
SIDE_H = int(2 * Z_HALF * PX)
HEAD = 46
W, H = TOP_W, HEAD + TOP_H + 10 + SIDE_H

GREEN, MAGENTA, BLUE, RED = (40, 200, 60), (230, 40, 230), (60, 140, 255), (240, 50, 40)
GRID = (70, 70, 70)


def dist(c):
    return math.hypot(c[0], c[1])


def bearing(c):
    return abs(math.degrees(math.atan2(c[1], c[0])))


def hull(pts):
    """Convex hull of 2D points (monotone chain), counter-clockwise."""
    pts = sorted(map(tuple, pts))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, hi = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0:
            hi.pop()
        hi.append(p)
    return lo[:-1] + hi[:-1]


def top_px(x, y):                    # x forward -> up, y left -> left
    return (TOP_W / 2 - y * PX, HEAD + TOP_H - x * PX)


def side_px(x, z):                   # x forward -> right, z up -> up
    return ((TOP_W - X_MAX * PX) / 2 + x * PX,      # same metres-per-pixel as the vertical axis
            HEAD + TOP_H + 10 + SIDE_H / 2 - z * PX)


def match_frame(scorer, gts, preds):
    """Scorer-style greedy matching at 3D IoU 0.35. Returns (pred matched?, gt matched?)."""
    preds = sorted(preds, key=lambda b: -b.score)
    if not preds or not gts:
        return preds, [False] * len(preds), [False] * len(gts)
    m = scorer._greedy_match(scorer._iou_matrix(preds, gts, scorer.iou_3d), MATCH_IOU)
    gt_hit = [False] * len(gts)
    for j in m[m >= 0]:
        gt_hit[j] = True
    return preds, [bool(j >= 0) for j in m], gt_hit


def categorize(gts, preds, p_ok, g_ok):
    cats = []
    if any(not ok and dist(p.center) < 2 for p, ok in zip(preds, p_ok)):
        cats.append('near_false_alarm')
    if any(not ok and 2 <= dist(g.center) < 4 and bearing(g.center) >= 15 for g, ok in zip(gts, g_ok)):
        cats.append('offcentre_miss')
    if gts and all(g_ok) and all(p_ok):
        cats.append('good')
    return cats


def draw(points, gts, preds, p_ok, g_ok, title, path):
    img = Image.new('RGB', (W, H), (12, 12, 16))
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    # top view: range rings and beam lines
    for r in (2, 4, 6, 8):
        x0, y0 = top_px(r, r)
        x1, y1 = top_px(-r, -r)
        d.arc([x0, y0, x1, y1], start=180, end=360, fill=GRID)
        d.text(top_px(r, 0), f'{r} m', fill=GRID, font=font)
    for a in (15, 30, 45):
        for s in (-1, 1):
            t = math.radians(a * s)
            d.line([top_px(0, 0), top_px(X_MAX * math.cos(t), X_MAX * math.sin(t))], fill=GRID)
    # side view frame
    d.line([side_px(0, 0), side_px(X_MAX, 0)], fill=GRID)
    for r in (2, 4, 6, 8):
        d.text(side_px(r, -Z_HALF + 0.25), f'{r} m', fill=GRID, font=font)

    # points, brightness from intensity (per-frame percentiles)
    if len(points):
        inten = points[:, 3]
        lo, hi = np.percentile(inten, [5, 99]) if len(inten) > 20 else (inten.min(), inten.max())
        lev = np.clip((inten - lo) / max(hi - lo, 1e-6), 0, 1)
        for (x, y, z), v in zip(points[:, :3], lev):
            g = int(70 + 185 * v)
            px, py = top_px(x, y)
            if 0 <= px < W and HEAD <= py < HEAD + TOP_H:
                d.point((px, py), fill=(g, g, g))
            sx, sy = side_px(x, z)
            if 0 <= sx < W and HEAD + TOP_H + 10 <= sy < H:
                d.point((sx, sy), fill=(g, g, g))

    # boxes: footprint (hull of the 8 corners) in each view
    def box(b, colour, width, label=None):
        c = b.corners
        d.polygon([top_px(x, y) for x, y in hull(c[:, :2])], outline=colour, width=width)
        d.polygon([side_px(x, z) for x, z in hull(c[:, [0, 2]])], outline=colour, width=width)
        if label:
            d.text(top_px(c[:, 0].max() + 0.05, b.center[1]), label, fill=colour, font=font)
    for g, ok in zip(gts, g_ok):                       # truth thick, detections thin on top: both visible
        box(g, GREEN if ok else MAGENTA, 4)
    for p, ok in zip(preds, p_ok):
        box(p, BLUE if ok else RED, 1, f'{p.score:.2f}')

    d.text((6, 4), title, fill=(230, 230, 230), font=font)
    d.text((6, 22), 'green=diver found  magenta=diver missed  blue=correct detection  red=false alarm',
           fill=(170, 170, 170), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default=str(REPO / 'third_party' / 'OpenPCDet' / 'data' / 'diver3d'))
    ap.add_argument('--output-root', default=str(REPO / 'third_party' / 'OpenPCDet' / 'output' / 'diver3d_models'))
    ap.add_argument('--run', default='centerpoint_sparseconv/runA_s666', help='<config>/<tag>')
    ap.add_argument('--epoch', type=int, default=10)
    ap.add_argument('--pred', help='use this prediction file instead of --run/--epoch')
    ap.add_argument('--per-category', type=int, default=12)
    ap.add_argument('--tokens', help='comma-separated frames to draw instead of sampling, e.g. scene_0009/000120')
    ap.add_argument('--out', help='output folder (default logs/examples/<config>__<tag>__ep<N>)')
    args = ap.parse_args()

    data = Path(args.data)
    sys.path.insert(0, str(data / 'tools'))
    import diver3d_eval as scorer
    pred_path = Path(args.pred) if args.pred else Path(args.output_root) / args.run / PRED_FILE.format(args.epoch)
    if not pred_path.is_file():
        sys.exit(f'no predictions at {pred_path}')
    name = pred_path.stem if args.pred else f'{args.run.replace("/", "__")}__ep{args.epoch}'
    out = Path(args.out) if args.out else REPO / 'logs' / 'examples' / name

    gt_all = scorer.load_boxes(str(data / 'annotations' / 'gt_val.json'), require_score=False, max_file_mb=0)
    pred_all = scorer.load_boxes(str(pred_path), require_score=True, max_file_mb=0)
    tokens = sorted(t for t in pred_all if t in gt_all)
    print(f'{len(tokens)} frames in {pred_path}')

    rows, by_cat, matched = [], {}, {}
    for t in tokens:
        preds, p_ok, g_ok = match_frame(scorer, gt_all[t], pred_all[t])
        cats = categorize(gt_all[t], preds, p_ok, g_ok)
        matched[t] = (preds, p_ok, g_ok)
        for c in cats:
            by_cat.setdefault(c, []).append(t)
        rows.append([t, len(gt_all[t]), len(preds), sum(not ok for ok in p_ok),
                     sum(not ok for ok in g_ok), ' '.join(cats)])
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'index.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['frame', 'divers', 'detections', 'false_alarms', 'missed', 'categories'])
        w.writerows(rows)

    if args.tokens:
        todo = [('selected', t) for t in args.tokens.split(',')]
    else:
        todo = []
        for c in ('near_false_alarm', 'offcentre_miss', 'good'):
            ts = by_cat.get(c, [])
            print(f'  {c:18} {len(ts):5d} frames')
            k = min(args.per_category, len(ts))
            for i in np.linspace(0, len(ts) - 1, k).round().astype(int) if k else []:
                todo.append((c, ts[i]))           # evenly spread over scenes, same picks every time

    for cat, t in todo:
        if t not in matched:
            print(f'  skipping {t}: not in the predictions')
            continue
        sc, fr = t.split('/')
        pts = np.fromfile(data / 'scenes' / sc / 'sonar' / f'frame_{fr}.bin', dtype=np.float32).reshape(-1, 4)
        preds, p_ok, g_ok = matched[t]
        gts = gt_all[t]
        title = (f'{name}   {t}   [{cat}]   divers {len(gts)}, found {sum(g_ok)}, '
                 f'detections {len(preds)}, false alarms {sum(not ok for ok in p_ok)}')
        draw(pts, gts, preds, p_ok, g_ok, title, out / cat / f'{sc}_{fr}.png')
    print(f'drew {len(todo)} frames -> {out}')


if __name__ == '__main__':
    main()
