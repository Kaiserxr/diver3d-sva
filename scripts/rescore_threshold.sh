#!/usr/bin/env bash
# Re-score saved checkpoints with a different detection score cut-off, and compare.
#
#   bash scripts/rescore_threshold.sh                 # cut-off 0.1, epoch 10, every run
#   bash scripts/rescore_threshold.sh 0.05 "6 10 12"  # other cut-off / epochs
#
# Why: the configs keep only detections with score >= 0.3 (the SonarVoxNet paper's protocol).
# AP is computed over all detections a model outputs, so a higher cut-off can only lower recall
# and AP. This re-runs evaluation (GPU, inference only, ~1-2 min per checkpoint) with
# MODEL.DENSE_HEAD.POST_PROCESSING.SCORE_THRESH overridden on the command line; the configs are
# not changed. Same val frames as training-time scoring (every 4th frame).
#
# Output: logs/<config>__<tag>__ep<N>__thresh<T>.log per checkpoint, predictions under
# output/.../eval/epoch_<N>/val/thresh<T>/, and a table: score at 0.3 (from the training log)
# next to score at the new cut-off. Already-finished checkpoints are skipped on a rerun.
set -uo pipefail

THRESH="${1:-0.1}"
EPOCHS="${2:-10}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS="$REPO/third_party/OpenPCDet/tools"
OUT="$REPO/third_party/OpenPCDet/output/diver3d_models"
mkdir -p "$REPO/logs"

for run_dir in "$OUT"/*/run[A-Z]_s*; do
    [[ -d "$run_dir" ]] || continue
    cfg="$(basename "$(dirname "$run_dir")")"; tag="$(basename "$run_dir")"
    for e in $EPOCHS; do
        ckpt="$run_dir/ckpt/checkpoint_epoch_$e.pth"
        log="$REPO/logs/${cfg}__${tag}__ep${e}__thresh${THRESH}.log"
        if [[ ! -f "$ckpt" ]]; then echo "missing checkpoint: $ckpt"; continue; fi
        if [[ -f "$log" ]] && grep -q "AP3D@0.35" "$log"; then echo "already done: $(basename "$log")"; continue; fi
        echo "scoring $cfg/$tag epoch $e with score >= $THRESH ..."
        (cd "$TOOLS" && python test.py --cfg_file "cfgs/diver3d_models/$cfg.yaml" --extra_tag "$tag" \
            --ckpt "$ckpt" --eval_tag "thresh$THRESH" --workers 8 \
            --set MODEL.DENSE_HEAD.POST_PROCESSING.SCORE_THRESH "$THRESH") > "$log" 2>&1 \
            || echo "  FAILED, see $log"
    done
done

python3 - "$REPO/logs" "$THRESH" $EPOCHS <<'EOF'
import glob, os, re, sys
logs, thresh, epochs = sys.argv[1], sys.argv[2], [int(e) for e in sys.argv[3:]]
KEYS = ['SDS', 'AP3D@0.35', 'AP3D@0.50', 'APBEV@0.35', 'TPs']

def block(lines, epoch):
    """Metrics printed after '*** EPOCH <epoch> EVALUATION ***' (last such block)."""
    out = None
    for i, l in enumerate(lines):
        if re.search(rf'EPOCH {epoch} EVALUATION', l):
            res = {}
            for m in lines[i + 1:]:              # progress bars can sit between header and scores
                if 'EVALUATION' in m:
                    break                        # next epoch's block: this one had no scores
                p = re.sub(r'\x1b\[[0-9;]*m', '', m).split()
                if len(p) == 2 and p[0] in KEYS:
                    res[p[0]] = float(p[1])
                t = re.search(r'\((\d+) TPs / (\d+) GT', m)
                if t:
                    res['TPs'] = int(t.group(1)); res['GT'] = int(t.group(2))
                    break
            if 'AP3D@0.35' in res:
                out = res
    return out

def read(paths):
    lines = []
    for p in sorted(paths):
        lines += open(p, errors='replace').read().splitlines()
    return lines

rows = []
for f in sorted(glob.glob(os.path.join(logs, f'*__ep*__thresh{thresh}.log'))):
    cfg, tag, ep = re.match(r'(.+?)__(.+?)__ep(\d+)__thresh', os.path.basename(f)).groups()
    if int(ep) not in epochs:
        continue
    new = block(read([f]), ep)
    old = block(read(glob.glob(os.path.join(logs, f'{cfg}__{tag}__2*.log'))), ep)
    rows.append((f'{tag} ep{ep}', old, new))

print(f'\n=== score cut-off 0.3 (training-time) -> {thresh}   (val, every 4th frame)')
print(f'{"":14}' + ''.join(f'{k:>22}' for k in KEYS))
for name, old, new in rows:
    cells = []
    for k in KEYS:
        a = old.get(k) if old else None
        b = new.get(k) if new else None
        fmt = (lambda v: f'{v:.0f}') if k == 'TPs' else (lambda v: f'{v:.3f}')
        cells.append(f'{(fmt(a) if a is not None else "?"):>9} -> {(fmt(b) if b is not None else "?"):<9}')
    print(f'{name:14}' + ''.join(f'{c:>22}' for c in cells))
print('\nTPs = divers found at 3D IoU >= 0.35. "?" = not found in the logs (see the per-checkpoint log).')
EOF
