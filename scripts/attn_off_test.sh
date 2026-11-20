#!/usr/bin/env bash
# Primus test: does a trained dense attention model actually use its attention?
#
#   bash scripts/attn_off_test.sh            # epochs 10 and 20 of every dense_na* / dense_hybrid* run
#   bash scripts/attn_off_test.sh "10"       # other epochs (only kept checkpoints: 10 and the last)
#
# Re-evaluates each checkpoint with MODEL.BACKBONE_3D.ATTN_OFF True: the trained weights are kept but
# every attention part is removed (neighborhood-attention mixers become identity; the hybrid's global
# attention branch is skipped). If the score barely drops, the model was not relying on attention
# (Primus, Wald et al.: in some transformer models, removing the transformer changes nothing).
# The configs are not changed. Same val frames as training-time scoring (every 4th frame).
#
# Output: logs/<config>__<tag>__ep<N>__attnoff.log per checkpoint and a table: score with attention
# (from the training log) next to score without it. Finished checkpoints are skipped on a rerun.
set -uo pipefail

EPOCHS="${1:-10 20}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS="$REPO/third_party/OpenPCDet/tools"
OUT="$REPO/third_party/OpenPCDet/output/diver3d_models"
mkdir -p "$REPO/logs"

for run_dir in "$OUT"/dense_na*/run[A-Z]_s* "$OUT"/dense_hybrid*/run[A-Z]_s*; do
    [[ -d "$run_dir" ]] || continue
    cfg="$(basename "$(dirname "$run_dir")")"; tag="$(basename "$run_dir")"
    for e in $EPOCHS; do
        ckpt="$run_dir/ckpt/checkpoint_epoch_$e.pth"
        log="$REPO/logs/${cfg}__${tag}__ep${e}__attnoff.log"
        if [[ ! -f "$ckpt" ]]; then echo "missing checkpoint: $ckpt"; continue; fi
        if [[ -f "$log" ]] && grep -q "AP3D@0.35" "$log"; then echo "already done: $(basename "$log")"; continue; fi
        echo "scoring $cfg/$tag epoch $e without attention ..."
        (cd "$TOOLS" && python test.py --cfg_file "cfgs/diver3d_models/$cfg.yaml" --extra_tag "$tag" \
            --ckpt "$ckpt" --eval_tag attnoff --workers 8 \
            --set MODEL.BACKBONE_3D.ATTN_OFF True) > "$log" 2>&1 \
            || echo "  FAILED, see $log"
    done
done

python3 - "$REPO/logs" $EPOCHS <<'EOF'
import glob, os, re, sys
logs, epochs = sys.argv[1], [int(e) for e in sys.argv[2:]]
KEYS = ['SDS', 'AP3D@0.35', 'AP3D@0.50']

def block(lines, epoch):
    out = None
    for i, l in enumerate(lines):
        if re.search(rf'EPOCH {epoch} EVALUATION', l):
            res = {}
            for m in lines[i + 1:]:
                if 'EVALUATION' in m:
                    break
                p = re.sub(r'\x1b\[[0-9;]*m', '', m).split()
                if len(p) == 2 and p[0] in KEYS:
                    res[p[0]] = float(p[1])
                if re.search(r'\(\d+ TPs / \d+ GT', m):
                    break
            if 'AP3D@0.35' in res:
                out = res
    return out

def read(paths):
    lines = []
    for p in sorted(paths):
        lines += open(p, errors='replace').read().splitlines()
    return lines

print('\n=== with attention -> attention removed   (val, every 4th frame)')
print(f'{"":36}' + ''.join(f'{k:>24}' for k in KEYS))
for f in sorted(glob.glob(os.path.join(logs, '*__ep*__attnoff.log'))):
    cfg, tag, ep = re.match(r'(.+?)__(.+?)__ep(\d+)__attnoff', os.path.basename(f)).groups()
    if int(ep) not in epochs:
        continue
    new = block(read([f]), ep)
    old = block(read(glob.glob(os.path.join(logs, f'{cfg}__{tag}__2*.log'))), ep)
    cells = []
    for k in KEYS:
        a = old.get(k) if old else None
        b = new.get(k) if new else None
        d = f' ({b - a:+.3f})' if a is not None and b is not None else ''
        cells.append(f'{(f"{a:.3f}" if a is not None else "?")} -> {(f"{b:.3f}" if b is not None else "?")}{d}')
    print(f'{cfg + " " + tag + " ep" + ep:36}' + ''.join(f'{c:>24}' for c in cells))
print('\nA large drop = the model uses its attention. About zero = attention was not doing anything.')
EOF
