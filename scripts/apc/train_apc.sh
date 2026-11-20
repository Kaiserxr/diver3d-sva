#!/usr/bin/env bash
# Train one APC_TDC model and keep everything needed to reproduce it.
#
#   bash scripts/apc/train_apc.sh <run name> <seed> [extra train.py args]
#   e.g. bash scripts/apc/train_apc.sh single 0 --epochs 24
#
# Rerun-safe: a finished run is skipped; an interrupted one resumes from its last epoch (last.pt).
# Scores: train.py scores the raw model (no ensemble, tracking or rescoring) on the full official
# val split with the official scorer after every validated epoch -> output/apc/<name>_s<seed>/metrics.csv
#
# Outputs:
#   logs/apc__<name>_s<seed>__<time>.log   console log (git commits, command, losses, scores)
#   output/apc/<name>_s<seed>/             metrics.csv, train_log.txt, last.pt, best_sds.pt, best_ap35.pt,
#                                          predictions of the best-SDS epoch
set -euo pipefail
[[ $# -ge 2 ]] || { echo "usage: bash scripts/apc/train_apc.sh <run name> <seed> [extra train.py args]"; exit 1; }
NAME="$1"; SEED="$2"; shift 2

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APC="$REPO/third_party/APC_TDC"
DATA="${DATA:-$HOME/data/diver3d_v2}"
PROC="${APC_PROC:-$HOME/data/apc_processed}"
OUT="$REPO/output/apc/${NAME}_s${SEED}"
mkdir -p "$REPO/logs" "$OUT"

if grep -q '^=== done' "$OUT/train_log.txt" 2>/dev/null; then
    echo "already finished: $OUT"; exit 0
fi
RESUME_ARGS=()
[[ -f "$OUT/last.pt" ]] && RESUME_ARGS=(--resume "$OUT/last.pt")

LOG="$REPO/logs/apc__${NAME}_s${SEED}__$(date +%Y%m%d-%H%M%S).log"
export D3D_VOX_Z=0.25          # APC's grid: 0.1 x 0.1 x 0.25 m (120 x 100 x 20)
{
    echo "date: $(date -Iseconds)"
    echo "repo commit: $(git -C "$REPO" rev-parse HEAD)   APC commit: $(git -C "$APC" rev-parse HEAD)"
    echo "command: train.py --seed $SEED $* ${RESUME_ARGS[*]}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
} | tee "$LOG"

cd "$APC/code/d3d"
set +e
python train.py --data_dir "$PROC" --tools_dir "$DATA/tools" --out_dir "$OUT" \
    --seed "$SEED" --workers "${WORKERS:-8}" "$@" "${RESUME_ARGS[@]}" 2>&1 | tee -a "$LOG"
code=${PIPESTATUS[0]}
set -e
echo "finished $(date -Iseconds) with exit code $code" | tee -a "$LOG"
exit "$code"
