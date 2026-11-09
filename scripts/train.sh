#!/usr/bin/env bash
# Train one experiment and record everything needed to reproduce it.
#
#   bash scripts/train.sh <config name> <run tag> [extra train.py args]
#   e.g. bash scripts/train.sh pointpillar baseline_s666 --num_epochs_to_eval 20
#
# Run it inside tmux (see LAB_RUNBOOK.md) so it survives a dropped AnyDesk session.
#
# Outputs:
#   logs/<config>__<tag>__<time>.log          full console log (git commit, command, losses, scores)
#   third_party/OpenPCDet/output/diver3d_models/<config>/<tag>/
#       <config>.yaml   copy of the config used
#       log_train_*.txt OpenPCDet's own log, including the fully merged config
#       ckpt/           checkpoints, one per epoch
#       tensorboard/    curves for TensorBoard
#       eval/           per-epoch scores when --num_epochs_to_eval is given
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: bash scripts/train.sh <config name> <run tag> [extra train.py args]"; exit 1
fi
CFG_NAME="$1"; TAG="$2"; shift 2

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS="$REPO/third_party/OpenPCDet/tools"
CFG="cfgs/diver3d_models/$CFG_NAME.yaml"
OUT="$REPO/third_party/OpenPCDet/output/diver3d_models/$CFG_NAME/$TAG"

[[ -f "$TOOLS/$CFG" ]] || { echo "no such config: $TOOLS/$CFG"; exit 1; }
if [[ -d "$OUT/ckpt" ]] && ls "$OUT/ckpt"/*.pth >/dev/null 2>&1; then
    echo "run '$TAG' already has checkpoints in $OUT"
    echo "pick a new tag, or delete that folder if you really want to start over"; exit 1
fi

mkdir -p "$REPO/logs"
LOG="$REPO/logs/${CFG_NAME}__${TAG}__$(date +%Y%m%d-%H%M%S).log"
CMD=(python train.py --cfg_file "$CFG" --extra_tag "$TAG" --fix_random_seed --workers 8 "$@")

{
    echo "date:        $(date -Is)"
    echo "host:        $(hostname)"
    echo "repo commit: $(git -C "$REPO" rev-parse HEAD) ($(git -C "$REPO" status --porcelain | wc -l) uncommitted changes)"
    echo "openpcdet:   $(git -C "$REPO/third_party/OpenPCDet" rev-parse --short HEAD)"
    echo "gpu:         $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
    echo "command:     (cd $TOOLS && ${CMD[*]})"
    echo "output dir:  $OUT"
    echo
} | tee "$LOG"

cd "$TOOLS"
set +e
"${CMD[@]}" 2>&1 | tee -a "$LOG"
STATUS=${PIPESTATUS[0]}
echo -e "\nfinished $(date -Is) with exit code $STATUS" | tee -a "$LOG"
exit "$STATUS"
