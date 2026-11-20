#!/usr/bin/env bash
# APC_TDC baselines, unattended, one run after another. Start once setup_apc.sh has finished:
#
#   tmux new -s apc
#   conda activate apc && cd ~/projects/diver3d-sva && bash scripts/apc/queue_apc.sh
#
# Check progress and results at any time, from any terminal:
#   cat ~/projects/diver3d-sva/logs/REPORT_APC.md        (rewritten after every run)
#   logs/apc_curves.png                                  (train loss + val AP3D per epoch, every run)
#
# Rerun-safe: finished runs are skipped, an interrupted run continues from its last epoch.
#
# Order:
#   0. smoke test: 48 frames, 2 epochs (a few minutes): checks the code runs on this PC
#   1. single-frame baseline = APC's exp002 recipe (24 epochs), 3 seeds
#   2. 5-frame baseline      = APC's exp010/011 recipe (12 epochs, frames t-2..t+2, EMA,
#                              IoU head, elevation copy-paste, z shift), 3 seeds
# Both are scored as raw single models: no ensemble, tracking, rescoring or calibration.
# Recipe changes from APC: every epoch is validated (they validated every 2nd epoch in exp002) and
# per-epoch checkpoints are not kept (disk); the per-epoch scores are all in metrics.csv.
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
mkdir -p logs
QLOG="logs/apc_queue__$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$QLOG") 2>&1
PLAN=logs/apc_plan.tsv

say() { echo "[$(date '+%m-%d %H:%M')] $*"; }
report() { python scripts/apc/report_apc.py > /dev/null 2>&1 || say "(report update failed)"; }

SINGLE=(--epochs 24 --bs 8 --val_every 1 --save_epochs 0)
FIVE=(--epochs 12 --bs 8 --val_every 1 --save_epochs 0 --ema 0.999 --paste_elev 1.0 --paste_elev_max 17
      --zshift 0.3 --iou_head 1 --frame_offsets=-2,-1,0,1,2)

run() {   # run <name> <seed> <args...>
    local name=$1 seed=$2; shift 2
    say "start  ${name}_s${seed}"
    bash scripts/apc/train_apc.sh "$name" "$seed" "$@"
    say "end    ${name}_s${seed} (exit $?)"
    report
}

# the plan, for the report (name, seed, epochs, stage)
{
    printf '# name\tseed\tepochs\tstage\n'
    printf 'smoke\t0\t2\t0 smoke test\n'
    for s in 0 1 2; do printf 'single\t%s\t24\t1 single-frame\n' "$s"; done
    for s in 0 1 2; do printf 'five\t%s\t12\t2 five-frame\n' "$s"; done
} > "$PLAN"

say "APC queue starting; free disk: $(df -h . | awk 'NR==2{print $4}')"
if ! bash scripts/apc/train_apc.sh smoke 0 --smoke; then
    say "smoke test FAILED: see the newest logs/apc__smoke_s0__*.log; stopping"; report; exit 1
fi
say "smoke test passed"; report

for s in 0 1 2; do run single "$s" "${SINGLE[@]}"; done
for s in 0 1 2; do run five "$s" "${FIVE[@]}"; done
say "APC queue finished"
