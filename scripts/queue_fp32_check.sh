#!/usr/bin/env bash
# Precision check for "dense beats sparse" (docs/DENSE_PLAN.md): dense Run A in fp32 without
# torch.compile, exactly like sparse Run A, 3 seeds, epochs 1-12 of the 20-epoch schedule.
#
#   tmux new -s fp32
#   conda activate sva && cd ~/projects/diver3d-sva && bash scripts/queue_fp32_check.sh
#
# Why: the same seed gave 0.329 in fp32 (runE) and 0.361 in bf16 + compile (runG) for dense Run A, so
# precision alone moves the score as much as the dense-vs-sparse gap. Seed 666 here should reproduce
# ~0.329. ~1.5 h per run. Rerun-safe like the other queues. Results appear in logs/REPORT.md as
# "Dense Run A, fp32".
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
mkdir -p logs
exec > >(tee -a "logs/queue_fp32__$(date +%Y%m%d-%H%M%S).log") 2>&1
say() { echo "[$(date '+%m-%d %H:%M')] $*"; }

say "fp32 check starting; free disk: $(df -h . | awk 'NR==2{print $4}')"
for s in 666 1234 2024; do
    say "start  dense_densea_fp32 runG_s$s"
    PCDET_STOP_EPOCH=12 RESUME=1 SEED="$s" KEEP_CKPTS="10 last" bash scripts/train.sh dense_densea_fp32 "runG_s$s" \
        --num_epochs_to_eval 20 --wo_gpu_stat
    say "end    dense_densea_fp32 runG_s$s (exit $?)"
    python scripts/report.py --data third_party/OpenPCDet/data/diver3d > /dev/null 2>&1 || say "(report update failed)"
done
say "fp32 check finished"
