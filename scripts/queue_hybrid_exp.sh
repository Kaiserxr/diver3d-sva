#!/usr/bin/env bash
# Overnight queue for experiments 0 and 1 of docs/HYBRID_PLAN.md, one run after another.
#
#   bash scripts/queue_hybrid_exp.sh            (inside tmux)
#
# Safe to rerun at any time (e.g. after a reboot): every run uses RESUME=1, so finished runs are
# skipped and an interrupted run continues from its last saved epoch. Order = priority, so if the
# night is too short the most important runs are done first.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

run() {   # run <config> <tag> <seed> <epochs>
    RESUME=1 SEED="$3" bash scripts/train.sh "$1" "$2" --num_epochs_to_eval "$4"
}

# experiment 0: tuning the sparse-conv baseline (seed 666; Run A itself is the LR 1e-3 / 20-epoch point)
run centerpoint_sparseconv_lr5e-4 runA_s666 666 20
run centerpoint_sparseconv_lr2e-3 runA_s666 666 20
run centerpoint_sparseconv_ep12   runA_s666 666 12

# experiment 1: hybrid middle stage, both position encodings, 3 seeds each
for s in 666 1234 2024; do
    run hybrid_global_rope runH_s$s $s 20
    run hybrid_global_cpe  runH_s$s $s 20
done

# third seed for the original four middle stages (A-D)
run centerpoint_sparseconv runA_s2024 2024 20
run dsvt_voxel             runB_s2024 2024 20
run ptv3_voxel             runC_s2024 2024 20
run sst_voxel              runD_s2024 2024 20

echo "queue finished $(date -Is)"
