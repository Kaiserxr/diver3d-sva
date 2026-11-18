#!/usr/bin/env bash
# Everything for the 19/11 presentation, in priority order, unattended. Replaces queue_hybrid_exp.sh.
#
#   tmux new -s queue
#   conda activate sva && cd ~/projects/diver3d-sva && bash scripts/queue_until_presentation.sh
#
# Check progress and results at any time, from any terminal:
#   cat ~/projects/diver3d-sva/logs/REPORT.md          (rewritten after every run)
#
# Rerun-safe: every run uses RESUME=1, so finished runs are skipped and an interrupted run (Ctrl+C,
# reboot) continues from its last saved epoch. Just start the same command again.
#
# Order (most important first):
#   1. experiment 0: tuning the sparse-conv baseline (3 runs)
#   2. experiment 1: hybrid RoPE + hybrid CPE, 3 seeds each
#   3. experiment 2: Control-RF (sanity check first), 3 seeds
#   4. Run A third seed
#   5. FSHNet: pulls the latest code from GitHub and, if the FSHNet port is there
#      (config fshnet_diver3d.yaml), runs its sanity check and then 3 seeds; otherwise skips
#   6. third seed for B, C, D
#   7. FSHNet again, if it was not available at step 5
# A sanity check must reach APBEV@0.35 >= 0.8 on its 105 frames, or that model's runs are skipped.
# Each finished run keeps only its epoch-10 and final checkpoints (disk); all scores are kept.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
DATA="${DATA:-$HOME/data/diver3d_v2}"
mkdir -p logs
QLOG="logs/queue__$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$QLOG") 2>&1
PLAN=logs/queue_plan.tsv

say() { echo "[$(date '+%m-%d %H:%M')] $*"; }
report() { python scripts/report.py --data third_party/OpenPCDet/data/diver3d > /dev/null 2>&1 || say "(report update failed)"; }

run() {   # run <config> <tag> <seed> <epochs>
    say "start  $1 $2"
    RESUME=1 SEED="$3" KEEP_CKPTS="10 last" bash scripts/train.sh "$1" "$2" --num_epochs_to_eval "$4"
    say "end    $1 $2 (exit $?)"
    report
}

sanity_ok() {   # sanity_ok <config>: memorise 105 frames; pass if APBEV@0.35 at the last epoch >= 0.8
    say "sanity ${1}_overfit"
    RESUME=1 KEEP_CKPTS="last" bash scripts/train.sh "${1}_overfit" sanity --num_epochs_to_eval 5 \
        --max_ckpt_save_num 5 || { say "sanity ${1}: training failed"; return 1; }
    local v
    v=$(grep -hE '^\s+APBEV@0.35' logs/"${1}"_overfit__sanity__*.log 2>/dev/null | tail -1 | awk '{print $2}')
    if awk -v v="${v:-0}" 'BEGIN{exit !(v + 0 >= 0.8)}'; then
        say "sanity ${1}: passed (APBEV@0.35 = $v)"; return 0
    fi
    say "sanity ${1}: NOT passed (APBEV@0.35 = ${v:-none}); skipping its runs"; return 1
}

FSHNET_DONE=0
fshnet_stage() {
    [[ $FSHNET_DONE == 1 ]] && return
    if [[ "${NO_PULL:-0}" != 1 ]]; then
        say "pulling latest code for FSHNet"
        git pull --ff-only -q && bash scripts/setup_openpcdet.sh "$DATA" > /dev/null \
            || say "(git pull / setup failed; using the code already here)"
    fi
    if [[ ! -f overlay/tools/cfgs/diver3d_models/fshnet_diver3d.yaml ]]; then
        say "FSHNet port not available yet; skipping for now"; return
    fi
    FSHNET_DONE=1
    if sanity_ok fshnet_diver3d; then
        for s in 666 1234 2024; do run fshnet_diver3d "runF_s$s" "$s" 20; done
    fi
}

# the plan, for the report (config, tag, seed, epochs, stage)
{
    printf '# config\ttag\tseed\tepochs\tstage\n'
    printf 'centerpoint_sparseconv_lr5e-4\trunA_s666\t666\t20\t1 tuning\n'
    printf 'centerpoint_sparseconv_lr2e-3\trunA_s666\t666\t20\t1 tuning\n'
    printf 'centerpoint_sparseconv_ep12\trunA_s666\t666\t12\t1 tuning\n'
    for s in 666 1234 2024; do
        printf 'hybrid_global_rope\trunH_s%s\t%s\t20\t2 hybrid\n' "$s" "$s"
        printf 'hybrid_global_cpe\trunH_s%s\t%s\t20\t2 hybrid\n' "$s" "$s"
    done
    for s in 666 1234 2024; do printf 'centerpoint_sparseconv_rf\trunR_s%s\t%s\t20\t3 control\n' "$s" "$s"; done
    printf 'centerpoint_sparseconv\trunA_s2024\t2024\t20\t4 seed 3\n'
    for s in 666 1234 2024; do printf 'fshnet_diver3d\trunF_s%s\t%s\t20\t5 FSHNet (if ported)\n' "$s" "$s"; done
    printf 'dsvt_voxel\trunB_s2024\t2024\t20\t6 seed 3\n'
    printf 'ptv3_voxel\trunC_s2024\t2024\t20\t6 seed 3\n'
    printf 'sst_voxel\trunD_s2024\t2024\t20\t6 seed 3\n'
} > "$PLAN"

say "queue started (log: $QLOG)"
report

# 1. tuning
run centerpoint_sparseconv_lr5e-4 runA_s666 666 20
run centerpoint_sparseconv_lr2e-3 runA_s666 666 20
run centerpoint_sparseconv_ep12   runA_s666 666 12

# 2. hybrids
for s in 666 1234 2024; do
    run hybrid_global_rope "runH_s$s" "$s" 20
    run hybrid_global_cpe  "runH_s$s" "$s" 20
done

# 3. control
if sanity_ok centerpoint_sparseconv_rf; then
    for s in 666 1234 2024; do run centerpoint_sparseconv_rf "runR_s$s" "$s" 20; done
fi

# 4. third seed for the baseline
run centerpoint_sparseconv runA_s2024 2024 20

# 5. FSHNet, if ported by now
fshnet_stage

# 6. third seed for B, C, D
run dsvt_voxel runB_s2024 2024 20
run ptv3_voxel runC_s2024 2024 20
run sst_voxel  runD_s2024 2024 20

# 7. FSHNet, second chance
fshnet_stage

report
say "queue finished"
