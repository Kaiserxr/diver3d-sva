#!/usr/bin/env bash
# Dense middle-stage experiment (docs/DENSE_PLAN.md), unattended, most important first.
#
#   tmux new -s dense
#   conda activate sva && cd ~/projects/diver3d-sva && bash scripts/queue_dense.sh
#
# Check progress and results at any time:  cat ~/projects/diver3d-sva/logs/REPORT.md
#                                          logs/dense_curves.png (train loss + val AP per epoch)
# Rerun-safe: finished runs are skipped and an interrupted run continues from its last epoch.
#
# Every run trains epochs 1-12 of the usual 20-epoch schedule (PCDET_STOP_EPOCH=12: same learning-rate
# curve, it just stops; the report averages epochs 6-12 anyway) with the middle stage in bf16 (configs:
# BF16: True). Tag runG. The first dense runs (runE: fp32, 20 epochs) were too slow and are not used.
#
# Order:
#   1. sanity checks (memorise 105 frames, APBEV@0.35 >= 0.8) for the 5 base variants (passed: skipped)
#   2. seed 666 of the 5 variants: dense Run A, identity, conv, neighborhood attention, hybrid
#      (so a first comparison exists after ~1 day even if the rest is still running)
#   3. seeds 1234 and 2024 of the same 5
#   4. a fair shot for attention (seed 666): LSA for NA and hybrid, LR 5e-4 / 2e-3 for NA and hybrid
#   5. the Primus test: trained NA / hybrid models re-scored with their attention removed
# A variant that fails its sanity check is skipped (its log says why).
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
mkdir -p logs
QLOG="logs/queue_dense__$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$QLOG") 2>&1
PLAN=logs/queue_plan.tsv

say() { echo "[$(date '+%m-%d %H:%M')] $*"; }
report() { python scripts/report.py --data third_party/OpenPCDet/data/diver3d > /dev/null 2>&1 || say "(report update failed)"; }

run() {   # run <config> <tag> <seed>
    say "start  $1 $2"
    PCDET_STOP_EPOCH=12 RESUME=1 SEED="$3" KEEP_CKPTS="10 last" bash scripts/train.sh "$1" "$2" \
        --num_epochs_to_eval 20 --wo_gpu_stat
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

BASE_VARIANTS=(dense_densea dense_identity dense_conv dense_na dense_hybrid)
FAIR=(dense_na_lsa dense_hybrid_lsa dense_na_lr5e-4 dense_na_lr2e-3 dense_hybrid_lr5e-4 dense_hybrid_lr2e-3)

# the plan, for the report (config, tag, seed, epochs, stage)
{
    printf '# config\ttag\tseed\tepochs\tstage\n'
    for s in 666 1234 2024; do
        st=$([[ $s == 666 ]] && echo '2 dense seed 1' || echo '3 dense seeds 2-3')
        for c in "${BASE_VARIANTS[@]}"; do printf '%s\trunG_s%s\t%s\t12\t%s\n' "$c" "$s" "$s" "$st"; done
    done
    for c in "${FAIR[@]}"; do printf '%s\trunG_s666\t666\t12\t4 fair shot\n' "$c"; done
} > "$PLAN"

say "dense queue starting; free disk: $(df -h . | awk 'NR==2{print $4}')"
OK=()
for c in "${BASE_VARIANTS[@]}"; do
    if sanity_ok "$c"; then OK+=("$c"); fi
done
say "passed sanity: ${OK[*]:-none}"
report

for s in 666 1234 2024; do
    for c in "${OK[@]}"; do run "$c" "runG_s$s" "$s"; done
done

for c in "${FAIR[@]}"; do
    base=$(echo "$c" | sed -E 's/_(lsa|lr5e-4|lr2e-3)$//')
    if [[ " ${OK[*]} " == *" $base "* ]]; then run "$c" runG_s666 666; fi
done

say "Primus test: attention removed at test time"
bash scripts/attn_off_test.sh "10 12" | tee logs/attn_off_table.txt
say "dense queue finished"
