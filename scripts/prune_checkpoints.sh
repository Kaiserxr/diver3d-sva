#!/usr/bin/env bash
# Delete checkpoints that are no longer needed, keeping only a few per run. Saves disk space.
#
#   bash scripts/prune_checkpoints.sh                    # DRY RUN: show what would be deleted, all runs
#   bash scripts/prune_checkpoints.sh --yes              # really delete
#   bash scripts/prune_checkpoints.sh --keep "10 last" --sanity-keep "last" [--yes] [RUN_DIR ...]
#
# What is kept:
#   * every run's per-epoch scores and predictions (eval/), logs, config copy and TensorBoard
#     curves: all our analysis scripts (eval_by_range.py, failure_stats.py, ...) read these
#   * checkpoints of the epochs in --keep (default "10 last": epoch 10 for re-scoring with
#     rescore_threshold.sh, and the final model). "last" = the highest saved epoch.
#   * sanity runs (*_overfit configs) keep --sanity-keep (default "last")
# What is deleted: all other checkpoint_epoch_*.pth and latest_model.pth files of the run.
#
# Safety: a run is only touched if its last saved epoch has been scored (its eval predictions
# exist) or it has a FINISHED marker. Unfinished or crashed runs are skipped, so they can still be
# resumed with RESUME=1. Without --yes nothing is deleted.
# A deleted checkpoint can be recreated by rerunning that training (fixed seeds).
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$REPO/third_party/OpenPCDet/output/diver3d_models"
KEEP="10 last"; SANITY_KEEP="last"; YES=0; RUNS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --yes) YES=1 ;;
        --keep) KEEP="$2"; shift ;;
        --sanity-keep) SANITY_KEEP="$2"; shift ;;
        --root) ROOT="$2"; shift ;;
        *) RUNS+=("$1") ;;
    esac
    shift
done
if [[ ${#RUNS[@]} -eq 0 ]]; then
    for d in "$ROOT"/*/*/; do [[ -d "$d/ckpt" ]] && RUNS+=("${d%/}"); done
fi

total=0
for run in "${RUNS[@]}"; do
    name="$(basename "$(dirname "$run")")/$(basename "$run")"
    shopt -s nullglob
    ckpts=("$run"/ckpt/*.pth)
    shopt -u nullglob
    [[ ${#ckpts[@]} -eq 0 ]] && continue
    last=$(ls "$run"/ckpt/checkpoint_epoch_*.pth 2>/dev/null | sed -E 's/.*checkpoint_epoch_([0-9]+)\.pth/\1/' | sort -n | tail -1)
    if [[ -z "$last" ]]; then echo "skip   $name: no epoch checkpoints"; continue; fi
    if [[ ! -f "$run/FINISHED" ]] && ! ls "$run"/eval/eval_with_train/epoch_"$last"/*/final_result/data/*.json >/dev/null 2>&1; then
        echo "skip   $name: last epoch ($last) not scored yet (unfinished or crashed run)"; continue
    fi
    keep="$KEEP"; [[ "$name" == *_overfit/* ]] && keep="$SANITY_KEEP"
    keep_files=()
    for k in $keep; do
        [[ "$k" == last ]] && k="$last"
        keep_files+=("$run/ckpt/checkpoint_epoch_$k.pth")
    done
    del=()
    for f in "${ckpts[@]}"; do
        skip=0
        for kf in "${keep_files[@]}"; do [[ "$f" == "$kf" ]] && skip=1; done
        [[ $skip == 0 ]] && del+=("$f")
    done
    [[ ${#del[@]} -eq 0 ]] && { echo "ok     $name: nothing to delete"; continue; }
    bytes=$(du -cb "${del[@]}" | tail -1 | cut -f1)
    total=$((total + bytes))
    kept=$(for kf in "${keep_files[@]}"; do [[ -f "$kf" ]] && basename "$kf" .pth | sed 's/checkpoint_//'; done | tr '\n' ' ')
    if [[ $YES == 1 ]]; then
        rm -f -- "${del[@]}"
        printf 'pruned %s: deleted %d files (%.2f GB), kept %s\n' "$name" ${#del[@]} "$(awk "BEGIN{print $bytes/1073741824}")" "$kept"
    else
        printf 'would  %s: delete %d files (%.2f GB), keep %s\n' "$name" ${#del[@]} "$(awk "BEGIN{print $bytes/1073741824}")" "$kept"
    fi
done
if [[ $YES == 1 ]]; then
    printf 'freed %.2f GB\n' "$(awk "BEGIN{print $total/1073741824}")"
else
    printf 'would free %.2f GB  (dry run: add --yes to delete)\n' "$(awk "BEGIN{print $total/1073741824}")"
fi
