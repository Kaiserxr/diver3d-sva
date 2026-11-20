#!/usr/bin/env bash
# Set up APC_TDC (top Diver3D challenge entry, MIT licence) as the new baseline. Safe to rerun.
#
#   bash scripts/apc/setup_apc.sh [/path/to/diver3d_v2]      (default: ~/data/diver3d_v2)
#
# What it does:
#   1. Downloads APC_TDC's code into third_party/APC_TDC, pinned to one commit (same idea as OpenPCDet).
#   2. Creates a separate conda env "apc" (Python 3.12, torch cu128, numpy 2, scipy, matplotlib).
#      Separate so APC's numpy >= 2 cannot break the OpenPCDet env "sva".
#   3. Packs the Diver3D train/val point clouds into two .npz files under ~/data/apc_processed
#      (APC's own pack_data.py). It only READS the dataset; the original data is never modified.
set -euo pipefail

APC_URL="https://github.com/seantangth/Diver3D-MaCVi2027-APC_TDC.git"
APC_COMMIT="7d444da52eebfb1daebe5cf29d44719eb533d895"   # main, 2026-10-30
DATA="${1:-$HOME/data/diver3d_v2}"
PROC="${APC_PROC:-$HOME/data/apc_processed}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APC="$REPO/third_party/APC_TDC"

[[ -f "$DATA/annotations/gt_train.json" && -f "$DATA/tools/diver3d_eval.py" ]] \
    || { echo "not a diver3d_v2 folder (needs annotations/ and tools/): $DATA"; exit 1; }

# 1. code at the pinned commit
[[ -d "$APC/.git" ]] || git clone -q "$APC_URL" "$APC"
git -C "$APC" fetch -q origin
git -C "$APC" checkout -q "$APC_COMMIT"
echo "[1/3] APC_TDC at $(git -C "$APC" rev-parse --short HEAD)"

# 2. conda env. Uses the env's own python by path ("conda activate" inside a script can silently
#    leave the base env active), and installs whatever is missing, so a half-made env is repaired.
CONDA_BASE="$(conda info --base)"
PY="$CONDA_BASE/envs/apc/bin/python"
[[ -x "$PY" ]] || conda create -y -q -n apc python=3.12
"$PY" -c "import torch" 2>/dev/null \
    || "$PY" -m pip install -q torch --index-url https://download.pytorch.org/whl/cu128
"$PY" -c "import numpy, scipy, matplotlib; assert int(numpy.__version__.split('.')[0]) >= 2" 2>/dev/null \
    || "$PY" -m pip install -q "numpy>=2.0" "scipy>=1.13" matplotlib
"$PY" - <<'EOF'
import torch, numpy, scipy
print(f"[2/3] env apc: torch {torch.__version__}, CUDA ok: {torch.cuda.is_available()}, "
      f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}, numpy {numpy.__version__}")
x = torch.randn(2, 6, 20, 100, 120, device='cuda')
torch.nn.Conv3d(6, 16, 3, padding=1).cuda()(x).sum().backward()
print("      conv3d forward/backward on the GPU: ok")
EOF

# 3. packed data (train + val only; test has no labels)
mkdir -p "$PROC"
if [[ -f "$PROC/d3d_train.npz" && -f "$PROC/d3d_val.npz" ]]; then
    echo "[3/3] packed data already in $PROC"
else
    echo "      free disk before packing: $(df -h "$PROC" | awk 'NR==2{print $4}')"
    "$PY" "$APC/code/d3d/pack_data.py" "$DATA" "$PROC" --splits train,val
    echo "[3/3] packed data in $PROC"
fi
ls -lh "$PROC"
echo
echo "Done. Next:  tmux new -s apc   then   bash scripts/apc/queue_apc.sh"
