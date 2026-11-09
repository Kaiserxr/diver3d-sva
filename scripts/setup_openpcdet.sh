#!/usr/bin/env bash
# Set up OpenPCDet with our Diver3D add-ons. Safe to run more than once.
#
#   bash scripts/setup_openpcdet.sh /path/to/diver3d_v2
#
# What it does, in plain terms:
#   1. Downloads OpenPCDet into third_party/OpenPCDet, pinned to one exact commit
#      so every machine runs the same library code.
#   2. Links our files (overlay/) into it. They are *links*, not copies, so an edit
#      on either side is the same file and git sees it in this repo.
#   3. Registers Diver3DDataset in OpenPCDet's dataset list, and makes the unused
#      Argoverse 2 loader optional (it would otherwise require av2 + kornia).
#   4. Replaces np.int / np.float (removed in numpy >= 1.24) with int / float.
#   5. Links the Diver3D data folder to third_party/OpenPCDet/data/diver3d.
#      The data itself is never copied or modified.
#
# It does NOT build the CUDA extensions; that is `python setup.py develop`,
# run afterwards inside the conda environment (see LAB_RUNBOOK.md).
set -euo pipefail

OPENPCDET_URL="https://github.com/open-mmlab/OpenPCDet.git"
OPENPCDET_COMMIT="233f849829b6ac19afb8af8837a0246890908755"   # master, 2025-10-08

DATA_ROOT="${1:-}"
if [[ -z "$DATA_ROOT" || ! -f "$DATA_ROOT/splits.json" ]]; then
    echo "usage: bash scripts/setup_openpcdet.sh /path/to/diver3d_v2"
    echo "       (the folder that contains splits.json, scenes/, annotations/, tools/)"
    exit 1
fi
DATA_ROOT="$(cd "$DATA_ROOT" && pwd)"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PCDET="$REPO/third_party/OpenPCDet"

# 1. OpenPCDet at the pinned commit
if [[ ! -d "$PCDET/.git" ]]; then
    git clone "$OPENPCDET_URL" "$PCDET"
fi
git -C "$PCDET" fetch --quiet origin
git -C "$PCDET" checkout --quiet "$OPENPCDET_COMMIT"
echo "[1/5] OpenPCDet at $(git -C "$PCDET" rev-parse --short HEAD)"

# 2. link overlay files into OpenPCDet
link() {   # link <path relative to overlay/ and to OpenPCDet/>
    local src="$REPO/overlay/$1" dst="$PCDET/$1"
    mkdir -p "$(dirname "$dst")"
    if [[ -e "$dst" && ! -L "$dst" ]]; then
        echo "refusing to replace a real file/folder: $dst"; exit 1
    fi
    ln -sfn "$src" "$dst"
}
link pcdet/datasets/diver3d
link tools/cfgs/dataset_configs/diver3d_dataset.yaml
link tools/cfgs/diver3d_models
echo "[2/5] overlay linked"

# 3. register the dataset class
python3 - "$PCDET/pcdet/datasets/__init__.py" <<'EOF'
import sys
p = sys.argv[1]
s = open(p, encoding='utf-8').read()
if 'Diver3DDataset' not in s:
    s = s.replace("from .custom.custom_dataset import CustomDataset\n",
                  "from .custom.custom_dataset import CustomDataset\n"
                  "from .diver3d.diver3d_dataset import Diver3DDataset\n", 1)
    s = s.replace("    'Argo2Dataset': Argo2Dataset\n",
                  "    'Argo2Dataset': Argo2Dataset,\n"
                  "    'Diver3DDataset': Diver3DDataset\n", 1)
    if 'Diver3DDataset' not in s or "'Diver3DDataset': Diver3DDataset" not in s:
        sys.exit('could not register Diver3DDataset: OpenPCDet __init__.py changed shape')
# The Argoverse 2 loader needs the av2 + kornia packages just to import pcdet.datasets.
# Diver3D never uses it, so make that one import optional instead of installing them.
argo = "from .argo2.argo2_dataset import Argo2Dataset\n"
if argo in s and 'Argo2Dataset = None' not in s:
    s = s.replace(argo, "try:\n    " + argo +
                  "except ImportError:  # needs av2 + kornia; unused for Diver3D\n"
                  "    Argo2Dataset = None\n", 1)
open(p, 'w', encoding='utf-8').write(s)
EOF
echo "[3/5] Diver3DDataset registered (Argoverse 2 import made optional)"

# 4. numpy >= 1.24 compatibility (np.int32 / np.float32 are untouched)
grep -rlE 'np\.(int|float)\b' "$PCDET/pcdet" --include='*.py' \
    | xargs -r sed -i -E 's/np\.int\b/int/g; s/np\.float\b/float/g'
echo "[4/5] np.int / np.float replaced"

# 5. data link (configs use DATA_PATH: ../data/diver3d, relative to tools/)
mkdir -p "$PCDET/data"
ln -sfn "$DATA_ROOT" "$PCDET/data/diver3d"
echo "[5/5] data: $PCDET/data/diver3d -> $DATA_ROOT"

echo
echo "Done. Next, inside the conda env:  cd $PCDET && python setup.py develop"
