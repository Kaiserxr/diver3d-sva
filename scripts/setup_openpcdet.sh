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
#   3. Registers Diver3DDataset (dataset) and SonarSparseMiddle (3D backbone) in
#      OpenPCDet's lists, and makes the unused
#      Argoverse 2 loader optional (it would otherwise require av2 + kornia).
#      Also lets tools/train.py read its random seed from $PCDET_SEED (default 666).
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
link pcdet/models/backbones_3d/sonar_sparse_middle.py
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

# ... and register our sparse-conv middle stage among the 3D backbones
python3 - "$PCDET/pcdet/models/backbones_3d/__init__.py" <<'EOF'
import sys
p = sys.argv[1]
s = open(p, encoding='utf-8').read()
if 'SonarSparseMiddle' not in s:
    s = s.replace("from .dsvt import DSVT\n",
                  "from .dsvt import DSVT\n"
                  "from .sonar_sparse_middle import SonarSparseMiddle\n", 1)
    s = s.replace("    'DSVT': DSVT,\n",
                  "    'DSVT': DSVT,\n"
                  "    'SonarSparseMiddle': SonarSparseMiddle,\n", 1)
    if s.count('SonarSparseMiddle') != 3:
        sys.exit('could not register SonarSparseMiddle: OpenPCDet backbones_3d/__init__.py changed shape')
    open(p, 'w', encoding='utf-8').write(s)
EOF
echo "[3/5] Diver3DDataset and SonarSparseMiddle registered (Argoverse 2 import made optional)"

# 3b. let train.py take the seed from $PCDET_SEED (default 666, OpenPCDet's fixed value),
#     so the same experiment can be repeated with a second seed
python3 - "$PCDET/tools/train.py" <<'EOF'
import sys
p = sys.argv[1]
s = open(p, encoding='utf-8').read()
if 'PCDET_SEED' not in s:
    old1 = "common_utils.set_random_seed(666 + cfg.LOCAL_RANK)"
    old2 = "seed=666 if args.fix_random_seed else None"
    if s.count(old1) != 1 or s.count(old2) != 1:
        sys.exit('could not patch the seed: OpenPCDet tools/train.py changed shape')
    s = s.replace(old1, "common_utils.set_random_seed(int(os.environ.get('PCDET_SEED', 666)) + cfg.LOCAL_RANK)")
    s = s.replace(old2, "seed=int(os.environ.get('PCDET_SEED', 666)) if args.fix_random_seed else None")
    open(p, 'w', encoding='utf-8').write(s)
EOF
echo "[3b]  train.py seed now read from \$PCDET_SEED (default 666)"

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
