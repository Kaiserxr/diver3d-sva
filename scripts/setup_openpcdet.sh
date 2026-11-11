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
#   3. Registers our classes in OpenPCDet's lists: Diver3DDataset (dataset);
#      SonarSparseMiddle, PTv3Backbone, SSTBackbone (3D middle stages);
#      VoxelColumnMaxScatter (voxels -> BEV). Also makes the unused
#      Argoverse 2 loader optional (it would otherwise require av2 + kornia).
#      Also lets tools/train.py read its random seed from $PCDET_SEED (default 666).
#   4. Replaces np.int / np.float (removed in numpy >= 1.24) with int / float.
#   5. Links the Diver3D data folder to third_party/OpenPCDet/data/diver3d.
#      The data itself is never copied or modified.
#
# It does NOT build the CUDA extensions; that is `pip install --no-build-isolation -e .`,
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
link pcdet/models/backbones_3d/ptv3_backbone.py
link pcdet/models/backbones_3d/ptv3
link pcdet/models/backbones_3d/sst_backbone.py
link pcdet/models/backbones_2d/map_to_bev/voxel_column_max.py
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

# ... and register our model modules in OpenPCDet's lookup tables (an `__all__` dict per folder)
register() {   # register <__init__.py relative to OpenPCDet/pcdet/models> <module>:<Class> ...
    python3 - "$PCDET/pcdet/models/$1" "${@:2}" <<'EOF'
import sys
p, entries = sys.argv[1], [e.split(':') for e in sys.argv[2:]]
s = open(p, encoding='utf-8').read()
for module, cls in entries:
    if f"'{cls}': {cls}" in s:
        continue                                   # already registered (rerun)
    head = s.index('\n__all__ = {')
    s = s[:head] + f'\nfrom .{module} import {cls}' + s[head:]
    close = s.index('\n}', s.index('\n__all__ = {'))
    s = s[:close] + f"\n    '{cls}': {cls}," + s[close:]
    if s.count(f"'{cls}': {cls}") != 1:
        sys.exit(f'could not register {cls} in {p}: the file changed shape')
open(p, 'w', encoding='utf-8').write(s)
EOF
}
register backbones_3d/__init__.py \
    sonar_sparse_middle:SonarSparseMiddle ptv3_backbone:PTv3Backbone sst_backbone:SSTBackbone
register backbones_2d/map_to_bev/__init__.py voxel_column_max:VoxelColumnMaxScatter
echo "[3/5] registered: Diver3DDataset, SonarSparseMiddle, PTv3Backbone, SSTBackbone, VoxelColumnMaxScatter"

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

# 3c. DSVT's attention pooling builds an all-zero *int* padding mask; PyTorch >= 2.x only
#     accepts bool/float masks. Older PyTorch silently converted it to bool, so .bool() is the
#     same behaviour (all False = "nothing is padding").
DSVT_PY="$PCDET/pcdet/models/backbones_3d/dsvt.py"
OLD_MASK='key_padding_mask = torch.zeros((pooled_voxel_num, pool_volume)).to(prepool_features.device).int()'
if grep -qF "$OLD_MASK" "$DSVT_PY"; then
    sed -i 's/(pooled_voxel_num, pool_volume)).to(prepool_features.device).int()/(pooled_voxel_num, pool_volume)).to(prepool_features.device).bool()/' "$DSVT_PY"
fi
grep -qF '(pooled_voxel_num, pool_volume)).to(prepool_features.device).bool()' "$DSVT_PY" \
    || { echo "could not patch the DSVT pooling mask: dsvt.py changed shape"; exit 1; }
echo "[3c]  DSVT pooling mask is now bool (PyTorch 2.x)"

# 3d. PyTorch >= 2.6 loads checkpoints with weights_only=True by default, which rejects the
#     numpy values OpenPCDet stores in them, so evaluation and resuming crash. These are
#     checkpoints our own runs write, so load them the pre-2.6 way (weights_only=False).
TEMPLATE_PY="$PCDET/pcdet/models/detectors/detector3d_template.py"
sed -i 's/map_location=loc_type)/map_location=loc_type, weights_only=False)/' "$TEMPLATE_PY"
[[ $(grep -c 'map_location=loc_type, weights_only=False)' "$TEMPLATE_PY") -eq 4 && \
   $(grep -c 'map_location=loc_type)' "$TEMPLATE_PY") -eq 0 ]] \
    || { echo "could not patch checkpoint loading: detector3d_template.py changed shape"; exit 1; }
echo "[3d]  checkpoints load with weights_only=False (PyTorch 2.6+)"

# 4. numpy >= 1.24 compatibility (np.int32 / np.float32 are untouched)
grep -rlE 'np\.(int|float)\b' "$PCDET/pcdet" --include='*.py' \
    | xargs -r sed -i -E 's/np\.int\b/int/g; s/np\.float\b/float/g'
echo "[4/5] np.int / np.float replaced"

# 5. data link (configs use DATA_PATH: ../data/diver3d, relative to tools/)
mkdir -p "$PCDET/data"
ln -sfn "$DATA_ROOT" "$PCDET/data/diver3d"
echo "[5/5] data: $PCDET/data/diver3d -> $DATA_ROOT"

echo
echo "Done. Next, inside the conda env:  cd $PCDET && pip install --no-build-isolation -e ."
