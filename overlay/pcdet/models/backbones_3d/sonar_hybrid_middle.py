"""Hybrid middle stage: the Run A sparse-conv stage, then global self-attention (experiment 1,
"Hybrid-global", in docs/HYBRID_PLAN.md).

Recipe from LitePT (Yue et al., CVPR 2026): convolution where the features are local and
high-resolution, attention where context matters. Here the sparse-conv stage of SonarSparseMiddle
runs unchanged (height 10 -> 5 -> 2), then ATTN_LAYERS transformer layers let every remaining
voxel (~1,500-2,000 per frame) attend to every other voxel in its frame. The output is the same
sparse tensor layout, so HeightCompression gives the identical 128-channel BEV map as Run A and
the BEV backbone, head and loss are shared.

Position encoding (POS_ENC):
    'cpe'   a SubM 3x3x3 sparse conv (+ linear + LayerNorm) added before each attention layer,
            as PTv3/LitePT's conditional position encoding. (spconv has no depthwise conv, so
            this is a full conv, ~110k parameters per layer at 64 channels.)
    'rope'  3D rotary encoding on voxel indices inside attention; no extra parameters.

The attention output projections start at zero, so at initialisation this network computes
exactly what Run A computes; attention is learned on top.

Config keys (NUM_FILTERS / NUM_SUBM as in SonarSparseMiddle):
    ATTN_LAYERS: 2      POS_ENC: 'rope' | 'cpe'      NHEAD: 4
    DIM_FEEDFORWARD: 256      DROPOUT: 0.0      ROPE_BASE: 100.0
"""
import torch.nn as nn

from ...utils.spconv_utils import spconv
from .sonar_sparse_middle import SonarSparseMiddle
from .voxel_global_attention import GlobalAttentionLayer, frame_layout


class SonarHybridMiddle(SonarSparseMiddle):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__(model_cfg, input_channels, grid_size, **kwargs)
        c = model_cfg
        d = self.num_point_features                    # 64: channels after down2
        self.pos_enc = c.get('POS_ENC', 'rope')
        assert self.pos_enc in ('rope', 'cpe'), self.pos_enc
        n_layers = c.get('ATTN_LAYERS', 2)
        self.attn = nn.ModuleList([
            GlobalAttentionLayer(d, c.get('NHEAD', 4), c.get('DIM_FEEDFORWARD', 256), c.get('DROPOUT', 0.0),
                                 pos_enc='rope' if self.pos_enc == 'rope' else 'none',
                                 rope_base=c.get('ROPE_BASE', 100.0))
            for _ in range(n_layers)])
        if self.pos_enc == 'cpe':
            self.cpe_conv = nn.ModuleList([
                spconv.SubMConv3d(d, d, 3, padding=1, bias=True, indice_key='hybrid_cpe') for _ in range(n_layers)])
            self.cpe_lin = nn.ModuleList([nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d)) for _ in range(n_layers)])
            for seq in self.cpe_lin:                   # LayerNorm scale 0: the CPE branch also starts at zero
                nn.init.zeros_(seq[1].weight)

    def forward(self, batch_dict):
        batch_dict = super().forward(batch_dict)
        x = batch_dict['encoded_spconv_tensor']        # (batch, z, y, x) indices, 64 features
        idx = x.indices.long()
        pos = idx[:, [3, 2, 1]]                        # voxel indices as (x, y, z)
        rows, slots, n_frames, max_len = frame_layout(idx[:, 0])
        f = x.features
        for i, layer in enumerate(self.attn):
            if self.pos_enc == 'cpe':
                f = f + self.cpe_lin[i](self.cpe_conv[i](x.replace_feature(f)).features)
            f = layer(f, pos, rows, slots, n_frames, max_len)
        batch_dict['encoded_spconv_tensor'] = x.replace_feature(f)
        return batch_dict
