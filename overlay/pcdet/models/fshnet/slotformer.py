"""SlotFormer block of FSHNet (Liu et al., CVPR 2025), plain-PyTorch version.

Source: FSHNet pcdet/models/model_utils/slotformer_3d_utils.py (https://github.com/Say2L/FSHNet,
commit 8e8f23e, Apache-2.0). What it does: voxels are grouped into "slots", long strips that
span the whole scene along one axis (all voxels whose y index, then x index, falls in the same
band of WIN_SIZE cells; the second layer shifts the bands by half). Inside each slot it runs
linear attention (ReLU feature map): kv = sum_i k_i^T v_i, s = sum_i k_i,
y_i = q_i kv / (q_i . s). Each attention is followed by a strip-convolution FFN (CWI_FFN).

Changes from the original:
  * the per-slot sums ran in custom Triton kernels (scatter_matmul_kv / scatter_matmul_qc); here
    they are computed with index_add / einsum in plain PyTorch. Same maths (tested against a
    per-slot loop); no Triton or einops needed. Diver3D frames are small, so this is cheap.
  * each SlotFormer instance gets its own spconv indice_key prefix (the original reused the same
    keys in stages with different voxel layouts)
  * torch.cuda.amp.autocast(float32) wrapper removed (we train in float32); checkpointing kept,
    with use_reentrant=False as PyTorch 2.x requires.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ...utils.spconv_utils import replace_feature, spconv


def slot_linear_attention(q, k, v, win_inds, n_win, eps=1e-3):
    """q, k, v: (N, heads, c), already ReLU'd q/k; win_inds: (N,) slot of each voxel."""
    kv = q.new_zeros(n_win, q.shape[1], q.shape[2], v.shape[2])
    kv.index_add_(0, win_inds, torch.einsum('nhc,nhd->nhcd', k, v))     # sum_i k_i^T v_i per slot
    s = q.new_zeros(n_win, q.shape[1], q.shape[2])
    s.index_add_(0, win_inds, k)                                       # sum_i k_i per slot
    y = torch.einsum('nhc,nhcd->nhd', q, kv[win_inds])
    z = (s[win_inds] * q).sum(-1, keepdim=True)
    return y / (z + eps)


class SelfAttnLayer(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False):
        super().__init__()
        self.dim, self.num_heads, self.head_dim = dim, num_heads, dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, win_inds, n_win, eps=1e-3):
        N, C = x.shape
        q, k, v = self.qkv(x).reshape(N, 3, self.num_heads, self.head_dim).unbind(1)
        y = slot_linear_attention(F.relu(q), F.relu(k), v, win_inds, n_win, eps)
        return self.proj(y.reshape(N, C))


class CWI_FFN_Layer_3D(spconv.SparseModule):
    """Feature quarters: 3x3x3 conv | strip conv along x | strip conv along y | identity; then MLP."""

    def __init__(self, embed_dim, indice_key='former', conv_size=13):
        super().__init__()
        g = embed_dim // 4
        self.group_dim = g
        self.bn = nn.BatchNorm1d(embed_dim, eps=1e-3, momentum=0.01)
        self.conv_k = spconv.SubMConv3d(g, g, kernel_size=3, stride=1, padding=1, indice_key=indice_key + 'k')
        self.conv_h = spconv.SubMConv3d(g, g, kernel_size=(1, 1, conv_size), stride=1,
                                        padding=(0, 0, conv_size // 2), indice_key=indice_key + 'h')
        self.conv_w = spconv.SubMConv3d(g, g, kernel_size=(1, conv_size, 1), stride=1,
                                        padding=(0, conv_size // 2, 0), indice_key=indice_key + 'w')
        self.bn2 = nn.BatchNorm1d(embed_dim, eps=1e-3, momentum=0.01)
        self.fc1 = nn.Linear(embed_dim, embed_dim * 2)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, src):
        g = self.group_dim
        src = replace_feature(src, self.bn(src.features))
        f = src.features
        src_k = self.conv_k(replace_feature(src, f[:, :g])).features
        src_h = self.conv_h(replace_feature(src, f[:, g:2 * g])).features
        src_w = self.conv_w(replace_feature(src, f[:, 2 * g:3 * g])).features
        src = replace_feature(src, f + torch.cat([src_k, src_h, src_w, f[:, 3 * g:]], 1))
        return replace_feature(src, src.features + self.fc2(self.act(self.fc1(self.bn2(src.features)))))


class SFLayer(nn.Module):
    def __init__(self, embed_dim, nhead, directions=2, win_size=12, shift=False, indice_key='former'):
        super().__init__()
        self.directions, self.win_size, self.shift = directions, win_size, shift
        self.self_norm_layers = nn.ModuleList([nn.BatchNorm1d(embed_dim, eps=1e-3, momentum=0.01) for _ in range(directions)])
        self.self_attn_layers = nn.ModuleList([SelfAttnLayer(embed_dim, nhead) for _ in range(directions)])
        self.ffn_layers = nn.ModuleList([CWI_FFN_Layer_3D(embed_dim, indice_key=indice_key, conv_size=win_size + 1)
                                         for _ in range(directions)])

    def slots(self, indices, direction):
        """Slot id per voxel: (batch, band of the y [direction 0] or x [direction 1] index)."""
        coord = indices[:, direction + 2:direction + 3]               # indices are (b, z, y, x)
        if self.shift:
            coord = coord + self.win_size // 2
        _, win_inds = torch.unique(torch.cat([indices[:, :1], coord // self.win_size], 1),
                                   return_inverse=True, dim=0)
        return win_inds, int(win_inds.max()) + 1 if len(win_inds) else 0

    def forward(self, src):
        for i in range(self.directions):
            win_inds, n_win = self.slots(src.indices.long(), i)
            norm_features = self.self_norm_layers[i](src.features)
            x = checkpoint(self.self_attn_layers[i], norm_features, win_inds, n_win, use_reentrant=False)
            src = replace_feature(src, src.features + x)
            src = self.ffn_layers[i](src)
        return src


class SlotFormer(spconv.SparseModule):
    def __init__(self, embed_dim, nhead, num_layers=2, win_size=12, shift=True, indice_key='former'):
        super().__init__()
        shifts = [False, shift]
        self.sla_layers = nn.ModuleList([
            SFLayer(embed_dim, nhead=nhead, directions=2, win_size=win_size, shift=shifts[i % 2], indice_key=indice_key)
            for i in range(num_layers)])

    def forward(self, src):
        for enc in self.sla_layers:
            src = enc(src)
        return src
