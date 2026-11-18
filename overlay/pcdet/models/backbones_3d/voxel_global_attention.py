"""Global self-attention over all non-empty voxels of each frame (pure PyTorch, no spconv).

Used by SonarHybridMiddle (experiment 1 in docs/HYBRID_PLAN.md). A Diver3D frame has only
~1,500-2,000 voxels after the sparse-conv stage, so full attention over a whole frame is cheap:
every voxel can attend to every other voxel, i.e. the receptive field is the whole scene.

Frames are padded to the longest frame in the batch and padding is masked out, so the result is
exactly what attending within each frame separately would give.

Position encoding options (the conv-style one lives in SonarHybridMiddle because it needs spconv):
    'rope'  3D rotary position encoding on the voxel indices (x, y, z): queries and keys are
            rotated by angles proportional to their coordinates, so attention scores depend on
            the relative offset between two voxels. No learned parameters (LitePT's PointROPE idea).
    'none'  no position information inside attention (used with the conv encoding).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def rope_angles(pos, head_dim, base):
    """Rotation angles (N, head_dim // 2) for 3D positions pos (N, 3) = (x, y, z) in voxels.

    The head_dim // 2 rotation pairs are shared out over the axes: z gets a quarter (at least 1;
    the grid is only a few cells tall there), x and y split the rest.
    """
    pairs = head_dim // 2
    nz = max(1, pairs // 4)
    nx = (pairs - nz + 1) // 2
    ny = pairs - nz - nx
    angles = []
    for axis, n in ((0, nx), (1, ny), (2, nz)):
        if n == 0:
            continue
        freq = base ** (-torch.arange(n, device=pos.device, dtype=torch.float32) / n)
        angles.append(pos[:, axis:axis + 1].float() * freq)
    return torch.cat(angles, 1)                                    # (N, pairs)


def apply_rope(x, angles):
    """Rotate consecutive pairs of x (..., N, head_dim) by angles (N, head_dim // 2)."""
    cos, sin = angles.cos(), angles.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)
    return out.flatten(-2)


class GlobalAttentionLayer(nn.Module):
    """Pre-norm transformer layer. The output projections start at zero, so a freshly added layer
    passes its input through unchanged and training starts from the plain sparse-conv network."""

    def __init__(self, d_model, nhead, dim_ff, dropout=0.0, pos_enc='rope', rope_base=100.0):
        super().__init__()
        assert d_model % nhead == 0, (d_model, nhead)
        self.nhead, self.head_dim = nhead, d_model // nhead
        assert pos_enc != 'rope' or self.head_dim % 2 == 0, 'RoPE needs an even head dimension'
        self.pos_enc, self.rope_base, self.dropout = pos_enc, rope_base, dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(dim_ff, d_model))
        for lin in (self.proj, self.ffn[-1]):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, x, pos, rows, slots, n_frames, max_len):
        """x (N, C) voxel features; pos (N, 3) voxel indices (x, y, z);
        rows/slots (N,) frame index and position within the frame's padded row."""
        N, C = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(N, 3, self.nhead, self.head_dim)
        dense = x.new_zeros(n_frames, max_len, 3, self.nhead, self.head_dim)
        dense[rows, slots] = qkv
        q, k, v = dense.permute(2, 0, 3, 1, 4)                    # each (B, heads, L, head_dim)
        if self.pos_enc == 'rope':
            ang = x.new_zeros(n_frames, max_len, self.head_dim // 2)
            ang[rows, slots] = rope_angles(pos, self.head_dim, self.rope_base)
            ang = ang[:, None]                                    # (B, 1, L, pairs)
            q, k = apply_rope(q, ang), apply_rope(k, ang)
        valid = torch.zeros(n_frames, max_len, dtype=torch.bool, device=x.device)
        valid[rows, slots] = True
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :],
                                             dropout_p=self.dropout if self.training else 0.0)
        out = out.permute(0, 2, 1, 3).reshape(n_frames, max_len, C)[rows, slots]
        x = x + self.proj(out)
        return x + self.ffn(self.norm2(x))


def frame_layout(batch_idx):
    """Row (frame) and slot (position within the frame) for each voxel, plus the padded length."""
    order = torch.argsort(batch_idx, stable=True)
    counts = torch.bincount(batch_idx)
    starts = torch.cumsum(counts, 0) - counts
    slots = torch.empty_like(batch_idx)
    slots[order] = torch.arange(len(batch_idx), device=batch_idx.device) - starts[batch_idx[order]]
    return batch_idx, slots, int(counts.numel()), int(counts.max()) if len(counts) else 0
