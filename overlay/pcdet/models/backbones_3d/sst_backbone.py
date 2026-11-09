"""SST-style single-stride window attention (RUN D in docs/DSVT_PLAN.md).

A compact re-implementation of the core of SST, "Embracing Single Stride 3D Object Detector
with Sparse Transformer" (Fan et al., CVPR 2022, arXiv 2112.06375). The official code lives in an
mmdetection3d fork; this file reproduces its mechanism in plain PyTorch so it plugs into
OpenPCDet and needs no custom CUDA:

  * single stride: no downsampling at all; every layer works on the original voxels
  * fixed windows (here full-height 3D windows on our 3D voxels; SST used 2D pillars)
  * region batching: windows are grouped by how many voxels they hold and padded to their
    group's size, so attention runs as a few dense batches with padding masks
  * Swin-style shifted windows on every other layer, so information crosses window borders

Deliberate differences from the official SST, stated so results are reported honestly:
  * 3D voxels with full-height windows instead of 2D pillars (keeps the sonar's elevation)
  * no voxel dropping: SST randomly drops voxels from crowded windows to cap the padded size;
    a Diver3D frame has only ~1,700 voxels, so we keep them all (largest group is padded to its
    actual maximum)
  * learned (MLP) position embedding of in-window coordinates instead of sine embeddings

Output: one feature per input voxel, same order (use VoxelColumnMaxScatter for the BEV map).

Config keys:
    WINDOW_SHAPE: [wx, wy]        window size in voxels (full height is always used)
    SHIFT: [sx, sy]               shift of the odd layers, usually half the window
    NUM_LAYERS: 8                 attention layers (alternating plain / shifted)
    D_MODEL: 128, NHEAD: 8, DIM_FEEDFORWARD: 256, DROPOUT: 0.0
    BUCKETS: [32, 64, 128, 256]   region-batching group sizes; larger windows go to a final group
"""
import torch
import torch.nn as nn


class _WindowLayout:
    """Which window each voxel falls in, its slot inside that window, and the padded batches."""

    def __init__(self, coords, grid_xyz, window, shift, buckets):
        b, z, y, x = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
        wx, wy = window
        sx, sy = shift
        nwx = (grid_xyz[0] + sx) // wx + 1
        nwy = (grid_xyz[1] + sy) // wy + 1
        xs, ys = x + sx, y + sy
        win = (b * nwy + ys // wy) * nwx + xs // wx                # global window id per voxel
        self.local = torch.stack([(xs % wx).float() / wx, (ys % wy).float() / wy,
                                  z.float() / max(grid_xyz[2] - 1, 1)], 1) - 0.5   # in-window position

        uniq, inv, counts = torch.unique(win, return_inverse=True, return_counts=True)
        order = torch.argsort(inv, stable=True)
        starts = torch.cumsum(counts, 0) - counts
        slot = torch.empty_like(inv)
        slot[order] = torch.arange(len(inv), device=inv.device) - starts[inv[order]]

        # region batching: group windows by size, pad each group to its size limit
        limits = list(buckets) + [int(counts.max())]
        self.groups, lo = [], 0
        for hi in limits:
            if hi <= lo:
                continue
            win_mask = (counts > lo) & (counts <= hi)
            lo = hi
            if not win_mask.any():
                continue
            win_ids = torch.nonzero(win_mask, as_tuple=True)[0]
            row = torch.full((len(uniq),), -1, dtype=torch.long, device=inv.device)
            row[win_ids] = torch.arange(len(win_ids), device=inv.device)
            vox = torch.nonzero(win_mask[inv], as_tuple=True)[0]  # voxels in this group
            r, s = row[inv[vox]], slot[vox]
            pad = torch.ones(len(win_ids), hi, dtype=torch.bool, device=inv.device)
            pad[r, s] = False
            self.groups.append((vox, r, s, hi, len(win_ids), pad))


class _SSTLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dropout):
        super().__init__()
        self.pos = nn.Sequential(nn.Linear(3, d_model), nn.ReLU(inplace=True), nn.Linear(d_model, d_model))
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(dim_ff, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, layout):
        q = x + self.pos(layout.local)
        out = torch.zeros_like(x)
        for vox, r, s, size, n_win, pad in layout.groups:
            dense_q = x.new_zeros(n_win, size, x.shape[1])
            dense_v = x.new_zeros(n_win, size, x.shape[1])
            dense_q[r, s] = q[vox]
            dense_v[r, s] = x[vox]
            att, _ = self.attn(dense_q, dense_q, dense_v, key_padding_mask=pad, need_weights=False)
            out[vox] = att[r, s]
        x = self.norm1(x + self.drop(out))                     # post-norm, as in SST / DETR encoders
        return self.norm2(x + self.drop(self.ffn(x)))


class SSTBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        c = model_cfg
        self.model_cfg = model_cfg
        d_model = c.get('D_MODEL', 128)
        self.window = list(c.WINDOW_SHAPE)
        self.shift = list(c.get('SHIFT', [w // 2 for w in self.window]))
        self.buckets = list(c.get('BUCKETS', [32, 64, 128, 256]))
        self.grid_xyz = [int(v) for v in grid_size]
        self.input_proj = nn.Identity() if input_channels == d_model else nn.Linear(input_channels, d_model)
        self.layers = nn.ModuleList([
            _SSTLayer(d_model, c.get('NHEAD', 8), c.get('DIM_FEEDFORWARD', 256), c.get('DROPOUT', 0.0))
            for _ in range(c.get('NUM_LAYERS', 8))])
        self.num_point_features = d_model
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch_dict):
        coords = batch_dict['voxel_coords'].long()           # (batch, z, y, x)
        x = self.input_proj(batch_dict['voxel_features'])
        layouts = [_WindowLayout(coords, self.grid_xyz, self.window, [0, 0], self.buckets),
                   _WindowLayout(coords, self.grid_xyz, self.window, self.shift, self.buckets)]
        for i, layer in enumerate(self.layers):
            x = layer(x, layouts[i % 2])                       # even layers plain, odd layers shifted
        batch_dict['voxel_features'] = x
        return batch_dict
