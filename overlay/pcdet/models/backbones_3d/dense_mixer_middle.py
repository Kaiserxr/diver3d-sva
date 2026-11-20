"""Dense middle stages for the "is dense attention OK on Diver3D?" experiment (docs/DENSE_PLAN.md).

Every voxel of the 120 x 100 x 10 grid is a token, empty ones included (the sparse stages only see
the ~1,700 occupied voxels). Two families:

1. MetaFormer skeleton with a swappable token mixer (MetaFormer Baselines, Yu et al., TPAMI 2023).
   Everything except the mixer is identical, so the variants differ in exactly one thing:

       scatter VFE features to a dense grid, 1x1x1 conv 128 -> C                       Z=10
       DEPTHS[0] blocks   x = x + mixer(norm(x));  x = x + mlp(norm(x))               Z=10
       conv (3,1,1) stride z2                                                         Z=5
       DEPTHS[1] blocks                                                               Z=5
       [hybrid only] global attention on a (1,4,4)-pooled copy, added back            Z=5
       conv (3,1,1) stride z2, no z padding                                           Z=2
       fold height into channels: 2 x C = 128 BEV channels (DenseBEVPassthrough)

   MIXER:
     'identity'  no spatial mixing (MetaFormer's IdentityFormer): the floor, what the skeleton alone gives
     'conv'      1x1x1 conv C -> 2C, depthwise 3x3x3, 1x1x1 conv 2C -> C   (~4C^2 parameters)
     'na'        3x3x3 neighborhood attention (NAT, Hassani et al., CVPR 2023): each voxel attends to
                 its 27 neighbours, with a learned relative position bias. Same footprint and ~4C^2
                 parameters as 'conv': fixed weights vs content-dependent weights is the only difference.
     'hybrid'    'conv' blocks at full resolution, then ATTN_LAYERS global self-attention layers over
                 the grid pooled by (1,4,4): 5 x 25 x 30 = 3,750 tokens, 3D RoPE (conv early, attention
                 late: LitePT; attention at low resolution: CAFormer / BoTNet / Primus). Its output
                 starts at zero, so at initialisation 'hybrid' computes exactly what 'conv' computes.
   LSA: True   locality self-attention for small datasets (Lee et al., 2021): a learned temperature per
               head and no attention from a token to itself. Applies to 'na' and to 'hybrid's attention.
   ATTN_OFF: True   remove all attention (the 'na' mixer becomes identity, the hybrid branch is skipped).
               Set it at test time on a trained model (test.py --set MODEL.BACKBONE_3D.ATTN_OFF True)
               to check whether the model actually uses its attention (the Primus test).

2. MIXER: 'densea'  Run A's exact layers (SonarSparseMiddle) as ordinary dense 3D convolutions:
   2 x conv 3^3, conv 3^3 stride z2, 2 x conv 3^3, conv (3,1,1) stride z2. Same channels and kernel
   sizes, so sparse vs dense is the only difference to Run A.

Config keys: MIXER, DIM (64), DEPTHS ([2, 2]), MLP_RATIO (4), NHEAD (4), ATTN_LAYERS (2),
             POOL (4), ROPE_BASE (100.0), LSA (False), ATTN_OFF (False),
             BF16 (False): run this middle stage in bfloat16 autocast (~2x faster on the dense grid;
             norms, softmax and the low-resolution global attention stay float32). The rest of the
             detector is unchanged (float32).
             COMPILE (False): torch.compile the middle stage on the GPU (fuses the 27-neighbour loop
             of 'na' and the norm / activation passes into fewer kernels). Same maths, only faster.
             If compiling fails, it falls back to the normal code and logs a warning.
"""
import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .voxel_global_attention import GlobalAttentionLayer


def to_dense(feats, coords, batch_size, shape):
    """(N, C) voxel features + (N, 4) (batch, z, y, x) indices -> (B, C, Z, Y, X), zeros where empty."""
    Z, Y, X = shape
    c = coords.long()
    flat = ((c[:, 0] * Z + c[:, 1]) * Y + c[:, 2]) * X + c[:, 3]
    dense = feats.new_zeros(batch_size * Z * Y * X, feats.shape[1])
    dense[flat] = feats
    return dense.view(batch_size, Z, Y, X, -1).permute(0, 4, 1, 2, 3).contiguous()


class ChannelNorm(nn.Module):
    """LayerNorm over the channels of each voxel, for (B, C, Z, Y, X) tensors."""

    def __init__(self, c):
        super().__init__()
        self.norm = nn.LayerNorm(c)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)


class ConvMixer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.pw1 = nn.Conv3d(c, 2 * c, 1)
        self.dw = nn.Conv3d(2 * c, 2 * c, 3, padding=1, groups=2 * c)
        self.act = nn.GELU()
        self.pw2 = nn.Conv3d(2 * c, c, 1)

    def forward(self, x):
        return self.pw2(self.act(self.dw(self.pw1(x))))


class NeighborhoodAttention3d(nn.Module):
    """Each voxel attends to the k^3 voxels around it (zero padding outside the grid is masked out).
    Pure PyTorch: one shifted view of the padded keys/values per neighbour, so nothing k^3 times the
    size of the feature map is ever stored."""

    def __init__(self, c, nhead, k=3, lsa=False):
        super().__init__()
        assert c % nhead == 0 and k % 2 == 1
        self.h, self.d, self.k, self.lsa = nhead, c // nhead, k, lsa
        self.qkv = nn.Conv3d(c, 3 * c, 1)
        self.proj = nn.Conv3d(c, c, 1)
        self.rpb = nn.Parameter(torch.zeros(nhead, k ** 3))        # relative position bias
        self.offsets = list(itertools.product(range(k), repeat=3))  # (dz, dy, dx) into the padded grid
        self.centre = self.offsets.index((k // 2,) * 3)
        if lsa:                                                     # learned temperature, init 1/sqrt(d)
            self.log_scale = nn.Parameter(torch.full((nhead,), -0.5 * math.log(self.d)))
        self._valid = {}

    def valid(self, shape, device):
        key = (shape, str(device))
        if key not in self._valid:
            r = self.k // 2
            ones = F.pad(torch.ones(1, 1, *shape, device=device), (r, r) * 3)
            Z, Y, X = shape
            v = torch.stack([ones[0, 0, dz:dz + Z, dy:dy + Y, dx:dx + X] for dz, dy, dx in self.offsets]).bool()
            if self.lsa:
                v[self.centre] = False                              # LSA: no attention to oneself
            self._valid[key] = v                                    # (k^3, Z, Y, X)
        return self._valid[key]

    def forward(self, x):
        B, C, Z, Y, X = x.shape
        q, k, v = self.qkv(x).view(B, 3, self.h, self.d, Z, Y, X).unbind(1)
        scale = self.log_scale.exp().view(1, self.h, 1, 1, 1, 1) if self.lsa else self.d ** -0.5
        r = self.k // 2
        kp, vp = F.pad(k, (r, r) * 3), F.pad(v, (r, r) * 3)
        logits = torch.stack([(q * kp[..., dz:dz + Z, dy:dy + Y, dx:dx + X]).sum(2)
                              for dz, dy, dx in self.offsets], 2)   # (B, h, k^3, Z, Y, X)
        logits = logits * scale + self.rpb.view(1, self.h, -1, 1, 1, 1)
        logits = logits.masked_fill(~self.valid((Z, Y, X), x.device), float('-inf'))
        attn = logits.float().softmax(2).to(vp.dtype)
        out = 0
        for i, (dz, dy, dx) in enumerate(self.offsets):
            out = out + attn[:, :, i:i + 1] * vp[..., dz:dz + Z, dy:dy + Y, dx:dx + X]
        return self.proj(out.reshape(B, C, Z, Y, X))


class MetaBlock(nn.Module):
    def __init__(self, c, mixer, mlp_ratio, nhead, lsa):
        super().__init__()
        self.kind = mixer
        self.norm1 = ChannelNorm(c)
        self.mixer = {'identity': lambda: nn.Identity(), 'conv': lambda: ConvMixer(c),
                      'na': lambda: NeighborhoodAttention3d(c, nhead, lsa=lsa)}[mixer]()
        self.norm2 = ChannelNorm(c)
        self.mlp = nn.Sequential(nn.Conv3d(c, mlp_ratio * c, 1), nn.GELU(), nn.Conv3d(mlp_ratio * c, c, 1))

    def forward(self, x, attn_off=False):
        if not (attn_off and self.kind == 'na'):
            x = x + self.mixer(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class LowResGlobalAttention(nn.Module):
    """Global self-attention on a (1, p, p) average-pooled copy of the grid (every cell is a token,
    empty ones too), 3D RoPE on the pooled cell indices; the change is upsampled and added back.
    GlobalAttentionLayer zero-initialises its output projections, so this starts as the identity."""

    def __init__(self, c, nhead, n_layers, pool, rope_base, lsa):
        super().__init__()
        self.pool = pool
        self.layers = nn.ModuleList([GlobalAttentionLayer(c, nhead, 4 * c, pos_enc='rope', rope_base=rope_base)
                                     for _ in range(n_layers)])
        self.lsa = lsa
        if lsa:                                                    # LSA: learned temperature on q
            self.q_scale = nn.Parameter(torch.zeros(n_layers))
        self._cache = {}

    def layout(self, B, z, y, x, device):
        key = (B, z, y, x, str(device))
        if key not in self._cache:
            L = z * y * x
            zz, yy, xx = torch.meshgrid(torch.arange(z), torch.arange(y), torch.arange(x), indexing='ij')
            pos = torch.stack([xx, yy, zz], -1).reshape(L, 3).repeat(B, 1).to(device)   # (x, y, z)
            rows = torch.arange(B, device=device).repeat_interleave(L)
            slots = torch.arange(L, device=device).repeat(B)
            self._cache[key] = (pos, rows, slots, L)
        return self._cache[key]

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):         # few tokens: keep RoPE angles in fp32
            return self._forward(x.float()).to(x.dtype)

    def _forward(self, x):
        B, C, Z, Y, X = x.shape
        p = self.pool
        assert Y % p == 0 and X % p == 0, (Y, X, p)
        low = F.avg_pool3d(x, (1, p, p))                           # (B, C, Z, Y/p, X/p)
        _, _, z, y, xx = low.shape
        pos, rows, slots, L = self.layout(B, z, y, xx, x.device)
        f0 = low.permute(0, 2, 3, 4, 1).reshape(B * L, C)
        f = f0
        for i, layer in enumerate(self.layers):
            if self.lsa:                                           # temperature via a scaled query copy
                f = _lsa_layer(layer, f, pos, rows, slots, B, L, self.q_scale[i])
            else:
                f = layer(f, pos, rows, slots, B, L)
        delta = (f - f0).view(B, z, y, xx, C).permute(0, 4, 1, 2, 3)
        delta = delta.repeat_interleave(p, 3).repeat_interleave(p, 4)
        return x + delta


def _lsa_layer(layer, x, pos, rows, slots, B, L, log_t):
    """GlobalAttentionLayer.forward with LSA: learned temperature exp(log_t) and the diagonal masked
    (all tokens are valid here, so no padding mask is needed)."""
    from .voxel_global_attention import apply_rope, rope_angles
    N, C = x.shape
    h = layer.norm1(x)
    q, k, v = layer.qkv(h).view(B, L, 3, layer.nhead, layer.head_dim).permute(2, 0, 3, 1, 4)
    ang = rope_angles(pos[:L], layer.head_dim, layer.rope_base)[None, None]
    q, k = apply_rope(q, ang), apply_rope(k, ang)
    eye = torch.eye(L, dtype=torch.bool, device=x.device)
    mask = torch.zeros(L, L, device=x.device, dtype=q.dtype).masked_fill(eye, float('-inf'))
    out = F.scaled_dot_product_attention(q * log_t.exp(), k, v, attn_mask=mask)
    out = out.permute(0, 2, 1, 3).reshape(N, C)
    x = x + layer.proj(out)
    return x + layer.ffn(layer.norm2(x))


class DenseRunA(nn.Module):
    """Run A's layers as dense 3D convolutions (conv + BN + ReLU, as spconv's post_act_block)."""

    def __init__(self, cin, c1, c2, n_subm):
        super().__init__()

        def cbr(i, o, k, stride=1, pad=1):
            return nn.Sequential(nn.Conv3d(i, o, k, stride=stride, padding=pad, bias=False),
                                 nn.BatchNorm3d(o, eps=1e-3, momentum=0.01), nn.ReLU())
        self.net = nn.Sequential(
            *[cbr(cin if i == 0 else c1, c1, 3) for i in range(n_subm)],
            cbr(c1, c2, 3, stride=(2, 1, 1), pad=1),
            *[cbr(c2, c2, 3) for _ in range(n_subm)],
            cbr(c2, c2, (3, 1, 1), stride=(2, 1, 1), pad=0))

    def forward(self, x):
        return self.net(x)


class DenseMixerMiddle(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        c = model_cfg
        self.model_cfg = model_cfg
        self.shape = tuple(int(v) for v in grid_size[::-1])      # (Z, Y, X)
        self.mixer = c.get('MIXER', 'conv')
        assert self.mixer in ('identity', 'conv', 'na', 'hybrid', 'densea'), self.mixer
        self.attn_off = bool(c.get('ATTN_OFF', False))
        self.bf16 = bool(c.get('BF16', False))
        self.compile = bool(c.get('COMPILE', False))
        self._compiled = None
        if self.mixer == 'densea':
            c1, c2 = c.get('NUM_FILTERS', [64, 64])
            self.net = DenseRunA(input_channels, c1, c2, c.get('NUM_SUBM', 2))
            dim = c2
        else:
            dim, depths = c.get('DIM', 64), c.get('DEPTHS', [2, 2])
            mlp, nhead, lsa = c.get('MLP_RATIO', 4), c.get('NHEAD', 4), bool(c.get('LSA', False))
            block_mixer = 'conv' if self.mixer == 'hybrid' else self.mixer
            self.stem = nn.Conv3d(input_channels, dim, 1)
            self.stage1 = nn.ModuleList([MetaBlock(dim, block_mixer, mlp, nhead, lsa) for _ in range(depths[0])])
            self.down1 = nn.Sequential(ChannelNorm(dim), nn.Conv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)))
            self.stage2 = nn.ModuleList([MetaBlock(dim, block_mixer, mlp, nhead, lsa) for _ in range(depths[1])])
            self.global_attn = (LowResGlobalAttention(dim, nhead, c.get('ATTN_LAYERS', 2), c.get('POOL', 4),
                                                      c.get('ROPE_BASE', 100.0), lsa)
                                if self.mixer == 'hybrid' else None)
            self.down2 = nn.Sequential(ChannelNorm(dim), nn.Conv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=0))
        nz = self.shape[0]
        self.out_height = ((nz + 2 - 3) // 2 + 1 - 3) // 2 + 1     # 10 -> 5 -> 2
        self.num_point_features = dim
        self.num_bev_features = dim * self.out_height

    def forward(self, batch_dict):
        x = to_dense(batch_dict['voxel_features'], batch_dict['voxel_coords'], batch_dict['batch_size'], self.shape)
        on = self.bf16 and x.is_cuda
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=on):
            x = self._run_body(x)
        B, C, Z, Y, X = x.shape
        batch_dict['spatial_features'] = x.float().reshape(B, C * Z, Y, X)   # same folding as HeightCompression
        batch_dict['spatial_features_stride'] = 1
        return batch_dict

    def _run_body(self, x):
        if not (self.compile and x.is_cuda):
            return self._body(x)
        if self._compiled is None:
            self._compiled = torch.compile(self._body)
        try:
            return self._compiled(x)
        except Exception as e:                                     # e.g. no Triton for this GPU
            import logging
            logging.getLogger(__name__).warning(f'DenseMixerMiddle: torch.compile failed ({e}); running uncompiled')
            self.compile = False
            return self._body(x)

    def _body(self, x):
        if self.mixer == 'densea':
            x = self.net(x)
        else:
            x = self.stem(x)
            for blk in self.stage1:
                x = blk(x, self.attn_off)
            x = self.down1(x)
            for blk in self.stage2:
                x = blk(x, self.attn_off)
            if self.global_attn is not None and not self.attn_off:
                x = self.global_attn(x)
            x = self.down2(x)
        return x
