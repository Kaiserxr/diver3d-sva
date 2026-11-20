"""Map-to-BEV step for dense middle stages that already produce the BEV map (DenseMixerMiddle).

DenseMixerMiddle folds height into channels itself and writes spatial_features (B, C, ny, nx), the
same layout HeightCompression produces, so this step only checks it and tells the BEV backbone how
many channels to expect.

Config keys:
    NUM_BEV_FEATURES: C, must equal the middle stage's output channels x output height (64 x 2 = 128)
"""
import torch.nn as nn


class DenseBEVPassthrough(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict, **kwargs):
        bev = batch_dict['spatial_features']
        assert bev.shape[1] == self.num_bev_features, (bev.shape, self.num_bev_features)
        return batch_dict
