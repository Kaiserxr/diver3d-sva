"""CenterPoint detector that also adds FSHNet's backbone loss (adaptive feature diffusion).

FSHNet edits OpenPCDet's CenterPoint.get_training_loss to add backbone_3d.get_loss(); here that is
a subclass instead, so OpenPCDet's own files stay untouched. Source: FSHNet
pcdet/models/detectors/centerpoint.py (https://github.com/Say2L/FSHNet, commit 8e8f23e, Apache-2.0).
"""
from ..detectors.centerpoint import CenterPoint


class FSHNetCenterPoint(CenterPoint):
    def get_training_loss(self):
        loss, tb_dict, disp_dict = super().get_training_loss()
        if getattr(self.backbone_3d, 'adaptive_feature_diffusion', False):
            loss_afd, tb_dict_afd = self.backbone_3d.get_loss()
            loss = loss + loss_afd
            tb_dict.update(tb_dict_afd)
        return loss, tb_dict, disp_dict
