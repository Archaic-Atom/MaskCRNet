# -*- coding: utf-8 -*-
import torch
from torch import nn
from .BackBone.restormer import Restormer
from .CreStereo.cascade_raft import CREStereo
# from .PSMNet.model import PSMNet
# from .BackBone.swin_transformer import SwinTransformerV2
from .PSMNet.submodule import feature_extraction
from .BackBone.extraction import BasicEncoder
from .LacGwcNet.stackhourglass import PSMNet


class MaskStereoMatching(nn.Module):
    RECONSTRUCTION_CHANNELS = 1

    def __init__(self, in_channles: int, start_disp: int, disp_num: int,
                 pre_train_opt: bool) -> None:
        super().__init__()
        self.start_disp, self.disp_num = start_disp, disp_num
        self.pre_train_opt = pre_train_opt
        self.feature_extraction = Restormer(
            inp_channels = in_channles, out_channels = self.RECONSTRUCTION_CHANNELS,
            dim = 48, pre_train_opt = pre_train_opt)

        # self.feature_extraction = SwinTransformerV2(img_size=(224, 224), patch_size=4, in_chans=1)
        # self.feature_extraction = BasicEncoder(in_channles, output_dim=32, norm_fn='batch')
        # self.feature_extraction = feature_extraction()

        if not self.pre_train_opt:
            # self.feature_matching = CREStereo(64)
            self.feature_matching = CREStereo(192)
            # self.feature_matching = PSMNet(1, 384, start_disp = start_disp, maxdisp = disp_num, udc=True, refine='csr')

    def _mask_pre_train_proc(self, left_img: torch.Tensor, mask_img_patch: torch.Tensor,
                             random_sample_list: torch.Tensor) -> torch.Tensor:

        output, _, _, _ = self.feature_extraction(mask_img_patch, left_img, random_sample_list)
        return output

    def _mask_fine_tune_proc(self, left_img: torch.Tensor, right_img: torch.Tensor) -> tuple:
        _, _, _, left_level3 = self.feature_extraction(left_img)
        _, _, _, right_level3 = self.feature_extraction(right_img)
        # return self.feature_matching(left_img, left_level3, right_level3)
        return self.feature_matching(left_level3, right_level3)

    def forward(self, left_img: torch.Tensor, right_img: torch.Tensor,
                random_sample_list: torch.Tensor = None) -> torch.Tensor:
        if self.pre_train_opt:
            return self._mask_pre_train_proc(left_img, right_img, random_sample_list)
        return self._mask_fine_tune_proc(left_img, right_img)
