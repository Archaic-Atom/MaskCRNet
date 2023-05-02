# -*- coding: utf-8 -*-
import torch
from torch import nn
from .restormer import Restormer
from .stereo_matching_basic import DispRegression
from .cascade_raft import CREStereo


class MaskStereoMatching(nn.Module):
    RECONSTRUCTION_CHANNELS = 1

    def __init__(self, in_channles: int, start_disp: int, disp_num: int,
                 pre_train_opt: bool) -> None:
        super().__init__()
        self.pre_train_opt = pre_train_opt
        self.feature_extraction = Restormer(
            inp_channels=in_channles, out_channels=self.RECONSTRUCTION_CHANNELS,
            dim = 16, pre_train_opt=pre_train_opt)
        self.regression = DispRegression([start_disp, start_disp + disp_num])
        if not self.pre_train_opt:
            self.feature_matching = CREStereo(64)

    def forward(self, left_img: torch.Tensor, right_img: torch.Tensor) -> torch.Tensor:
        output, _, _, left_level3 = self.feature_extraction(left_img)
        if self.pre_train_opt:
            return torch.sigmoid(output)
        else:
            _, _, _, right_level3 = self.feature_extraction(right_img)
            output = self.feature_matching(left_level3, right_level3)
            return output
