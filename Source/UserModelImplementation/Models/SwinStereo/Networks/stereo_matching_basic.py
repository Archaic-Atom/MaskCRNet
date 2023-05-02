# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F


class DispRegression(nn.Module):
    """docstring for DispRegression"""
    ID_START_DISP, ID_END_DISP = 0, 1
    ID_DISP_DIM = 1

    def __init__(self, disp_range: list) -> object:
        super().__init__()
        assert disp_range[self.ID_END_DISP] > disp_range[self.ID_START_DISP]
        self.__start_disp, off_set = disp_range[self.ID_START_DISP], 1
        self.__disp_num = disp_range[self.ID_END_DISP] - disp_range[self.ID_START_DISP] + off_set

    def _disp_regression(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 4
        disp_values = torch.arange(
            self.__start_disp, self.__start_disp + self.__disp_num,
            dtype=x.dtype, device=x.device)
        disp_values = disp_values.view(1, self.__disp_num, 1, 1)
        return torch.sum(F.softmax(x, dim=1) * disp_values, 1, keepdim=True)

    def forward(self, x: torch.Tensor, probability: float = 0.65,
                need_mask: bool = False) -> torch.Tensor:
        if need_mask:
            mask = torch.sum(
                (F.softmax(x, dim=self.ID_DISP_DIM) > probability).float(),
                self.ID_DISP_DIM, keepdim=True)
            return self._disp_regression(x), mask
        return self._disp_regression(x)
