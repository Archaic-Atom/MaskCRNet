# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


def convbn(in_planes: int, out_planes: int,
           kernel_size: int, stride: int, pad: int, dilation: int) -> object:
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                                   stride=stride, padding=dilation if dilation > 1 else pad,
                                   dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


def convbn_3d(in_planes: int, out_planes: int, kernel_size: int,
              stride: int, pad: int) -> object:
    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size,
                                   padding=pad, stride=stride, bias=False),
                         nn.BatchNorm3d(out_planes))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int,
                 downsample: bool, pad: int, dilation: int) -> object:
        super().__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.tensor) -> torch.tensor:
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class matchshifted(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, left: torch.tensor, right: torch.tensor, shift: int) -> torch.tensor:
        batch, filters, height, width = left.size()
        shifted_left = F.pad(
            torch.index_select(left, 3, Variable(torch.LongTensor([i for i in range(shift, width)])).cuda()),
            (shift, 0, 0, 0)
        )
        shifted_right = F.pad(
            torch.index_select(right, 3, Variable(torch.LongTensor([i for i in range(width - shift)])).cuda()),
            (shift, 0, 0, 0)
        )
        return torch.cat((shifted_left, shifted_right), 1).view(
            batch, filters * 2, 1, height, width
        )


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
        return torch.sum(F.softmax(x, dim=1) * disp_values, 1, keepdim=False)

    def forward(self, x: torch.Tensor, probability: float = 0.65,
                need_mask: bool = False) -> torch.Tensor:
        if need_mask:
            mask = torch.sum(
                (F.softmax(x, dim=self.ID_DISP_DIM) > probability).float(),
                self.ID_DISP_DIM, keepdim=False)
            return self._disp_regression(x), mask
        return self._disp_regression(x)


class feature_extraction(nn.Module):
    def __init__(self) -> object:
        super().__init__()
        self.inplanes = 32
        self.firstconv = self._first_layer()

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64, 64)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32, 32)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16, 16)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 32, kernel_size=1,
                                                padding=0, stride=1, bias=False))

    def _first_layer(self) -> object:
        return nn.Sequential(convbn(1, 32, 3, 2, 1, 1),
                             nn.ReLU(inplace=True),
                             convbn(32, 32, 3, 1, 1, 1),
                             nn.ReLU(inplace=True),
                             convbn(32, 32, 3, 1, 1, 1),
                             nn.ReLU(inplace=True))

    def _make_layer(self, block: object, planes: int, blocks: int,
                    stride: int, pad: int, dilation: int) -> object:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)

        layers = [block(self.inplanes, planes, stride, downsample, pad, dilation)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x: torch.tensor) -> torch.tensor:
        output = self.firstconv(x)
        output = self.layer1(output)
        output_raw = self.layer2(output)
        output = self.layer3(output_raw)
        output_skip = self.layer4(output)

        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.upsample(output_branch1, (output_skip.size()[
                                    2], output_skip.size()[3]), mode='bilinear')

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.upsample(output_branch2, (output_skip.size()[
                                    2], output_skip.size()[3]), mode='bilinear')

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.upsample(output_branch3, (output_skip.size()[
                                    2], output_skip.size()[3]), mode='bilinear')

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.upsample(output_branch4, (output_skip.size()[
                                    2], output_skip.size()[3]), mode='bilinear')

        output_feature = torch.cat((output_raw, output_skip, output_branch4,
                                    output_branch3, output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)

        return output_feature


class RecoverySize(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_0 = nn.Sequential(convbn(32, 16, 3, 1, 1, 1),
                                    nn.ReLU(inplace=True),
                                    convbn(16, 16, 3, 1, 1, 1),
                                    nn.ReLU(inplace=True),
                                    convbn(16, 16, 3, 1, 1, 1),
                                    nn.ReLU(inplace=True))
        self.conv_1 = nn.Sequential(convbn(16, 3, 3, 1, 1, 1),
                                    nn.ReLU(inplace=True),
                                    convbn(3, 3, 3, 1, 1, 1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(3, 3, kernel_size=3, padding=1, stride=1, bias=False))

    def forward(self, x):
        x = F.interpolate(x, [x.size()[2] * 2, x.size()[3] * 2], mode='bilinear', align_corners=True)
        x = self.conv_0(x)
        x = F.interpolate(x, [x.size()[2] * 2, x.size()[3] * 2], mode='bilinear', align_corners=True)
        x = self.conv_1(x)
        x = torch.sigmoid(x)
        return x
