# -*- coding: utf-8 -*-
import torch
from torch import nn

import time
try:
    from .encoder import mae_vit_base_patch16
except ImportError:
    from encoder import mae_vit_base_patch16


class StereoMatching(nn.Module):
    """docstring for ClassName"""

    def __init__(self, img_size: tuple or int, in_channles: int,
                 start_disp: int, disp_num: int) -> None:
        super().__init__()
        self.start_disp, self.disp_num = start_disp, disp_num
        self.feature_extraction = mae_vit_base_patch16(
            img_size=img_size, in_chans = in_channles, out_chans=1)
        self.feature_matching = mae_vit_base_patch16(
            img_size=img_size, in_chans = disp_num, out_chans=disp_num)

    def gen_cost(self, left_img: torch.Tensor, right_img: torch.Tensor) -> torch.Tensor:
        b, _, h, w = left_img.shape
        cost = torch.zeros(b, self.disp_num, h, w).cuda()
        for i in range(self.disp_num):
            d = self.start_disp + i
            if d > 0:
                cost[:, i, :, d:] = left_img[:, :, :, d:] * right_img[:, :, :, :-d]
            elif d < 0:
                cost[:, i, :, :d] = left_img[:, :, :, d:] * right_img[:, :, :, :-d]
            else:
                cost[:, i, :, :] = left_img[:, :, :, :] * right_img[:, :, :, :]
        return cost.contiguous()

    def regress(self, x: torch.Tensor) -> torch.Tensor:
        disp_values = torch.arange(
            self.start_disp, self.start_disp + self.disp_num).view(1, -1, 1, 1).float().to(x.device)
        return torch.sum(x * disp_values, 1)

    def forward(self, left_img: torch.Tensor, right_img: torch.Tensor) -> torch.Tensor:
        left_img = self.feature_extraction(left_img)
        right_img = self.feature_extraction(right_img)
        cost = self.gen_cost(left_img, right_img)
        cost = self.feature_matching(cost)
        return self.regress(cost)


if __name__ == '__main__':
    model = StereoMatching(img_size=(448, 448), in_channles=3, start_disp=1, disp_num=192)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(num_params)
    num_params = sum(param.numel() for param in model.parameters())
    print(num_params)
    model = model.cuda()

    img_l = torch.rand(1, 3, 448, 448).cuda()
    img_r = torch.rand(1, 3, 448, 448).cuda()

    for _ in range(100):
        time_start = time.time()
        res = model(img_l, img_r)
        time_end = time.time()
        print('totally cost', time_end - time_start, res.shape)

    # image = model.unpatchify(res[1])
    # print(image.shape)
