# -*- coding: utf-8 -*-
import torch


def reconstruct_img(img_size: tuple,
                    mask_img_patch: torch.Tensor,
                    random_sample_list: torch.Tensor) -> torch.Tensor:
    b, c, h, w = img_size
    _, p_n, _, block_h, block_w = mask_img_patch.shape
    _, r_n = random_sample_list.shape
    new_img = torch.zeros(b, c, h, w).to(mask_img_patch.device)
    assert p_n == r_n
    for i in range(b):
        for j in range(p_n):
            height_id = torch.div(random_sample_list[i, j], int(h / block_w), rounding_mode='floor')
            width_id = random_sample_list[i, j] % int(w / block_h)
            new_img[i, :, height_id * block_h:height_id * block_h + block_h,
                    width_id * block_w: width_id * block_w + block_w] =\
                mask_img_patch[i, j, :, :, :]
    return new_img
