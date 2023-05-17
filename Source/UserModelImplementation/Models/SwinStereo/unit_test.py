# -*- coding: utf-8 -*-
import torch
import numpy as np
try:
    from .Networks.mask_stereo_matching import MaskStereoMatching
except ImportError:
    from Networks.mask_stereo_matching import MaskStereoMatching


class SwinStereoUnitTest(object):
    def __init__(self):
        super().__init__()

    def exec(self, args: object) -> None:
        pre_train_opt = False
        left_img = torch.rand(1, 1, 256, 256).cuda()
        right_img = torch.rand(1, 2, 1, 32, 32).cuda() if pre_train_opt else torch.rand(1, 1, 256, 256).cuda()
        # right_img =
        range_list = torch.from_numpy(np.array([[0, 1]]))
        print(range_list.shape)
        model = MaskStereoMatching(1, -112, 128, pre_train_opt).cuda()
        num_params = sum(param.numel() for param in model.parameters())
        print(num_params)
        res = model(left_img, right_img, range_list)

        print(res[0].shape)
        # print(model)

        return 0
        for name, param in model.named_parameters():
            print(name)
            if "feature_extraction" in name:
                print(name)
                param.requires_grad = False


def main() -> None:
    unit_test = SwinStereoUnitTest()
    unit_test.exec(None)


if __name__ == '__main__':
    main()
