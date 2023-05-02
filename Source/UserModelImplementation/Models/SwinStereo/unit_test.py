# -*- coding: utf-8 -*-
import torch
try:
    from .Networks.mask_stereo_matching import MaskStereoMatching
except ImportError:
    from Networks.mask_stereo_matching import MaskStereoMatching


class SwinStereoUnitTest(object):
    def __init__(self):
        super().__init__()

    def exec(self, args: object) -> None:
        left_img = torch.rand(1, 1, 32, 128)
        right_img = torch.rand(1, 1, 32, 128)
        model = MaskStereoMatching(1, -112, 128, False)
        num_params = sum(param.numel() for param in model.parameters())
        print(num_params)
        output = model(left_img, right_img)
        print(len(output))
        print(output[0].shape)
        print(output[0])

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
