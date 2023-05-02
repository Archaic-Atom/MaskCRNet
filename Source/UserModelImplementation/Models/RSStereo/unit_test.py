# -*- coding: utf-8 -*-
import argparse
import torch

try:
    from .Networks.sttr import STTR
    from .Networks.misc import NestedTensor
except ImportError:
    from Networks.sttr import STTR
    from Networks.misc import NestedTensor


class RSStereoNetUnitTest(object):
    def __init__(self):
        super().__init__()

    def create_args(self) -> object:
        parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

        # * STTR
        parser.add_argument('--channel_dim', default=128, type=int,
                            help="Size of the embeddings (dimension of the transformer)")

        # * Positional Encoding
        parser.add_argument('--position_encoding', default='sine1d_rel', type=str,
                            choices=('sine1d_rel', 'none'),
                            help="Type of positional embedding to use on top of the image features")

        # * Transformer
        parser.add_argument('--num_attn_layers', default=6, type=int,
                            help="Number of attention layers in the transformer")
        parser.add_argument('--nheads', default=8, type=int,
                            help="Number of attention heads inside the transformer's attentions")

        # * Regression Head
        parser.add_argument('--regression_head', default='ot', type=str, choices=('softmax', 'ot'),
                            help='Normalization to be used')
        parser.add_argument('--context_adjustment_layer', default='cal', choices=['cal', 'none'], type=str)
        parser.add_argument('--cal_num_blocks', default=8, type=int)
        parser.add_argument('--cal_feat_dim', default=16, type=int)
        parser.add_argument('--cal_expansion_ratio', default=4, type=int)

        # * Loss
        parser.add_argument('--px_error_threshold', type=int, default=3,
                            help='Number of pixels for error computation (default 3 px)')
        parser.add_argument('--loss_weight', type=str,
                            default='rr:1.0, l1_raw:1.0, l1:1.0, occ_be:1.0',
                            help='Weight for losses')
        parser.add_argument('--validation_max_disp', type=int, default=-1)

        return parser.parse_args()

    @staticmethod
    def gen_input_data() -> tuple:
        left_img = torch.rand(1, 3, 258, 513).cuda()
        right_img = torch.rand(1, 3, 258, 513).cuda()
        bs, _, h, w = left_img.size()
        downsample = 3
        col_offset = int(downsample / 2)
        row_offset = int(downsample / 2)
        sampled_cols = torch.arange(col_offset, w, downsample)[None, ].expand(bs, -1).cuda()
        sampled_rows = torch.arange(row_offset, h, downsample)[None, ].expand(bs, -1).cuda()
        return left_img, right_img, sampled_cols, sampled_rows

    def exec(self, args: object) -> None:
        model = STTR(args)
        model = model.cuda()
        num_params = sum(param.numel() for param in model.parameters())
        print(num_params)
        left_img, right_img, sampled_cols, sampled_rows = self.gen_input_data()
        inputs = NestedTensor(left_img, right_img, sampled_cols=sampled_cols,
                              sampled_rows=sampled_rows)
        outputs = model(inputs)
        print(outputs['disp_pred'].shape)


def main() -> None:
    unit_test = RSStereoNetUnitTest()
    args = unit_test.create_args()
    unit_test.exec(args)


if __name__ == '__main__':
    main()
