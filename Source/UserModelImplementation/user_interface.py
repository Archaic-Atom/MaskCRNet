# -*- coding: utf-8 -*-
import argparse
import JackFramework as jf
import UserModelImplementation.user_define as user_def

# model and dataloader
from UserModelImplementation import Models
from UserModelImplementation import Dataloaders


class UserInterface(jf.UserTemplate.NetWorkInferenceTemplate):
    """docstring for UserInterface"""

    def __init__(self) -> object:
        super().__init__()

    def inference(self, args: object) -> object:
        dataloader = Dataloaders.dataloaders_zoo(args, args.dataset)
        model = Models.model_zoo(args, args.modelName)
        return model, dataloader

    def user_parser(self, parser: object) -> object:
        # parser.add_argument('--startDisp', type=int, default=user_def.START_DISP,
        #                    help='start disparity')
        # return parser
        #
        parser.add_argument('--startDisp', type=int, default=user_def.START_DISP,
                            help='start disparity')
        parser.add_argument('--dispNum', type=int, default=user_def.DISP_NUM,
                            help='disparity number')
        parser.add_argument('--lr_scheduler', type=UserInterface.__str2bool,
                            default=user_def.LR_SCHEDULER,
                            help='use or not use lr scheduler')
        parser.add_argument('--pre_train_opt', type=UserInterface.__str2bool,
                            default=user_def.PRE_TRAIN_OPT,
                            help='pre-trained option')

        parser.add_argument('--block_size', type=int, default=32, help='masked block size')
        parser.add_argument('--mask_ratio', type=float, default=0.65, help='mask ratio')

        parser.add_argument('--weight_decay', default=1e-4, type=float)
        parser.add_argument('--lr_decay_rate', default=0.99, type=float)

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
        return parser

    @staticmethod
    def __str2bool(arg: str) -> bool:
        if arg.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
