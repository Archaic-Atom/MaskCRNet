# -*- coding: utf-8 -*-
import argparse
import JackFramework as jf
# import UserModelImplementation.user_define as user_def

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
        return None

    @staticmethod
    def __str2bool(arg: str) -> bool:
        if arg.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
