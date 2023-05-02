# -*- coding: utf-8 -*-
import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

import JackFramework as jf
# import UserModelImplementation.user_define as user_def

try:
    from .Networks.sttr import STTR
    from .Networks.misc import NestedTensor
    from .Networks.loss import build_criterion
except ImportError:
    from Networks.sttr import STTR
    from Networks.misc import NestedTensor
    from Networks.loss import build_criterion


class RSStereoInterface(jf.UserTemplate.ModelHandlerTemplate):
    """docstring for RSStereoInterface"""
    MODEL_ID, DISP_IMG_ID = 0, 2
    LEFT_IMG_ID, RIGHT_IMG_ID = 0, 1

    def __init__(self, args: object) -> object:
        super().__init__(args)
        self.__args = args
        self.criterion = build_criterion(args)

    @staticmethod
    def _convert_input(left_img: torch.tensor, right_img: torch.tensor,
                       disp_img: torch.tensor) -> NestedTensor:
        bs, _, h, w = left_img.size()
        downsample = 3
        col_offset, row_offset = int(downsample / 2), int(downsample / 2)
        sampled_cols = torch.arange(
            col_offset, w, downsample)[None, ].expand(bs, -1).cuda()
        sampled_rows = torch.arange(
            row_offset, h, downsample)[None, ].expand(bs, -1).cuda()
        return NestedTensor(left_img, right_img, disp_img,
                            sampled_cols=sampled_cols, sampled_rows=sampled_rows)

    @staticmethod
    def _convert_label(disp_img: torch.tensor) -> NestedTensor:
        bs, h, w = disp_img.size()
        downsample = 3
        col_offset, row_offset = int(downsample / 2), int(downsample / 2)
        sampled_cols = torch.arange(
            col_offset, w, downsample)[None, ].expand(bs, -1).cuda()
        sampled_rows = torch.arange(
            row_offset, h, downsample)[None, ].expand(bs, -1).cuda()
        occ_mask = torch.zeros_like(disp_img).byte().cuda()
        occ_mask_right = torch.zeros_like(disp_img).byte().cuda()
        return NestedTensor(None, None, disp=disp_img, sampled_cols=sampled_cols,
                            sampled_rows=sampled_rows, occ_mask=occ_mask, occ_mask_right=occ_mask_right)

    def get_model(self) -> list:
        # args = self.__args
        # return model
        model = STTR(self.__args)
        return [model]

    def optimizer(self, model: list, lr: float) -> list:
        args = self.__args
        opt = torch.optim.AdamW(model[self.MODEL_ID].parameters(), lr=lr,
                                weight_decay=self.__args.weight_decay)
        if args.lr_scheduler:
            sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=args.lr_decay_rate)
        else:
            sch = None
        # return opt and sch
        return [opt], [None]

    def lr_scheduler(self, sch: object, ave_loss: list, sch_id: int) -> None:
        # how to do schenduler
        # if self.MODEL_ID == sch_id:
        #    sch.step()
        #    print("current learning rate", sch.get_lr())
        pass

    def inference(self, model: list, input_data: list, model_id: int) -> list:
        # args = self.__args
        # return output
        outputs = None
        if self.MODEL_ID == model_id:
            nested_tensor = self._convert_input(
                input_data[self.LEFT_IMG_ID], input_data[self.RIGHT_IMG_ID],
                input_data[self.DISP_IMG_ID])
            if self.__args.mode == 'test':
                outputs = None
            else:
                outputs = model(nested_tensor)
        return [outputs]

    def accuracy(self, output_data: list, label_data: list, model_id: int) -> list:
        # return acc
        # args = self.__args
        res = []
        if self.MODEL_ID == model_id:
            for item in output_data:
                if len(item['disp_pred'].shape) == 3:
                    acc, mae = jf.acc.SMAccuracy.d_1(item['disp_pred'], label_data[0])
                    res.append(acc[1])
                    res.append(mae)
        return res

    def loss(self, output_data: list, label_data: list, model_id: int) -> list:
        # return loss
        if self.MODEL_ID == model_id:
            labels = self._convert_label(label_data[0])
            loss = self.criterion(labels, output_data[0])
        return [loss['aggregated']]

    # Optional
    def pretreatment(self, epoch: int, rank: object) -> None:
        # do something before training epoch
        pass

    # Optional
    def postprocess(self, epoch: int, rank: object,
                    ave_tower_loss: list, ave_tower_acc: list) -> None:
        # do something after training epoch
        pass

    # Optional
    def load_model(self, model: object, checkpoint: dict, model_id: int) -> bool:
        # return False
        return False

    # Optional
    def load_opt(self, opt: object, checkpoint: dict, model_id: int) -> bool:
        # return False
        return False

    # Optional
    def save_model(self, epoch: int, model_list: list, opt_list: list) -> dict:
        # return None
        return None
