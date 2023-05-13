# -*- coding: utf-8 -*-
import torch
# import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import JackFramework as jf
# import UserModelImplementation.user_define as user_def

from . import loss_functions as lf
from .Networks.mask_stereo_matching import MaskStereoMatching


class SwinStereoInterface(jf.UserTemplate.ModelHandlerTemplate):
    """docstring for RSStereoInterface"""
    MODEL_ID, DISP_IMG_ID = 0, 2
    LEFT_IMG_ID, RIGHT_IMG_ID = 0, 1
    IMG_ID, MASK_IMG_ID, RANDOM_SAMPLE_LIST_ID = 0, 1, 2

    def __init__(self, args: object) -> object:
        super().__init__(args)
        self.__args = args

    def get_model(self) -> list:
        args = self.__args
        if 'whu' == args.dataset:
            model = MaskStereoMatching(1, args.startDisp, args.dispNum, args.pre_train_opt)
        else:
            model = MaskStereoMatching(3, args.startDisp, args.dispNum, args.pre_train_opt)

        if not args.pre_train_opt:
            for name, param in model.named_parameters():
                if "feature_extraction" in name:
                    param.requires_grad = False

        return [model]

    def optimizer(self, model: list, lr: float) -> list:
        args = self.__args
        opt = optim.Adam(model[0].parameters(), lr=lr)
        if args.lr_scheduler:
            sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=args.lr_decay_rate)
        else:
            sch = None
        return [opt], [sch]

    def lr_scheduler(self, sch: object, ave_loss: list, sch_id: int) -> None:
        # how to do schenduler
        # if self.MODEL_ID == sch_id:
        #    sch.step()
        #    print("current learning rate", sch.get_lr())
        pass

    def inference(self, model: list, input_data: list, model_id: int) -> list:
        args, outputs = self.__args, []
        if self.MODEL_ID == model_id:
            if args.pre_train_opt:
                outputs.append(model(input_data[self.IMG_ID],
                                     input_data[self.MASK_IMG_ID],
                                     input_data[self.RANDOM_SAMPLE_LIST_ID]))
            else:
                if args.mode == 'test':
                    disp = model(input_data[self.LEFT_IMG_ID], input_data[self.RIGHT_IMG_ID])
                    outputs.append(disp)
                else:
                    outputs = jf.Tools.convert2list(model(input_data[self.LEFT_IMG_ID],
                                                          input_data[self.RIGHT_IMG_ID]))
        return outputs

    def accuracy(self, output_data: list, label_data: list, model_id: int) -> list:
        # return acc
        args, res = self.__args, []
        if self.MODEL_ID == model_id:
            if args.pre_train_opt:
                mask = label_data[0] > 0
                acc = (torch.abs(output_data[0][mask] - label_data[0][mask]) * 255).sum()
                res.append(acc / mask.int().sum())
            else:
                for item in output_data:
                    if len(item.shape) == 3:
                        acc, mae = jf.acc.SMAccuracy.d_1(item, label_data[0], invalid_value=-999)
                        res.append(acc[1])
                        res.append(mae)
        return res

    def loss(self, output_data: list, label_data: list, model_id: int) -> list:
        # return loss
        args = self.__args
        if self.MODEL_ID == model_id:
            if args.pre_train_opt:
                mask = label_data[0] > 0
                loss = ((output_data[0][mask] - label_data[0][mask])) ** 2
                loss = loss.sum() / mask.int().sum()
                return [loss]
            gt_left = label_data[0]
            mask = (gt_left < args.startDisp + args.dispNum) & (gt_left > args.startDisp)
            loss_1 = 0.5 * F.smooth_l1_loss(output_data[0][mask], gt_left[mask]) + \
                0.7 * F.smooth_l1_loss(output_data[1][mask], gt_left[mask]) + \
                F.smooth_l1_loss(output_data[2][mask], gt_left[mask])

            gt_distribute = lf.disp2distribute(args.startDisp, gt_left, args.dispNum, b=2)
            loss_2 = 0.5 * lf.CEloss(args.startDisp, gt_left, args.dispNum, gt_distribute, output_data[3]) + \
                0.7 * lf.CEloss(args.startDisp, gt_left, args.dispNum, gt_distribute, output_data[4]) + \
                lf.CEloss(args.startDisp, gt_left, args.dispNum, gt_distribute, output_data[5])

            loss_3 = F.smooth_l1_loss(output_data[6][mask], gt_left[mask])
            loss_4 = lf.CEloss(args.startDisp, gt_left, args.dispNum, gt_distribute, output_data[7])

            # loss_0 = jf.loss.SMLoss.smooth_l1(
            #     output_data[0], label_data[0], args.startDisp, args.startDisp + args.dispNum)
            # loss_1 = jf.loss.SMLoss.smooth_l1(
            #    output_data[1], label_data[0], args.startDisp, args.startDisp + args.dispNum)
            # loss_2 = jf.loss.SMLoss.smooth_l1(
            #    output_data[2], label_data[0], args.startDisp, args.startDisp + args.dispNum)
            return [loss_3 + loss_1 + loss_2 + loss_4, loss_3]

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
        args = self.__args
        #  return False
        if not args.pre_train_opt:
            model.load_state_dict(checkpoint['model_0'], strict = False)
            jf.log.info("Model loaded successfully_add")
            return True
        return False

    # Optional
    def load_opt(self, opt: object, checkpoint: dict, model_id: int) -> bool:
        args = self.__args
        if not args.pre_train_opt:
            return True
        return False

    # Optional
    def save_model(self, epoch: int, model_list: list, opt_list: list) -> dict:
        # return None
        return None
