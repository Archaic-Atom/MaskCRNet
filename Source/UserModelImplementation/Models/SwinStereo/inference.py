# -*- coding: utf-8 -*-
import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
import JackFramework as jf
# import UserModelImplementation.user_define as user_def

try:
    from .Networks.mask_stereo_matching import MaskStereoMatching
    from .loss_functions import sequence_loss
except ImportError:
    from Networks.mask_stereo_matching import MaskStereoMatching
    from loss_functions import sequence_loss


class SwinStereoInterface(jf.UserTemplate.ModelHandlerTemplate):
    """docstring for RSStereoInterface"""
    MODEL_ID, DISP_IMG_ID = 0, 2
    LEFT_IMG_ID, RIGHT_IMG_ID = 0, 1

    def __init__(self, args: object) -> object:
        super().__init__(args)
        self.__args = args

    def get_model(self) -> list:
        args = self.__args
        # return model
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

        opt = torch.optim.AdamW(model[self.MODEL_ID].parameters(), lr=lr,
                                weight_decay=self.__args.weight_decay)
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
                outputs.append(model(
                    input_data[self.LEFT_IMG_ID], input_data[self.RIGHT_IMG_ID]))
            else:
                if args.mode == 'test':
                    outputs.append(model(
                        input_data[self.LEFT_IMG_ID], input_data[self.RIGHT_IMG_ID]))
                else:
                    outputs.append(model(
                        input_data[self.LEFT_IMG_ID], input_data[self.RIGHT_IMG_ID]))
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

                disp = output_data[0][0][:, 0, :, :]
                if len(disp.shape) == 3:
                    acc, mae = jf.acc.SMAccuracy.d_1(disp, label_data[0], invalid_value=-999)
                    res.append(acc[1])
                    res.append(mae)
        return res

    def loss(self, output_data: list, label_data: list, model_id: int) -> list:
        # return loss
        args = self.__args
        if self.MODEL_ID == model_id:
            if args.pre_train_opt:
                mask = label_data[0] > 0
                loss = (255 * (output_data[0][mask] - label_data[0][mask])) ** 2
                loss = loss.sum() / mask.int().sum()
            else:
                gt_left, gt_mask = label_data[0], label_data[0] > -999
                gt_left = gt_left.unsqueeze(1)
                gt_flow = torch.cat([gt_left, gt_left * 0], dim=1)
                loss = sequence_loss(
                    output_data[0], gt_flow, gt_mask, gamma=0.8)
        return [loss]

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
            model.load_state_dict(checkpoint['model_0'], strict=False)
            jf.log.info("Model loaded successfully_add")
            return True
        return False

    # Optional
    def load_opt(self, opt: object, checkpoint: dict, model_id: int) -> bool:
        # return False
        args = self.__args
        #  return False
        if not args.pre_train_opt:
            return True
        return False

    # Optional
    def save_model(self, epoch: int, model_list: list, opt_list: list) -> dict:
        # return None
        return None