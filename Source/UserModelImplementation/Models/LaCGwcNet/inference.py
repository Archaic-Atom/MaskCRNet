# -*- coding: utf-8 -*-
import torch.optim as optim

import JackFramework as jf
# import UserModelImplementation.user_define as user_def
from .Networks.stackhourglass import PSMNet
import torch.nn.functional as F
from . import loss_functions as lf


class LacGwcNetworkInterface(jf.UserTemplate.ModelHandlerTemplate):
    """docstring for DeepLabV3Plus"""
    MODEL_ID = 0  # only NLCANetv3
    LABEL_ID = 0  # only a label
    LEFT_IMG_ID = 0
    RIGHT_IMG_ID = 1

    def __init__(self, args: object) -> object:
        super().__init__(args)
        self.__args = args

    @staticmethod
    def lr_lambda(epoch: int) -> float:
        convert_epoch_one, convert_epoch_two, reduce_lr, lr_factor = 200, 300, 0.1, 1.0

        factor = lr_factor if epoch < convert_epoch_one \
            else reduce_lr * lr_factor if convert_epoch_one <= epoch < convert_epoch_two \
            else reduce_lr * reduce_lr * lr_factor

        return factor

    def get_model(self) -> list:
        args = self.__args
        affinity_settings = {}
        affinity_settings['win_w'] = 3
        affinity_settings['win_h'] = 3
        affinity_settings['dilation'] = [1, 2, 4, 8]
        model = PSMNet(3, start_disp=args.startDisp, maxdisp=args.dispNum,
                       struct_fea_c=4, fuse_mode='separate',
                       affinity_settings=affinity_settings, udc=True, refine='csr', mask=False)
        return [model]

    def optimizer(self, model: list, lr: float) -> list:
        args = self.__args
        opt = optim.Adam(model[0].parameters(), lr=lr)
        if args.lr_scheduler:
            sch = optim.lr_scheduler.LambdaLR(opt, lr_lambda=self.lr_lambda)
        else:
            sch = None

        return [opt], [sch]

    def lr_scheduler(self, sch: object, ave_loss: list, sch_id: int) -> None:
        if self.MODEL_ID == sch_id:
            sch.step()

    def inference(self, model: list, input_data: list, model_id: int) -> list:
        args, disp = self.__args, None
        if self.MODEL_ID == model_id:
            disp = model(input_data[self.LEFT_IMG_ID], input_data[self.RIGHT_IMG_ID])

        if args.mode == 'test':
            disp = disp.unsqueeze(0)
        return jf.Tools.convert2list(disp)

    def accuracy(self, output_data: list, label_data: list, model_id: int) -> list:
        # return acc
        # args = self.__args
        res = []

        if self.MODEL_ID == model_id:
            for item in output_data:
                if len(item.shape) == 3:
                    acc, mae = jf.acc.SMAccuracy.d_1(item, label_data[0], invalid_value=-999)
                    res.append(acc[1])
                    res.append(mae)

        return res

    def loss(self, output_data: list, label_data: list, model_id: int) -> list:
        # return loss
        args, gt_left = self.__args, label_data[0]

        if self.MODEL_ID == model_id:
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

            # loss_5 = F.smooth_l1_loss(output_data[8] * 255, label_data[1]) if args.mask else 0

        return [loss_1 + loss_2 + loss_3 + loss_4]

    def pretreatment(self, epoch: int, rank: object) -> None:
        # do something before training epoch
        pass

    def postprocess(self, epoch: int, rank: object,
                    ave_tower_loss: list, ave_tower_acc: list) -> None:
        # do something after training epoch
        pass

    def load_model(self, model: object, checkpoint: dict, model_id: int) -> bool:
        model.load_state_dict(checkpoint['model_0'], strict=False)
        jf.log.info("Model loaded successfully_add")
        return True

    def load_opt(self, opt: object, checkpoint: dict, model_id: int) -> bool:
        # opt.load_state_dict(checkpoint['opt_0'], strict=False)
        # jf.log.info("Opt loaded successfully_add")
        return True

    def save_model(self, epoch: int, model_list: list, opt_list: list) -> dict:
        return None
