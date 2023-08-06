# -*- coding: utf-8 -*-
import torch
# import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import JackFramework as jf
# import UserModelImplementation.user_define as user_def
import timm.optim.optim_factory as optim_factory

import math
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

    @staticmethod
    def lr_lambda(epoch: int) -> float:
        warmup_epochs = 40
        cos_epoch = 1000
        if epoch < warmup_epochs:
            factor = epoch / warmup_epochs
        else:
            factor = 0.5 * \
                (1. + math.cos(math.pi * (epoch - warmup_epochs) / cos_epoch))
        return factor

    def get_model(self) -> list:
        args = self.__args
        if 'whu' == args.dataset:
            model = MaskStereoMatching(1, 1, args.startDisp, args.dispNum, args.pre_train_opt)
        else:
            model = MaskStereoMatching(3, 3, args.startDisp, args.dispNum, args.pre_train_opt)

        if not args.pre_train_opt:
            for name, param in model.named_parameters():
                if "feature_extraction" in name:
                    param.requires_grad = False

        return [model]

    def optimizer(self, model: list, lr: float) -> list:
        args = self.__args
        opt = torch.optim.AdamW(model[0].parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
        #opt = optim.Adam(model[0].parameters(), lr=lr)
        if args.lr_scheduler:
            sch = optim.lr_scheduler.LambdaLR(opt, lr_lambda=self.lr_lambda)
        else:
            sch = None
        return [opt], [sch]

    def lr_scheduler(self, sch: object, ave_loss: list, sch_id: int) -> None:
        # how to do schenduler
        if self.MODEL_ID == sch_id:
            sch.step()
           # print("current learning rate", sch.get_lr())

    def inference(self, model: list, input_data: list, model_id: int) -> list:
        args, outputs = self.__args, []
        if self.MODEL_ID == model_id:
            if args.pre_train_opt:
                outputs = jf.Tools.convert2list(
                    model(input_data[self.IMG_ID],
                          input_data[self.MASK_IMG_ID],
                          input_data[self.RANDOM_SAMPLE_LIST_ID]))
            else:
                if args.mode == 'test':
                    disp = model(input_data[self.LEFT_IMG_ID],
                                 input_data[self.RIGHT_IMG_ID])
                    outputs.append(disp[:, 0, :, :])
                else:
                    outputs = jf.Tools.convert2list(model(input_data[self.LEFT_IMG_ID],
                                                          input_data[self.RIGHT_IMG_ID]))
        return outputs

    def accuracy(self, output_data: list, label_data: list, model_id: int) -> list:
        # return acc
        args, res = self.__args, []
        if self.MODEL_ID == model_id:
            if args.pre_train_opt:
                # mask = label_data[0] > 0
                # res.append((torch.abs(output_data[0][mask] - label_data[0][mask]) * 255).mean())
                res.append(output_data[1])
            else:
                gt_left = label_data[0]
                mask = (gt_left < args.startDisp + args.dispNum) & (gt_left > args.startDisp)
                for idx, item in enumerate(output_data):
                    disp = item[:, 0, :, :]
                    if len(disp.shape) == 3 and idx > len(output_data) - 3:
                        acc, mae = jf.acc.SMAccuracy.d_1(disp, gt_left * mask, invalid_value=0)
                        res.append(acc[1])
                        res.append(mae)
        return res

    def loss(self, output_data: list, label_data: list, model_id: int) -> list:
        # return loss
        args = self.__args
        if self.MODEL_ID == model_id:
            if args.pre_train_opt:
                # print(output_data[2].shape)
                # print(output_data[2] * 255)
                # print(label_data[0] * 255)
                # mask = label_data[0] > 0
                '''
                import cv2
                img = output_data[2][0, :, :, :].cpu().detach().numpy()
                print(img.shape)
                img = img.transpose(1, 2, 0)
                cv2.imwrite('/home2/raozhibo/Documents/Programs/RSStereo/Tmp/imgs/1.png', img * 255)
                print(label_data[0].shape)
                img = label_data[0][0, :, :, :].cpu().detach().numpy()
                img = img.transpose(1, 2, 0)
                cv2.imwrite('/home2/raozhibo/Documents/Programs/RSStereo/Tmp/imgs/2.png', img * 255)
                '''
                # mask = label_data[0] > 0
                # loss = ((output_data[0][mask] - label_data[0][mask])) ** 2
                # loss = loss.mean()
                return [output_data[0]]

            gt_left = label_data[0]
            mask = (gt_left < args.startDisp + args.dispNum) & (gt_left > args.startDisp)
            gt_left = gt_left.unsqueeze(1)
            gt_flow = torch.cat([gt_left, gt_left * 0], dim=1)
            loss = lf.sequence_loss(output_data, gt_flow, mask, gamma=0.8)
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
            model.load_state_dict(checkpoint['model_0'], strict = False)
            jf.log.info("Model loaded successfully_add")
            return True
        # print(checkpoint['model_0'])
        # checkpoint['model_0']['pos_embed']
        #
        #state_dict = model.state_dict()
        # print(checkpoint['model_0']['feature_extraction']['pos_embed'])

        checkpoint['model_0']['module.feature_extraction.pos_embed'] = self.interpolate_pos_embed(
            checkpoint['model_0']['module.feature_extraction.pos_embed'],
            448 * 448 / 16 / 16)
        checkpoint['model_0']['module.feature_extraction.decoder_pos_embed'] = self.interpolate_pos_embed(
            checkpoint['model_0']['module.feature_extraction.decoder_pos_embed'],
            448 * 448 / 16 / 16)
        model.load_state_dict(checkpoint['model_0'], strict = True)
        jf.log.info("Model loaded successfully_add")
        return True

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

    @staticmethod
    def interpolate_pos_embed(pos_embed_checkpoint, num_patches) -> None:
        embedding_size, num_extra_tokens = pos_embed_checkpoint.shape[-1], 1
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            return torch.cat((extra_tokens, pos_tokens), dim=1)
        return pos_embed_checkpoint
