# -*- coding: utf-8 -*-
import torch
# import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import JackFramework as jf
# import UserModelImplementation.user_define as user_def
import math
from .Networks.stereo_matching_model import StereoMatching


class StereoTInterface(jf.UserTemplate.ModelHandlerTemplate):
    """docstring for RSStereoInterface"""
    MODEL_ID = 0
    LEFT_IMG_ID, RIGHT_IMG_ID, DISP_IMG_ID = 0, 1, 2
    IMG_ID, MASK_IMG_ID, RANDOM_SAMPLE_LIST_ID = 0, 1, 2

    def __init__(self, args: object) -> object:
        super().__init__(args)
        self.__args = args

    @staticmethod
    def lr_lambda(epoch: int) -> float:
        warmup_epochs = 40
        cos_epoch = 1000
        return (epoch / warmup_epochs
                if epoch < warmup_epochs
                else 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / cos_epoch)))

    def get_model(self) -> list:
        args = self.__args
        model = StereoMatching((args.imgHeight, args.imgWidth), 3, args.startDisp, args.dispNum)

        return [model]

    def optimizer(self, model: list, lr: float) -> list:
        args = self.__args
        opt = torch.optim.AdamW(model[0].parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
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
        if self.MODEL_ID == model_id:
            outputs = jf.Tools.convert2list(model(input_data[self.LEFT_IMG_ID],
                                                  input_data[self.RIGHT_IMG_ID]))
        return outputs

    def accuracy(self, output_data: list, label_data: list, model_id: int) -> list:
        # return acc
        args, res = self.__args, []
        if self.MODEL_ID == model_id:

            gt_left = label_data[0]
            mask = (gt_left < args.startDisp + args.dispNum) & (gt_left > args.startDisp)
            for idx, disp in enumerate(output_data):
                if len(disp.shape) == 3 and idx > len(output_data) - 3:
                    acc, mae = jf.acc.SMAccuracy.d_1(disp, gt_left * mask, invalid_value=0)
                    res.extend((acc[1], mae))
        return res

    def loss(self, output_data: list, label_data: list, model_id: int) -> list:
        if self.MODEL_ID == model_id:
            gt_left = label_data[0]
            args = self.__args
            mask = (gt_left < args.startDisp + args.dispNum) & (gt_left > args.startDisp)
            loss = F.smooth_l1_loss(output_data[0][mask], gt_left[mask])
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
            return self._extracted_from_load_model_5(model, checkpoint, False)
        # print(checkpoint['model_0'])
        # checkpoint['model_0']['pos_embed']
        #
        # state_dict = model.state_dict()
        # print(checkpoint['model_0']['feature_extraction']['pos_embed'])

        checkpoint['model_0']['module.feature_extraction.pos_embed'] = self.interpolate_pos_embed(
            checkpoint['model_0']['module.feature_extraction.pos_embed'],
            448 * 448 / 16 / 16)
        checkpoint['model_0']['module.feature_extraction.decoder_pos_embed'] = self.interpolate_pos_embed(
            checkpoint['model_0']['module.feature_extraction.decoder_pos_embed'],
            448 * 448 / 16 / 16)
        return self._extracted_from_load_model_5(model, checkpoint, True)

    # TODO Rename this here and in `load_model`
    def _extracted_from_load_model_5(self, model, checkpoint, strict):
        model.load_state_dict(checkpoint['model_0'], strict=strict)
        jf.log.info("Model loaded successfully_add")
        return True

    # Optional
    def load_opt(self, opt: object, checkpoint: dict, model_id: int) -> bool:
        args = self.__args
        return not args.pre_train_opt

    # Optional
    def save_model(self, epoch: int, model_list: list, opt_list: list) -> dict:
        # return None
        return None

    @ staticmethod
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
