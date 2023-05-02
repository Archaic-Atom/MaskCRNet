# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import JackFramework as jf


import os
import re
import argparse
from PIL import Image
import numpy as np
import pandas as pd
import tifffile


DEPTH_DIVIDING = 256.0
ACC_EPSILON = 1e-9


def read_label_list(list_path: str):
    input_dataframe = pd.read_csv(list_path)
    gt_dsp_path = input_dataframe["gt_disp"].values
    return gt_dsp_path


def d_1(res: torch.tensor, gt: torch.tensor, start_threshold: int = 2,
        threshold_num: int = 4, relted_error: float = 0.05,
        invaild_value: int = 0, max_disp: int = 192) -> torch.tensor:
    mask = (gt != invaild_value) & (gt < max_disp)
    mask.detach_()
    acc_res = []
    with torch.no_grad():
        total_num = mask.int().sum()
        error = torch.abs(res[mask] - gt[mask])
        related_threshold = gt[mask] * relted_error
        for i in range(threshold_num):
            threshold = start_threshold + i
            # acc = (error > threshold) & (error > related_threshold)
            acc = (error > threshold)
            acc_num = acc.int().sum()
            error_rate = acc_num / (total_num + ACC_EPSILON)
            acc_res.append(error_rate)
        mae = error.sum() / (total_num + ACC_EPSILON)
    return acc_res, mae


def read_pfm(filename: str) -> tuple:
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


class Evalution(nn.Module):
    """docstring for Evalution"""

    def __init__(self, start_threshold: int = 2,
                 threshold_num: int = 4, relted_error: float = 0.05,
                 invaild_value: int = 0):
        super().__init__()
        self._start_threshold = start_threshold
        self._threshold_num = threshold_num
        self._relted_error = relted_error
        self._invaild_value = invaild_value

    def forward(self, res, gt):
        return d_1(res, gt, self._start_threshold,
                   self._threshold_num, self._relted_error,
                   self._invaild_value)


def read_dsp(path: str) -> np.array:
    file_type = os.path.splitext(path)[-1]
    if file_type == ".png":
        img = np.array(Image.open(path), dtype=np.float32) / float(DEPTH_DIVIDING)
    elif file_type == '.pfm':
        img, _ = read_pfm(path)
    elif file_type == '.tiff':
        img = np.array(tifffile.imread(path))
    else:
        print('gt file name error!')
    return img


def get_data(img_path: str, gt_path: str) -> np.array:
    img = read_dsp(img_path)
    img_gt = read_dsp(gt_path)
    return img, img_gt


def data2cuda(img: np.array, img_gt: np.array) -> torch.tensor:
    img = torch.from_numpy(img).float()
    img_gt = torch.from_numpy(img_gt.copy()).float()

    img = Variable(img, requires_grad=False)
    img_gt = Variable(img_gt, requires_grad=False)
    return img, img_gt


def print_total(total: np.array, err_total: int,
                total_img_num: int, threshold_num: int) -> str:
    offset = 1
    str_data = 'total '
    for j in range(threshold_num):
        d1_str = '%dpx: %f ' % (j + offset, total[j] / total_img_num)
        str_data = str_data + d1_str
    str_data = str_data + 'mae: %f' % (err_total / total_img_num)
    print(str_data)
    return str_data


def cal_total(id_num: int, total: np.array, err_total: int,
              acc_res: torch.tensor, mae: torch.tensor,
              threshold_num: int) -> None:
    str_data = str(id_num) + ' '
    for i in range(threshold_num):
        d1_res = acc_res[i].cpu()
        d1_res = d1_res.detach().numpy()
        total[i] = total[i] + d1_res
        str_data = str_data + str(d1_res) + ' '

    mae_res = mae.cpu()
    mae_res = mae_res.detach().numpy()
    err_total = err_total + mae_res

    str_data = str_data + str(mae_res)
    print(str_data)

    return total, err_total


def parser_args() -> object:
    parser = argparse.ArgumentParser(
        description="The Evalution process")
    parser.add_argument('--img_path_format', type=str,
                        default='./ResultImg/%06d_10.png',
                        help='img_path_format')
    parser.add_argument('--gt_list_path', type=str,
                        default='./Datasets/kitti2015_training_list.csv',
                        help='gt list path')
    parser.add_argument('--epoch', type=int,
                        default=0, help='epoch num')
    parser.add_argument('--output_path', type=str,
                        default='./Result/test_output.txt',
                        help='output file')
    parser.add_argument('--invaild_value', type=int,
                        default=0,
                        help='invaild value')
    args = parser.parse_args()
    return args


def evalution(epoch: int, img_path_format: str, gt_list_path: str,
              output_path: str, invaild_value: int) -> None:
    gt_dsp_path = read_label_list(gt_list_path)
    total_img_num = len(gt_dsp_path)

    start_threshold = 1
    threshold_num = 5

    # Variable
    total = np.zeros(threshold_num)
    err_total = 0

    # push model to CUDA
    eval_model = Evalution(start_threshold=start_threshold,
                           threshold_num=threshold_num,
                           invaild_value=invaild_value)
    eval_model = torch.nn.DataParallel(eval_model).cuda()

    for i in range(total_img_num):
        img_path = img_path_format % (i)
        gt_path = gt_dsp_path[i]

        img, img_gt = get_data(img_path, gt_path)
        img, img_gt = data2cuda(img, img_gt)

        acc_res, mae = eval_model(img, img_gt)
        total, err_total = cal_total(i, total,
                                     err_total, acc_res,
                                     mae, threshold_num)

    data_str = print_total(total, err_total, total_img_num, threshold_num)
    fd_file = jf.FileHandler.open_file(output_path, True)
    jf.FileHandler.write_file(fd_file, f'epoch: {epoch}, {data_str}')


def main():
    args = parser_args()
    evalution(args.epoch, args.img_path_format, args.gt_list_path,
              args.output_path, args.invaild_value)
    # setting
    # img_path_format = args.img_path_format
    # gt_list_path = './Datasets/scene_flow_testing_list.csv'
    # gt_list_path = './Datasets/kitti2012_training_list.csv'
    # gt_list_path = args.gt_list_path


if __name__ == '__main__':
    main()
