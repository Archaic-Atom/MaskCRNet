# -*- coding: utf-8 -*-
import os
import torch
import pandas as pd
import numpy as np
import re
import argparse
from PIL import Image
import tifffile
import JackFramework as jf

DEPTH_DIVIDING = 256.0
ACC_EPSILON = 1e-9


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


def read_label_list(list_path: str) -> list:
    input_dataframe = pd.read_csv(list_path)
    return input_dataframe["gt_disp"].values


def read_disp(path: str) -> np.array:
    file_type = os.path.splitext(path)[-1]
    if file_type == ".png":
        img = np.array(Image.open(path), dtype=np.float32) / float(DEPTH_DIVIDING)
    elif file_type == '.pfm':
        img, _ = read_pfm(path)
    elif file_type == '.tiff':
        img = np.array(tifffile.imread(path))
    elif file_type == '.tif':
        img = np.array(tifffile.imread(path))
    else:
        print('gt file name error!')
    return img


def parser_args() -> object:
    parser = argparse.ArgumentParser(
        description="The Evalution process")
    parser.add_argument('--list_path', type=str,
                        default='./Datasets/kitti2015_training_list.csv',
                        help='list path')

    parser.add_argument('--output_path', type=str,
                        default='./Datasets/kitti2015_training_list.csv',
                        help='output path')
    return parser.parse_args()


def count_disp(list_path: str, output_path: str, start_disp: int = -128, disp_num: int = 448) -> None:
    disp_list = read_label_list(list_path)
    res_list = [0] * disp_num
    print('total:', len(disp_list))
    for idx, path in enumerate(disp_list):
        print(idx, path)
        disp = read_disp(path).astype(np.int32)
        for i in range(disp_num):
            disp_values = start_disp + i
            res = (disp_values == disp).astype(np.int32).sum()
            res_list[i] = res_list[i] + res

    fd_file = jf.FileHandler.open_file(output_path, False)
    for data in res_list:
        jf.FileHandler.write_file(fd_file, str(data))


def main() -> None:
    args = parser_args()
    count_disp(args.list_path, args.output_path)


if __name__ == '__main__':
    main()
