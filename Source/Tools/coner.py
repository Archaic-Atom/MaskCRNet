# -*- coding: utf-8 -*-
import tifffile
import cv2
import numpy as np


def draw_line(img: np.array, line_no: int) -> np.array:
    h, _, _ = img.shape

    for i in range(h // line_no):
        img[i * line_no, :, 0] = 111
        img[i * line_no, :, 1] = 127
        img[i * line_no, :, 2] = 250

    return img


def main() -> None:
    img_l = tifffile.imread('/Users/rhc/Downloads/DFC2019_track2_test/Test-Track2/JAX_160_001_015_LEFT_RGB.tif')
    img_r = tifffile.imread('/Users/rhc/Downloads/DFC2019_track2_test/Test-Track2/JAX_160_001_015_RIGHT_RGB.tif')
    img_l = draw_line(img_l, 25)
    img_r = draw_line(img_r, 25)
    # img_l = img_l[2:122, 800:920, :]
    # img_r = img_r[2:122, 800:920, :]
    img_l = img_l[2 + 45:122 - 45, 800 + 45:920 - 45, :]
    img_r = img_r[2 + 45:122 - 45, 800 + 45:920 - 45, :]
    cv2.imwrite('/Users/rhc/WorkSpace/Programs/RSStereo/Tmp/example/10.png', img_l)
    cv2.imwrite('/Users/rhc/WorkSpace/Programs/RSStereo/Tmp/example/11.png', img_r)


if __name__ == '__main__':
    main()
