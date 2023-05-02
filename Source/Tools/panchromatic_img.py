# -*- coding: utf-8 -*
from matplotlib import pyplot as plt
import tifffile
import cv2
import numpy as np


def show_panchromatic_img(path: str) -> None:
    img = tifffile.imread(path)
    img = (img - img.min()) / (img.max() - img.min())
    img = np.array(img)
    cv2.imshow('1', img)
    cv2.waitKey(0)


def main() -> None:
    path = '/Users/rhc/Downloads/Dataset/experimental data/with ground truth/train/right/YD_right_147.tiff'
    show_panchromatic_img(path)


if __name__ == "__main__":
    main()
