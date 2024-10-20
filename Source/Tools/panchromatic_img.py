# -*- coding: utf-8 -*
# from matplotlib import pyplot as plt
import tifffile
import cv2
import numpy as np


def show_panchromatic_img(path: str, save_path: str) -> None:
    img = tifffile.imread(path)
    img = (img - img.min()) / (img.max() - img.min())
    img = np.array(img)
    cv2.imwrite(save_path, img * 255)


def vis_disp(img: np.array) -> np.array:
    print(img.min(), img.max())
    img = (img - img.min()) / (img.max() - img.min()) * 255
    img = img.astype(np.uint8)
    img = cv2.convertScaleAbs(img, alpha=1.0)
    img = cv2.applyColorMap(img, cv2.COLORMAP_DEEPGREEN)
    return img


def main() -> None:
    root_path = '/Users/rhc/WorkSpace/Programs/RSStereo/Tmp/result/'
    rgb_path = '1.tiff'
    rgb_png_path = '1.png'
    disp_path = '2.tiff'
    disp_png_path = '2.png'
    show_panchromatic_img(root_path + rgb_path, root_path + rgb_png_path)
    disp_img = tifffile.imread(root_path + disp_path)
    disp_img = vis_disp(disp_img)
    cv2.imwrite(root_path + disp_png_path, disp_img)


if __name__ == "__main__":
    main()
