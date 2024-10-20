# -*- coding: utf-8 -*-
import tifffile
import numpy as np
import cv2


def vis_disp(img: np.array) -> np.array:
    # print(img.min(), img.max())
    # img = (img - img.min()) / (img.max() - img.min()) * 255
    img = img.astype(np.uint8)
    img = cv2.convertScaleAbs(img, alpha=1.0)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img


def main() -> None:
    root_path = '/Users/rhc/Documents/Scholar/Ph.D./MyPaper/019_MaskRemoteSensing/MaskRemoteSensing/img/result/whu/'
    img_r = root_path + '1.drawio.png'
    img_o = root_path + '2.drawio.png'

    img_r = cv2.imread(img_r)
    img_o = cv2.imread(img_o)

    error = np.array(img_r - img_o)

    print(error.shape)
    error = np.mean(error, axis=2)
    error = vis_disp(error)

    cv2.imwrite(root_path + '3.drawio.png', error)


if __name__ == "__main__":
    main()
