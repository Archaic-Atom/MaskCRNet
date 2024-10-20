# -*- coding: utf-8 -*
import tifffile
import cv2
import numpy as np


def vis_disp(img: np.array) -> np.array:
    print(img.min(), img.max())
    img = (img - img.min()) / (img.max() - img.min()) * 255
    img = img.astype(np.uint8)
    img = cv2.convertScaleAbs(img, alpha=1.0)
    img = cv2.applyColorMap(img, cv2.COLORMAP_DEEPGREEN)
    return img


def main() -> None:
    # img = tifffile.imread('/Users/rhc/WorkSpace/Programs/RSStereo/ResultImg/000017_10.tiff')
    name = 'OMA_285_031_032_'

    img = tifffile.imread('/Users/rhc/Downloads/DFC2019_track2_test/Test-Track2/'
                          + name + 'LEFT_RGB.tif')
    disp_img = tifffile.imread('/Users/rhc/WorkSpace/Programs/RSStereo/Submission/us3d/'
                               + name + 'LEFT_DSP.tif')

    disp_img = vis_disp(disp_img)

    cv2.imwrite('/Users/rhc/Documents/Scholar/Ph.D./MyPaper/019_MaskRemoteSensing/MaskRemoteSensing/img/result/us3d/us3d_'
                + name + 'rgb.png', img)
    cv2.imwrite('/Users/rhc/Documents/Scholar/Ph.D./MyPaper/019_MaskRemoteSensing/MaskRemoteSensing/img/result/us3d/us3d_'
                + name + 'dsp.png', disp_img)


if __name__ == '__main__':
    main()
