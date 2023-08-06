# -*- coding: utf-8 -*-
import tifffile
import numpy as np
import cv2


def main() -> None:
    left_img = tifffile.imread('/Users/rhc/WorkSpace/Programs/RSStereo/Tmp/example/JAX_160_001_015_LEFT_RGB.tif')
    right_img = tifffile.imread('/Users/rhc/WorkSpace/Programs/RSStereo/Tmp/example/QC_left_689.tiff')
    disp_img = tifffile.imread('/Users/rhc/WorkSpace/Programs/RSStereo/Tmp/example/QC_left_689.tiff')

    disp_img = np.array(disp_img) * 70
    disp_img = (disp_img - disp_img.min()) / (disp_img.max() - disp_img.min())
    print(disp_img.shape)

    crop_h, crop_w = 384, 384
    org_x, org_y = 512, 512
    img = left_img[org_y:org_y + crop_h, org_x:org_x + crop_w, :]
    # print(img)

    #rgb_img = cv2.applyColorMap(cv2.convertScaleAbs(img, alpha=1), cv2.COLORMAP_JET)

    cv2.imwrite('/Users/rhc/WorkSpace/Programs/RSStereo/Tmp/example/5.png', img)

    patch_size = 128
    for j in range(crop_h // patch_size):
        for i in range(crop_w // patch_size):
            patch = img[j * patch_size:j * patch_size + patch_size,
                        i * patch_size:i * patch_size + patch_size, :]
            cv2.imwrite('/Users/rhc/WorkSpace/Programs/RSStereo/Tmp/example/' +
                        str(i) + '_' + str(j) + '.png', patch)


if __name__ == '__main__':
    main()
