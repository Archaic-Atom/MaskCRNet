# -*- coding: utf-8 -*
import tifffile
import cv2
import numpy as np


def main() -> None:
    img = tifffile.imread('/Users/rhc/WorkSpace/Programs/RSStereo/Tmp/example/JAX_160_001_015_LEFT_RGB.tif')
    #img = img - img.min()
    #img = ((img / img.max()) * 255.0).astype(np.uint8)
    #img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.imwrite('/Users/rhc/WorkSpace/Programs/RSStereo/Tmp/example/8.png', img)


if __name__ == '__main__':
    main()
