# -*- coding: utf-8 -*-
import JackFramework as jf

from .stereo_dataloader import StereoDataloader


def dataloaders_zoo(args: object, name: str) -> object:
    for case in jf.Switch(name):
        if case('sceneflow') or case('kitti2012') or case('kitti2015')\
                or case('crestereo') or case('eth3d') or case('rob') \
                or case('middlebury') or case('US3D') or case('whu'):
            jf.log.info("Enter the StereoDataloader")
            dataloader = StereoDataloader(args)
            break
        if case(''):
            dataloader = None
            jf.log.error("The dataloader's name is error!!!")
    return dataloader
