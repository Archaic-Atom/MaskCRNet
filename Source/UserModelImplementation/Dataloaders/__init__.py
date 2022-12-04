# -*- coding: utf-8 -*-
import JackFramework as jf

from .your_dataloader import YourDataloader


def dataloaders_zoo(args: object, name: str) -> object:
    for case in jf.Switch(name):
        if case('YourDataloader'):
            jf.log.info("Enter the your dataloader")
            dataloader = YourDataloader(args)
            break
        if case(''):
            dataloader = None
            jf.log.error("The dataloader's name is error!!!")
    return dataloader
