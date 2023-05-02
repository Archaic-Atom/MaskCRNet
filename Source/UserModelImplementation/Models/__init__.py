# -*- coding: utf-8 -*
import JackFramework as jf

from .RSStereo import RSStereoInterface
from .LaCGwcNet import LacGwcNetworkInterface
from .SwinStereo import SwinStereoInterface


def model_zoo(args: object, name: str) -> object:
    for case in jf.Switch(name):
        if case('RSStereo'):
            jf.log.info("Enter the RSStereo model")
            model = RSStereoInterface(args)
            break
        if case('SwinStereo'):
            jf.log.info("Enter the SwinStereo model")
            model = SwinStereoInterface(args)
            break
        if case('LacGwcNet'):
            jf.log.info("Enter the LacGwcNet model")
            model = LacGwcNetworkInterface(args)
            break
        if case(''):
            model = None
            jf.log.error("The model's name is error!!!")
    return model
