# -*- coding: utf-8 -*
import JackFramework as jf

from .Your_Model.inference import YourModelInterface


def model_zoo(args: object, name: str) -> object:
    for case in jf.Switch(name):
        if case('YourMode'):
            jf.log.info("Enter the YourMode model")
            model = YourModelInterface(args)
            break
        if case(''):
            model = None
            jf.log.error("The model's name is error!!!")
    return model
