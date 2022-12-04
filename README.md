# Template-jf
[![Use the JackFramework Demo](https://github.com/Archaic-Atom/FrameworkTemplate/actions/workflows/build_env.yml/badge.svg?event=push)](https://github.com/Archaic-Atom/FrameworkTemplate/actions/workflows/build_env.yml)
![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)
![Pytorch 1.7](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)
![cuDnn 7.3.6](https://img.shields.io/badge/cudnn-7.3.6-green.svg?style=plastic)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg?style=plastic)

>This is template project for JackFramework (https://github.com/Archaic-Atom/JackFramework). **It is used to rapidly build the model, without caring about the training process (such as DDP or DP, Tensorboard, et al.)**

Document：https://www.wolai.com/archaic-atom/rqKJVi7M1x44mPT8CdM1TL

Demo Project: https://github.com/Archaic-Atom/Demo-jf

---
### Software Environment
1. OS Environment
```
os >= linux 16.04
cudaToolKit == 10.1
cudnn == 7.3.6
```

2. Python Environment (We provide the whole env in )
```
python >= 3.8.5
pythorch >= 1.15.0
numpy >= 1.14.5
opencv >= 3.4.0
PIL >= 5.1.0
```
---
### Hardware Environment
The framework only can be used in GPUs.

### Train the model by running:
0. Install the JackFramework lib from Github (https://github.com/Archaic-Atom/JackFramework)
```
$ cd JackFramework/
$ ./install.sh
```

1. Get the Training list or Testing list （You need rewrite the code by your path, and my related demo code can be found in Source/Tools/genrate_**_traning_path.py）
```
$ ./GenPath.sh
```
Please check the path. The source code in Source/Tools.

2. Implement the model's interface and dataloader's interface of JackFramework in Source/UserModelImplementation/Models/your_model/inference.py and Source/UserModelImplementation/Dataloaders/your_dataloader.py.

The template of model is shown in follows:
```python
# -*- coding: utf-8 -*-
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

import JackFramework as jf
# import UserModelImplementation.user_define as user_def


class YourModelInterface(jf.UserTemplate.ModelHandlerTemplate):
    """docstring for DeepLabV3Plus"""

    def __init__(self, args: object) -> object:
        super().__init__(args)
        self.__args = args

    def get_model(self) -> list:
        # args = self.__args
        # return model
        return []

    def optimizer(self, model: list, lr: float) -> list:
        # args = self.__args
        # return opt and sch
        return [], []

    def lr_scheduler(self, sch: object, ave_loss: list, sch_id: int) -> None:
        # how to do schenduler
        pass

    def inference(self, model: list, input_data: list, model_id: int) -> list:
        # args = self.__args
        # return output
        return []

    def accuary(self, output_data: list, label_data: list, model_id: int) -> list:
        # return acc
        # args = self.__args
        return []

    def loss(self, output_data: list, label_data: list, model_id: int) -> list:
        # return loss
        # args = self.__args
        return []

    # Optional
    def pretreatment(self, epoch: int, rank: object) -> None:
        # do something before training epoch
        pass

    # Optional
    def postprocess(self, epoch: int, rank: object,
                    ave_tower_loss: list, ave_tower_acc: list) -> None:
        # do something after training epoch
        pass

    # Optional
    def load_model(self, model: object, checkpoint: dict, model_id: int) -> bool:
        # return False
        return False

    # Optional
    def load_opt(self, opt: object, checkpoint: dict, model_id: int) -> bool:
        # return False
        return False

    # Optional
    def save_model(self, epoch: int, model_list: list, opt_list: list) -> dict:
        # return None
        return None

```

The template of Dataloader is shown in follows:
```python
# -*- coding: utf-8 -*-
import time
import JackFramework as jf
# import UserModelImplementation.user_define as user_def


class YourDataloader(jf.UserTemplate.DataHandlerTemplate):
    """docstring for DataHandlerTemplate"""

    def __init__(self, args: object) -> object:
        super().__init__(args)
        self.__args = args
        self.__result_str = jf.ResultStr()
        self.__train_dataset = None
        self.__val_dataset = None
        self.__imgs_num = 0
        self.__start_time = 0

    def get_train_dataset(self, path: str, is_training: bool = True) -> object:
        # args = self.__args
        # return dataset
        return None

    def get_val_dataset(self, path: str) -> object:
        # return dataset
        # args = self.__args
        # return dataset
        return None

    def split_data(self, batch_data: tuple, is_training: bool) -> list:
        self.__start_time = time.time()
        if is_training:
            # return input_data_list, label_data_list
            return [], []
            # return input_data, supplement
        return [], []

    def show_train_result(self, epoch: int, loss:
                          list, acc: list,
                          duration: float) -> None:
        assert len(loss) == len(acc)  # same model number
        info_str = self.__result_str.training_result_str(epoch, loss[0], acc[0], duration, True)
        jf.log.info(info_str)

    def show_val_result(self, epoch: int, loss:
                        list, acc: list,
                        duration: float) -> None:
        assert len(loss) == len(acc)  # same model number
        info_str = self.__result_str.training_result_str(epoch, loss[0], acc[0], duration, False)
        jf.log.info(info_str)

    def save_result(self, output_data: list, supplement: list,
                    img_id: int, model_id: int) -> None:
        assert self.__train_dataset is not None
        # args = self.__args
        # save method
        pass

    def show_intermediate_result(self, epoch: int,
                                 loss: list, acc: list) -> str:
        assert len(loss) == len(acc)  # same model number
        return self.__result_str.training_intermediate_result(epoch, loss[0], acc[0])


```

you must implement the related class for using JackFramework, the demo can be find in Source/UserModelImplementation/Models/Your_Model/inference.py or Source/UserModelImplementation/Dataloaders/your_dataloader.py. Or you can find the other demo in Demo project.

Next, you need implement the interface file Source/user_interface.py (you can add some parameters in user\_parser function of this file ), as shown in follows:
```python
# -*- coding: utf-8 -*-
import argparse
import JackFramework as jf
# import UserModelImplementation.user_define as user_def

# model and dataloader
from UserModelImplementation import Models
from UserModelImplementation import Dataloaders


class UserInterface(jf.UserTemplate.NetWorkInferenceTemplate):
    """docstring for UserInterface"""

    def __init__(self) -> object:
        super().__init__()

    def inference(self, args: object) -> object:
        dataloader = Dataloaders.dataloaders_zoo(args, args.dataset)
        model = Models.model_zoo(args, args.modelName)
        return model, dataloader

    def user_parser(self, parser: object) -> object:
        # parser.add_argument('--startDisp', type=int, default=user_def.START_DISP,
        #                    help='start disparity')
        # return parser
        return None

    @staticmethod
    def __str2bool(arg: str) -> bool:
        if arg.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
```

Finally, you need pass this object to JackFramework, as shown in follows:
```python
# -*coding: utf-8 -*-
import JackFramework as jf
from UserModelImplementation.user_interface import UserInterface


def main()->None:
    app = jf.Application(UserInterface(), "Stereo Matching Models")
    app.start()


# execute the main function
if __name__ == "__main__":
    main()

```

3. Run the program, like:
```
$ ./Scripts/start_debug_stereo_net.sh
```
---
### File Structure
```
Template-jf
├── Datasets # Get it by ./generate_path.sh, you need build folder
│   ├── dataset_example_training_list.csv
│   └── ...
├── Scripts # Get it by ./generate_path.sh, you need build folder
│   ├── clean.sh         # clean the project
│   ├── generate_path.sh # generate the tranining or testing list like kitti2015_val_list
│   ├── start_train_dataset_model.sh # start training command
│   └── ...
├── Source # source code
│   ├── UserModelImplementation
│   │   ├── Models            # any models in this folder
│   │   ├── Dataloaders       # any dataloaders in this folder
│   │   ├── user_define.py    # any global variable in this fi
│   │   └── user_interface.py # to use model and Dataloader
│   ├── Tools # put some tools in this folder
│   ├── main.py
│   └── ...
├── LICENSE
└── README.md
```
---
### Update log
#### 2021-05-29
1. Add the depth for transformer;
2. Fork the JackFramework to a new project;
3. Remove the JackFramework from this project.

#### 2021-04-08
1. Add the stereo;
2. Add transformer.

#### 2021-01-13
1. Fork a new prject (based on pythorch);
2. Use a new code style;
3. Build the frameworks for pythorch;
4. Write ReadMe
