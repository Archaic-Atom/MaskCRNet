[![Use the JackFramework Demo](https://github.com/Archaic-Atom/FrameworkTemplate/actions/workflows/build_env.yml/badge.svg?event=push)](https://github.com/Archaic-Atom/FrameworkTemplate/actions/workflows/build_env.yml)
![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)
![Pytorch 1.7](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)
![cuDnn 7.3.6](https://img.shields.io/badge/cudnn-7.3.6-green.svg?style=plastic)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg?style=plastic)


# Cascaded Recurrent Networks with Masked Representation Learning for Stereo Matching of High-Resolution Satellite Images

## Project Overview
This project presents Masked Cascaded Recurrent Networks (MaskCRNet), a method for stereo matching of high-resolution satellite images. It employs masked representation learning to enhance feature extraction and uses cascaded recurrent modules to improve robustness against imperfect rectification, achieving accurate stereo matching for high-resolution satellite images.

## Key Contributions

- **Masked Representation Learning Pre-training Strategy**: Addresses challenges in remote sensing stereo datasets by improving data utilization and feature representation on small datasets.
- **Improved Correlation Computation**: Based on self-attention, cross-attention, and deformable convolutions, it handles imperfect rectification to enhance performance.
- **State-of-the-Art Performance**: Achieves state-of-the-art results on the US3D and WHU-Stereo datasets.

## Code Structure

```
MaskCRNet
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

## Dataset Preparation
1. US3D Dataset: Download from the US3D official website and organize according to the dataset's README.
2. WHU-Stereo Dataset: Download from the WHU-Stereo GitHub page and organize according to the dataset's README.

## Environment Dependencies

Ensure you have the following Python libraries installed:

- torch
- torchvision
- numpy
- JackFramework 
- DatasetHandler

## Training the Model
1. Get the Training list or Testing list （You need rewrite the code by your path, and my related demo code can be found in Source/Tools/genrate_**_traning_path.py）
```
$ ./Scripts/GenPath.sh
```


2. Run the program, like:
```
$ ./Scripts/start_debug_stereo_net.sh
```

## Testing the Model

1. Run the program, like:
```
$ ./Scripts/start_test_stereo_net.sh
```

## Citation
If you use this code or method, please cite the following paper:
```
@article{rao2024cascaded,  
  title={Cascaded Recurrent Networks with Masked Representation Learning for Stereo Matching of High-Resolution Satellite Images},  
  author={Rao, Zhibo and Li, Xing and Xiong, Bangshu and Dai, Yuchao and Shen, Zhelun and Li, Hangbiao and Lou, Yue},  
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},  
  year={2024},  
  url={https://github.com/Archaic-Atom/MaskCRNet}  
}
```

## Contact Us
For any questions or suggestions, please contact us at:

- Email: raoxi36@foxmail.com

Thank you for using our code!