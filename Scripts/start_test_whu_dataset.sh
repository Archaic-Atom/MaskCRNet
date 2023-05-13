#!/bin/bash
test_gpus_id=7
eva_gpus_id=7
test_list_path='./Datasets/whu_stereo_testing_list.csv'
evalution_format='training'

CUDA_VISIBLE_DEVICES=${eva_gpus_id} python  Source/main.py \
                        --mode test \
                        --batchSize 1 \
                        --gpu 1 \
                        --trainListPath ${test_list_path} \
                        --imgWidth 1024 \
                        --imgHeight 1024 \
                        --dataloaderNum 16 \
                        --maxEpochs 45 \
                        --imgNum 4370 \
                        --sampleNum 1 \
                        --lr 0.001 \
                        --port 8808 \
                        --log ./TestLog/ \
                        --dist False \
                        --pre_train_opt false \
                        --modelName SwinStereo \
                        --outputDir ./TestResult/ \
                        --modelDir ./Checkpoint/ \
                        --dataset whu 

CUDA_VISIBLE_DEVICES=${eva_gpus_id} python ./Source/Tools/evalution_stereo_net.py --gt_list_path ${test_list_path}