#!/bin/bash
# parameter
test_gpus_id=2,3
eva_gpus_id=0
test_list_path='./Datasets/kitti2015_training_list.csv'
evalution_format='training'

echo "test gpus id: "${test_gpus_id}
echo "the list path is: "${test_list_path}
echo "start to predict disparity map"
CUDA_VISIBLE_DEVICES=${test_gpus_id} python -u Source/main.py \
                        --mode test \
                        --batchSize 4 \
                        --gpu 4 \
                        --trainListPath ${test_list_path} \
                        --imgWidth 1536 \
                        --imgHeight 512 \
                        --dataloaderNum 16 \
                        --maxEpochs 45 \
                        --imgNum 200 \
                        --sampleNum 1 \
                        --lr 0.0001 \
                        --log ./TestLog/ \
                        --dist False \
                        --modelName model_name \
                        --outputDir ./DebugResult/ \
                        --modelDir ./Checkpoint/ \
                        --dataset dataset_name
echo "Finish!"