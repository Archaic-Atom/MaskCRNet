#!/bin/bash
# parameters
tensorboard_port=6006
dist_port=8800
tensorboard_folder='./log/'
echo "The tensorboard_port:" ${tensorboard_port}
echo "The dist_port:" ${dist_port}

# command
# delete the previous tensorboard files
if [ -d "${tensorboard_folder}" ]; then
    rm -r ${tensorboard_folder}
fi

echo "Begin to train the model!"
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -u Source/main.py \
                        --mode train \
                        --batchSize 2 \
                        --gpu 4 \
                        --trainListPath ./Datasets/scene_flow_training_list.csv \
                        --imgWidth 512 \
                        --imgHeight 256 \
                        --dataloaderNum 24 \
                        --maxEpochs 200 \
                        --imgNum 35454 \
                        --sampleNum 1 \
                        --log ${tensorboard_folder} \
                        --lr 0.001 \
                        --dist True \
                        --modelName your_model \
                        --port ${dist_port} \
                        --dataset dataset_name > TrainRun.log 2>&1 &
echo "You can use the command (>> tail -f TrainRun.log) to watch the training process!"

echo "Start the tensorboard at port:" ${tensorboard_port}
nohup tensorboard --logdir ${tensorboard_folder} --port ${tensorboard_port} \
                        --bind_all --load_fast=false > Tensorboard.log 2>&1 &
echo "All processes have started!"

echo "Begin to watch TrainRun.log file!"
tail -f TrainRun.log