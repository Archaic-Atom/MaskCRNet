#!/bin/bash
# parameters
tensorboard_port=6234
dist_port=8809
tensorboard_folder='./log/'
echo "The tensorboard_port:" ${tensorboard_port}
echo "The dist_port:" ${dist_port}

# command
# delete the previous tensorboard files
if [ -d "${tensorboard_folder}" ]; then
    rm -r ${tensorboard_folder}
fi

echo "Begin to train the model!"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python -u Source/main.py \
                        --batchSize 1 \
                        --gpu 8 \
                        --trainListPath ./Datasets/whu_stereo_training_list.csv \
                        --valListPath ./Datasets/whu_stereo_testing_list.csv \
                        --imgWidth 448 \
                        --imgHeight 448 \
                        --dataloaderNum 12 \
                        --maxEpochs 10000 \
                        --imgNum 1220 \
                        --valImgNum 450 \
                        --sampleNum 1 \
                        --log ${tensorboard_folder} \
                        --lr 0.00004 \
                        --dist true \
                        --startDisp -128 \
                        --dispNum 192 \
                        --modelDir ./Checkpoint/ \
                        --modelName SwinStereo \
                        --port ${dist_port} \
                        --auto_save_num 1 \
                        --lr_scheduler false \
                        --pre_train_opt false \
                        --dataset whu > TrainRun.log 2>&1 &
echo "You can use the command (>> tail -f TrainRun.log) to watch the training process!"

echo "Start the tensorboard at port:" ${tensorboard_port}
nohup tensorboard --logdir ${tensorboard_folder} --port ${tensorboard_port} \
                        --bind_all --load_fast=false > Tensorboard.log 2>&1 &
echo "All processes have started!"

echo "Begin to watch TrainRun.log file!"
tail -f TrainRun.log
