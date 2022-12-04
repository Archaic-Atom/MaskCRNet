#!/bin/bash
# parameters
tensorboard_port=6006
dist_port=8800

echo "The tensorboard_port:"
echo ${tensorboard_port}
echo "The dist_port:"
echo ${dist_port}

# command
echo "Start to kill process!"
echo "Start to kill tensorboard process!"
echo "We find the PID of tensorboard:" $(lsof -i:${tensorboard_port} | awk '{if(NR==2) print $2}')
kill -9 $(lsof -i:${tensorboard_port} | awk '{if(NR==2) print $2}')
echo "We have killed tensorboard process!"

echo "Start to kill traning process!"
echo "We find the PID:" $(lsof -i:${dist_port} | awk '{if(NR>=2) print $2}')
kill -9 $(lsof -i:${dist_port} | awk '{if(NR>=2) print $2}')
echo "We have killed traning process!"
echo "Finish!"