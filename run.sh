#!/usr/bin/env bash
export DATA_DIR='/data/rali5/Tmp/lupeng/data/cnn-dailymail'
export SAVE_DIR='/u/lupeng/Project/code/Discourse_summ/saved'

python -u -c 'import torch; print(torch.__version__)'

MODE=$1
SCORE=$2
SAVEID=$3
DATASET=$4



if [[ $MODE == "train" ]]
then

echo "test success!"

else
echo "Unknown issue"

fi