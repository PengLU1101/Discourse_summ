#!/usr/bin/env bash
export DATA_DIR='/data/rali5/Tmp/lupeng/data/cnn-dailymail'
export SAVE_DIR='/u/lupeng/Project/code/Discourse_summ/saved'

python -u -c 'import torch; print(torch.__version__)'

MODE=$1
SCORE=$2
SAVEID=$3
DATASET=$4

FULL_DATA_PATH=$DATA_PATH/$DATASET
SAVE=$SAVE_PATH/"$MODEL"_"$DATASET"_"$SAVE_ID"

#Only used in training
BATCH_SIZE=$5
HIDDEN_DIM=$6
LEARNING_RATE=${7}
MAX_STEPS=${8}
TEST_BATCH_SIZE=${9}

if [[ $MODE == "train" ]]
then
echo "Start Training......"

python -u $CODE_PATH/run.py --do_train \
    --cuda \
    --do_valid \
    --do_test \
    --data_path $FULL_DATA_PATH \
    --model $MODEL \
    -b $BATCH_SIZE -md $HIDDEN_DIM -ed $EMBEDDING_DIM \
    -lr $LEARNING_RATE --max_steps $MAX_STEPS \
    -save $SAVE --test_batch_size $TEST_BATCH_SIZE \
    ${14} ${15} ${16} ${17} ${18} ${19} ${20}

elif [ $MODE == "valid" ]
then

echo "Start Evaluation on Valid Data Set......"

python -u $CODE_PATH/run.py --do_valid --cuda -init $SAVE

elif [ $MODE == "test" ]
then

echo "Start Evaluation on Test Data Set......"

python -u $CODE_PATH/run.py --do_test --cuda -init $SAVE

else
   echo "Unknown MODE" $MODE
fi