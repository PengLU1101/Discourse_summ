#!/usr/bin/env bash
#export DATA_PATH='/u/lupeng/Project/dataset'
export DATA_PATH='/data/rali5/Tmp/lupeng/data'
export SAVE_PATH='/u/lupeng/Project/code/Discourse_summ/saved'
export CODE_PATH='/u/lupeng/Project/code/Discourse_summ/codes'

#export DATA_PATH='/content'
#export SAVE_PATH='/content/saved'
#export CODE_PATH='/content/Tree_thought/codes'

python -u -c 'import torch; print(torch.__version__)'

MODE=$1
MODEL=$2
SAVEID=$3
DATASET=$4
BATCH_SIZE=$5
HIDDEN_DIM=$6
LEARNING_RATE=${7}
MAX_STEPS=${8}
WARM_UP_STEPS=${9}
PARSER_TYPE=${10}
PREDICTOR_TYPE=${11}
MACHINE=${12}
DROPOUT=${13}
TUNE_STOP=${14}
NUM_LAYER=${15}
OPTIM=${16}



FULL_DATA_PATH=$DATA_PATH/$DATASET
SAVE=$SAVE_PATH/$MACHINE/"$MODEL"_"$DATASET"_"$SAVEID"

if [[ $MODE == "train" ]]
then
echo "Start Training......"

python -u $CODE_PATH/run.py --do_train \
    --save_path $SAVE \
    --data_path $FULL_DATA_PATH \
    --encoder_type $MODEL \
    --score_type_parser $PARSER_TYPE \
    --score_type_predictor $PREDICTOR_TYPE\
    -b $BATCH_SIZE -md $HIDDEN_DIM -ed 128 --dropout $DROPOUT --n_layer $NUM_LAYER\
    -lr $LEARNING_RATE --max_steps $MAX_STEPS --warm_up_steps $WARM_UP_STEPS --optim $OPTIM\
    --quick_thought_step $TUNE_STOP
    #${14} ${15} ${16} ${17} ${18} ${19} ${20}

elif [ $MODE == "resume" ]
then
echo "Resume training..."

python -u $CODE_PATH/run.py --do_train \
    --data_path $FULL_DATA_PATH \
    -init $SAVE \
    #--encoder_type $MODEL \
    #--score_type_parser $PARSER_TYPE \
    #--score_type_predictor $PREDICTOR_TYPE\
    #-b $BATCH_SIZE -md $HIDDEN_DIM -ed $EMBEDDING_DIM \
    #-lr $LEARNING_RATE --max_steps $MAX_STEPS --warm_up_steps $WARM_UP_STEPS\



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