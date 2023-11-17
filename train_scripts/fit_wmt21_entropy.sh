#!/bin/bash
#SBATCH --job-name=fit_grid_ar
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=140G

source /ikerlariak/aormazabal024/PhD/LM-Combination/trainenv-trumoi/bin/activate


COMBLM_DIR=/ikerlariak/aormazabal024/PhD/CombLM-release/CombLM

export PYTHONPATH=$COMBLM_DIR/src:$PYTHONPATH

train_path=$COMBLM_DIR/src/train/fit_combined_model.py

dataset_path=$COMBLM_DIR/prepared_datasets/wmt21_10k/opt/valid_test_split/
model_1_path=$COMBLM_DIR/models/opt-1.3b_wmt21/

save_base_path=$COMBLM_DIR/models/fit_combinations_wmt21_entropy/

lr="2e-3"
train_bs="1024"


max_train=100
model_type=entropy

echo  DATASET IS $dataset_path
echo MODEL IS  $model_1_path

echo $model_type
echo $max_train
echo "Running command for in domain valid:"

save_path=$save_base_path/${model_type}_${max_train}

echo RUNNING $model_type $max_train  no_mixin 
python3 $train_path \
    --model_1_path $model_1_path \
    --model_2_path facebook/opt-30b \
    --train_datasets $dataset_path/validation1 \
    --validation_datasets $dataset_path/validation2 \
    --max_samples $max_train \
    --model_type $model_type \
    --max_train $max_train \
    --max_valid 500 \
    --lr $lr \
    --train_bs $train_bs \
    --save_dir $save_path \
    --n_epochs 1 \
