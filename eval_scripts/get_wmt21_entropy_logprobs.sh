#!/bin/bash
#SBATCH --job-name=fit_grid_500
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=140G

COMBLM_DIR=/ikerlariak/aormazabal024/PhD/CombLM-release/CombLM

export PYTHONPATH=$COMBLM_DIR/src:$PYTHONPATH
source /ikerlariak/aormazabal024/PhD/LM-Combination/trainenv-trumoi/bin/activate


#Loop through all datasets, model paths and names
dataset_path=$COMBLM_DIR/prepared_datasets/wmt21_10k/opt/valid_test_split/validation2
model_path=$COMBLM_DIR/models/opt-1.3b_wmt21/
fit_path=$COMBLM_DIR/models/fit_combinations_wmt21_entropy/entropy_100/checkpoint_0/

save_dir=$COMBLM_DIR/result_logprobs/wmt21_entropy
mkdir -p $save_dir
save_path=${save_dir}/validation2

set -x
python3 /ikerlariak/aormazabal024/PhD/LM-Combination/src/eval/get_logprobs.py \
    --model_1_path $model_path \
    --model_2_path "facebook/opt-30b" \
    --model_type entropy \
    --checkpoint  $fit_path \
    --eval_path ${dataset_path} \
    --output_dir ${save_path}  \
    --eval_max_samples 200 \
