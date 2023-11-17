#!/bin/bash
#SBATCH --job-name=fit_grid_500
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=140G

COMBLM_DIR=/ikerlariak/aormazabal024/PhD/CombLM-release/CombLM

export PYTHONPATH=$COMBLM_DIR/src:$PYTHONPATH
source /ikerlariak/aormazabal024/PhD/LM-Combination/trainenv-trumoi/bin/activate


#Loop through all datasets, model paths and names
model_path=$COMBLM_DIR/models/opt-1.3b_wmt21/
fit_path=$COMBLM_DIR/models/fit_combinations_wmt21_entropy/entropy_100/checkpoint_0/

save_dir=$COMBLM_DIR/result_translation/wmt21_csen_greedy
mkdir -p $save_dir
save_path=${save_dir}/validation2


l1=cs
l2=en
model_type=entropy 

prompt_file=$COMBLM_DIR/prompts/${l1}-${l2}_5shot.txt
model_1=$model_path 
model_2=facebook/opt-30b
dataset=$COMBLM_DIR/data/wmt21_10k/cs-en_test.src

translate_script=$COMBLM_DIR/src/eval/eval_combined_translation.py

set -x 
python3 $translate_script  \
    --model_1_path $model_1 \
    --model_2_path $model_2 \
    --model_type $model_type \
    --checkpoint $fit_path  \
    --source_sentences $dataset \
    --l1 $l1 \
    --l2 $l2 \
    --prompt_file $prompt_file \
    --save_dir $save_dir  \
    --beam_size 1 
