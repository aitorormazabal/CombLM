#!/bin/bash
#SBATCH --job-name=ft-opt1.3b
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=140G
export PMI_SIZE=1
export OMPI_COMM_WORLD_SIZE=1
export MV2_COMM_WORLD_SIZE=1
export WORLD_SIZE=1


COMBLM_DIR=/ikerlariak/aormazabal024/PhD/CombLM-release/CombLM


source /ikerlariak/aormazabal024/PhD/LM-Combination/trainenv-trumoi/bin/activate


train_script=$COMBLM_DIR/src/train/train_model.py

train_dataset=$COMBLM_DIR/prepared_datasets/enron_10k/opt/train
valid_dataset=$COMBLM_DIR/prepared_datasets/enron_10k/opt/validation

config_file=$COMBLM_DIR/train_scripts/opt_fsdp_config.yaml



#Using FSDP
#srun accelerate launch --config_file $config_file  $train_script \

#Not using FSDP
srun accelerate launch --multi_gpu --mixed_precision bf16  --main_process_port 20655  $train_script \
    --prepared_train_dataset $train_dataset \
    --prepared_validation_dataset $valid_dataset \
    --output_dir $COMBLM_DIR/models/opt-1.3b_enron \
    --model_name_or_path facebook/opt-1.3b \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 3 \
    --gradient_accumulation_steps 15 \
    --with_tracking \
    --report_to wandb \
    --lr_scheduler constant_with_warmup \
    --num_warmup_steps 100 \
    --learning_rate 4e-5 \
    --checkpointing_steps 100 \
    --eval_log_steps 400 \
    --train_log_steps 2 \
    --max_validation_steps 500 \
    --max_train_steps 100 \

