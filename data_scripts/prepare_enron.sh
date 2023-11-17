#!/bin/bash

COMBLM_DIR=/ikerlariak/aormazabal024/PhD/CombLM-release/CombLM
export TMPDIR=$COMBLM_DIR/tmp
export TMP=$COMBLM_DIR/tmp

train_file=$COMBLM_DIR/data/enron_10k/train.jsonl
valid_file=$COMBLM_DIR/data/enron_10k/valid.jsonl

python3 $COMBLM_DIR/src/datasets/prepare_dataset.py \
    --train_file $train_file \
    --validation_file $valid_file \
    --tokenizer_name facebook/opt-1.3b \
    --sep_token '</s>' \
    --preprocessing_num_workers 40 \
    --output_dir $COMBLM_DIR/prepared_datasets/enron_10k/opt \

python3 /ikerlariak/aormazabal024/PhD/LM-Combination/src/datasets/split_valid_into_valid_test.py \
    --validation_datase $COMBLM_DIR/prepared_datasets/enron_10k/opt/validation \
    --output_dir $COMBLM_DIR/prepared_datasets/enron_10k/opt/valid_test_split \
