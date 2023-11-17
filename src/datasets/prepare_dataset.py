import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path
from datetime import timedelta


import datasets
import torch
from datasets import load_dataset, disable_caching, DatasetDict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version

check_min_version("4.26.0.dev0")


require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)



def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a dataset for fine-tuning")
    parser.add_argument(
        "--train_file", type=str, required=True, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, required=False, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path",
        required=True
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
        required=True
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        type=bool,
        default=False,
        help="If True, use a slow tokenizer",
    )
    parser.add_argument(
        "--no_add_special_tokens",
        type=bool,
        default=True,
        help="If True, no special tokens will be added by the tokenizer (i.e. EOS at beginning for OPT)",
    )
    parser.add_argument(
        "--text_column_name",
        type=str,
        default="text",
        help="The column name of text to input in the model (for CSV/JSON files).",
    )
    parser.add_argument(
        "--sep_token",
        type=str,
        required=True,
        help="Document separator token",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. ."
        ),
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")

    args = parser.parse_args()

    return args

def delete_dataset(dataset):
    if type(dataset) == datasets.DatasetDict:
        for k in dataset:
            delete_dataset(dataset[k])
    else:
        logging.warning(f"Deleting {len(dataset.cache_files)} cache files:")
        cached_files = [cache_file["filename"] for cache_file in dataset.cache_files]
        del dataset
        for cached_file in cached_files:
            logging.warning(f"Deleting {cached_file}")
            try:
                os.remove(cached_file)
            except OSError:
                pass


def main():
    disable_caching()
    args = parse_args()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()

    os.makedirs(args.output_dir, exist_ok=True)


    # Load the dataset
    data_files =  {}

    data_files["train"] = args.train_file

    train_ext = args.train_file.split(".")[-1]
    if train_ext=="jsonl":
        train_ext = "json"
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
        valid_ext = args.validation_file.split(".")[-1]
        if valid_ext=="jsonl":
            valid_ext = "json"
        assert train_ext==valid_ext, "Train and validation files must have the same extension"
    
    ext = train_ext 

    assert ext=="json", "Only json train files are supported"


    raw_datasets = load_dataset(ext, data_files=data_files)
    if "validation" not in raw_datasets:
        logging.warning("No validation file is provided, using a percentage of the training set")
        raw_datasets = raw_datasets["train"].train_test_split(test_size=float(args.validation_split_percentage)/100)
        raw_datasets = DatasetDict({
            "train": raw_datasets["train"],
            "validation": raw_datasets["test"],
        })
    for k in raw_datasets:
        logging.info(f"Dataset {k} has {len(raw_datasets[k])} samples")

    for k in raw_datasets:
        assert args.text_column_name in raw_datasets[k].column_names, f"Could not find {args.text_column_name} in dataset {k}"
    
        if args.text_column_name != "text":
            raw_datasets[k].rename_column(args.text_column_name, "text")

        to_remove = list(set(raw_datasets[k].column_names) - set(["text"]))
        raw_datasets[k].remove_columns(to_remove)
        

    assert "train" in raw_datasets.keys(), "Could not find training dataset"
    assert "validation" in raw_datasets.keys(), "Could not find validation dataset"

    # Tokenize the dataset
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)

    column_names = raw_datasets["train"].column_names
    assert "text" in column_names, f"Column names are {', '.join(column_names)}, but 'text' is not one of them."

    tokenized_sep = tokenizer(args.sep_token, add_special_tokens=False)["input_ids"]
    assert len(tokenized_sep) == 1, "Separator token must be a single token"

    sep_token_id = tokenized_sep[0]
    logging.info(f"Using {(args.sep_token,sep_token_id)} as document separator token")

    def tokenize_function(examples):
        examples["text"] = [args.sep_token + doc for doc in examples["text"]]
        tokenized =  tokenizer(examples["text"], add_special_tokens=not args.no_add_special_tokens)
        assert all([tokens[0]==sep_token_id for tokens in tokenized["input_ids"]]) # Check that all documents start with the separator token
        return tokenized
    

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )
    delete_dataset(raw_datasets)
    
    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logging.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
        block_size = 1024
    else:
        if args.block_size > tokenizer.model_max_length:
            logging.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        result["ids"] = list(range(len(result["labels"])))
        return result
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        desc=f"Grouping texts in chunks of {block_size}",
    )
    delete_dataset(tokenized_datasets)

    first_train = lm_datasets["train"][0]
    logging.info(f"First train example: {first_train}\n{tokenizer.decode(first_train['input_ids'])}")
    first_valid = lm_datasets["validation"][0]
    logging.info(f"First validation example: {first_valid}\n{tokenizer.decode(first_valid['input_ids'])}")


    logging.info(f"Saving processed datasets to {args.output_dir}")



    # Save the processed datasets
    lm_datasets.save_to_disk(args.output_dir)

    delete_dataset(lm_datasets)

if __name__ == "__main__":
    main()
