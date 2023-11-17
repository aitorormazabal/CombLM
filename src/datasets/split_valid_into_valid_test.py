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
from datasets import load_dataset, disable_caching, DatasetDict, load_from_disk
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
        "--validation_dataset_path", type=str, required=True, help="Prepared validation dataset path"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Where to save the split dataset"
    )

    args = parser.parse_args()
    return args



def main():
    disable_caching()
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    dataset = load_from_disk(args.validation_dataset_path)
    #Split valid into valid and test
    split = dataset.train_test_split(test_size=0.666)
    valid1_dataset = split["train"]
    valid2_test_dataset = split["test"]
    split = valid2_test_dataset.train_test_split(test_size=0.5)
    valid2_dataset = split["train"]
    
    test_dataset = split["test"]

    valid1_valid2_test = DatasetDict({"validation1": valid1_dataset, "validation2": valid2_dataset, "test": test_dataset})
    logging.info(f"Original size of validation set: {len(dataset)}")
    logging.info(f"Size of split validation1 set: {len(valid1_dataset)}")
    logging.info(f"Size of split validation2 set: {len(valid2_dataset)}")
    logging.info(f"Size of split test set: {len(test_dataset)}")

    valid1_valid2_test.save_to_disk( args.output_dir)

    
    

if __name__ == "__main__":
    main()
