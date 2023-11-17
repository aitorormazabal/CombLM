#File to prepare a dataset consisting of verbalizations of wmt datasets

import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path
from datetime import timedelta
import itertools


import datasets
import torch
from datasets import load_dataset, disable_caching, DatasetDict, concatenate_datasets
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
TEMPLATE_L1="$L1"
TEMPLATE_L2="$L2"
TEMPLATE_S1="$S1"
TEMPLATE_S2="$S2"

VERBALIZATION_PREFIXES = [
    """Translate the following sentences from $L1 to $L2:\n\n""",
    """Given a sentence in $L1, translate it to $L2:\n\n""",
]


language_mapping = {"eng":"English", 
                    "deu":"German",
                    "fra":"French",
                    "hin":"Hindi",
                    "rus":"Russian",
                    "ces":"Czech",}

def parse_args():
    parser = argparse.ArgumentParser(description="Verbalize a wmt translation dataset")
    parser.add_argument(
        "--l1", type=str, required=True, help="Source language of the dataset to verbalize."
    )
    parser.add_argument(
        "--l2", type=str, required=True, help="Target language of the dataset to verbalize."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="The directory containing the  dataset to verbalize."
    )
    parser.add_argument(
        "--verbalization_mix", type=str, choices=["all", "alternate"], default="all", help="How to mix the verbalizations."
    )
    parser.add_argument(
        "--k_shot", type=int, default=5, help="How many examples to verbalize for each language pair."
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Truncate the number of train examples from each language pair to this "
    )
    parser.add_argument("--output_dir", type=str, default=None, required=True, help="Where to store the final model.")

    args = parser.parse_args()

    return args

def verbalize_pairs(verbalization_prefix, l1, l2, *pairs):
    l1_full = language_mapping[l1]
    l2_full = language_mapping[l2]
    #v_1 =  verbalization.replace(TEMPLATE_L1, l1).replace(TEMPLATE_L2, l2).replace(TEMPLATE_S1, s1).replace(TEMPLATE_S2, s2)
    #v_2 = verbalization.replace(TEMPLATE_L1, l2).replace(TEMPLATE_L2, l1).replace(TEMPLATE_S1, s2).replace(TEMPLATE_S2, s1)
    v_1_prefix = verbalization_prefix.replace(TEMPLATE_L1, l1_full).replace(TEMPLATE_L2, l2_full)
    v_2_prefix = verbalization_prefix.replace(TEMPLATE_L1, l2_full).replace(TEMPLATE_L2, l1_full)

    v_1_body = "\n".join(itertools.chain( *[(l1_full+' : '+s1, l2_full+' : '+s2) for s1, s2 in pairs ]) )
    v_2_body = "\n".join(itertools.chain( *[(l2_full+' : '+s2, l1_full+' : '+s1) for s1, s2 in pairs ]) )
    return v_1_prefix + v_1_body, v_2_prefix + v_2_body

def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    disable_caching()
    args = parse_args()

    def verbalize_items(examples, l1, l2):
        pairs = [ (example[l1].strip(), example[l2].strip()) for example in  examples["translation"]  ]
        pairs_grouped = [pairs[i:i+args.k_shot] for i in range(0, len(pairs), args.k_shot)]

        result = {"text": []}
        for i, group in enumerate(pairs_grouped):
            if args.verbalization_mix == "all":
                verbalization_prefixes = VERBALIZATION_PREFIXES
            elif args.verbalization_mix == "alternate":
                verbalization_prefixes = [VERBALIZATION_PREFIXES[i%len(VERBALIZATION_PREFIXES)]]
            else:
                raise ValueError("Invalid verbalization mix")
            
            for verbalization_prefix in verbalization_prefixes:
                verbalized_pairs = verbalize_pairs(verbalization_prefix, l1, l2, *group)
                result["text"].append(verbalized_pairs[0])
                result["text"].append(verbalized_pairs[1])

        return result

    # Load the datasets
    # Load training files fro directory 
    l1_train = os.path.join(args.data_dir, f"train.{args.l1}")
    l2_train = os.path.join(args.data_dir, f"train.{args.l2}")
    l1_dev = os.path.join(args.data_dir, f"dev.{args.l1}")
    l2_dev = os.path.join(args.data_dir, f"dev.{args.l2}")

    l1_dataset = load_dataset("text", data_files={"train": l1_train, "validation": l1_dev})
    l2_dataset = load_dataset("text", data_files={"train": l2_train, "validation": l2_dev})

    l1_dataset = l1_dataset.rename_column("text", "l1_text")
    l2_dataset = l2_dataset.rename_column("text", "l2_text")

    logging.info("Dataset sizes:")
    logging.info(f"l1: {len(l1_dataset['train'])} train, {len(l1_dataset['validation'])} validation")
    logging.info(f"l2: {len(l2_dataset['train'])} train, {len(l2_dataset['validation'])} validation")

    def merge_into_pairs(examples):
        return {"translation": list({args.l1: s1, args.l2: s2} for s1, s2 in zip(examples["l1_text"], examples["l2_text"]))}
    
    #dataset = concatenate_datasets([l1_dataset, l2_dataset]).map(merge_into_pairs, batched=True, remove_columns=["l1_text", "l2_text"])
    #Concatenate each split separately 

    dataset = DatasetDict({
        split: concatenate_datasets([l1_dataset[split].flatten_indices(), l2_dataset[split].flatten_indices()], axis=1)
        for split in l1_dataset.keys()
    })
    dataset = dataset.map(merge_into_pairs, batched=True, remove_columns=["l1_text", "l2_text"])

    dataset = dataset.shuffle()

    verbalized_dataset = dataset.map(verbalize_items, batched=True, fn_kwargs={"l1": args.l1, "l2": args.l2}, remove_columns=["translation"])
    for split in "train", "validation":
        verbalized_dataset[split].to_json(os.path.join(args.output_dir, f"{args.l1}-{args.l2}-{split}.jsonl" ))


if __name__ == "__main__":
    main()
