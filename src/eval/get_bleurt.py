import argparse
import json
import logging
from logging import getLogger   
import math
import os
import random
from itertools import chain
from pathlib import Path
from filelock import FileLock
from datetime import timedelta
from evaluate import load

import datasets
import torch
from datasets import load_dataset, load_from_disk, disable_caching
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from transformers import OPTForCausalLM, AutoTokenizer
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version



def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to the source sentences",
    )
    parser.add_argument(
        "--references",
        type=str,
        required=True,
        help="Path to the target sentences",
    )
    args = parser.parse_args()

    return args


logger = getLogger(__name__)


def main():
    disable_caching()
    args = parse_args()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    #Read source and target sentences
    with open(args.predictions, "r") as f:
        predictions = f.readlines()
    with open(args.references, "r") as f:
        references = f.readlines()
    #Remove last if empty
    if predictions[-1] == "":
        predictions = predictions[:-1]
    if references[-1] == "":
        references = references[:-1]

    #Load bleurt
    bleurt = load("bleurt", "BLEURT-20",module_type="metric" )
    results = bleurt.compute(predictions=predictions, references=references)

    logging.info(f"BLEURT scores: {results['scores']} for {len(results['scores'])} sentences")
    #Get mean score 
    scores = results['scores']
    mean_score = sum(scores)/len(scores)
    logging.info(f"Mean BLEURT score: {mean_score}")

if __name__ == "__main__":
    main()