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
import glob
import re 

import datasets
import torch
from datasets import load_dataset, load_from_disk, disable_caching
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from transformers import OPTForCausalLM, AutoTokenizer
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version
import itertools


logger = logging.getLogger(__name__)
def parse_args():
    parser = argparse.ArgumentParser(description="Compare perplexities")
    parser.add_argument(
        "--logprobs_path",
        type=str,
        help="Path to evaluated logprobs",
    )
    args = parser.parse_args()

    return args
def calc_ppl(logprobs):
    return torch.exp(-torch.mean(logprobs.flatten()))

def main():
    args = parse_args() 
    model_1_logprobs = torch.load(os.path.join(args.logprobs_path, "model_1_logprobs.pt"))
    model_2_logprobs = torch.load(os.path.join(args.logprobs_path, "model_2_logprobs.pt"))
    model_logprobs = torch.load(os.path.join(args.logprobs_path, "model_logprobs.pt"))

    model_1_ppl = calc_ppl(model_1_logprobs)
    model_2_ppl = calc_ppl(model_2_logprobs)
    model_ppl = calc_ppl(model_logprobs)
    print(f'Model 1 PPL: {model_1_ppl}')
    print(f'Model 2 PPL: {model_2_ppl}')
    print(f'Combined model PPL: {model_ppl}')

if __name__ == "__main__":
    main()
