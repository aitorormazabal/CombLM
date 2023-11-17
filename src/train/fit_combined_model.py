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

import datasets
import torch
from datasets import load_dataset, load_from_disk, disable_caching
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from transformers import OPTForCausalLM
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version

from eval.eval_utils import calc_perplexity

from models.combined_model import  EntropyTrainedCombinedOPTForCausalLM, LinearTrainedCombinedOPTForCausalLM, OutputsAndEntropyTrainedCombinedOPTForCausalLM, AdaptiveLinearTrainedCombinedOPTForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--model_1_path",
        type=str,
        default=None,
        required=True,
        help="Path to the model 1 checkpoint.",
    )
    parser.add_argument(
        "--model_2_path",
        type=str,
        default=None,
        required=True,
        help="Path to the model 2 checkpoint.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        required=True,
        help="Combined model type",
    )
    parser.add_argument(
        "--train_datasets",
        type=str,
        nargs="+",
        default=None,
        required=True,
        help="Path to the pre-processed training dataset.",
    )
    parser.add_argument(
        "--validation_datasets",
        type=str,
        nargs="+",
        default=None,
        required=True,
        help="Path to the pre-processed validation dataset.",
    )
    parser.add_argument(
        "--no_mixed_precision",
        action="store_true",
        help="Whether to disable mixed precision training.",
    )
    parser.add_argument(
        "--only_run_datasets",
        action="store_true",
        help="Whether to only run the dataset preparation part, without keeping each part in memory.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to use for training and validation",
    )

    parser.add_argument(
        "--max_train",
        type=int,
        default=500,
        help="Maximum number of samples to use for training",
    )
    parser.add_argument(
        "--max_valid",
        type=int,
        default=500,
        help="Maximum number of samples to use for validation",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate to use for training.",
    )
    parser.add_argument(
        "--train_bs",
        type=int,
        default=1024,
        help="Batch size to use for training.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Path to save the model to.",
    )
    parser.add_argument(
        "--try_model_gpu",
        action="store_true",
        help="Whether to try to use GPU for models to combine.",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=2,
        help="Number of epochs to train for.",
    )
    parser.add_argument(
        "--save_only_last",
        action="store_true",
        help="Whether to only save the last model.",
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

    dtype = torch.float16 if not args.no_mixed_precision else torch.float32
    
    model_1 = OPTForCausalLM.from_pretrained(args.model_1_path, torch_dtype=dtype, device_map="auto")
    model_2 = OPTForCausalLM.from_pretrained(args.model_2_path, torch_dtype=dtype, device_map="auto")

    logger.info(f"Model 1 device map: {model_1.hf_device_map}")
    logger.info(f"Model 2 device map: {model_2.hf_device_map}")

    logger.info(f"Loaded models, instantiating combined model")

    

    train_datasets = []
    for dataset in args.train_datasets:
        ds = load_from_disk(dataset).remove_columns(["ids"])
        if args.max_samples is not None and len(ds) > args.max_samples:
            ds = ds.select(range(args.max_samples))
        train_datasets.append(ds)
    valid_datasets = []
    for dataset in args.validation_datasets:
        ds = load_from_disk(dataset).remove_columns(["ids"])
        if args.max_samples is not None and len(ds) > args.max_samples:
            ds = ds.select(range(args.max_samples))
        valid_datasets.append(ds)

    extra_args = {}
    if args.model_type == "linear":
        model = LinearTrainedCombinedOPTForCausalLM(model_1, model_2)
    elif args.model_type == "entropy":
        model =  OutputsAndEntropyTrainedCombinedOPTForCausalLM(model_1, model_2, with_entropies=True, with_outputs=False)
        if torch.cuda.is_available():
            logger.info("Using GPU")
            model.comb_cuda()
    elif args.model_type == "outputs":
        model = OutputsAndEntropyTrainedCombinedOPTForCausalLM(model_1, model_2, with_entropies=False)
        if torch.cuda.is_available():
            logger.info("Using GPU")
            model.comb_cuda()
    elif args.model_type == "outputs_and_entropy":
        model = OutputsAndEntropyTrainedCombinedOPTForCausalLM(model_1, model_2, with_entropies=True)
        if torch.cuda.is_available():
            logger.info("Using GPU")
            model.comb_cuda()
    elif args.model_type == "adaptive_outputs":
        model = OutputsAndEntropyTrainedCombinedOPTForCausalLM(model_1, model_2, with_entropies=False, adaptive=True)
        if torch.cuda.is_available():
            logger.info("Using GPU")
            model.comb_cuda()
    elif args.model_type == "adaptive_outputs_and_entropy":
        model = OutputsAndEntropyTrainedCombinedOPTForCausalLM(model_1, model_2, with_entropies=True, adaptive=True)
        if torch.cuda.is_available():
            logger.info("Using GPU")
            model.comb_cuda()
    elif args.model_type == "adaptive_linear":
        model = AdaptiveLinearTrainedCombinedOPTForCausalLM(model_1, model_2)
    elif args.model_type == "noadaptive_linear":
        model = AdaptiveLinearTrainedCombinedOPTForCausalLM(model_1, model_2, adaptive=False)
    elif args.model_type == "adaptive_entropy":
        model =  OutputsAndEntropyTrainedCombinedOPTForCausalLM(model_1, model_2, with_entropies=True, with_outputs=False, adaptive=True)
        if torch.cuda.is_available():
            logger.info("Using GPU")
            model.comb_cuda()


 
    logger.info("Training combined model")
    model.fit(train_datasets, valid_datasets, process_bs=3, train_bs = args.train_bs, n_epochs = args.n_epochs, lr=args.lr, only_run_datasets=args.only_run_datasets, max_train=args.max_train*1024, max_valid=args.max_valid*1024, save_dir = args.save_dir, save_only_last=args.save_only_last, **extra_args)




if __name__ == "__main__":
    main()
