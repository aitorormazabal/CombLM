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
import os

import datasets
import time 
import torch
from datasets import load_dataset, load_from_disk, disable_caching
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from transformers import OPTForCausalLM, AutoTokenizer
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
        "--checkpoint",
        type=str,
        default=None,
        required=True,
        help="Path to the combined model checkpoint, containing combination parameters.",
    )
    parser.add_argument(
        "--eval_path",
        type=str,
        default=None,
        required=True,
        help="Path to the pre-processed validation dataset.",
    )
    parser.add_argument(
        "--eval_max_samples",
        type=int,
        default=1000,
        help="Number of samples to evaluate on.",
    )
    parser.add_argument(
        "--no_mixed_precision",
        action="store_true",
        help="Whether to disable mixed precision training.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path to save the results to.",
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


    #Loading input_ids
    eval_dataset = load_from_disk(args.eval_path)
    if args.eval_max_samples is not None:
        eval_dataset = eval_dataset.select(range(args.eval_max_samples))
    
    logger.info(f"Loaded eval dataset with {len(eval_dataset)} samples")
    input_ids = torch.LongTensor(eval_dataset["input_ids"])

    #Loading models 

    dtype = torch.float16 if not args.no_mixed_precision else torch.float32
    
    model_1 = OPTForCausalLM.from_pretrained(args.model_1_path, torch_dtype=dtype, device_map="auto")
    model_2 = OPTForCausalLM.from_pretrained(args.model_2_path, torch_dtype=dtype, device_map="auto")

    model_1.eval()
    model_2.eval()

    logger.info(f"Model 1 device map: {model_1.hf_device_map}")
    logger.info(f"Model 2 device map: {model_2.hf_device_map}")

    logger.info(f"Loaded models, instantiating combined model")

    if args.model_type == "linear":
        model = LinearTrainedCombinedOPTForCausalLM(model_1, model_2)
    if args.model_type == "mean":
        model = LinearTrainedCombinedOPTForCausalLM(model_1, model_2)
    elif args.model_type == "entropy":
        model =  OutputsAndEntropyTrainedCombinedOPTForCausalLM(model_1, model_2, with_entropies=True, with_outputs=False)
        # if torch.cuda.is_available():
        #     logger.info("Using GPU")
        #     model.comb_cuda()
    elif args.model_type == "outputs":
        model = OutputsAndEntropyTrainedCombinedOPTForCausalLM(model_1, model_2, with_entropies=False)
        # if torch.cuda.is_available():
        #     logger.info("Using GPU")
        #     model.comb_cuda()
    elif args.model_type == "outputs_and_entropy":
        model = OutputsAndEntropyTrainedCombinedOPTForCausalLM(model_1, model_2, with_entropies=True)
        # if torch.cuda.is_available():
        #     logger.info("Using GPU")
        #     model.comb_cuda()
    elif args.model_type == "adaptive_outputs":
        model = OutputsAndEntropyTrainedCombinedOPTForCausalLM(model_1, model_2, with_entropies=False, adaptive=True)
        # if torch.cuda.is_available():
        #     logger.info("Using GPU")
        #     model.comb_cuda()
    elif args.model_type == "adaptive_outputs_and_entropy":
        model = OutputsAndEntropyTrainedCombinedOPTForCausalLM(model_1, model_2, with_entropies=True, adaptive=True)
        # if torch.cuda.is_available():
        #     logger.info("Using GPU")
        #     model.comb_cuda()
    elif args.model_type == "adaptive_linear":
        model = AdaptiveLinearTrainedCombinedOPTForCausalLM(model_1, model_2)
    elif args.model_type == "noadaptive_linear":
        model = AdaptiveLinearTrainedCombinedOPTForCausalLM(model_1, model_2, adaptive=False)
    elif args.model_type == "adaptive_entropy":
        model =  OutputsAndEntropyTrainedCombinedOPTForCausalLM(model_1, model_2, with_entropies=True, with_outputs=False, adaptive=True)
        # if torch.cuda.is_available():
        #     logger.info("Using GPU")
        #     model.comb_cuda()

    #if torch.cuda.is_available():
    #    logger.info("Using GPU")
    #    model.comb_cuda()

    logger.info(f"Loading checkpoint from {args.checkpoint}")
    model.load_from_disk(args.checkpoint)

    logger.info(f"Loaded checkpoint comb_params: {model.comb_params}")
    if hasattr(model, "lambd"):
        logger.info(f"Loaded checkpoint lambd: {model.lambd}")

    # model.comb_cpu()

    model.comb_cuda()
    #Move to model 1 device

    #Set to eval 
    model.eval()
    model.comb_params.eval()

    #Print training status of all models
    logger.info(f"Model 1 training status: {model_1.training}")
    logger.info(f"Model 2 training status: {model_2.training}")
    logger.info(f"Combined model training status: {model.training}")
    logger.info(f"Combined model comb_params training status: {model.comb_params.training}")

    def logprobs_from_model(model, input_ids):

        with torch.no_grad():
            all_logprobs = []
            for i in tqdm(range(len(input_ids))):
                curr_input_ids = input_ids[i].clone().cuda()
                logging.info(f"curr_input_ids shape: {curr_input_ids.shape}")
                logits = model(curr_input_ids, return_dict=True).logits

                logits = logits.cpu().to(dtype=torch.float32)
                logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
                logprobs = logprobs.cpu()
                curr_input_ids = curr_input_ids.cpu()
                shift_lps = logprobs[:, :-1, :]
                shift_labels = curr_input_ids[:, 1:]

                logger.info(f"shift_lps shape: {shift_lps.shape}, shift_labels shape: {shift_labels.shape}")
                curr_logprobs = shift_lps.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

                on_cpu = curr_logprobs.clone().detach().cpu()
                all_logprobs.append(on_cpu)
                logging.info(f"Logprobs shape {on_cpu.shape}, full logprobs shape {logprobs.shape}")
                logging.info(f"Batch perplexity {torch.exp(-on_cpu.mean())}")

            all_logprobs = torch.vstack(all_logprobs)

            return all_logprobs


    #Batch input_ids to save memory
    batch_size = 4
    input_ids = [input_ids[i:i+batch_size] for i in range(0, len(input_ids), batch_size)]


    
    #Get logprobs for combined model
    logger.info(f"Getting logprobs for combined model")
    model_logprobs = logprobs_from_model(model, input_ids)

    model_ppl = torch.exp(-model_logprobs.mean())
    logger.info(f"Combined model ppl: {model_ppl}")

    #Get logprobs for model 2
    logger.info(f"Getting logprobs for model 2")
    model_2_logprobs = logprobs_from_model(model_2, input_ids)

    #Get logprobs for model 1 
    logger.info(f"Getting logprobs for model 1")
    model_1_logprobs = logprobs_from_model(model_1, input_ids)


    #Get  ppls
    model_1_ppl = torch.exp(-model_1_logprobs.mean())
    model_2_ppl = torch.exp(-model_2_logprobs.mean())

    logger.info(f"Model 1 ppl: {model_1_ppl}")
    logger.info(f"Model 2 ppl: {model_2_ppl}")

    #Save logprobs to disk
    logger.info(f"Saving logprobs to {args.output_dir}")
    #Make path if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    torch.save(model_1_logprobs, os.path.join(args.output_dir, "model_1_logprobs.pt"))
    torch.save(model_2_logprobs, os.path.join(args.output_dir, "model_2_logprobs.pt"))
    torch.save(model_logprobs, os.path.join(args.output_dir, "model_logprobs.pt"))



if __name__ == "__main__":
    main()
