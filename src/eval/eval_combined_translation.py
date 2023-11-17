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
        "--source_sentences",
        type=str,
        nargs="+",
        default=None,
        required=True,
        help="Path to the pre-processed validation dataset.",
    )

    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        required=True,
        help="Path to the prompt file, in plaintext.",
    )
    parser.add_argument(
        "--l1",
        type=str,
        default=None,
        required=True,
        help="Language 1",
    )
    parser.add_argument(
        "--l2",
        type=str,
        default=None,
        required=True,
        help="Language 2",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=3,
        help="Batch size for translation.",
    )
    parser.add_argument(
        "--beam_size",
        type=int, 
        default=1
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Path to save the model to.",
    )
    parser.add_argument(
        "--sampling",
        action="store_true"
    )
    parser.add_argument(
        "--no_space",
        action="store_true"
    )
    
    args = parser.parse_args()

    return args


logger = getLogger(__name__)

language_mapping = {"en":"English", 
                    "de":"German",
                    "fr":"French",
                    "hi":"Hindi",
                    "ru":"Russian",
                    "cs":"Czech",}

def main():
    disable_caching()
    args = parse_args()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    dtype = torch.float16 
    
    model_1 = OPTForCausalLM.from_pretrained(args.model_1_path, torch_dtype=dtype, device_map="auto")
    model_2 = OPTForCausalLM.from_pretrained(args.model_2_path, torch_dtype=dtype, device_map="auto")

    logger.info(f"Model 1 device map: {model_1.hf_device_map}")
    logger.info(f"Model 2 device map: {model_2.hf_device_map}")

    logger.info(f"Loaded models, instantiating combined model")

    if args.model_type == "linear":
        model = LinearTrainedCombinedOPTForCausalLM(model_1, model_2)
    if args.model_type == "mean":
        model = LinearTrainedCombinedOPTForCausalLM(model_1, model_2)
    elif args.model_type == "entropy":
        model =  OutputsAndEntropyTrainedCombinedOPTForCausalLM(model_1, model_2, with_entropies=True, with_outputs=False)
    elif args.model_type == "outputs":
        model = OutputsAndEntropyTrainedCombinedOPTForCausalLM(model_1, model_2, with_entropies=False)
    elif args.model_type == "outputs_and_entropy":
        model = OutputsAndEntropyTrainedCombinedOPTForCausalLM(model_1, model_2, with_entropies=True)

    elif args.model_type == "adaptive_outputs":
        model = OutputsAndEntropyTrainedCombinedOPTForCausalLM(model_1, model_2, with_entropies=False, adaptive=True)
  
    elif args.model_type == "adaptive_outputs_and_entropy":
        model = OutputsAndEntropyTrainedCombinedOPTForCausalLM(model_1, model_2, with_entropies=True, adaptive=True)

    elif args.model_type == "adaptive_linear":
        model = AdaptiveLinearTrainedCombinedOPTForCausalLM(model_1, model_2)
    elif args.model_type == "noadaptive_linear":
        model = AdaptiveLinearTrainedCombinedOPTForCausalLM(model_1, model_2, adaptive=False)
    elif args.model_type == "adaptive_entropy":
        model =  OutputsAndEntropyTrainedCombinedOPTForCausalLM(model_1, model_2, with_entropies=True, with_outputs=False, adaptive=True)
model.comb_cuda()

    model.eval()
    model.comb_params.eval()

    if args.beam_size*args.batch_size < 4: 
        logger.info(f"Small batch*beam {args.beam_size*args.batch_size} size, Using GPU")
        if torch.cuda.is_available():
            logger.info("Using GPU")
            model.comb_cuda()
    else:
        logger.info(f"Large batch*beam {args.beam_size*args.batch_size}, Using CPU")
        model.comb_cpu()
    sampling = False
    if args.sampling:
        sampling=True

    logger.info(f"Loading checkpoint from {args.checkpoint}")
    model.load_from_disk(args.checkpoint)

    logger.info(f"Loaded checkpoint comb_params: {model.comb_params}")
    if hasattr(model, "lambd"):
        logger.info(f"Loaded checkpoint lambd: {model.lambd}")

    logger.info(f"Loaded checkpoint, instantiating tokenizer")
    tok = AutoTokenizer.from_pretrained("facebook/opt-1.3b", padding_side="left", pad_token="<pad>")
    #Loading prompt file
    with open(args.prompt_file, "r") as f:
        kshot_prompt = f.read().strip()
    
    #Remove last four linse from kshot_prompt, to reduce shot by 2
    kshot_prompt = "\n".join(kshot_prompt.split("\n")[:-4])


    logger.info(f"Loaded kshot_prompt: {kshot_prompt}")

    l1 = language_mapping[args.l1]
    l2 = language_mapping[args.l2]

    #Check that languages appear in prompt, in the correct order
    assert l1 in kshot_prompt, f"Language 1 not in prompt, {l1}"
    assert l2 in kshot_prompt, f"Language 2 not in prompt, {l2}"
    assert kshot_prompt.index(l1) < kshot_prompt.index(l2), "Languages not in correct order"


    logger.info(f"Loading source sentences")
    with open(args.source_sentences[0], "r") as f:
        source_sentences = [s.strip() for s in f.readlines()]
        #Remove last sentence if empty
        if source_sentences[-1] == "":
            source_sentences = source_sentences[:-1]
    logger.info(f"Loaded {len(source_sentences)} source sentences")

    logger.info(f"Generating translation prompts")
    if args.no_space:
        translation_prompts = [ kshot_prompt + "\n{}: {}\n{}:".format(l1, sentence, l2) for sentence in source_sentences]
    else:
        translation_prompts = [ kshot_prompt + "\n{} : {}\n{} :".format(l1, sentence, l2) for sentence in source_sentences]
    logger.info(f"Generated {len(translation_prompts)} translation prompts")
    #Print random translation prompt
    logger.info(f"Random translation prompt: {random.choice(translation_prompts)}")

    #Group into batches
    group_translation_prompts = [translation_prompts[i:i+args.batch_size] for i in range(0, len(translation_prompts), args.batch_size)]
    #Tokenize translation prompts

    logger.info(f"Tokenizing translation prompts")
    tokenized_translation_prompts = [tok(p, return_tensors="pt", padding=True, truncation=True) for p in group_translation_prompts]
    

    logger.info(f"Model 1 device is {model_1.device}")
    for m in [model_1, model_2, model]:
        m.config.eos_token = '\n'
        m.config.eos_token_id = tok('\n', add_special_tokens=False).input_ids[0]

    max_length = 830 #Avoid OOM

    logger.info(f"Max length is {max_length}")

    #Combined model generation
    logger.info(f"Generating translations with combined model")
    combined_model_outputs = []

    with torch.no_grad():
        for i in tqdm(range(len(tokenized_translation_prompts))):
            inputs = tokenized_translation_prompts[i]
            logger.info(f"Inputs keys and shapes: {[(k, v.shape) for k, v in inputs.items()]}")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            combined_model_outputs.append(model.generate(**inputs, max_length=max_length, num_beams=args.beam_size, do_sample=sampling))
            #Delete inputs to free memory
            inputs.clear()

        lengths = [len(o) for o in list(chain(*combined_model_outputs))]
        for i,l in enumerate(lengths):
            if l==max_length:
                logger.info(f"Found max_length output in combined model, index {i}")


    #Model 1 generation
    logger.info(f"Generating translations with model 1")
    model_1_outputs = []
    for i in tqdm(range(len(tokenized_translation_prompts))):
        inputs = tokenized_translation_prompts[i]
        logger.info(f"Inputs keys and shapes: {[(k, v.shape) for k, v in inputs.items()]}")
        #Move inputs to GPU
        inputs = {k: v.to(model_1.device) for k, v in inputs.items()}
        model_1_outputs.append(model_1.generate(**inputs, max_length=max_length, num_beams=args.beam_size, do_sample=sampling))
        #Delete inputs to free memory
        inputs.clear()
    
    lengths = [len(o) for o in list(chain(*model_1_outputs))]
    for i,l in enumerate(lengths):
        if l==max_length:
            logger.info(f"Found max_length output in model 1, index {i}")

    #Model 2 generation
    logger.info(f"Generating translations with model 2")
    model_2_outputs = []
    for i in tqdm(range(len(tokenized_translation_prompts))):
        inputs = tokenized_translation_prompts[i]

        logger.info(f"Inputs keys and shapes: {[(k, v.shape) for k, v in inputs.items()]}")
        inputs = {k: v.to(model_2.device) for k, v in inputs.items()}
        model_2_outputs.append(model_2.generate(**inputs, max_length=max_length, num_beams=args.beam_size, do_sample=sampling))
        #Delete inputs to free memory
        inputs.clear()


    

    lengths = [len(o) for o in list(chain(*model_2_outputs))]
    for i,l in enumerate(lengths):
        if l==max_length:
            logger.info(f"Found max_length output in model 2, index {i}")


    logger.info(f"Decoding translations")
    logger.info(f"Raw translations are {model_1_outputs}, {combined_model_outputs}, {model_2_outputs}")
    #Decode all batches and chain 
    model_1_outputs = list(chain(*[tok.batch_decode(outputs) for outputs in model_1_outputs]))
    combined_model_outputs = list(chain(*[tok.batch_decode(outputs) for outputs in combined_model_outputs]))
    model_2_outputs = list(chain(*[tok.batch_decode(outputs) for outputs in model_2_outputs]))

    logger.info(f"Raw translations are {model_1_outputs}, {combined_model_outputs}, {model_2_outputs}")

    def get_clean_translation(translation, prompt):
        res =  translation.strip('<pad>').strip('/s>').removeprefix('</s>').removeprefix(prompt).split('</s>')[0].strip()
        logger.info(f"Orig is {translation}, cleaned is {res}")
        return res
    model_1_translations = [get_clean_translation(translation, prompt) for translation, prompt in zip(model_1_outputs, translation_prompts)]
    combined_model_translations = [get_clean_translation(translation, prompt) for translation, prompt in zip(combined_model_outputs, translation_prompts)]
    model_2_translations = [get_clean_translation(translation, prompt) for translation, prompt in zip(model_2_outputs, translation_prompts)]


    #Save translations
    #Model 1
    logger.info(f"Saving translations with model 1")
    model_name = args.model_1_path.replace("/", "_")+f"_{args.model_type}_{args.l1}_{args.l2}"
    with open(os.path.join(args.save_dir, f"{model_name}.txt"), "w") as f:
        for translation in model_1_outputs:
            f.write(translation + "\n")
    with open(os.path.join(args.save_dir, f"{model_name}_clean.txt"), "w") as f:
        for translation in model_1_translations:
            f.write(translation + "\n")

    #Combined model
    logger.info(f"Saving translations with combined model")
    model_name = f"combined_{args.model_type}_{args.l1}_{args.l2}"
    with open(os.path.join(args.save_dir, f"{model_name}.txt"), "w") as f:
        for translation in combined_model_outputs:
            f.write(translation + "\n")
    with open(os.path.join(args.save_dir, f"{model_name}_clean.txt"), "w") as f:
        for translation in combined_model_translations:
            f.write(translation + "\n")

    #Model 2
    logger.info(f"Saving translations with model 2")
    model_name = args.model_2_path.replace("/", "_")+f"_{args.model_type}_{args.l1}_{args.l2}"
    with open(os.path.join(args.save_dir, f"{model_name}.txt"), "w") as f:
        for translation in model_2_outputs:
            f.write(translation + "\n")
    with open(os.path.join(args.save_dir, f"{model_name}_clean.txt"), "w") as f:
        for translation in model_2_translations:
            f.write(translation + "\n")


if __name__ == "__main__":
    main()
