
import argparse
import json
import logging
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

import transformers
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
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


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.26.0.dev0")

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--prepared_train_dataset",
        type=str,
        default=None,
        required=True,
        help="Path to the pre-processed prepared train dataset.",
    )
    parser.add_argument(
        "--prepared_validation_dataset",
        type=str,
        default=None,
        required=True,
        help="Path to the pre-processed validation dataset.",
    )
    parser.add_argument(
        "--extra_validation_datasets",
        type=str,
        nargs="+",
        help="Path to the pre-processed extra validation datasets.",
        required=False,
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_validation_steps",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of validation examples to this "
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--checkpoint_activations",
        action="store_true",
        help="Whether to enable activation checkpointing.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--train_log_steps",
        type=int, 
        default=10, 
        help=(
            'Train logging frequency, in steps'
        ),
    )
    parser.add_argument(
        "--eval_log_steps",
        type=int, 
        default=10, 
        help=(
            'Validation logging frequency, in steps'
        ),
    )
    args = parser.parse_args()

    return args




def main():
    disable_caching()
    args = parse_args()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir

    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=180*60))
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs, kwargs_handlers=[init_kwargs])

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    


    train_dataset = load_from_disk(args.prepared_train_dataset)
    validation_dataset = load_from_disk(args.prepared_validation_dataset)
    if args.extra_validation_datasets:
        extra_validation_datasets = [load_from_disk(args.extra_validation_dataset) for args.extra_validation_dataset in args.extra_validation_datasets]
    else:
        extra_validation_datasets = []

    assert train_dataset.column_names == validation_dataset.column_names, "Train and validation datasets must have the same columns"

    all_validation_datasets = {'validation' : validation_dataset, **{f"extra_validation_{i}" : extra_validation_datasets[i] for i in range(len(extra_validation_datasets)) }}
    #for key in all_validation_datasets.keys():
        #if args.max_validation_samples is not None and args.max_validation_samples < len(all_validation_datasets[key]):
        #    logger.info(f"Truncating extra validation dataset {key} to {args.max_validation_samples} samples.")
        #    all_validation_datasets[key] = all_validation_datasets[key].select(range(args.max_validation_samples))



    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}. Decoded: {tokenizer.decode(train_dataset[index]['input_ids'])}")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloaders = {}
    for key in all_validation_datasets.keys():
        assert key not in eval_dataloaders, f"Extra validation dataset name {key} is already used"
        eval_dataloaders[key] = DataLoader(
            all_validation_datasets[key], collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size
        )
        

    if args.checkpoint_activations:
        model.gradient_checkpointing_enable()
    model = accelerator.prepare(model)


    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`. Modle was previously prepared
    optimizer, train_dataloader, lr_scheduler  = accelerator.prepare(
         optimizer, train_dataloader, lr_scheduler
    )
    #for key in eval_dataloaders.keys():
    #    eval_dataloaders[key] = accelerator.prepare(eval_dataloaders[key])

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        run_name = args.output_dir
        accelerator.init_trackers("LMCOMB_finetunes", config=experiment_config, init_kwargs={"wandb":{"name":run_name}})

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    resume_step = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        accelerator.print(f"Got resume from checkpoint: {args.resume_from_checkpoint}, finding path to resuem from")
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`

            accelerator.print(f"Resumed from checkpoint path automatically detected: {path}")
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    completed_steps = starting_epoch * num_update_steps_per_epoch


    def log_to_file(log_file, log_string):
        logger.info(f"Logging to {log_file}: {log_string}")
        path = os.path.join(args.output_dir, log_file)
        with FileLock(path+'.lock'):
            with open(path, "a") as f:
                f.write(log_string)

    #log_to_file("ids_log_file.txt", f"--- Resuming training at epoch {starting_epoch}, step {resume_step} ---\n")
    
    


    #Get validation for initi model

    model.eval()
    to_log = {}
    
    for key, eval_dataloader in eval_dataloaders.items():
        losses = []
        
        for step, batch in enumerate(eval_dataloader):
            
            with torch.no_grad():
                batch.pop("ids", None)
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(loss.repeat(args.per_device_eval_batch_size))

        
        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        logger.info(f"For validation set {key} at init model perplexity: {perplexity} eval_loss: {eval_loss}")
        to_log[f"{key}_perplexity"] = perplexity
        to_log[f"{key}_eval_loss"] = eval_loss

    if args.with_tracking:
        accelerator.log(
            {
                **to_log,
                "epoch": starting_epoch,
                "step": completed_steps,
            },
            step=0,
        )
    model.train()


    for epoch in range(starting_epoch, args.num_train_epochs):

        model.train()
        if args.with_tracking:
            total_loss = 0
        #log_to_file("ids_log_file.txt", f"--- Epoch {epoch} ---\n")

        step_loss = 0
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                        completed_steps += 1
                    continue
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step == resume_step:
                    logger.info(f"Resuming training at step {resume_step}, first batch after skipped steps: {batch}")

            ids = batch.pop("ids")
            
            #log_to_file("ids_log_file.txt", f"{hash_batch(batch)}\n")
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                step_loss += loss.detach()
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                step_loss/=accelerator.gradient_accumulation_steps #Gonna log only step loss corresponding to rank 0 GPU
                step_perplexity = math.exp(step_loss)
                if (completed_steps - 1) % args.train_log_steps == 0:
                    logger.info(f"epoch {epoch}: train perplexity: {step_perplexity} train_loss: {step_loss} lr: {lr_scheduler.get_last_lr()[0]}")
                    if args.with_tracking:
                        accelerator.log(
                            {
                                "train_perplexity": step_perplexity,
                                "train_loss": step_loss,
                                "epoch": epoch,
                                "lr": lr_scheduler.get_last_lr()[0],
                                "step": completed_steps,
                                "train_batch_size": total_batch_size, 
                            },
                            step=completed_steps,
                        )
                
                step_loss = 0
                if (completed_steps -1) % args.eval_log_steps == 0:
                    model.eval()
                    to_log = {}
                    
                    for key, eval_dataloader in eval_dataloaders.items():
                        losses = []
                        
                        for step, batch in enumerate(eval_dataloader):
                            if step>args.max_validation_steps:
                                break
                            with torch.no_grad():
                                batch.pop("ids", None)
                                outputs = model(**batch)

                            loss = outputs.loss
                            losses.append(loss.repeat(args.per_device_eval_batch_size))

                        
                        losses = torch.cat(losses)
                        try:
                            eval_loss = torch.mean(losses)
                            perplexity = math.exp(eval_loss)
                        except OverflowError:
                            perplexity = float("inf")

                        logger.info(f"For validation set {key} epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")
                        to_log[f"{key}_perplexity"] = perplexity
                        to_log[f"{key}_eval_loss"] = eval_loss

                    if args.with_tracking:
                        accelerator.log(
                            {
                                **to_log,
                                "epoch": epoch,
                                "step": completed_steps,
                            },
                            step=completed_steps,
                        )
                    model.train()


                if  isinstance(checkpointing_steps, int):
                    if (completed_steps-1) % checkpointing_steps == 0:
                        logger.info(f"Saving checkpoint at step {completed_steps}")
                        output_dir = f"step_{completed_steps }"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir)


            if completed_steps >= args.max_train_steps:
                break


        if args.with_tracking:
            
            train_loss = total_loss.item() / len(train_dataloader)
            try:
                train_ppl = math.exp(train_loss)
            except OverflowError:
                train_ppl = float("inf")
            accelerator.log(
                {
                    "train_loss": train_loss, #Accumulated epoch train loss corresponding to rank 0 GPU
                    "train_ppl": train_ppl,
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )
        logger.info("epoch {} completed".format(epoch))
        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, state_dict=accelerator.get_state_dict(model)
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump({"perplexity": perplexity}, f)


if __name__ == "__main__":
    main()
