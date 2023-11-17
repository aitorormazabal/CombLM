
import random, logging
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
import itertools 
from transformers.models.opt.modeling_opt import OPTForCausalLM,  OPTPreTrainedModel
from torch.utils.data import DataLoader 
from torch.utils.data.dataset import TensorDataset
from tqdm import tqdm
from torch.utils.data import Subset

from transformers import default_data_collator, AutoTokenizer

from transformers.modeling_outputs import  CausalLMOutputWithPast
from eval.eval_utils import calc_perplexity, calc_entropy
import os 
import copy
from tqdm import tqdm 

logger = logging.getLogger(__name__)
COMBLM_DIR="/ikerlariak/aormazabal024/PhD/CombLM-release/CombLM"
CACHE_DIRS = [os.path.join(COMBLM_DIR, "cache"),]


#We need a custom clone function to make sure we don't carry accelerate hooks from previous model, which might interfere with proper device placement
def clone_linear(linear):
    trg_dim = linear.weight.shape[0]
    src_dim = linear.weight.shape[1]
    bias = linear.bias is not None 
    new_linear = torch.nn.Linear(src_dim, trg_dim, bias)
    new_linear.load_state_dict(linear.state_dict())
    return new_linear

class CombinedOPTForCausalLM(OPTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
    
    def fit(self, *args, **kwargs):
        raise NotImplementedError("CombinedOPTForCausalLM cannot be trained directly, it must be initialized from two pre-trained OPTForCausalLM models")
    def save_fit(self, save_path):
        raise NotImplementedError("CombinedOPTForCausalLM does not implement a save_fit method.")
    def load_fit(self, load_path):
        raise NotImplementedError("CombinedOPTForCausalLM does not implement a load_fit method.")
    def from_pretrained(self, pretrained_model_name_or_path, **kwargs):
        raise NotImplementedError("CombinedOPTForCausalLM cannot be loaded from a pre_trained checkpoint, it must be initialized from two pre-trained OPTForCausalLM models")

    def get_input_embeddings(self):
        raise NotImplementedError("Cannot get input embeddings for CombinedOPTForCausalLM")

    def set_input_embeddings(self, value):
        raise NotImplementedError("Cannot set input embeddings for CombinedOPTForCausalLM")

    def get_output_embeddings(self):
        raise NotImplementedError("Cannot get output embeddings for CombinedOPTForCausalLM")

    def set_output_embeddings(self, new_embeddings):
        raise NotImplementedError("Cannot set output embeddings for CombinedOPTForCausalLM")

    def set_decoder(self, decoder):
        raise NotImplementedError("Cannot set decoder for CombinedOPTForCausalLM")

    def get_decoder(self):
        raise NotImplementedError("Cannot get decoder for CombinedOPTForCausalLM")

    def save_to_disk(self, save_directory):
        raise NotImplementedError("Cannot save CombinedOPTForCausalLM to disk")
    def load_from_disk(self, load_directory):
        raise NotImplementedError("Cannot load CombinedOPTForCausalLM from disk")
    
    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, use_cache=None, **kwargs):
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        if past:
            input_ids = input_ids[:, -1:]
        # first step, decoder_cached_states are empty
        return {
            "input_ids": input_ids,  # encoder_outputs is defined. input_ids not needed
            "attention_mask": attention_mask,
            "past_key_values": past,
            "use_cache": use_cache,
        }
    @staticmethod
    def _reorder_cache(past, beam_idx):
        

        reordered_past = ()
        for model_past in past:
            reordered_model_past = ()
            for layer_past in model_past:
                reordered_model_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)

            reordered_past += (reordered_model_past,)

        return reordered_past
    


class TrainedCombinedOPTForCausalLM(CombinedOPTForCausalLM):

    #Model can only be instantiated directly from two OPTForCausalLM models
    def __init__(self, opt_model_1, opt_model_2):
        super().__init__(opt_model_1.config) #Smaller model config is used as combined model config for general decoder-only LM configs needed by generation (is_encoder_decoder etc.).
        self.opt_model_1 = opt_model_1
        self.opt_model_2 = opt_model_2
        self.comb_params = None 

        assert isinstance(self.opt_model_1, OPTForCausalLM) and isinstance(self.opt_model_2, OPTForCausalLM)
        assert self.opt_model_1.config.vocab_size == self.opt_model_2.config.vocab_size
    
    def get_config_dict(self):
        return {"opt_model_1": self.opt_model_1.config._name_or_path, "opt_model_2": self.opt_model_2.config._name_or_path, "class": self.__class__.__name__}

    def save_to_disk(self, save_directory, checkpoint_data={}, overwrite=False):
        #Make sure directory does not exist
        if os.path.exists(save_directory):
            if overwrite:
                logger.warning(f"Directory {save_directory} already exists, overwriting {self.__class__.__name__} to disk")
            else:
                raise ValueError(f"Directory {save_directory} already exists, and overwrite is set to False.")
        os.makedirs(save_directory, exist_ok=True)
        config = self.get_config_dict()
        torch.save(config, os.path.join(save_directory, "config.bin"))
        torch.save(checkpoint_data, os.path.join(save_directory, "checkpoint_data.bin"))
        assert self.comb_params is not None, f"Cannot save {self.__class__.__name__} to disk, model has not been fit"
        orig_device, orig_dtype = self.get_comb_device_dtype()
        fp16_cpu_model = self.comb_params.cpu().to(dtype=torch.float16)
        sate_dict = fp16_cpu_model.state_dict()
        torch.save(sate_dict, os.path.join(save_directory, "comb_params.bin"))
        self.comb_params.to(device=orig_device, dtype=orig_dtype)

    def load_from_disk(self, load_directory, strict=True):
        config = torch.load(os.path.join(load_directory, "config.bin"))
        if config != self.get_config_dict():
            logging.info(f"WARNING: Config of {self.__class__.__name__} does not match config of saved model")
            logging.info(f"WARNING: Config of {self.__class__.__name__}: {self.get_config_dict()}")
            logging.info(f"WARNING: Config of saved model: {config}")
            if strict:
                raise ValueError(f"Config of {self.__class__.__name__} does not match config of saved model\n Config of {self.__class__.__name__}: {self.get_config_dict()}\n Config of saved model: {config}")

        assert self.comb_params is not None, f"Cannot load {self.__class__.__name__} from disk, model has not been initialized"
        state_dict = torch.load(os.path.join(load_directory, "comb_params.bin"))

        assert set(state_dict.keys()) == set(self.comb_params.state_dict().keys()), f"Saved state dict does not match state dict of {self.__class__.__name__}"

        state_dict = {key: state_dict[key].to(self.comb_params.state_dict()[key].device, dtype=self.comb_params.state_dict()[key].dtype) for key in state_dict.keys()}

        for key in state_dict.keys():
            if "model_1" in key or "model_2" in key: #Parameters of the form model_i_* must match the current ones 
                assert torch.allclose(state_dict[key], self.comb_params.state_dict()[key]), f"Saved state dict does not match state dict of {self.__class__.__name__} for key {key}, that should match"

            if hasattr(state_dict[key], "shape"):
                assert state_dict[key].shape == self.comb_params.state_dict()[key].shape, f"Shape of saved state dict does not match shape of {self.__class__.__name__} state dict for key {key}. Saved shape: {state_dict[key].shape}, current shape: {self.comb_params.state_dict()[key].shape}"


        self.comb_params.load_state_dict(state_dict)
    
    
    def process_dataset_for_fit(self, dataset, model, bs=1, return_hidden=False, dtype=None, disable_cache=False): #Run forward on dataset and return logits, on OPTForCausalLM dataset 
        for cache_dir in CACHE_DIRS:
            cache_path = os.path.join(cache_dir, "processed_for_fit", model.config._name_or_path.replace("/", "."))
            dataset_id = dataset._fingerprint+str(return_hidden)+str(dtype)
            full_path = os.path.join(cache_path, dataset_id)

            if os.path.exists(full_path) and not disable_cache:
                logging.info(f"Loading processed dataset from cache path {full_path}")
                if dtype is None:
                    dtype = torch.float32
                entropies, logprobs, labels, hidden =  torch.load(full_path)
                entropies = entropies.to(dtype=dtype)
                logprobs = logprobs.to(dtype=dtype)
                if return_hidden:
                    assert hidden is not None, "Hidden states were not cached"
                    hidden = hidden.to(dtype=dtype)
                else:
                    assert hidden is None, "Hidden states were cached, but not requested"
                logging.info(f"Loaded processed dataset from cache path {full_path}")
                return entropies, logprobs, labels, hidden
        
            elif not os.path.exists(cache_path): #Make sure cache dir exists
                logging.info(f"Creating cache dir {cache_path}")
                os.makedirs(cache_path, exist_ok=True)

        # If we haven't found yet set full path to the first cache dir
        cache_path = os.path.join(CACHE_DIRS[0], "processed_for_fit", model.config._name_or_path.replace("/", "."))
        dataset_id = dataset._fingerprint+str(return_hidden)+str(dtype)
        full_path = os.path.join(cache_path, dataset_id)
        logging.info(f"Did not find in any cache, set save path to the first cache dir, in {full_path}")
        assert isinstance(model, OPTForCausalLM)
        dataloader = DataLoader(
            dataset, shuffle=False, collate_fn=default_data_collator, batch_size=bs
        )

        

        entropies = []
        labels = []
        logprobs = []
        hidden_states = []
        for batch in tqdm(dataloader):
            if torch.cuda.is_available():
                batch["input_ids"] = batch["input_ids"].cuda()
            with torch.no_grad():
                batch.pop("return_dict", None)
                batch.pop("labels", None)
                #outputs = model(**batch, return_dict = True, output_hidden_states=return_hidden)
                hidden = model.model.decoder(**batch, return_dict = True, output_hidden_states=False)[0]
                logits = model.lm_head(hidden)
                
                if return_hidden:
                    curr_states = hidden.detach().cpu()[..., :-1, :]
                else:
                    curr_states = None
                curr_labels = batch["input_ids"].cpu()
                curr_logits = logits.detach().cpu()
                to_dtype = dtype
                if to_dtype is None:
                    to_dtype = curr_logits.dtype
                curr_logits = curr_logits.float()
                curr_entropies = calc_entropy(curr_logits[:, :-1, :])
                curr_logits = curr_logits[:, :-1, :].contiguous()
                curr_labels = curr_labels[:, 1:].contiguous()
                curr_logprobs = torch.log_softmax(curr_logits, dim=-1).gather(-1, curr_labels.unsqueeze(-1))
                
                entropies.append(curr_entropies.to(dtype=to_dtype))
                labels.append(curr_labels)
                logprobs.append(curr_logprobs.to(dtype=to_dtype))
                if return_hidden:
                    hidden_states.append(curr_states.to(dtype=to_dtype))

        

        entropies = torch.vstack(entropies)
        logprobs = torch.vstack(logprobs)
        labels = torch.vstack(labels)
        if len(hidden_states)>0:
            hidden_states = torch.vstack(hidden_states)
        else:
            hidden_states = None

        # Save to cache
        if not disable_cache:
            logging.info(f"Saving processed dataset to cache path {full_path}")
            save_entropies = entropies.to(dtype=torch.float16)
            save_logprobs = logprobs.to(dtype=torch.float16)
            save_labels = labels
            save_hidden = hidden_states
            if save_hidden is not None:
                save_hidden_states = hidden_states.to(dtype=torch.float16)
            torch.save((save_entropies, save_logprobs, save_labels, save_hidden), full_path)
            logging.info(f"Saved")
        return entropies, logprobs, labels, hidden_states
        
    def prepare_dataset_for_fit(self, processed_1, processed_2):

        processed_1_entropies, processed_1_logprobs, processed_1_labels, processed_1_hidden = processed_1
        processed_2_entropies, processed_2_logprobs, processed_2_labels, processed_2_hidden = processed_2


        assert (processed_1_labels == processed_2_labels).all

        labels = processed_1_labels
        entropies = torch.cat([processed_1_entropies, processed_2_entropies], dim=-1)

        logprobs = torch.cat([processed_1_logprobs, processed_2_logprobs], dim=-1)
        
        if processed_1_hidden is not None:
            hidden_1 = processed_1_hidden.view(-1, processed_1_hidden.shape[-1])
            logging.info(f"Using hidden states for Model 1")
        if processed_2_hidden is not None:
            hidden_2 = processed_2_hidden.view(-1, processed_2_hidden.shape[-1])
            logging.info(f"Using hidden states for Model 2")

        entropies = entropies.view(-1, entropies.shape[-1])
        logprobs = logprobs.view(-1, logprobs.shape[-1])
        labels = labels.view(-1)

        return (entropies, logprobs, labels, hidden_1, hidden_2)
    
    def combine_batch_for_fit(self, batch):
        assert len(batch) == 4, "Batch should be tuple of (entropies, logprobs, labels, hidden_states)"
        raise NotImplementedError("Combine batch not implemented for this model")

    def combine_for_forward(self, *args):
        raise NotImplementedError("Combine for forward not implemented for this model")
    def get_comb_device_dtype(self):
        assert self.comb_params is not None, "Combination params not initialized"
        assert len(list(self.comb_params.parameters())) > 0, "Combination model has no parameters"
        #Returns the dtype of the first parameter found in the ffn
        dtypes = [param.dtype for param in self.comb_params.parameters()]
        assert len(set(dtypes)) == 1, "All dtypes must be the same"
        devices = [param.device for param in self.comb_params.parameters()]
        assert len(set(devices)) == 1, "All dtypes must be the same"
        for param in self.comb_params.parameters():
            return param.device, param.dtype

    def comb_cuda(self):
        self.comb_params = self.comb_params.cuda()
    def comb_cpu(self):
        self.comb_params = self.comb_params.cpu()

    def get_logprobs_from_prepared_batch(self, batch):
        return batch[1]
    def fit(self, trains, valids, process_bs = 1,train_bs=1000, n_epochs = 10, lr = 1e-5, early_stopping = False, only_run_datasets=False, max_train=None, max_valid=None, disable_cache=False, save_dir=None, save_only_last=False, **kwargs):
        assert "input_ids" in trains[0][0] and "input_ids" in valids[0][0]

        prepared_trains = []
        prepared_valids = []
        for i in range(len(trains)):
            logger.info(f"Processing dataset for fit, train {i} model 1")
            processed_train_1 = self.process_dataset_for_fit(trains[i], self.opt_model_1, bs=process_bs, return_hidden=True, dtype=torch.float32)
            if only_run_datasets:
                del processed_train_1
                processed_train_1 = None
            logger.info(f"Processing dataset for fit, train {i} model 2")
            processed_train_2 = self.process_dataset_for_fit(trains[i], self.opt_model_2, bs=process_bs, return_hidden=True, dtype=torch.float32)
            if only_run_datasets:
                del processed_train_2
                processed_train_2 = None
            train = self.prepare_dataset_for_fit(processed_train_1, processed_train_2)
            prepared_trains.append(train)
            processed_train_1 = None 
            processed_train_2 = None 
        for i in range(len(valids)):
            logger.info(f"Processing dataset for fit, valid {i} model 1")
            processed_valid_1 = self.process_dataset_for_fit(valids[i], self.opt_model_1, bs=process_bs, return_hidden=True, dtype=torch.float32)
            if only_run_datasets:
                del processed_valid_1
                processed_valid_1 = None
            logger.info(f"Processing dataset for fit, valid {i} model 2")
            processed_valid_2 = self.process_dataset_for_fit(valids[i], self.opt_model_2, bs=process_bs, return_hidden=True, dtype=torch.float32)
            if only_run_datasets:
                del processed_valid_2
                processed_valid_2 = None
            valid = self.prepare_dataset_for_fit(processed_valid_1, processed_valid_2)
            prepared_valids.append(valid)
            processed_valid_1 = None 
            processed_valid_2 = None 


        if torch.cuda.is_available():
            self.opt_model_1 = self.opt_model_1.cpu()
            self.opt_model_2 = self.opt_model_2.cpu()
            logger.info("Using GPU")
            self.comb_cuda()
            #self.comb_params.to(dtype=torch.float16)

        

        train_logprobs_all = [self.get_logprobs_from_prepared_batch(train) for train in prepared_trains]
        valid_logprobs_all = [self.get_logprobs_from_prepared_batch(valid) for valid in prepared_valids]

        train_logprobs_all = [train_logprobs[:max_train] for train_logprobs in train_logprobs_all]
        valid_logprobs_all = [valid_logprobs[:max_valid] for valid_logprobs in valid_logprobs_all]

        train_logprobs = torch.cat(train_logprobs_all, dim=0)
        valid_logprobs = torch.cat(valid_logprobs_all, dim=0)

        train_1_ppl = torch.exp(-train_logprobs[...,0].mean())
        train_2_ppl = torch.exp(-train_logprobs[...,1].mean())

        all_valid_1_ppl = [torch.exp(-valid_logprobs[...,0].mean()) for valid_logprobs in valid_logprobs_all]
        all_valid_2_ppl = [torch.exp(-valid_logprobs[...,1].mean()) for valid_logprobs in valid_logprobs_all]


        logging.info(f"Model 1 of {self.opt_model_1.__class__.__name__} type, with parameter count: {self.opt_model_1.num_parameters()}")
        logging.info(f"Model 2 of {self.opt_model_2.__class__.__name__} type, with parameter count: {self.opt_model_2.num_parameters()}")

        logging.info(f"Train Model 1 PPL: {train_1_ppl}, Train Model 2 PPL: {train_2_ppl}, Valid Model 1 PPLS: {all_valid_1_ppl}, Valid Model 2 PPLS: {all_valid_2_ppl}")


        first_best = train_logprobs[...,0] > train_logprobs[...,1]
        second_best = train_logprobs[...,1] > train_logprobs[...,0]
        logging.info(f"First best: {first_best.sum()}, second best: {second_best.sum()}, total {len(train_logprobs)}")

        train_datasets = []
        valid_datasets = []
        for train in prepared_trains:
            train_ds = TensorDataset(*train)
            if max_train is not None and max_train < len(train_ds):
                logging.info(f"Using only {max_train} samples for training, from original {len(train_ds)}")
                train_ds = Subset(train_ds, range(max_train))
            train_datasets.append(train_ds)
        
        for valid in prepared_valids:
            valid_ds = TensorDataset(*valid)
            if max_valid is not None and max_valid < len(valid_ds):
                logging.info(f"Using only {max_valid} samples for validation, from original {len(valid_ds)}")
                valid_ds = Subset(valid_ds, range(max_valid))
            valid_datasets.append(valid_ds)

        
        train_dataset = torch.utils.data.ConcatDataset(train_datasets)

        train_dataloader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True)
        valid_dataloaders = [DataLoader(valid_dataset, batch_size=train_bs, shuffle=True) for valid_dataset in valid_datasets]

        optimizer = torch.optim.Adam(self.comb_params.parameters(), lr=lr)

        best_valid_loss = None
        for epoch in range(n_epochs):
            self.comb_params.train()
            if torch.cuda.is_available():
                self.comb_cuda()
            for i,batch in enumerate(tqdm(train_dataloader)):
                
                comb_losses, model_1_losses, model_2_losses = self.combine_batch_for_fit(batch)
                loss = comb_losses.mean()
                model_1_loss = model_1_losses.mean().detach()
                model_2_loss = model_2_losses.mean().detach()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            self.comb_params.eval()

            checkpoint_data= {}

            valid0_comb_loss = None
            for valid_set, valid_dataloader in enumerate(valid_dataloaders):
                logger.info(f"Validating on dataset {valid_set}")

                comb_loss = 0
                model_1_loss = 0
                model_2_loss = 0
                eval_num = 0

                for i,batch in enumerate(tqdm(valid_dataloader)):
                    with torch.no_grad():
                        comb_losses, model_1_losses, model_2_losses = self.combine_batch_for_fit(batch)
                    
                    comb_loss += comb_losses.sum()
                    model_1_loss += model_1_losses.sum()
                    model_2_loss += model_2_losses.sum()
                    eval_num += len(model_1_losses)


                comb_loss /= eval_num
                model_1_loss /= eval_num
                model_2_loss /= eval_num

                comb_ppl = torch.exp(comb_loss)
                model_1_ppl = torch.exp(model_1_loss)
                model_2_ppl = torch.exp(model_2_loss)



                logger.info(f"Epoch {epoch} Valid set {valid_set}: Combined model Loss {comb_loss} Combined model  PPL {comb_ppl}")
                logger.info(f"Epoch {epoch} Valid set {valid_set}: Model 1 Loss {model_1_loss} Model 1 PPL {model_1_ppl}")
                logger.info(f"Epoch {epoch} Valid set {valid_set}: Model 2 Loss {model_2_loss} Model 2 PPL {model_2_ppl}")

                checkpoint_data[f"valid_{valid_set}_comb_loss"] = comb_loss.detach().cpu()
                checkpoint_data[f"valid_{valid_set}_comb_ppl"] = comb_ppl.detach().cpu()
                checkpoint_data[f"valid_{valid_set}_model_1_loss"] = model_1_loss.detach().cpu()
                checkpoint_data[f"valid_{valid_set}_model_1_ppl"] = model_1_ppl.detach().cpu()
                checkpoint_data[f"valid_{valid_set}_model_2_loss"] = model_2_loss.detach().cpu()
                checkpoint_data[f"valid_{valid_set}_model_2_ppl"] = model_2_ppl.detach().cpu()

                if valid_set == 0:
                    valid0_comb_loss = comb_loss


            if save_dir is not None:
                checkpoint_data = {
                    "epoch": epoch,
                    "max_train": max_train,
                    "max_valid": max_valid,
                    "train_bs": train_bs,
                    "lr": lr,
                    "n_epochs": n_epochs,
                    **checkpoint_data
                }
                save_path = os.path.join(save_dir, f"checkpoint_{epoch}")
                if save_only_last:
                    logging.info(f"Saving only last checkpoint to {save_path}")
                    save_path = os.path.join(save_dir, "checkpoint_last")
                logging.info(f"Saving checkpoint to {save_path}")
                self.save_to_disk(save_path, checkpoint_data=checkpoint_data, overwrite=True)
                if best_valid_loss is None or valid0_comb_loss < best_valid_loss:
                    save_path = os.path.join(save_dir, "best_model")
                    logging.info(f"New best model, saving to {save_path}")
                    self.save_to_disk(save_path, checkpoint_data=checkpoint_data, overwrite=True)
                    best_valid_loss = valid0_comb_loss



    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if past_key_values is not None:
            model_1_past, model_2_past = past_key_values
        else:
            model_1_past = model_2_past = None
        

        model_1_outputs = self.opt_model_1.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=model_1_past,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=return_dict,
        )
        model_1_hidden = model_1_outputs[0]
        model_2_outputs = self.opt_model_2.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=model_2_past,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=return_dict,
        )
        model_2_hidden = model_2_outputs[0]


        model_1_logits = self.opt_model_1.lm_head(model_1_hidden)
        model_2_logits = self.opt_model_2.lm_head(model_2_hidden)

        model_1_logprobs = F.log_softmax(model_1_logits, dim=-1)
        model_2_logprobs = F.log_softmax(model_2_logits, dim=-1)


        model_1_entropy = calc_entropy(model_1_logits) 
        model_2_entropy = calc_entropy(model_2_logits) 

        entropies = torch.cat((model_1_entropy, model_2_entropy), dim=-1)
        hidden = (model_1_hidden, model_2_hidden)
        logprobs = (model_1_logprobs, model_2_logprobs)

        #Combine
        logits = self.combine_for_forward(entropies, logprobs, hidden)

        past_key_values = None 
        loss = None
        
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            logger.info(f"Logits shape: {logits.shape}, shift_logits shape: {shift_logits.shape}, shift_labels shape: {shift_labels.shape}")

            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            if use_cache:
                assert len(model_1_outputs) == 2 and len(model_1_outputs) == 2, "Unexpected output format of underlying OPTForCausalLM model."
                past_key_values = (model_1_outputs[1], model_2_outputs[1])
            else:
                past_key_values = None
            return tuple(v for v in [loss, logits, past_key_values] if v is not None)

        if use_cache: 
            assert model_1_outputs.past_key_values is not None and model_2_outputs.past_key_values is not None, "Unexpected output format of underlying OPTForCausalLM model."
            past_key_values = (model_1_outputs.past_key_values, model_2_outputs.past_key_values)
        else:
            past_key_values = None
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
        )


class OutputsAndEntropyTrainedCombinedOPTForCausalLM(TrainedCombinedOPTForCausalLM):
    def __init__(self, opt_model_1, opt_model_2, n_layers = 2, hidden_size = 512, only_hidden = False, with_entropies = True, with_outputs = True, adaptive = False):
        super().__init__(opt_model_1, opt_model_2)
        self.only_hidden = only_hidden
        self.with_entropies = with_entropies
        self.with_outputs = with_outputs
        self.adaptive = adaptive

        input_size = 0
        if with_outputs:
            if only_hidden:
                input_size += self.opt_model_1.config.hidden_size + self.opt_model_2.config.hidden_size
            else:
                input_size += self.opt_model_1.config.vocab_size + self.opt_model_2.config.vocab_size
        if with_entropies:
            input_size += 2 #Add 2 for the entropies

        logging.info(f"Input size for network is {input_size}")
        assert input_size > 0, "At least one of with_outputs and with_entropies must be True."

        if adaptive:
            output_size = self.opt_model_1.config.vocab_size 
            assert self.opt_model_1.config.vocab_size == self.opt_model_2.config.vocab_size
        else: 
            output_size = 1
        layers = []
        bn = torch.nn.BatchNorm1d(input_size, affine=False)
        layers.append(bn)
        fc1 = torch.nn.Linear(input_size, hidden_size, bias=True)
        layers.append(fc1)
        layers.append(torch.nn.ReLU())
        for layer in range(n_layers):
            fc = torch.nn.Linear(hidden_size, hidden_size, bias=True)
            layers.append(fc)
            layers.append(torch.nn.ReLU())
        fc2 = torch.nn.Linear(hidden_size, output_size, bias=True)
        layers.append(fc2)
        layers.append(torch.nn.Sigmoid())
        

        self.ffn = torch.nn.Sequential(*layers)
        device, dtype = fc2.weight.device, fc2.weight.dtype
        with torch.no_grad():
            self.model_1_head = clone_linear(self.opt_model_1.lm_head)
            self.model_2_head = clone_linear(self.opt_model_2.lm_head)
            self.model_1_head = self.model_1_head.to(device=device, dtype=dtype)
            self.model_2_head = self.model_2_head.to(device=device, dtype=dtype)

        
        self.comb_params = torch.nn.ModuleDict(
            {
                'ffn':self.ffn,
                'model_1_head':self.model_1_head,
                'model_2_head':self.model_2_head,
            }
        )

        for param in self.model_1_head.parameters():
            param.requires_grad = False
        for param in self.model_2_head.parameters():
            param.requires_grad = False


        assert isinstance(self.opt_model_1, OPTForCausalLM) and isinstance(self.opt_model_2, OPTForCausalLM)
        assert self.opt_model_1.config.vocab_size == self.opt_model_2.config.vocab_size

    def combine_batch_for_fit(self, batch):
        entropies, logprobs, labels, hidden_1, hidden_2 = batch
        logprobs_1 = logprobs[..., 0]
        logprobs_2 = logprobs[..., 1]

        orig_device, orig_dtype = entropies.device, entropies.dtype
        entropies = entropies.to(*self.get_comb_device_dtype())
        hidden_1 = hidden_1.to(*self.get_comb_device_dtype())
        hidden_2 = hidden_2.to(*self.get_comb_device_dtype())

        comb_input = None
        if self.with_outputs:
            if self.only_hidden:
                comb_input = torch.cat([hidden_1, hidden_2], dim=-1)
            else:
                logits_1 = self.model_1_head(hidden_1)
                logits_2 = self.model_2_head(hidden_2)

                probs_1 = torch.softmax(logits_1, dim=-1)
                probs_2 = torch.softmax(logits_2, dim=-1)
                probs = torch.cat([probs_1, probs_2], dim=-1)
                comb_input  = torch.cat([probs], dim=-1)
        if self.with_entropies:
            if comb_input is None:
                comb_input = entropies
            else:
                comb_input = torch.cat([comb_input, entropies], dim=-1)
        combination = self.ffn(comb_input).squeeze(-1)
        combination = combination.to(orig_device, orig_dtype)

        if not self.adaptive:
            logprobs_comb = torch.log(torch.exp(logprobs_1) * combination + torch.exp(logprobs_2) * (1-combination)) 
            return -logprobs_comb, -logprobs_1, -logprobs_2
        else:
            #Only label logprobs aren't enough, need to re-calculate all probs from hidden 
            logits_1 = self.model_1_head(hidden_1)
            logits_2 = self.model_2_head(hidden_2)
            logits_1 = logits_1.to(orig_device, orig_dtype)
            logits_2 = logits_2.to(orig_device, orig_dtype)


            all_probs_1 = torch.softmax(logits_1, dim=-1)
            all_probs_2 = torch.softmax(logits_2, dim=-1)
            
            all_logprobs_1 = torch.log_softmax(logits_1, dim=-1) 
            all_logprobs_2 = torch.log_softmax(logits_2, dim=-1)


            label_logprobs_1 = all_logprobs_1.gather(-1, labels.unsqueeze(-1)).view(-1)

            label_logprobs_2 = all_logprobs_2.gather(-1, labels.unsqueeze(-1)).view(-1)

            max_1, max_2 = torch.max(torch.abs(label_logprobs_1 - logprobs_1)), torch.max(torch.abs(label_logprobs_2 - logprobs_2))

            argmax_1, argmax_2 = torch.argmax(torch.abs(label_logprobs_1 - logprobs_1)), torch.argmax(torch.abs(label_logprobs_2 - logprobs_2))

            #logger.info(f"Max difference between label logprobs and recalculated logprobs: {max_1}, {max_2}, at {argmax_1}, {argmax_2}, with values {label_logprobs_1[argmax_1]}, {logprobs_1[argmax_1]} and {label_logprobs_2[argmax_2]}, {logprobs_2[argmax_2]}")
            #Since these are senstitive logprobs, allow five percent lee-way
            assert torch.allclose(label_logprobs_1, logprobs_1, rtol = 0.025, atol=1e-4), f"When recalculating logprobs, label ones should be the same as the ones passed in. Max difference for values  {label_logprobs_1[argmax_1]}, {logprobs_1[argmax_1]} is {max_1}"
            assert torch.allclose(label_logprobs_2, logprobs_2, rtol = 0.025, atol=1e-4), f"When recalculating logprobs, label ones should be the same as the ones passed in. Max difference for values  {label_logprobs_2[argmax_2]}, {logprobs_2[argmax_2]} is {max_2}"

            assert combination.shape == all_probs_1.shape, "For adaptive combination shape should be the same as probs shape"

            comb_probs = all_probs_1 * combination + all_probs_2 * (1-combination)

            comb_probs = comb_probs/(torch.sum(comb_probs, dim=-1, keepdim=True) + 1e-8)

            comb_logprobs = torch.log(comb_probs)

            label_logprobs_comb = comb_logprobs.gather(-1, labels.unsqueeze(-1))

            return -label_logprobs_comb, -label_logprobs_1, -label_logprobs_2

    def combine_for_forward(self, entropies, logprobs, hidden):
        model_1_logprobs, model_2_logprobs = logprobs

        input_dtype = model_1_logprobs.dtype

        model_1_logprobs, model_2_logprobs = model_1_logprobs.to(torch.float32), model_2_logprobs.to(torch.float32)
        entropies = entropies.to(torch.float32)

        model_1_probs, model_2_probs = torch.exp(model_1_logprobs), torch.exp(model_2_logprobs)

        comb_input=None
        if self.with_outputs:
            logging.info(f"Combinging for forward with inputs")
            if self.only_hidden:
                comb_input = torch.cat([hidden_1, hidden_2], dim=-1)
            else:
                comb_input = torch.cat([model_1_probs, model_2_probs], dim=-1)

        if self.with_entropies:
            if comb_input is not None:
                logging.info(f"Combinging for forward with inputs and entropy")
                comb_input = torch.cat([comb_input, entropies], dim=-1)
            else:
                
                logging.info(f"Combinging for forward with entropies only")
                comb_input = entropies

        logging.info(f"Comb input shape is {comb_input.shape}")
        shape = comb_input.shape
        comb_input = comb_input.view(-1, shape[-1])

        orig_device, orig_dtype = comb_input.device, comb_input.dtype 

        device, dtype = self.get_comb_device_dtype()
        comb_input = comb_input.to(device,dtype)
        combination = self.ffn(comb_input).view(shape[:-1]+ (-1,)).squeeze(-1)
        combination = combination.to(orig_device,orig_dtype)


        if not self.adaptive:
            logging.info(f"Combining with non adaptive combination {combination.shape}, {model_1_probs.shape}")
            assert combination.shape == model_1_probs.shape[:-1]
            combination = combination.unsqueeze(-1)
        else:
            logging.info(f"Combining with adaptive combination {combination.shape}, {model_1_probs.shape}")
            assert combination.shape == model_1_probs.shape


        #Print devices and dtypes of all operands
        # logging.info(f"Combination device is {combination.device}, model_1_probs device is {model_1_probs.device}, model_2_probs device is {model_2_probs.device}")
        # logging.info(f"Combination dtype is {combination.dtype}, model_1_probs dtype is {model_1_probs.dtype}, model_2_probs dtype is {model_2_probs.dtype}")
        
        #logging.info(f"Combination device is {combination.device}, model_1_probs device is {model_1_probs.device}, model_2_probs device is {model_2_probs.device}")
        combined_probs = combination*model_1_probs + (1-combination)*model_2_probs


        logits = torch.log(1e-10 + combined_probs) # Return to logit space, 1-e7 to avoid underflow in FP16

        logits = logits.to(input_dtype)

        return logits

class AdaptiveLinearTrainedCombinedOPTForCausalLM(TrainedCombinedOPTForCausalLM):
    def __init__(self, opt_model_1, opt_model_2, adaptive=True):
        super().__init__(opt_model_1, opt_model_2)

        self.adaptive = adaptive
        if not self.adaptive:
            output_size = 1
        else:
            output_size = self.opt_model_1.config.vocab_size
        comb = torch.nn.Linear(0, output_size) 
        self.ffn = torch.nn.Sequential(comb, torch.nn.Sigmoid())
        
        device, dtype = comb.weight.device, comb.weight.dtype
        
        with torch.no_grad():
            self.model_1_head = clone_linear(self.opt_model_1.lm_head)
            self.model_2_head = clone_linear(self.opt_model_2.lm_head)
            self.model_1_head = self.model_1_head.to(device=device, dtype=dtype)
            self.model_2_head = self.model_2_head.to(device=device, dtype=dtype)
        self.comb_params = torch.nn.ModuleDict(
            {
                "ffn": self.ffn,
                "model_1_head": self.model_1_head,
                "model_2_head": self.model_2_head,
            }
        )
    
        for param in self.model_1_head.parameters():
            param.requires_grad = False
        for param in self.model_2_head.parameters():
            param.requires_grad = False
        assert isinstance(self.opt_model_1, OPTForCausalLM) and isinstance(self.opt_model_2, OPTForCausalLM)
        assert self.opt_model_1.config.vocab_size == self.opt_model_2.config.vocab_size

    def combine_batch_for_fit(self, batch):
        entropies, logprobs, labels, hidden_1, hidden_2 = batch
        logprobs_1 = logprobs[..., 0]
        logprobs_2 = logprobs[..., 1]

        orig_device, orig_dtype = entropies.device, entropies.dtype
        comb_device, comb_dtype = self.get_comb_device_dtype()

        if self.adaptive:
            labels = labels.to(comb_device)
            logprobs_1 = logprobs_1.to(comb_device, comb_dtype)
            logprobs_2 = logprobs_2.to(comb_device, comb_dtype)
            #Only label logprobs aren't enough, need to re-calculate all probs from hidden 
            hidden_1 = hidden_1.to(comb_device, comb_dtype)
            hidden_2 = hidden_2.to(comb_device, comb_dtype)
            logits_1 = self.model_1_head(hidden_1)
            logits_2 = self.model_2_head(hidden_2)

            all_probs_1 = torch.softmax(logits_1, dim=-1)
            all_probs_2 = torch.softmax(logits_2, dim=-1)

            all_logprobs_1 = torch.log_softmax(logits_1, dim=-1) 
            all_logprobs_2 = torch.log_softmax(logits_2, dim=-1)

  
            label_logprobs_1 = all_logprobs_1.gather(-1, labels.unsqueeze(-1)).view(-1)

            label_logprobs_2 = all_logprobs_2.gather(-1, labels.unsqueeze(-1)).view(-1)



            max_1, max_2 = torch.max(torch.abs(label_logprobs_1 - logprobs_1)), torch.max(torch.abs(label_logprobs_2 - logprobs_2))
            argmax_1, argmax_2 = torch.argmax(torch.abs(label_logprobs_1 - logprobs_1)), torch.argmax(torch.abs(label_logprobs_2 - logprobs_2))

            assert torch.allclose(label_logprobs_1, logprobs_1, rtol = 0.025, atol=1e-4), "When recalculating logprobs, label ones should be the same as the ones passed in."
            assert torch.allclose(label_logprobs_2, logprobs_2, rtol = 0.025, atol=1e-4), "When recalculating logprobs, label ones should be the same as the ones passed in."

            empty_input = torch.tensor([], device=comb_device, dtype=comb_dtype)
            combination = self.ffn(empty_input).squeeze(-1)

            while combination.dim() < all_probs_1.dim():
                combination = combination.unsqueeze(0)

            assert combination.shape[-1] == all_probs_1.shape[-1] 


            comb_probs = all_probs_1 * combination + all_probs_2 * (1-combination)

            comb_probs = comb_probs/(torch.sum(comb_probs, dim=-1, keepdim=True) + 1e-8)

            comb_logprobs = torch.log(comb_probs)


            label_logprobs_comb = comb_logprobs.gather(-1, labels.unsqueeze(-1))

            label_logprobs_comb = label_logprobs_comb.to(orig_device, orig_dtype)
            label_logprobs_1 = label_logprobs_1.to(orig_device, orig_dtype)
            label_logprobs_2 = label_logprobs_2.to(orig_device, orig_dtype)

            return -label_logprobs_comb, -label_logprobs_1, -label_logprobs_2
        else: #Can just use the logprobs passed in

            empty_input = torch.tensor([], device=comb_device, dtype=comb_dtype)
            combination = self.ffn(empty_input).squeeze(-1)
            combination = combination.to(orig_device, orig_dtype).squeeze(-1)
            logger.info(f"Combination: {combination}")
            probs_1, probs_2 = torch.exp(logprobs_1), torch.exp(logprobs_2)
            comb_probs = probs_1 * combination + probs_2 * (1-combination)
            comb_logprobs = torch.log(comb_probs)

            return -comb_logprobs, -logprobs_1, -logprobs_2

    
    def combine_for_forward(self, entropies, logprobs, hidden):
        model_1_logprobs, model_2_logprobs = logprobs

        orig_dtype = model_1_logprobs.dtype
        model_1_logprobs, model_2_logprobs = model_1_logprobs.to(torch.float32), model_2_logprobs.to(torch.float32)

        model_1_probs, model_2_probs = torch.exp(model_1_logprobs), torch.exp(model_2_logprobs)

        comb_device, comb_dtype = self.get_comb_device_dtype()
        empty_input = torch.tensor([], device=comb_device, dtype=comb_dtype)
        combination = self.ffn(empty_input).squeeze(-1)

        combination = combination.to(model_1_probs.device, model_1_probs.dtype).squeeze(-1)

        if self.adaptive:
            assert combination.shape[-1] == model_1_probs.shape[-1] 
        else:
            assert combination.shape[-1] == 1


        combined_probs = combination*model_1_probs + (1-combination)*model_2_probs

        
        logits = torch.log(1e-20 + combined_probs) # Return to logit space 

        logits = logits.to(orig_dtype)

        return logits

















class LinearTrainedCombinedOPTForCausalLM(TrainedCombinedOPTForCausalLM):
    def __init__(self, opt_model_1, opt_model_2):
        super().__init__(opt_model_1, opt_model_2)

        self.lambd = 0.5
        self.comb_params = torch.nn.ModuleDict({})


    def load_from_disk(self, load_directory):
        super().load_from_disk(load_directory)
        checkpoint_data = torch.load(os.path.join(load_directory, "checkpoint_data.bin"),  map_location="cpu")
        assert "best_lambd" in checkpoint_data, "No best_lambd found in checkpoint data"
        self.lambd = checkpoint_data["best_lambd"].item()

    def get_comb_device_dtype(self):
        #Always return fp32 on CPU
        return torch.device("cpu"), torch.float32

    def fit(self, trains, valids, process_bs = 1,train_bs=1000, n_epochs = 10, lr = 1e-5, early_stopping = False, only_run_datasets=False, max_train=None, max_valid=None, save_dir=None, **kwargs):
        assert "input_ids" in trains[0][0] and "input_ids" in valids[0][0]

        prepared_trains = []
        prepared_valids = []
        for i in range(len(trains)):
            logger.info(f"Processing dataset for fit, train {i} model 1")
            processed_train_1 = self.process_dataset_for_fit(trains[i], self.opt_model_1, bs=process_bs, return_hidden=True, dtype=torch.float32)
            if only_run_datasets:
                del processed_train_1
                processed_train_1 = None
            logger.info(f"Processing dataset for fit, train {i} model 2")
            processed_train_2 = self.process_dataset_for_fit(trains[i], self.opt_model_2, bs=process_bs, return_hidden=True, dtype=torch.float32)
            if only_run_datasets:
                del processed_train_2
                processed_train_2 = None
            train = self.prepare_dataset_for_fit(processed_train_1, processed_train_2)
            prepared_trains.append(train)
        for i in range(len(valids)):
            logger.info(f"Processing dataset for fit, valid {i} model 1")
            processed_valid_1 = self.process_dataset_for_fit(valids[i], self.opt_model_1, bs=process_bs, return_hidden=True, dtype=torch.float32)
            if only_run_datasets:
                del processed_valid_1
                processed_valid_1 = None
            logger.info(f"Processing dataset for fit, valid {i} model 2")
            processed_valid_2 = self.process_dataset_for_fit(valids[i], self.opt_model_2, bs=process_bs, return_hidden=True, dtype=torch.float32)
            if only_run_datasets:
                del processed_valid_2
                processed_valid_2 = None
            valid = self.prepare_dataset_for_fit(processed_valid_1, processed_valid_2)
            prepared_valids.append(valid)


        

        train_logprobs_all = [self.get_logprobs_from_prepared_batch(train) for train in prepared_trains]
        valid_logprobs_all = [self.get_logprobs_from_prepared_batch(valid) for valid in prepared_valids]

        train_logprobs_all = [train_logprobs[:max_train] for train_logprobs in train_logprobs_all]
        valid_logprobs_all = [valid_logprobs[:max_valid] for valid_logprobs in valid_logprobs_all]

        train_logprobs = torch.cat(train_logprobs_all, dim=0)
        valid_logprobs = torch.cat(valid_logprobs_all, dim=0)
        
        train_1_ppl = torch.exp(-train_logprobs[...,0].mean())
        train_2_ppl = torch.exp(-train_logprobs[...,1].mean())

        all_valid_1_ppl = [torch.exp(-valid_logprobs[...,0].mean()) for valid_logprobs in valid_logprobs_all]
        all_valid_2_ppl = [torch.exp(-valid_logprobs[...,1].mean()) for valid_logprobs in valid_logprobs_all]


        logging.info(f"Model 1 of {self.opt_model_1.__class__.__name__} type, with parameter count: {self.opt_model_1.num_parameters()}")
        logging.info(f"Model 2 of {self.opt_model_2.__class__.__name__} type, with parameter count: {self.opt_model_2.num_parameters()}")

        logging.info(f"Train Model 1 PPL: {train_1_ppl}, Train Model 2 PPL: {train_2_ppl}, Valid Model 1 PPLS: {all_valid_1_ppl}, Valid Model 2 PPL: {all_valid_2_ppl}")

        first_best = train_logprobs[...,0] > train_logprobs[...,1]
        second_best = train_logprobs[...,1] > train_logprobs[...,0]
        logging.info(f"First best: {first_best.sum()}, second best: {second_best.sum()}, total {len(train_logprobs)}")

        

        logging.info(f"Fitting linear combination")
        best_lambd = 0
        best_ppl = train_1_ppl 
        train_1_probs = torch.exp(train_logprobs[...,0])
        train_2_probs = torch.exp(train_logprobs[...,1])


        all_valid_1_probs = [torch.exp(valid_logprobs[...,0]) for valid_logprobs in valid_logprobs_all]
        all_valid_2_probs = [torch.exp(valid_logprobs[...,1]) for valid_logprobs in valid_logprobs_all]
        
        all_train_ppl = []
        for lambd in torch.linspace(0, 1, 100):
            comb_probs = lambd * train_1_probs + (1 - lambd) * train_2_probs
            comb_ppl = torch.exp(-torch.log(comb_probs).mean())
            all_train_ppl.append(comb_ppl)
            if comb_ppl < best_ppl:
                best_ppl = comb_ppl
                best_lambd = lambd
        self.lambd = best_lambd
        logging.info(f"All train PPL: {all_train_ppl}")

        


        mean_comb_probs = 0.5 * train_1_probs + 0.5 * train_2_probs
        mean_comb_ppl = torch.exp(-torch.log(mean_comb_probs).mean())
        logging.info(f"Mean Train PPL: {mean_comb_ppl}")

        all_mean_valid_comb_probs = [0.5 * valid_1_probs + 0.5 * valid_2_probs for valid_1_probs, valid_2_probs in zip(all_valid_1_probs, all_valid_2_probs)]
        all_mean_valid_comb_ppl = [torch.exp(-torch.log(mean_valid_comb_probs).mean()) for mean_valid_comb_probs in all_mean_valid_comb_probs]

        logging.info(f"All mean valid PPL: {all_mean_valid_comb_ppl}")

        all_comb_valid_probs = [best_lambd * valid_1_probs + (1 - best_lambd) * valid_2_probs for valid_1_probs, valid_2_probs in zip(all_valid_1_probs, all_valid_2_probs)]
        all_comb_valid_ppl = [torch.exp(-torch.log(comb_valid_probs).mean()) for comb_valid_probs in all_comb_valid_probs]
        logging.info(f"All valid PPL: {all_comb_valid_ppl}")

        if save_dir is not None:
                checkpoint_data = {
                    "epoch": 0,
                    "max_train": max_train,
                    "max_valid": max_valid,
                    "mean_comb_ppl": mean_comb_ppl,
                    "best_lambd": best_lambd,
                    "all_mean_valid_comb_ppl": all_mean_valid_comb_ppl,
                    "all_comb_valid_ppl": all_comb_valid_ppl,
                    "all_train_ppl": all_train_ppl,
                }
                save_path = os.path.join(save_dir, f"checkpoint_0")
                logging.info(f"Saving checkpoint to {save_path}")
                self.save_to_disk(save_path, checkpoint_data=checkpoint_data, overwrite=True)

            
    
    def combine_for_forward(self, entropies, logprobs, hidden):
        model_1_logprobs, model_2_logprobs = logprobs
        model_1_probs, model_2_probs = torch.exp(model_1_logprobs), torch.exp(model_2_logprobs)
        comb_probs = self.lambd * model_1_probs + (1 - self.lambd) * model_2_probs
        comb_logprobs = torch.log(comb_probs)
        return comb_logprobs



class EntropyTrainedCombinedOPTForCausalLM(TrainedCombinedOPTForCausalLM):
    def __init__(self, opt_model_1, opt_model_2, n_layers = 2, hidden_size = 256):
        super().__init__(opt_model_1, opt_model_2)
        input_size = 2 #Add 2 for the entropies


        layers = []
        bn = torch.nn.BatchNorm1d(input_size, affine=False)
        layers.append(bn)
        fc1 = torch.nn.Linear(input_size, hidden_size, bias=True)
        layers.append(fc1)
        layers.append(torch.nn.ReLU())
        for layer in range(n_layers):
            fc = torch.nn.Linear(hidden_size, hidden_size, bias=True)
            layers.append(fc)
            layers.append(torch.nn.ReLU())
        fc2 = torch.nn.Linear(hidden_size, 1, bias=True)
        layers.append(fc2)
        layers.append(torch.nn.Sigmoid())

        self.ffn = torch.nn.Sequential(*layers)
        self.comb_params = torch.nn.ModuleDict({
            'ffn':self.ffn
        })

        assert isinstance(self.opt_model_1, OPTForCausalLM) and isinstance(self.opt_model_2, OPTForCausalLM)
        assert self.opt_model_1.config.vocab_size == self.opt_model_2.config.vocab_size

    

    def combine_batch_for_fit(self, batch):
        entropies, logprobs, labels, hidden_1, hidden_2 = batch
        logprobs_1 = logprobs[..., 0]
        logprobs_2 = logprobs[..., 1]
        orig_device, orig_dtype = entropies.device, entropies.dtype
        entropies = entropies.to(*self.get_comb_device_dtype())
        combination = self.ffn(entropies).squeeze(-1).to(orig_device, orig_dtype)

        logprobs_comb = torch.log(torch.exp(logprobs_1) * combination + torch.exp(logprobs_2) * (1-combination)) 
        return -logprobs_comb, -logprobs_1, -logprobs_2

    def combine_for_forward(self, entropies, logprobs, hidden):
        model_1_logprobs, model_2_logprobs = logprobs
        model_1_probs, model_2_probs = torch.exp(model_1_logprobs), torch.exp(model_2_logprobs)
        
        orig_device, orig_dtype = entropies.device, entropies.dtype
        entropies = entropies.to(*self.get_comb_device_dtype())
        combination = self.ffn(entropies).squeeze(-1).to(orig_device, orig_dtype)

        combined_probs = prediction*model_1_probs + (1-prediction)*model_2_probs

        
        logits = torch.log(1e-20 + combined_probs) # Return to logit space 


        return logits
