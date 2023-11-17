from tqdm import tqdm 
import torch 
import logging
import torch.nn.functional as F

logger = logging.getLogger(__name__)
def calc_entropy(logits):
    orig_dtype = logits.dtype
    logits = logits.to(dtype=torch.float32)
    logprobs = F.log_softmax(logits, dim=-1)

    res =  -torch.sum(torch.exp(logprobs)*logprobs, dim=-1).unsqueeze(-1)
    return res.to(dtype=orig_dtype)

def valid_from_checkpoint_data(checkpoint_data):
    if 'valid_0_comb_ppl' not in checkpoint_data:
        return None
    
    assert 'valid_0_comb_ppl' in checkpoint_data
    assert 'valid_1_comb_ppl' in checkpoint_data
    assert 'valid_0_model_1_ppl' in checkpoint_data
    assert 'valid_1_model_1_ppl' in checkpoint_data
    assert 'valid_0_model_2_ppl' in checkpoint_data
    assert 'valid_1_model_2_ppl' in checkpoint_data

    valid_0_comb_ppl = checkpoint_data['valid_0_comb_ppl']
    valid_1_comb_ppl = checkpoint_data['valid_1_comb_ppl']
    valid_0_model_1_ppl = checkpoint_data['valid_0_model_1_ppl']
    valid_1_model_1_ppl = checkpoint_data['valid_1_model_1_ppl']
    valid_0_model_2_ppl = checkpoint_data['valid_0_model_2_ppl']
    valid_1_model_2_ppl = checkpoint_data['valid_1_model_2_ppl']

    return valid_0_comb_ppl, valid_1_comb_ppl, valid_0_model_1_ppl, valid_1_model_1_ppl, valid_0_model_2_ppl, valid_1_model_2_ppl



def calc_perplexity(model, encoded, max_length=1024, stride=512, max_forward=100):
    logger.info(f"Calculating perplexity for model of type {type(model)}, with max_length={max_length}, stride={stride}. Max len in config: {model.config.max_position_embeddings}")
    stride = 512
    seq_len = encoded.size(1)
    assert encoded.size(0) == 1
    if seq_len > stride*max_forward:
        logger.info(f"Truncating sequence to {stride*max_forward} tokens for {max_forward} forward passes.")
        seq_len = stride*max_forward

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encoded[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            #logger.info(f"Vocab size {model.config.vocab_size} linear shape {model.opt_model_1.lm_head.weight.shape}")
            logger.info(f"Forward pass {begin_loc} to {end_loc}, input_ids size {input_ids.size()}, labels size {target_ids.size()}")
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over input tokens.
            # Multiply it with trg_len to get the summation instead of average.
            # We will take average over all the tokens to get the true average
            # in the last step of this example.
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl 