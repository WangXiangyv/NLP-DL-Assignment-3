import json
import torch
import torch.nn as nn
from src.decoding import golden_greedy_decoding_without_cache, model_warming
from typing import Callable
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers import PreTrainedModel

def eval_throughput(
    model,
    tokenizer,
    device,
    dataset,
    desired_length:int,
    bsz:int,
    decoding:Callable=golden_greedy_decoding_without_cache
):
    tot_time = 0
    total_tokens = 0
    model_warming(model, tokenizer, device, dataset[0:bsz]) # Warm up the model to avoid the compile cost 
    # Do inference
    for i in tqdm(range(0, (len(dataset) + bsz - 1) // bsz), leave=False):
        batch = dataset[i * bsz: (i + 1) * bsz]
        _, time_consumption = decoding(model, tokenizer, device, batch, desired_length)
        tot_time += time_consumption
        total_tokens += len(batch) * desired_length
    
    return total_tokens/tot_time

def eval_gpu_memory(
    model_name_or_path,
    tokenizer,
    device,
    dataset,
    desired_length:int,
    bsz:int,
    decoding:Callable=golden_greedy_decoding_without_cache,
    **kwargs
):
    """ Note that this function should be called when no memory has been allocated on the given device """
    assert torch.cuda.memory_allocated(device) == 0, "'eval_gpu_memory' should be called when no memory has been allocated on the given device"
    
    torch.cuda.reset_peak_memory_stats(device)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    for i in tqdm(range(0, (len(dataset) + bsz - 1) // bsz), leave=False):
        batch = dataset[i * bsz: (i + 1) * bsz]
        decoding(model, tokenizer, device, batch, desired_length)
    gpu_mem = torch.cuda.max_memory_allocated(device)
    return gpu_mem


def eval_gsm8k(model):
    raise NotImplementedError