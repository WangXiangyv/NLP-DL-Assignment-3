import torch
from src.decoding import golden_greedy_decoding_without_cache, model_warming
from typing import Callable
import src.utils as utils
import jsonlines
from src.data_helper import load_gsm8k_dataset
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
    for i in range(0, (len(dataset) + bsz - 1) // bsz):
        batch = dataset[i * bsz: (i + 1) * bsz]
        _, time_consumption = decoding(model, tokenizer, device, batch, desired_length)
        tot_time += time_consumption
        total_tokens += len(batch) * desired_length
    
    return total_tokens/tot_time

def eval_gpu_memory(
    cls,
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
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(device)
    model = cls.from_pretrained(model_name_or_path, device_map=device, **kwargs)
    for i in range(0, (len(dataset) + bsz - 1) // bsz):
        batch = dataset[i * bsz: (i + 1) * bsz]
        decoding(model, tokenizer, device, batch, desired_length)
    gpu_mem = torch.cuda.max_memory_allocated(device)
    return gpu_mem


def eval_gsm8k(predictions, references):
    acc = 0
    for p, r in zip(predictions, references):
        if p is not None and p == r:
            acc += 1
    acc = 100.0 * acc / len(references)
    return acc