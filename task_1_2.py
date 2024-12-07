from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from src.customized_gpt2 import CustomizedGPT2LMHeadModel
from src.decoding import golden_greedy_decoding_wo_cache, customized_greedy_decoding
from src.data_helper import load_prompt_dataset
from tqdm import tqdm


def eval_throughput():
    times = [0, 0]
    total_tokens = 0

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    original_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", attn_implementation="eager", device_map=device)
    custom_model = CustomizedGPT2LMHeadModel.from_pretrained("openai-community/gpt2", attn_implementation="eager", device_map=device)

    # Load dataset
    prompt_dataset = load_prompt_dataset("data/data.txt")

    # Do inference
    for i in tqdm(range(0, (len(prompt_dataset) + bsz - 1) // bsz)):
        batch = prompt_dataset[i * bsz: (i + 1) * bsz]
        golden_wo_cache_res, golden_wo_cache_time = golden_greedy_decoding_wo_cache(original_model, tokenizer, device, batch, MAX_NEW_LENGTH)
        custom_res, custom_time = customized_greedy_decoding(custom_model, tokenizer, device, batch, MAX_NEW_LENGTH)
        times[0] += golden_wo_cache_time
        times[1] += custom_time
        total_tokens += len(batch) * MAX_NEW_LENGTH
        assert torch.equal(golden_wo_cache_res, custom_res), "Decoding results are not equal"
    
    print(f"Totol output tokens is: {total_tokens} tokens")
    print(f"Time taken for golden greedy decoding without KV cache: {times[0]} - Throughput: {total_tokens/times[0]} tokens/s")
    print(f"Time taken for customized greedy decoding: {times[1]} - Throughput: {total_tokens/times[1]} tokens/s")

def eval_gpu_memeory():
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    # Load dataset
    prompt_dataset = load_prompt_dataset("data/data.txt")
    original_mem = customized_mem = None
    # torch.cuda.init()
    # torch.cuda.reset_peak_memory_stats(device) # Rest the record of GPU memory allocated peak

    # original_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", attn_implementation="eager", device_map=device)
    # for i in range(0, (len(prompt_dataset) + bsz - 1) // bsz):
    #     batch = prompt_dataset[i * bsz: (i + 1) * bsz]
    #     golden_greedy_decoding_wo_cache(original_model, tokenizer, device, batch, MAX_NEW_LENGTH)
    # original_mem = torch.cuda.max_memory_allocated(device)

    # torch.cuda.synchronize(device)
    # del original_model # Release the original model
    # torch.cuda.reset_peak_memory_stats(device) # Rest the peak GPU memory allocated
    
    custom_model = CustomizedGPT2LMHeadModel.from_pretrained("openai-community/gpt2", attn_implementation="eager", device_map=device)
    for i in range(0, (len(prompt_dataset) + bsz - 1) // bsz):
        batch = prompt_dataset[i * bsz: (i + 1) * bsz]
        customized_greedy_decoding(custom_model, tokenizer, device, batch, MAX_NEW_LENGTH)
    customized_mem = torch.cuda.max_memory_allocated(device) # Note that it would be better to test the two models separately to mitigate bias
    
    
    print(f"Max memeory allocated for golden greedy decoding without KV cache: {original_mem}")
    print(f"Max memeory allocated for customized greedy decoding: {customized_mem}")

if __name__ == "__main__":
    MAX_NEW_LENGTH = 100
    bsz = 16
    device = "cuda:1"
    # eval_throughput()
    eval_gpu_memeory()