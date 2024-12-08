from transformers import AutoTokenizer, AutoModelForCausalLM
from src.customized_gpt2 import CustomizedGPT2LMHeadModel
from src.decoding import golden_greedy_decoding_without_cache, customized_greedy_decoding, golden_greedy_decoding_with_cache
from src.evaluate import eval_throughput, eval_gpu_memory
from src.data_helper import load_prompt_dataset
import argparse

def get_args():
    parser = argparse.ArgumentParser(
        prog="Task-1.2",
        description="Script for conducting task 1.1 experiments",
    )
    parser.add_argument(
        "-l", "--desired_length",
        default=64,
        type=int
    )
    parser.add_argument(
        "-b", "--bsz",
        default=16,
        type=int
    )
    parser.add_argument(
        "-D", "--device",
        default="cuda"
    )
    parser.add_argument(
        "-t", "--task",
        choices=["eval_throughput", "eval_gpu_memory"],
        default="eval_throughput"
    )
    parser.add_argument(
        "-c", "--cache_strategy",
        choices=["no_kv_cache", "golden_kv_cache", "simple_kv_cache"]
    )
    parser.add_argument(
        "-d", "--dataset_path",
        default="data/data.txt"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt_data = load_prompt_dataset(args.dataset_path)
    match args.cache_strategy:
        case "no_kv_cache":
            decoding_method = golden_greedy_decoding_without_cache
        case "golden_kv_cache":
            decoding_method = golden_greedy_decoding_with_cache
        case "simple_kv_cache":
            decoding_method = customized_greedy_decoding

    if args.task == "eval_throughput":
        if args.cache_strategy == "simple_kv_cache":
            model = CustomizedGPT2LMHeadModel.from_pretrained(
                "openai-community/gpt2",
                attn_implementation="eager",
                device_map=args.device,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                "openai-community/gpt2",
                attn_implementation="eager",
                device_map=args.device,
            )
        throughput = eval_throughput(
            model,
            tokenizer,
            args.device,
            prompt_data,
            args.desired_length,
            args.bsz,
            decoding=decoding_method
        )
        print(f"Throughput: {throughput} tokens/s")
    else:
        mem = eval_gpu_memory(
            CustomizedGPT2LMHeadModel if args.cache_strategy == "simple_kv_cache" else AutoModelForCausalLM,
            "openai-community/gpt2",
            tokenizer,
            args.device,
            prompt_data,
            args.desired_length,
            args.bsz,
            decoding=decoding_method,
            attn_implementation="eager"
        )
        print(f"GPU Memory: {mem} bytes")