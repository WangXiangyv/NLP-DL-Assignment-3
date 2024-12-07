import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoConfig, GPT2LMHeadModel, LlamaForCausalLM, Conv1D
from optimum.quanto import QuantizedModelForCausalLM, qint4
from src.decoding import golden_greedy_decoding_without_cache, golden_greedy_decoding_with_cache
from src.data_helper import load_prompt_dataset
from src.evaluate import eval_throughput, eval_gpu_memory
import argparse

def get_args():
    parser = argparse.ArgumentParser(
        prog="Task-1.1",
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
        "-u", "--use_cache",
        action="store_true"
    )
    parser.add_argument(
        "-q", "--quantization_method",
        choices=["original", "int8", "int4", "int2"],
        default="original"
    )
    parser.add_argument(
        "-m", "--model_name_or_path",
        default="facebook/opt-1.3b"
    )
    parser.add_argument(
        "-d", "--dataset_path",
        default="data/data.txt"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization_config = QuantoConfig(weights=args.quantization_method) if args.quantization_method != "original" else None
    prompt_data = load_prompt_dataset(args.dataset_path)
    if args.task == "eval_throughput":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map=args.device,
            quantization_config=quantization_config
        )
        throughput = eval_throughput(
            model,
            tokenizer,
            args.device,
            prompt_data,
            args.desired_length,
            args.bsz,
            decoding=golden_greedy_decoding_with_cache if args.use_cache else golden_greedy_decoding_without_cache
        )
        print(throughput)
    else:
        eval_gpu_memory(
            args.model_name_or_path,
            tokenizer,
            args.device,
            prompt_data,
            args.desired_length,
            args.bsz,
            decoding=golden_greedy_decoding_with_cache if args.use_cache else golden_greedy_decoding_without_cache,
            quantization_config=quantization_config
        )