from openai import OpenAI, AsyncOpenAI
from src.openai_model import OpenAIModel, ReflexionModel, AsyncOpenAIModel
import argparse
from src.calculator import Calculator
from src.data_helper import load_gsm8k_dataset
import src.utils as utils
from tqdm import tqdm
from src.evaluate import eval_gsm8k


def get_args():
    parser = argparse.ArgumentParser(
        prog="Task-2",
        description="Script for conducting task 2 experiments",
    )
    parser.add_argument(
        "-o", "--output_path"
    )
    parser.add_argument(
        "-p", "--prompt",
        choices=["vanilla", "CoT", "ICL", "Reflexion"],
        default="vanilla"
    )
    parser.add_argument(
        "-a", "--api_key",
        default="empty"
    )
    parser.add_argument(
        "-u", "--base_url",
        default="https://api.deepseek.com"
    )
    parser.add_argument(
        "-m", "--model",
        default="deepseek-chat"
    )
    parser.add_argument(
        "-d", "--dataset_path",
        default="data/gsm8k"
    )
    parser.add_argument(
        "-s", "--async_size",
        type=int,
        default=1
    )
    parser.add_argument(
        "-N", "--num_test_cases",
        type=int,
        default=10
    )
    parser.add_argument(
        "-n", "--num_examples",
        type=int,
        default=1
    )
    return parser.parse_args()

def run_reflexion(args):
    ds = load_gsm8k_dataset(args.dataset_path)
    test_ds = ds["test"].select(range(args.num_test_cases))
    prompt_ds = ds["prompt"].select(range(args.num_examples))
    examples = utils.build_examples(prompt_ds)
    results = []
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url
    )
    actor_model = OpenAIModel(client, args.model)
    self_reflection_model = OpenAIModel(client, args.model)
    evaluator = Calculator(0.75, 1.0)
    model = ReflexionModel(actor_model,evaluator, self_reflection_model)
    for datum in tqdm(test_ds):
        results.append(model.reflexion(datum["question"], examples))
    utils.save_results(args.output_path, results)
    predictions = [utils.extract_answer(res[-1]["response"], transform=utils.str2int) for res in results]
    references = [utils.extract_answer(datum["answer"], transform=utils.str2int) for datum in test_ds]
    print(eval_gsm8k(predictions, references))
    
def run_vanilla(args):
    ds = load_gsm8k_dataset(args.dataset_path)
    test_ds = ds["test"].select(range(args.num_test_cases))
    results = []
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url
    )
    model = OpenAIModel(client, model=args.model)
    for datum in tqdm(test_ds):
        results.append(model.get_completion(OpenAIModel.build_vanilla_prompt(datum["question"])))
    utils.save_results(args.output_path, results)
    predictions = [utils.extract_answer(res, transform=utils.str2int) for res in results]
    references = [utils.extract_answer(datum["answer"], transform=utils.str2int) for datum in test_ds]
    print(eval_gsm8k(predictions, references))

def run_CoT(args):
    ds = load_gsm8k_dataset(args.dataset_path)
    test_ds = ds["test"].select(range(args.num_test_cases))
    results = []
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url
    )
    model = OpenAIModel(client, model=args.model)
    for datum in tqdm(test_ds):
        results.append(model.get_completion(OpenAIModel.build_CoT_prompt(datum["question"])))
    utils.save_results(args.output_path, results)
    predictions = [utils.extract_answer(res, transform=utils.str2int) for res in results]
    references = [utils.extract_answer(datum["answer"], transform=utils.str2int) for datum in test_ds]
    print(eval_gsm8k(predictions, references))

def run_ICL(args):
    ds = load_gsm8k_dataset(args.dataset_path)
    test_ds = ds["test"].select(range(args.num_test_cases))
    prompt_ds = ds["prompt"].select(range(args.num_examples))
    examples = utils.build_examples(prompt_ds)
    results = []
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url
    )
    model = OpenAIModel(client, model=args.model)
    for datum in tqdm(test_ds):
        results.append(model.get_completion(OpenAIModel.build_ICL_prompt(datum["question"]), examples))
    utils.save_results(args.output_path, results)
    predictions = [utils.extract_answer(res, transform=utils.str2int) for res in results]
    references = [utils.extract_answer(datum["answer"], transform=utils.str2int) for datum in test_ds]
    print(eval_gsm8k(predictions, references))

if __name__ == "__main__":
    args = get_args()

    match args.prompt:
        case "Reflexion":
            run_reflexion(args)
        case "vanilla":
            run_vanilla(args)
        case "CoT":
            run_CoT(args)
        case "ICL":
            run_ICL(args)