from typing import Callable
import jsonlines
import regex as re
def extract_answer(text:str, transform:Callable):
    try:
        ans = text.split("####")[-1].strip()
        ans = transform(ans)
    except:
        ans = None
    return ans

def str2bool(s:str):
    if "true" in s.lower():
        return True
    elif "false" in s.lower():
        return False
    else:
        return None
    
def str2int(s:str):
    pattern = r"\d"
    number = re.match(pattern, s)
    number = int(number) if number is not None else None
    return number

def save_results(path, res):
    with jsonlines.open(path, "w") as writer:
        writer.write_all(res)

def build_examples(prompt_ds):
    examples = [f"Problem: {ex["question"]}\nSolution: {ex["answer"]}" for ex in prompt_ds]
    return examples

if __name__ == "__main__":
    print(int('4545'))