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
    pattern = re.compile(r"""
        [+-]?
        (?:
            (?:\d{1,3}(?:,\d{3})+)
            |
            \d+
        )
        (?:\.\d+)?
        (?:[eE][+-]?\d+)?
    """, re.VERBOSE)
    matches = re.findall(pattern, s)
    if len(matches) == 0:
        return None
    number_str = "".join(matches[-1])
    number_str = re.sub(r",", "", number_str)
    try:
        number = int(number_str)
    except ValueError:
        number = float(number_str)
    
    return number

def save_results(path, res):
    with jsonlines.open(path, "w") as writer:
        writer.write_all(res)

def build_examples(prompt_ds):
    examples = [f"Problem: {ex["question"]}\nSolution: {ex["answer"]}" for ex in prompt_ds]
    return examples

if __name__ == "__main__":
    print(str2int("euriwery 7,765,000"))