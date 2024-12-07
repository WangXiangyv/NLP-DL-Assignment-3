import json
import os
from datasets import Dataset, DatasetDict, load_dataset
from collections import defaultdict

def load_prompt_dataset(path: os.PathLike="data/data.txt") -> list:
    with open(path, "r", encoding="utf-8") as f:
        prompt_dataset = [i.strip() for i in f.readlines()]
    return prompt_dataset

def load_gsm8k_dataset(pathDir: os.PathLike="data/gsm8k") -> DatasetDict:
    assert os.path.isdir(pathDir)
    ds = load_dataset(
        "json",
        data_files = {
            "train": os.path.join(pathDir, "train.jsonl"),
            "test": os.path.join(pathDir, "test.jsonl")
        }
    )
    ds["prompt"] = ds["train"].shuffle(seed=2024).select(range(10))
    return ds

def load_mbpp_dataset(pathDir: os.PathLike="data/mbpp") -> DatasetDict:
    assert os.path.isdir(pathDir)
    data_file = os.path.join(pathDir, "mbpp.jsonl")
    assert os.path.exists(data_file)
    ds = defaultdict(list)
    with open(data_file, "r", encoding="utf-8") as fin:
        cnt = 0
        for row in fin:
            datum = json.loads(row)
            cnt += 1
            if cnt <= 10:
                ds["prompt"].append(datum)
            elif cnt >= 11 and cnt <= 510:
                ds["test"].append(datum)
            elif cnt >= 511 and cnt <= 600:
                ds["dev"].append(datum)
            else:
                ds["train"].append(datum)
    for k in ds.keys():
        ds[k] = Dataset.from_list(ds[k])
    return DatasetDict(ds)

if __name__ == "__main__":
    pass