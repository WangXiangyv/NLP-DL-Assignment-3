# 2024 PKU NLP-DL Assignment 3 #

### Introduction of the structure of the project

- Files in `data/` folder are datasets downloaded from website for task2 as well as the provided dataset for task 1.

- Files in `scripts/` folder are three shell scripts for running experiments for task 1.1, task 1.2 and task 2 respectively.

- Files in `src/` folder are the main source code for the assignment, including:
    1. `src/decoding.py` for different decoding strategies in task 1.

    2. `src/customized-gpt2.py` for self-implememnted GPT2 model with KV-cache.

    3. `src/evaluate.py` for some functions responsible for measuring throughtput, memory, acc, etc.

    4. `src/data_helper.py` for some functions used for loading datasets.

    5. `src/openai_model.py` for the basic and concurrent versions of openai agent class, which can cooperate with openai style LLM API.

    6. `src/calculator.py` for a mathematic expression validator witch is used as the evaluator in Reflexion framework.

    7. `src/utils.py` for some utility functions.

- `task_1_1.py`, `task_1_2.py` and `task_2.py` are three python scripts for running experiments for task 1.1, task 1.2 and task 2 respectively.

- `requirements.txt` is a python env requirements file, which be used under python 3.12.x.

- Other files in the project.