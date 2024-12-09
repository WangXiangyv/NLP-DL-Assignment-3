from openai import OpenAI, OpenAIError, AsyncOpenAI
from typing import List, Tuple
import logging
from tqdm.asyncio import tqdm as atqdm
import asyncio
import src.utils as utils

logger = logging.getLogger(__name__)

class OpenAIModel:
    """
    Base model for OpenAI style LLM agent.
    In essence, a simple wrapper around OpenAi client, responsible for building prompts and get completions.
    """
    def __init__(self, client:OpenAI=None, model:str=None, frequency_penalty:float=0, temperature:float=1) -> None:
        self.client = client
        self.model = model
        self.frequency_penalty = frequency_penalty
        self.temperature = temperature
    
    @staticmethod
    def build_prompt(query):
        messages = [
            {
                "role": "user",
                "content": query
            },
        ]
        return messages

    @staticmethod
    def build_vanilla_prompt(query: str):
        content = (
            "Try to solve a math problem.\n"
            f"# Problem to solve #\n{query}\n\n"
            "You must output your final answer at the end with prefix ####.\n"
        )
        return OpenAIModel.build_prompt(content)

    @staticmethod
    def build_CoT_prompt(query: str):
        content = (
            "Try to solve a math problem.\n"
            f"# Problem to solve #\n{query}\n\n"
            "You must output your final answer at the end with prefix ####.\n"
            "Let's think step by step."
        )
        return OpenAIModel.build_prompt(content)
    
    @staticmethod
    def build_ICL_prompt(query: str, examples:List[str]):
        examples = "\n".join([f"## Example {i} ##\n" + ex for i, ex in enumerate(examples)])
        content = (
            "Try to solve a math problem.\n"
            "You can refer to the examples provided below for hints.\n\n"
            f"# Examples #\n{examples}\n\n"
            f"# Problem to solve #\n{query}\n\n"
            "You must output your final answer at the end with prefix ####.\n"
        )
        return OpenAIModel.build_prompt(content)

    def get_completion(self, prompt) -> str:
        completion = None
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=prompt,
                frequency_penalty = self.frequency_penalty,
                temperature = self.temperature
            )
            completion = response.choices[0].message.content
        except OpenAIError as e:
            logger.error(f"Error occurred when trying to get completion: {e}")
        return completion

class AsyncOpenAIModel(OpenAIModel):
    def __init__(self, client:AsyncOpenAI=None, model:str=None, frequency_penalty:float=0, temperature:float=1) -> None:
        super().__init__(client, model, frequency_penalty, temperature)
        
    async def get_completion(self, prompt, sem:asyncio.Semaphore) -> str:
        completion = None
        try:
            async with sem:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=prompt,
                    frequency_penalty=self.frequency_penalty,
                    temperature=self.temperature
                )
            completion = response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error occurred when trying to get async ompletion: {e}")
        return completion
    
    async def get_batch_completion(self, prompts, max_async_size:int=32):
        sem = asyncio.Semaphore(max_async_size)
        total = len(prompts)
        tasks = [self.get_completion(p, sem) for p in prompts]
        outputs = await atqdm.gather(*tasks, total=total, leave=True)
        return outputs


class ReflexionModel:
    def __init__(self, actor, evaluator, self_reflection, max_trial:int=5, max_memory: int=5) -> None:
        self.actor = actor
        self.evaluator = evaluator
        self.self_reflection = self_reflection
        self.max_trial = max_trial
        self.max_memory = max_memory
        self.memory = []
    
    def clear_memory(self):
        self.memory.clear()
    
    def _build_actor_prompt(self, query, examples:List[str]):
        examples = "\n".join([f"## Example {i} ##\n" + ex for i, ex in enumerate(examples)])
        records = "\n".join([f"## Record {i} ##\n" + rd for i, rd in enumerate(self.memory)]) if len(self.memory) > 0 else None
        content = (
            "Try to solve a math problem.\n"
            "You can refer to the examples and QA records along with corresponding analyses provided below for hints.\n\n"
            f"# Examples #\n{examples}\n\n"
            f"# Records #\n{records}\n\n"
            f"# Problem to solve #\n{query}\n\n"
            "You must surround all the math calculation expressions in your response with << on the left and >> on the right. "
            "Math calculation expressions surrounded by << and >> must be in plain text without format."
            "You must output your final answer at the end with prefix ####.\n"
            "Let's think step by step."
        )
        return OpenAIModel.build_prompt(content)
    
    def _build_self_reflection_prompt(self, query, response, evaluation):
        content = (
            "Here are a math problem and a corresponding solution.\n"
            f"# Problem #\n{query}\n\n"
            f"# Solution #\n{response}\n\n"
            f"Merely considering numeric calculation, the solution is roughly judged to be {evaluation}.\n"
            "Based on the rough judgement, present a further analysis of the solution.\n"
            "Note that you must output your final judgement (True or False) at the end with prefix ####"
        )
        return OpenAIModel.build_prompt(content)
    
    def _build_record(self, query, response, reflection):
        return (
            f"Problem: {query}\n"
            f"Solution: {response}\n"
            f"Analysis: {reflection}"
        )
    
    def trial(self, query, examples) -> Tuple[str, bool]:
        actor_prompt = self._build_actor_prompt(query, examples)
        response = self.actor.get_completion(actor_prompt)
        evaluation = self.evaluator.eval_answer(response)
        self_reflection_prompt = self._build_self_reflection_prompt(query, response, evaluation)
        reflection = self.self_reflection.get_completion(self_reflection_prompt)
        reflection_judgement = utils.extract_answer(reflection, utils.str2bool)
        record = self._build_record(query, response, reflection)
        if len(self.memory) >= self.max_memory:
            self.memory.pop(0)
        self.memory.append(record)
        return response, reflection, evaluation, reflection_judgement
        
    def reflexion(self, query:str, examples:List[str]):
        trajectory = []
        response = None
        reflection = None
        evaluation = False
        reflection_judgement = False
        trial_cnt = 0
        while(not (evaluation and reflection_judgement) and trial_cnt < self.max_trial):
            response, reflection, evaluation, reflection_judgement = self.trial(query, examples)
            trial_cnt += 1
            trajectory.append(
                {
                    "response":response,
                    "reflection":reflection,
                    "evaluation":evaluation,
                    "reflection_judgement":reflection_judgement
                }
            )
        return trajectory

# class SafeList:
#     def __init__(self):
#         self._list = []
#         self._lock = asyncio.Lock()

#     async def append(self, item):
#         async with self._lock:
#             self._list.append(item)

#     async def get_obj(self):
#         async with self._lock:
#             return list(self._list)

# class AsyncReflexionModel(ReflexionModel):
    # """
    # Note that this class is not available. Because the memory is not well managed
    # """
    # def __init__(self, actor:AsyncOpenAIModel, evaluator, self_reflection:AsyncOpenAIModel, max_trial = 5, max_memory = 5):
    #     assert isinstance(self.actor, AsyncOpenAIModel), "'self.actor' do not support sync"
    #     assert isinstance(self.self_reflection, AsyncOpenAIModel), "'self.self_reflection' do not support sync"
    #     super().__init__(actor, evaluator, self_reflection, max_trial, max_memory)
    #     self.memory = SafeList()
        
    # async def trial(self, query, examples, sem):
    #     actor_prompt = self._build_actor_prompt(query, examples)
    #     response = await self.actor.get_completion(actor_prompt, sem)
    #     evaluation = self.evaluator.eval_answer(response)
    #     self_reflection_prompt = self._build_self_reflection_prompt(query, response, evaluation)
    #     reflection = await self.self_reflection.get_completion(self_reflection_prompt, sem)
    #     record = self._build_record(query, response, reflection)
    #     if len(self.memory) >= self.max_memory:
    #         self.memory.pop(0)
    #     await self.memory.append(record)
    #     if evaluation:
    #         return response, True
    #     else:
    #         return response, False
    
    # async def reflexion(self, query:str, examples:List[str], sem):
    #     response = None
    #     evaluation = False
    #     trial_cnt = 0
    #     while(not evaluation and trial_cnt < self.max_trial):
    #         response, evaluation = await self.trial(query, examples, sem)
    #         trial_cnt += 1
    #     return response
    
    # async def batch_reflexion(self, queries, examples, max_async_size:int=32):
    #     sem = asyncio.Semaphore(max_async_size)
    #     total = len(queries)
    #     tasks = [self.reflexion(q, examples, sem) for q in queries]
    #     outputs = await atqdm.gather(*tasks, total=total, leave=True)
    #     return outputs