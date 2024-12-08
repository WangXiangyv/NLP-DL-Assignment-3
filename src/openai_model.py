from openai import OpenAI, OpenAIError
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

class OpenAIModel:
    """
    Base model for OpenAI style LLM agent.
    In essence, a simple wrapper around OpenAi client, responsible for building prompts and get completions.
    """
    def __init__(self, client: OpenAI = None, model: str = None, frequency_penalty: int = 0, temperature: int = 1) -> None:
        self.client = client
        self.model = model,
        self.frequency_penalty = frequency_penalty
        self.temperature = temperature
    
    @staticmethod
    def build_prompt(query):
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]
        return messages

    @staticmethod
    def build_vanilla_prompt(query: str):
        content = (
            "Try to solve a math problem.\n"
            f"# Problem to solve #\n{query}\n\n"
            "You must output your final answer in the end with prefix ####.\n"
        )
        return OpenAIModel.build_prompt(content)

    @staticmethod
    def build_CoT_prompt(query: str):
        content = (
            "Try to solve a math problem.\n"
            f"# Problem to solve #\n{query}\n\n"
            "You must output your final answer in the end with prefix ####.\n"
            "Let's think step by step."
        )
        return OpenAIModel.build_prompt(content)
    
    @staticmethod
    def build_ICL_prompt(query: str, examples:List[str]):
        examples = "\n".join([f"## Example {i} ##\n" + ex for i, ex in enumerate(examples)])
        content = (
            "Try to solve a math problem.\n"
            "You can refer to the examples and QA records along with corresponding analyses below for hints.\n\n"
            f"# Examples #\n{examples}\n\n"
            f"# Problem to solve #\n{query}\n\n"
            "You must output your final answer in the end with prefix ####.\n"
        )
        return OpenAIModel.build_prompt(content)

    def get_completion(self, prompt) -> str:
        completion = None
        try:
            response = self.client.chat.completions.create(
                model = self.model,
                messages = prompt,
                frequency_penalty = self.frequency_penalty,
                temperature = self.temperature
            )
            completion = response.choices[0].message.content
        except OpenAIError as e:
            logger.error(f"Error occurred when trying to get completion: {e}")
        return completion


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
        records = "\n".join([f"## Record {i} ##\n"] + rd for i, rd in enumerate(self.memory))
        content = (
            "Try to solve a math problem.\n"
            "You can refer to the examples and QA records along with corresponding analyses below for hints.\n\n"
            f"# Examples #\n{examples}\n\n"
            f"# Records #\n{records}\n\n"
            f"# Problem to solve #\n{query}\n\n"
            "Note that your response must be in plain text. "
            "You must surround all the math calculation expressions in your response with << on the left and >> on the right. "
            "You must output your final answer in the end with prefix ####.\n"
            "Let's think step by step."
        )
        return OpenAIModel.build_vanilla_prompt(content)
    
    def _build_self_reflection_prompt(self, query, response, evaluation):
        content = (
            "Here are a math problem and a corresponding solution.\n"
            f"# Problem #\n{query}\n\n"
            f"# Solution #\n{response}\n\n"
            f"The solution is roughly judged to be {evaluation}.\n"
            "Based on the rough judgement, present a detailed analysis of the solution."
        )
        return OpenAIModel.build_vanilla_prompt(content)
    
    def _build_record(self, query, response, reflection):
        return (
            f"Problem: {query}\n"
            f"Solution: {response}\n"
            f"Analysis: {reflection}\n"
        )
    
    def _trial(self, query, examples) -> Tuple[str, bool]:
        actor_prompt = self._build_actor_prompt(query, examples)
        response = self.actor.get_completion(actor_prompt)
        evaluation = self.evaluator.eval_answer(response)
        self_reflection_prompt = self._build_self_reflection_prompt(query, response, evaluation)
        reflection = self.self_reflection.get_completion(self_reflection_prompt)
        record = self._build_record(query, response, reflection)
        if len(self.memory) >= self.max_memory:
            self.memory.pop(0)
        self.memory.append(record)
        if evaluation:
            return response, True
        else:
            return response, False
        
    def reflexion(self, query:str, examples:List[str]):
        response = None
        evaluation = False
        trial_cnt = 0
        while(not evaluation and trial_cnt < self.max_trial):
            response, evaluation = self._trial(query, examples)
            trial_cnt += 1
        return response