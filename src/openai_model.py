from openai import OpenAI, OpenAIError
from typing import List
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """
    Abstract base model for OpenAI style LLM agent.
    In essence, a simple wrapper around OpenAi client, responsible for building prompts and get completions.
    """
    def __init__(self, client: OpenAI = None, model: str = None, frequency_penalty: int = 0, temperature: int = 1) -> None:
        self.client = client
        self.model = model,
        self.frequency_penalty = frequency_penalty
        self.temperature = temperature
    
    @staticmethod
    @abstractmethod
    def build_prompt():
        raise NotImplementedError

    def get_completion(self, prompt: dict|List[dict]) -> str:
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

class VanillaModel(BaseModel):
    @staticmethod
    def build_prompt(user_content):
        messages = [
            {
                "role": "user",
                "content": f"{user_content}"
            }
        ]
        return messages

class ZeroShotCoTModel(BaseModel):
    @staticmethod
    def build_prompt(user_content: str):
        messages = [
            {
                "role": "user",
                "content": f"{user_content}\nLet's think step by step."
            }
        ]
        return messages

class RAGModel(BaseModel):
    @staticmethod
    def build_prompt(user_content: str, retrieved_information: str|List[str]):
        if not isinstance(retrieved_information, str):
            retrieved_information = '\n'.join(retrieved_information)
        messages = [
            {
                "role": "user",
                "content": f"Given the following data:\n\n{retrieved_information}\n\n{user_content}"
            }
        ]
        return messages