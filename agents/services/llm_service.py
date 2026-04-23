import os
from openai import AsyncOpenAI

ASI_MODEL = "asi1"


def get_llm_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=os.getenv("ASI_ONE_API_KEY"),
        base_url="https://api.asi1.ai/v1",
    )
