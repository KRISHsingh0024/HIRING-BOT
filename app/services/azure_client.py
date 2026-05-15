import os
from typing import List
from openai import AzureOpenAI

AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-02-15-preview")


def _get_client():
    return AzureOpenAI(
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
    )


def get_embedding(text: str) -> List[float]:
    client = _get_client()
    resp = client.embeddings.create(model="text-embedding-3-small", input=text)
    return resp.data[0].embedding


def chat_completion(messages: List[dict], model: str = "gpt-4o-mini") -> dict:
    client = _get_client()
    resp = client.chat.completions.create(model=model, messages=messages)
    # return raw choice
    return resp.choices[0].message if getattr(resp, "choices", None) else {"content": str(resp)}
