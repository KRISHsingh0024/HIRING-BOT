"""NVIDIA NIM client wrapper for LLM and embeddings."""
import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
from app.utils.sample_loader import load_few_shot_examples

load_dotenv()

NIM_API_KEY = os.getenv("NVIDIA_API_KEY") or os.getenv("OPENAI_API_KEY")
NIM_BASE_URL = os.getenv("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")

DEFAULT_CHAT_MODEL = os.getenv("NVIDIA_CHAT_MODEL", "meta/llama-3.1-70b-instruct")
DEFAULT_EMBED_MODEL = os.getenv("NVIDIA_EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")
DEFAULT_RERANK_MODEL = os.getenv("NVIDIA_RERANK_MODEL", "nvidia/nv-reranker-qa-mistral-7b")


def _get_client():
    if not NIM_API_KEY:
        raise RuntimeError(
            "Missing NVIDIA/OpenAI API key. Set NVIDIA_API_KEY in .env (preferred) "
            "or OPENAI_API_KEY, then re-run."
        )
    return OpenAI(
        base_url=NIM_BASE_URL,
        api_key=NIM_API_KEY,
    )


def get_embedding(
    text: str,
    model: str = DEFAULT_EMBED_MODEL,
    input_type: str = "passage",
) -> List[float]:
    """Get embedding from NVIDIA NV-Embed model."""
    client = _get_client()
    resp = client.embeddings.create(
        model=model,
        input=text,
        extra_body={"input_type": input_type},
    )
    return resp.data[0].embedding


def chat_completion(
    messages: List[dict], model: str = DEFAULT_CHAT_MODEL
) -> dict:
    """Get chat completion from NVIDIA NIM Llama 3.1."""
    client = _get_client()
    # Prepend few-shot examples from SHL sample conversations to teach the model
    # the expected clarifying-question + recommendations style.
    few_shot = load_few_shot_examples(max_examples=4)
    if few_shot:
        # avoid modifying caller list
        messages = few_shot + messages

    resp = client.chat.completions.create(model=model, messages=messages)
    if getattr(resp, "choices", None):
        message = resp.choices[0].message
        return {
            "content": getattr(message, "content", ""),
            "role": getattr(message, "role", "assistant"),
        }

    return {"content": str(resp), "role": "assistant"}


def rerank(query: str, docs: List[str], model: str = DEFAULT_RERANK_MODEL) -> List[float]:
    """
    Rerank documents using NVIDIA reranker.
    Returns scores (higher is better).
    """
    query_terms = {term.lower() for term in query.split() if term.strip()}
    scores = []

    for doc in docs:
        doc_terms = {term.lower() for term in doc.split() if term.strip()}
        overlap = len(query_terms & doc_terms)
        length_bonus = min(len(doc_terms), 200) / 200.0
        scores.append(float(overlap) + 0.01 * length_bonus)

    return scores
