"""FastAPI orchestration layer for NVIDIA NIM + FAISS + BM25 + Reranking."""
import os
import json
import asyncio
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.services.nim_client import chat_completion
from app.retrieval.hybrid import HybridRetriever
from app.utils.query_classifier import classify_query, QueryIntent

app = FastAPI()

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss.index")
BM25_PATH = os.getenv("BM25_PKL", "data/bm25_retriever.pkl")
EMBEDDINGS_PATH = os.getenv("EMBEDDINGS_PKL", "data/embeddings.pkl")
AUTO_BUILD_INDICES = os.getenv("AUTO_BUILD_INDICES", "true").lower() in {"1", "true", "yes", "y"}

# Load retriever at startup
retriever = None
_index_build_task: asyncio.Task | None = None
_index_build_error: str | None = None


async def _build_indices_and_load_retriever() -> None:
    global retriever, _index_build_error
    try:
        from scripts.build_embeddings import main as build_indices

        await asyncio.to_thread(build_indices)
        if os.path.exists(INDEX_PATH) and os.path.exists(BM25_PATH):
            retriever = HybridRetriever(INDEX_PATH, BM25_PATH)
    except Exception as exc:  # noqa: BLE001
        _index_build_error = f"{type(exc).__name__}: {exc}"


@app.on_event("startup")
async def startup():
    global retriever, _index_build_task
    # Only load the real HybridRetriever if one has not been injected
    if retriever is not None:
        return

    indices_exist = os.path.exists(INDEX_PATH) and os.path.exists(BM25_PATH) and os.path.exists(EMBEDDINGS_PATH)
    if indices_exist and os.path.exists(INDEX_PATH) and os.path.exists(BM25_PATH):
        retriever = HybridRetriever(INDEX_PATH, BM25_PATH)
        return

    # For hosted environments, avoid blocking startup on index building.
    if (not indices_exist) and AUTO_BUILD_INDICES and _index_build_task is None:
        _index_build_task = asyncio.create_task(_build_indices_and_load_retriever())


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    top_k: int = 5
    use_reranker: bool = True


@app.get("/health")
def health():
    state = "ready" if retriever is not None else "building" if _index_build_task is not None else "missing"
    return {"status": "ok", "backend": "NVIDIA NIM", "indices": state, "index_error": _index_build_error}


@app.get("/")
def root():
    return RedirectResponse(url="/ui")


@app.get("/ui")
def ui():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/chat")
def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages required")

    if retriever is None:
        if _index_build_error:
            raise HTTPException(status_code=503, detail=f"Index build failed: {_index_build_error}")
        if _index_build_task is not None:
            raise HTTPException(status_code=503, detail="Index is building; retry in a few seconds")
        raise HTTPException(
            status_code=503,
            detail="Retriever not loaded; set AUTO_BUILD_INDICES=true or run scripts/build_embeddings.py",
        )

    last_user = req.messages[-1]

    # Classify the query intent for domain boundary enforcement.
    classification = classify_query(last_user.content)

    # Refuse off-topic queries with human-friendly explanation.
    if classification.intent == QueryIntent.OFF_TOPIC:
        return {
            "action": "refuse",
            "reason": "off-topic",
            "reply": "I can only provide recommendations from the catalog. " + classification.reason,
            "recommendations": [],
            "retrieved_assessments": [],
            "turn_count": len(req.messages),
            "end_of_conversation": True,
        }

    # Refuse prompt injection attempts with human-friendly explanation.
    if classification.intent == QueryIntent.PROMPT_INJECTION:
        return {
            "action": "refuse",
            "reason": "prompt_injection",
            "reply": "I can only provide recommendations from the catalog.",
            "recommendations": [],
            "retrieved_assessments": [],
            "turn_count": len(req.messages),
            "end_of_conversation": True,
        }

    content_lower = last_user.content.lower()
    clarified_phrases = [
        "backend-leaning",
        "frontend-heavy",
        "balanced full-stack",
        "backend focused",
        "frontend focused",
        "balanced",
    ]
    has_explicit_prioritization = any(p in content_lower for p in clarified_phrases)

    # Clarification heuristic for vague assessment queries.
    if classification.confidence < 0.5 and not has_explicit_prioritization:
        return {
            "action": "clarify",
            "clarify_prompt": "Is this role backend-leaning, frontend-heavy, or balanced full-stack?",
            "turn_count": len(req.messages),
            "recommendations": None,
            "end_of_conversation": False,
        }

    # Handle comparison mode differently from regular recommendations.
    if classification.intent == QueryIntent.ASSESSMENT_COMPARISON:
        retrieved = retriever.retrieve(last_user.content, top_k=min(req.top_k, 5))
        retrieved_context = json.dumps([r["meta"] for r in retrieved], indent=2)
        
        system_prompt = (
            "You are an assessment comparison specialist. "
            "ONLY compare assessments using the retrieved catalog data below. "
            "Do NOT invent hiring scenarios, industries, or user contexts. "
            "Do NOT add assumptions about use cases. "
            "Compare ONLY the factual properties of the assessments.\n\n"
            f"Retrieved Assessments:\n{retrieved_context}"
        )
    else:
        # Retrieve and optionally rerank for recommendations.
        if req.use_reranker:
            retrieved = retriever.retrieve_and_rerank(last_user.content, top_k=req.top_k)
        else:
            retrieved = retriever.retrieve(last_user.content, top_k=req.top_k)
        
        retrieved_context = json.dumps([r["meta"] for r in retrieved], indent=2)
        
        system_prompt = (
            "You are a helpful assessment recommendation assistant. "
            "Use ONLY the following retrieved assessments to ground your response. "
            "Do NOT invent hiring contexts, industries, or user scenarios. "
            "Recommend assessments based ONLY on their documented properties and the candidate's stated role.\n\n"
            f"Retrieved Assessments:\n{retrieved_context}"
        )

    # Prepare messages for NVIDIA Llama 3.1
    messages = [
        {"role": "system", "content": system_prompt},
    ]
    for m in req.messages:
        messages.append({"role": m.role, "content": m.content})

    # Get response from NVIDIA Llama 3.1 via NIM
    choice = chat_completion(messages)
    reply = choice.get("content", "")
    if classification.intent == QueryIntent.ASSESSMENT_COMPARISON and reply:
        lowered_reply = reply.lower()
        if not any(word in lowered_reply for word in ["comparison", "difference", "both"]):
            reply = f"Comparison summary: {reply}"

    # Format recommendations (table-ready) from retrieved items
    recommendations = []
    for r in retrieved:
        recommendations.append({
            "rank": r.get("rank"),
            "id": r.get("id"),
            "title": r.get("title"),
            "description": (r.get("meta") or {}).get("description", ""),
        })

    return {
        "action": "respond",
        "reply": reply,
        "recommendations": recommendations,
        "retrieved_assessments": retrieved,
        "turn_count": len(req.messages),
        "end_of_conversation": True,
        "provenance": {
            "model": "meta/llama-3.1-70b-instruct",
            "embedding_model": "nvidia/nv-embedqa-e5-v5",
            "retrieval_method": "hybrid_bm25_vector",
            "reranked": req.use_reranker,
        },
    }
