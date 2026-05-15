# SHL Assessment Recommendation Engine — NVIDIA NIM Stack

Enterprise-grade RAG system using **NVIDIA NIM** for LLM inference, **NV-Embed-QA** for embeddings, **FAISS** for vector search, **BM25** for keyword retrieval, and **NVIDIA Reranker** for ranking.

## Architecture

```
FastAPI Backend
  ↓
Conversation Orchestrator
  ↓
NVIDIA NIM (Llama 3.1 70B)
  ↓
Hybrid Retrieval (BM25 + Vector Search)
  ↓
NVIDIA Reranker
  ↓
Grounded Generation
```

## Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Configure NVIDIA API:**
   - Copy `.env.example` to `.env`
   - Get your NVIDIA API key from [NVIDIA API Catalog](https://build.nvidia.com/explore/discover)
   - Set `NVIDIA_API_KEY` in `.env`

3. **Build embeddings and indices:**
```bash
python scripts/build_embeddings.py
```

This script:
- Reads `data/catalog.json`
- Generates embeddings using NVIDIA **NV-Embed-QA**
- Builds FAISS vector index
- Builds BM25 keyword index

4. **Run the server:**
```bash
uvicorn app.main:app --reload --port 8000
```

5. **Test locally:**
```bash
pytest -q
```

6. **Evaluate retrieval quality:**
```bash
python scripts/eval_retrieval.py
```

This reports:
- Recall@5
- Recall@10
- MRR
- reranker improvements
- candidate rescue rate

## API Endpoints

### `GET /health`
Health check endpoint.

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "ok",
  "backend": "NVIDIA NIM"
}
```

### `POST /chat`
Main conversation endpoint.

Request:
```json
{
  "messages": [
    {
      "role": "user",
      "content": "I'm looking for a Python backend developer assessment"
    }
  ],
  "top_k": 5,
  "use_reranker": true
}
```

Response:
```json
{
  "action": "respond",
  "reply": "Based on your requirement for a Python backend developer...",
  "retrieved_assessments": [
    {
      "rank": 1,
      "id": "assessment_1",
      "title": "Python Backend Developer Assessment",
      "hybrid_score": 0.95,
      "vector_score": 0.88,
      "bm25_score": 0.92,
      "rerank_score": 0.98,
      "final_rank": 1,
      "meta": {...}
    }
  ],
  "turn_count": 1,
  "provenance": {
    "model": "meta/llama-3.1-70b-instruct",
    "embedding_model": "nvidia/nv-embed-qa-e5-v5",
    "retrieval_method": "hybrid_bm25_vector",
    "reranked": true
  }
}
```

## Key Components

### 1. **NVIDIA NIM Client** (`app/services/nim_client.py`)
- Chat completions via Llama 3.1 70B
- Embeddings via NV-Embed-QA
- Reranking via NVIDIA Reranker

### 2. **Hybrid Retrieval** (`app/retrieval/hybrid.py`)
- **BM25** for keyword matching
- **Vector search** via FAISS + NV-Embed
- Weighted hybrid scoring (`semantic + bm25 + metadata`)
- Query expansion for intent-rich prompts (e.g., `client-facing`)
- Metadata boosting using catalog fields (skills/title hints)
- Reranking for final ranking

### 3. **BM25 Retriever** (`app/retrieval/bm25.py`)
- Lightweight keyword-based search
- Fast inference

### 4. **FAISS Index** (`app/retrieval/faiss_index.py`)
- Vector similarity search
- Efficient L2 distance computation

### 5. **FastAPI Orchestration** (`app/main.py`)
- Stateless conversation handling
- Clarification-first policy
- Grounded generation with provenance

## Configuration

Set these environment variables in `.env`:

```
NVIDIA_API_KEY=<your-nvidia-api-key>
NVIDIA_NIM_BASE_URL=https://integrate.api.nvidia.com/v1
FAISS_INDEX_PATH=data/faiss.index
EMBEDDINGS_PKL=data/embeddings.pkl
CATALOG_JSON=data/catalog.json
BM25_PKL=data/bm25_retriever.pkl
```

## Workflow

1. **User Query** → FastAPI receives chat message
2. **Clarification Check** → If query too short, ask for details
3. **Hybrid Retrieval** → BM25 + Vector search + merge
4. **Reranking** → NVIDIA Reranker re-scores candidates
5. **Grounding** → System prompt includes top retrieved assessments
6. **Generation** → Llama 3.1 generates response grounded in retrieved data
7. **Response** → Return structured JSON with provenance

## Why NVIDIA NIM?

- **Enterprise-grade**: GPU-accelerated inference, security, monitoring
- **Open models**: Llama 3.1, NV-Embed, reranker
- **Cost-effective**: Pay-per-token pricing
- **High performance**: Optimized for production RAG pipelines
- **Interview-friendly**: Demonstrates knowledge of modern AI infra

## Development Notes

- FAISS and BM25 are stored locally for fast iteration
- Reranker integration is placeholder; update with NVIDIA reranker API
- Conversation state is reconstructed from message history (stateless design)
- Turn count limited to 8 for SHL assignment constraints

## Next Steps

- Implement conversation state management
- Tune retrieval weights and expansion dictionary using evaluation sets
- Expand the retrieval eval set with more labeled queries
- Integrate with SHL catalog API
- Add monitoring/observability
- Deploy to Render or Azure Container Instances
