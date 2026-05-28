# Assessment Recommendation Engine — NVIDIA NIM Stack

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

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Configure NVIDIA API:
- Copy `.env.example` to `.env`
- Get your NVIDIA API key from the NVIDIA API Catalog
- Set `NVIDIA_API_KEY` in `.env`

3. Build embeddings and indices:

```bash
python scripts/build_embeddings.py
```

This script:
- Reads `data/catalog.json`
- Generates embeddings using NVIDIA **NV-Embed-QA**
- Builds FAISS vector index
- Builds BM25 keyword index

4. Run the server:

```bash
uvicorn app.main:app --reload --port 8000
```

5. Test locally:

```bash
pytest -q
```

6. Evaluate retrieval quality:

```bash
python scripts/eval_retrieval.py
```

This reports:
- Recall@5
- Recall@10
- MRR
- reranker improvements
- candidate rescue rate

## Hosting

The app serves:
- API: `POST /chat`, `GET /health`
- UI: `/` (static files from `app/static/`)

### Option 1: Render (recommended)

- Push this repo to GitHub.
- In Render: **New → Blueprint** and select the repo.
- Render will read `render.yaml`.
- Add secret env var `NVIDIA_API_KEY` in the Render dashboard.

Notes for free hosting:
- Free instances may spin down when idle (cold start on first request).
- No persistent disk on free plans, so indices may be rebuilt after restarts.

### Option 2: Azure App Service (container)

See [azure-app-service.md](azure-app-service.md).

### Option 3: Docker (anywhere)

```bash
docker build -t shl-nim-rag .
docker run -p 8000:8000 \
  -e NVIDIA_API_KEY=YOUR_KEY \
  -e NVIDIA_NIM_BASE_URL=https://integrate.api.nvidia.com/v1 \
  -e AUTO_BUILD_INDICES=true \
  shl-nim-rag
```

Open:
- UI: `http://localhost:8000/`
- Health: `http://localhost:8000/health`

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

### NVIDIA NIM Client (`app/services/nim_client.py`)
- Chat completions via Llama 3.1 70B
- Embeddings via NV-Embed-QA
- Reranking via NVIDIA Reranker

### Hybrid Retrieval (`app/retrieval/hybrid.py`)
- BM25 for keyword matching
- Vector search via FAISS + NV-Embed
- Weighted hybrid scoring (`semantic + bm25 + metadata`)
- Query expansion for intent-rich prompts
- Metadata boosting using catalog fields
- Reranking for final ranking

### BM25 Retriever (`app/retrieval/bm25.py`)
- Lightweight keyword-based search
- Fast inference

### FAISS Index (`app/retrieval/faiss_index.py`)
- Vector similarity search
- Efficient L2 distance computation

### FastAPI Orchestration (`app/main.py`)
- Stateless conversation handling
- Clarification-first policy
- Grounded generation with provenance

## Configuration

Set these environment variables in `.env`:

```bash
NVIDIA_API_KEY=<your-nvidia-api-key>
NVIDIA_NIM_BASE_URL=https://integrate.api.nvidia.com/v1
FAISS_INDEX_PATH=data/faiss.index
EMBEDDINGS_PKL=data/embeddings.pkl
CATALOG_JSON=data/catalog.json
BM25_PKL=data/bm25_retriever.pkl
```

## Workflow

1. User query reaches FastAPI
2. Clarification check determines whether more detail is needed
3. Hybrid retrieval combines BM25 and vector search
4. Reranking re-scores the top candidates
5. Grounding prompt includes the retrieved assessments
6. Llama 3.1 generates the response grounded in retrieved data
7. The API returns structured JSON with provenance

## Why NVIDIA NIM?

- Enterprise-grade GPU-accelerated inference
- Open models such as Llama 3.1 and NV-Embed
- Cost-effective pay-per-token pricing
- High performance for production RAG pipelines
- Useful for demonstrating modern AI infrastructure knowledge

## Development Notes

- FAISS and BM25 are stored locally for fast iteration
- Reranker integration is placeholder; update with NVIDIA reranker API
- Conversation state is reconstructed from message history
- Turn count limited to 8 for assignment constraints

## Vercel Deployment

This repo is Vercel-ready as a Python ASGI app. Vercel uses `api/index.py` as the entrypoint, and that module imports the FastAPI app from `app/main.py`.

1. Push the repository to GitHub.
2. Import the repo into Vercel.
3. Set the required runtime variables, especially `NVIDIA_API_KEY`.
4. Make sure the built assets in `data/` are present so `/health` and `/chat` can load the retriever.
5. The UI is available at `/ui`, and `/` redirects there automatically.
