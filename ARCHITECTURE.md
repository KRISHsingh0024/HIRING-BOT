# NVIDIA NIM Architecture for SHL Assignment

## Overview

This SHL Assessment Recommendation Engine demonstrates **enterprise-grade AI infrastructure** using NVIDIA services for production-ready RAG (Retrieval-Augmented Generation) pipelines.

The stack is designed to answer: *"Which assessment should we recommend for this candidate?"* using semantic understanding, keyword matching, intelligent reranking, and grounded generation.

## Why NVIDIA Over Azure/OpenAI?

| Aspect | NVIDIA NIM | Azure OpenAI |
|--------|-----------|------------|
| **Model Control** | Open-weight (Llama 3.1) | Proprietary GPT |
| **Inference Speed** | GPU-optimized inference | API latency |
| **Cost Efficiency** | Pay-per-token, lower rates | Premium pricing |
| **Reranking** | Built-in NVIDIA Reranker | Manual implementation |
| **Interview Value** | Shows AI infrastructure knowledge | Standard approach |
| **Learning Value** | Deep understanding of RAG systems | API integration only |

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER REQUEST                              │
│        "Python backend developer with 5 years experience"       │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                    FASTAPI ORCHESTRATOR                          │
│  ✓ Validate input                                               │
│  ✓ Check if clarification needed                                │
│  ✓ Route to retrieval pipeline                                  │
│  ✓ Orchestrate generation                                       │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│              HYBRID RETRIEVAL PIPELINE                           │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Input: Query Text                                         │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           ↓                                      │
│           ┌───────────────────────────────┐                     │
│           │ NVIDIA NV-Embed Embeddings    │                     │
│           │ (Text → Dense Vector)         │                     │
│           └───────────────────────────────┘                     │
│                ↙                           ↘                    │
│      ┌──────────────┐          ┌──────────────────┐             │
│      │ FAISS Search │          │ BM25 Keyword     │             │
│      │ (Vector)     │          │ Matching         │             │
│      └──────────────┘          └──────────────────┘             │
│           ↓                           ↓                         │
│      Top-5 Docs        +        Top-5 Docs                      │
│      (Semantic)               (Keyword)                          │
│           ↓                           ↓                         │
│           └───────────────────┬───────────────────┘             │
│                               ↓                                  │
│                 ┌─────────────────────────┐                     │
│                 │  Merge & Score          │                     │
│                 │  (Hybrid)               │                     │
│                 │  α * vector +           │                     │
│                 │  (1-α) * bm25           │                     │
│                 └────────────────────┬────┘                     │
│                                      ↓                          │
│                        Top-5 Candidates (Hybrid)               │
└────────────────────────────────┬──────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│           NVIDIA RERANKER (Deep Relevance Scoring)              │
│                                                                  │
│  For each candidate: score_rerank = f(query, doc_context)      │
│  • Semantic alignment                                           │
│  • Query intent matching                                        │
│  • Document context relevance                                  │
│  • Multi-factor ranking                                         │
└────────────────────────────┬──────────────────────────────────┘
                             ↓
                   Final Top-3 Results
                   (Reranked & Sorted)
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│        NVIDIA LLAMA 3.1 (Grounded Generation)                   │
│                                                                  │
│  System Prompt:                                                  │
│  "You are an assessment recommendation assistant.               │
│   Use ONLY the following retrieved assessments..."              │
│                                                                  │
│  + Retrieved Assessments Context                                │
│  + User Messages History                                        │
│  → Llama 3.1 70B generates response grounded in docs            │
└────────────────────────────┬──────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│              STRUCTURED JSON RESPONSE                           │
│                                                                  │
│  {                                                               │
│    "action": "respond",                                          │
│    "reply": "...",                                              │
│    "retrieved_assessments": [...],                              │
│    "provenance": {                                              │
│      "model": "meta/llama-3.1-70b-instruct",                    │
│      "embedding_model": "nvidia/nv-embed-qa-e5-v5",            │
│      "retrieval_method": "hybrid_bm25_vector",                  │
│      "reranked": true                                           │
│    }                                                             │
│  }                                                               │
└─────────────────────────────────────────────────────────────────┘
```

## Component Breakdown

### 1. **NVIDIA NIM LLM Endpoint**
- **Model**: Llama 3.1 70B Instruct
- **Task**: Grounded generation, reasoning over retrieved context
- **Why**: Better for conversational understanding + instruction-following than smaller models

### 2. **NVIDIA NV-Embed-QA**
- **Model**: NV-Embed-QA (specialized for question-answering retrieval)
- **Task**: Convert text → dense vectors (768-dim embeddings)
- **Why**: Better semantic matching than generic embeddings; optimized for retrieval tasks

### 3. **FAISS Vector Database**
- **Task**: Fast similarity search (L2 distance)
- **Why**: Lightweight, explainable, interview-friendly; local inference with no API calls

### 4. **BM25 Keyword Retriever**
- **Task**: Exact keyword matching + TF-IDF scoring
- **Why**: Catches exact matches (e.g., "Python", "OOP") that semantic search might miss

### 5. **Hybrid Merging Strategy**
```python
hybrid_score = semantic_w * vector_score + bm25_w * bm25_score + metadata_w * metadata_score
```
- **Default weights**: semantic=0.6, bm25=0.3, metadata=0.1
- **Metadata boost**: uses catalog features such as `skills` overlap and title hints
- **Query expansion**: expands intent terms (e.g., `client-facing` → communication/stakeholder/interpersonal)
- **Result**: Best of both worlds — semantic + keyword robustness

### 6. **NVIDIA Reranker**
- **Model**: NVIDIA Reranker (or comparable cross-encoder)
- **Task**: Re-score top-K candidates using deep relevance model
- **Why**: Modern RAG systems use reranking for dramatic Recall@K improvements

### 7. **FastAPI Orchestrator**
- **Task**: Stateless orchestration of entire pipeline
- **Why**: 
  - Production-ready async framework
  - Easy to scale horizontally
  - Clear separation of concerns

---

## Data Flow Example

**User Input:**
```
"I need to assess a Python developer with stakeholder collaboration skills."
```

**Step 1: Clarification Check**
```
If len(query) < 8:
  → Ask for more details
Else:
  → Proceed to retrieval
```

**Step 2: Generate Embedding**
```
query_embedding = NIM.embed("I need to assess a Python developer...")
# → [0.224, -0.18, 0.56, ..., -0.09]  (768-dim vector)
```

**Step 3: Hybrid Retrieval**
```
# Vector search: top-3 by L2 distance
vector_results = [
  {"id": "py_backend", "score": 0.92},
  {"id": "py_junior", "score": 0.85},
  {"id": "full_stack", "score": 0.78}
]

# BM25 search: top-3 by keyword match
bm25_results = [
  {"id": "py_backend", "score": 0.88},
  {"id": "oop_assessment", "score": 0.72},
  {"id": "communication", "score": 0.65}
]

# Merge (α=0.5)
merged = {
  "py_backend": 0.5 * 0.92 + 0.5 * 0.88 = 0.90,
  "py_junior": 0.5 * 0.85 + 0.5 * 0 = 0.425,
  "full_stack": 0.5 * 0.78 + 0.5 * 0 = 0.39,
  "oop_assessment": 0.5 * 0 + 0.5 * 0.72 = 0.36,
  "communication": 0.5 * 0 + 0.5 * 0.65 = 0.325
}

# Top-5 after merge
top_5 = sorted by score desc = [py_backend, py_junior, full_stack, oop_assessment, communication]
```

**Step 4: Reranking**
```
For each top-5 result:
  rerank_score = NVIDIA_Reranker(query, doc_context)
  
# Results after reranking (deep model vs. simple heuristic):
[
  {"id": "py_backend", "rerank_score": 0.96},  # Much higher confidence
  {"id": "communication", "rerank_score": 0.75},  # Now relevant!
  {"id": "py_junior", "rerank_score": 0.68},
  {"id": "oop_assessment", "rerank_score": 0.52},
  {"id": "full_stack", "rerank_score": 0.48}
]
```

**Step 5: Grounded Generation**
```
system_prompt = """
You are an assessment recommendation assistant.
Use ONLY the following retrieved assessments to ground your response.

Retrieved Assessments:
{
  "id": "py_backend",
  "title": "Python Backend Developer Assessment",
  "seniority": "mid",
  "skills": ["Python", "FastAPI", "Testing"]
},
...
"""

messages = [
  {"role": "system", "content": system_prompt},
  {"role": "user", "content": "I need to assess a Python developer..."}
]

response = Llama3.1(messages)
# Llama generates: "Based on the retrieved assessments, I recommend
# the 'Python Backend Developer Assessment' because..."
```

**Step 6: Return Structured Response**
```json
{
  "action": "respond",
  "reply": "Based on the retrieved assessments, I recommend the 'Python Backend Developer Assessment'...",
  "retrieved_assessments": [
    {
      "rank": 1,
      "id": "py_backend",
      "title": "Python Backend Developer Assessment",
      "hybrid_score": 0.90,
      "vector_score": 0.92,
      "bm25_score": 0.88,
      "rerank_score": 0.96,
      "final_rank": 1
    },
    ...
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

---

## Interview-Ready Talking Points

### Q: "Why NVIDIA NIM instead of OpenAI?"
**A:** "NVIDIA NIM provides enterprise-grade GPU inference with open-weight models, better cost efficiency, and full control over the RAG pipeline. For this assignment, it showcases understanding of:
- Inference optimization
- Open-source LLM ecosystems
- Enterprise AI infrastructure
- Retrieval-augmented generation best practices

Specifically, I used NV-Embed for high-quality semantic embeddings, FAISS for efficient local search, BM25 for keyword robustness, and the NVIDIA reranker for deep relevance scoring—this mirrors production RAG architectures used in industry."

### Q: "Walk me through the retrieval pipeline."
**A:** "The system uses hybrid retrieval combining vector search and keyword matching:
1. **Vector search (FAISS)**: User query → NV-Embed → dense vector → similarity search
2. **Keyword search (BM25)**: Query → tokenize → TF-IDF scoring
3. **Merge**: Weighted combination (α·vector + (1-α)·bm25) balances semantic and exact matching
4. **Rerank**: Top-K candidates passed to NVIDIA reranker for deep relevance rescoring

This approach is standard in modern RAG systems and dramatically improves Recall@K."

### Q: "How does grounding work?"
**A:** "The system passes retrieved assessments directly into the system prompt, so Llama 3.1 generates responses strictly grounded in those documents. This prevents hallucinations and ensures all recommendations are backed by the catalog."

### Q: "What if the query is ambiguous?"
**A:** "The system implements a clarification-first policy: if the user's query is too short or lacks specificity, it asks for details before retrieving. This maximizes information gain while respecting the 8-turn conversation limit."

---

## Production Deployment Considerations

### Scalability
- **Stateless FastAPI**: Horizontal scaling via load balancing
- **NVIDIA NIM**: Managed service (handled by NVIDIA)
- **FAISS + BM25**: Local or distributed cache (Redis for multi-instance)

### Monitoring
- **Application Insights**: Track latency, error rates
- **Query analytics**: Log queries + rankings for feedback loops
- **Model performance**: A/B test different α values, reranker thresholds

### Cost Optimization
- **NVIDIA NIM**: Pay-per-token (typically $0.001–$0.01 per 1K tokens)
- **Local inference**: FAISS + BM25 run free locally
- **Caching**: Cache embeddings + rankings for repeated queries

### Security
- **API keys**: Store in Azure Key Vault or similar
- **Rate limiting**: Implement per-user limits
- **Input validation**: Pydantic models for request validation

---

## Files Overview

```
project/
├── app/
│   ├── __init__.py
│   ├── main.py                      # FastAPI orchestrator
│   ├── services/
│   │   ├── __init__.py
│   │   ├── nim_client.py            # NVIDIA NIM wrapper
│   │   └── azure_client.py          # Legacy (archived)
│   └── retrieval/
│       ├── __init__.py
│       ├── faiss_index.py           # FAISS utilities
│       ├── bm25.py                  # BM25 retriever
│       └── hybrid.py                # Hybrid retrieval orchestrator
├── data/
│   ├── catalog.json                 # SHL assessment catalog
│   ├── faiss.index                  # FAISS vector index (generated)
│   ├── embeddings.pkl               # Cached embeddings (generated)
│   └── bm25_retriever.pkl           # BM25 index (generated)
├── scripts/
│   └── build_embeddings.py          # Generate embeddings + indices
├── tests/
│   └── test_health.py               # Health check + basic tests
├── requirements.txt                 # Dependencies
├── .env.example                     # Environment config template
├── README.md                        # User-facing documentation
└── ARCHITECTURE.md                  # This file
```

---

## Next Steps / Future Enhancements

1. **Conversation State Management**
   - Track user context across turns
   - Implement "memory" of previous queries

2. **Advanced Ranking**
   - Metadata-aware scoring (seniority, skills match)
   - Threshold-based filtering

3. **Catalog Integration**
   - Real SHL API integration
   - Dynamic catalog updates

4. **Evaluation Framework**
   - Compute Recall@K metrics
   - A/B test retrieval strategies

5. **Fine-tuning**
   - Domain-specific embeddings on SHL data
   - Custom reranker training

---

## References

- [NVIDIA NIM Documentation](https://docs.nvidia.com)
- [LLaMA 3.1 Paper](https://arxiv.org/abs/2405.00663)
- [Retrieval-Augmented Generation Survey](https://arxiv.org/abs/2312.10997)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)
