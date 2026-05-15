"""Phase 2 retrieval quality tests."""
from app.retrieval.hybrid import HybridRetriever
import app.retrieval.hybrid as hybrid_module


def _make_retriever() -> HybridRetriever:
    return HybridRetriever.__new__(HybridRetriever)


def test_query_expansion_client_facing():
    retriever = _make_retriever()
    expanded = retriever._expand_query("Need a client-facing engineer")
    assert "communication" in expanded
    assert "stakeholder" in expanded
    assert "interpersonal" in expanded


def test_default_weights_sum_to_one():
    semantic_weight, bm25_weight, metadata_weight = HybridRetriever._resolve_weights(alpha=None, metadata_weight=0.1)
    assert round(semantic_weight + bm25_weight + metadata_weight, 6) == 1.0
    assert semantic_weight > bm25_weight


def test_alpha_overrides_semantic_bm25_split():
    semantic_weight, bm25_weight, metadata_weight = HybridRetriever._resolve_weights(alpha=0.75, metadata_weight=0.1)
    assert round(semantic_weight + bm25_weight + metadata_weight, 6) == 1.0
    assert semantic_weight > bm25_weight
    assert metadata_weight == 0.1


def test_metadata_boost_increases_with_skill_overlap():
    retriever = _make_retriever()
    entry = {
        "title": "Python Backend Developer Assessment",
        "meta": {"skills": ["Python", "FastAPI", "Testing"]},
    }
    no_overlap = retriever._metadata_boost("need javascript frontend", entry)
    overlap = retriever._metadata_boost("need python backend testing", entry)
    assert overlap > no_overlap


def test_retrieve_and_rerank_uses_thirty_candidates(monkeypatch):
    retriever = _make_retriever()
    retriever.entries = [
        {"id": f"item_{i}", "title": f"Item {i}", "meta": {"description": f"doc {i}", "skills": []}}
        for i in range(40)
    ]

    def fake_retrieve(query, top_k=5, alpha=None, metadata_weight=0.1):
        return [
            {
                "rank": i + 1,
                "id": f"item_{i}",
                "title": f"Item {i}",
                "meta": {"description": f"doc {i}", "skills": []},
                "hybrid_score": 1.0 - (i * 0.01),
                "vector_score": 0.5,
                "bm25_score": 0.5,
                "metadata_score": 0.0,
            }
            for i in range(top_k)
        ]

    seen = {}

    def fake_rerank(query, docs, model="nvidia/nv-reranker-qa-mistral-7b"):
        seen["count"] = len(docs)
        # Give earlier docs slightly higher scores so order is deterministic.
        return [float(len(docs) - i) for i in range(len(docs))]

    monkeypatch.setattr(retriever, "retrieve", fake_retrieve)
    monkeypatch.setattr(hybrid_module, "rerank", fake_rerank)

    results = retriever.retrieve_and_rerank("need python backend", top_k=5)

    assert seen["count"] == 30
    assert len(results) == 5
    assert all("final_rank" in item for item in results)
