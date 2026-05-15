"""Tests for retrieval evaluation metrics."""
from scripts.eval_retrieval import recall_at_k, reciprocal_rank, evaluate_cases, EvalCase


class DummyRetriever:
    def __init__(self, baseline, reranked):
        self._baseline = baseline
        self._reranked = reranked

    def retrieve(self, query, top_k=5, alpha=None, metadata_weight=0.1):
        return self._baseline[query][:top_k]

    def retrieve_and_rerank(self, query, top_k=5, alpha=None, metadata_weight=0.1, rerank_candidate_k=30):
        return self._reranked[query][:top_k]


def test_recall_and_mrr_helpers():
    results = ["a", "b", "c"]
    relevant = ["b"]

    assert recall_at_k(results, relevant, 1) == 0.0
    assert recall_at_k(results, relevant, 2) == 1.0
    assert reciprocal_rank(results, relevant) == 0.5


def test_evaluate_cases_computes_rescue_rate():
    cases = [EvalCase(query="q1", relevant_ids=["x"])]
    baseline = {"q1": [{"id": f"d{i}"} for i in range(1, 11)]}
    baseline["q1"][5] = {"id": "x"}
    reranked = {"q1": [{"id": "x"}] + [{"id": f"d{i}"} for i in range(1, 10)]}
    retriever = DummyRetriever(baseline, reranked)

    summary = evaluate_cases(cases, retriever, candidate_k=10, final_k=5)

    assert summary["baseline"]["recall@5"] == 0.0
    assert summary["reranked"]["recall@5"] == 1.0
    assert summary["candidate_rescue_rate"] == 1.0
    assert summary["candidate_rescued"] == 1
