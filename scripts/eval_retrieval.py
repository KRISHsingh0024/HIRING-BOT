"""Offline retrieval evaluation for SHL assessment ranking quality.

Reports:
- Recall@5
- Recall@10
- MRR
- reranker improvements
- candidate rescue rate
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from app.retrieval.hybrid import HybridRetriever


DEFAULT_EVAL_SET = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "retrieval_eval_set.json")
DEFAULT_FAISS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "faiss.index")
DEFAULT_BM25 = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "bm25_retriever.pkl")


@dataclass
class EvalCase:
    query: str
    relevant_ids: List[str]


def load_eval_cases(path: str) -> List[EvalCase]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    cases: List[EvalCase] = []
    for item in payload:
        cases.append(EvalCase(query=item["query"], relevant_ids=list(item["relevant_ids"])))
    return cases


def recall_at_k(results: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    top_k = results[:k]
    return 1.0 if any(result_id in relevant_ids for result_id in top_k) else 0.0


def reciprocal_rank(results: Sequence[str], relevant_ids: Sequence[str]) -> float:
    for index, result_id in enumerate(results, start=1):
        if result_id in relevant_ids:
            return 1.0 / index
    return 0.0


def evaluate_cases(cases: Iterable[EvalCase], retriever: HybridRetriever, candidate_k: int = 30, final_k: int = 5) -> dict:
    baseline_recall_5 = []
    baseline_recall_10 = []
    baseline_mrr = []
    reranked_recall_5 = []
    reranked_recall_10 = []
    reranked_mrr = []
    rescued = 0
    rescue_eligible = 0

    per_case = []

    for case in cases:
        baseline_results = retriever.retrieve(case.query, top_k=max(candidate_k, final_k), alpha=None, metadata_weight=0.1)
        reranked_results = retriever.retrieve_and_rerank(
            case.query,
            top_k=final_k,
            alpha=None,
            metadata_weight=0.1,
            rerank_candidate_k=candidate_k,
        )

        baseline_ids = [item["id"] for item in baseline_results]
        reranked_ids = [item["id"] for item in reranked_results]

        b_r5 = recall_at_k(baseline_ids, case.relevant_ids, 5)
        b_r10 = recall_at_k(baseline_ids, case.relevant_ids, 10)
        b_mrr = reciprocal_rank(baseline_ids, case.relevant_ids)

        r_r5 = recall_at_k(reranked_ids, case.relevant_ids, 5)
        r_r10 = recall_at_k(reranked_ids, case.relevant_ids, 10)
        r_mrr = reciprocal_rank(reranked_ids, case.relevant_ids)

        baseline_recall_5.append(b_r5)
        baseline_recall_10.append(b_r10)
        baseline_mrr.append(b_mrr)
        reranked_recall_5.append(r_r5)
        reranked_recall_10.append(r_r10)
        reranked_mrr.append(r_mrr)

        candidate_hit = any(result_id in case.relevant_ids for result_id in baseline_ids[:candidate_k])
        if candidate_hit and b_r5 == 0.0:
            rescue_eligible += 1
            if r_r5 == 1.0:
                rescued += 1

        per_case.append(
            {
                "query": case.query,
                "relevant_ids": case.relevant_ids,
                "baseline_top5": baseline_ids[:5],
                "reranked_top5": reranked_ids[:5],
                "baseline_hit@5": b_r5,
                "reranked_hit@5": r_r5,
                "rescued": bool(candidate_hit and b_r5 == 0.0 and r_r5 == 1.0),
            }
        )

    total = max(len(baseline_recall_5), 1)
    summary = {
        "cases": len(baseline_recall_5),
        "baseline": {
            "recall@5": sum(baseline_recall_5) / total,
            "recall@10": sum(baseline_recall_10) / total,
            "mrr": sum(baseline_mrr) / total,
        },
        "reranked": {
            "recall@5": sum(reranked_recall_5) / total,
            "recall@10": sum(reranked_recall_10) / total,
            "mrr": sum(reranked_mrr) / total,
        },
        "improvements": {
            "recall@5": (sum(reranked_recall_5) - sum(baseline_recall_5)) / total,
            "recall@10": (sum(reranked_recall_10) - sum(baseline_recall_10)) / total,
            "mrr": (sum(reranked_mrr) - sum(baseline_mrr)) / total,
        },
        "candidate_rescue_rate": (rescued / rescue_eligible) if rescue_eligible else 0.0,
        "candidate_rescue_eligible": rescue_eligible,
        "candidate_rescued": rescued,
        "per_case": per_case,
    }
    return summary


def format_summary(summary: dict) -> str:
    lines = [
        f"Cases: {summary['cases']}",
        "",
        "Baseline:",
        f"  Recall@5:  {summary['baseline']['recall@5']:.3f}",
        f"  Recall@10: {summary['baseline']['recall@10']:.3f}",
        f"  MRR:       {summary['baseline']['mrr']:.3f}",
        "",
        "Reranked:",
        f"  Recall@5:  {summary['reranked']['recall@5']:.3f}",
        f"  Recall@10: {summary['reranked']['recall@10']:.3f}",
        f"  MRR:       {summary['reranked']['mrr']:.3f}",
        "",
        "Improvements:",
        f"  Recall@5:  {summary['improvements']['recall@5']:+.3f}",
        f"  Recall@10: {summary['improvements']['recall@10']:+.3f}",
        f"  MRR:       {summary['improvements']['mrr']:+.3f}",
        "",
        "Reranker impact:",
        f"  Candidate rescue rate: {summary['candidate_rescue_rate']:.3f}",
        f"  Rescued / eligible: {summary['candidate_rescued']} / {summary['candidate_rescue_eligible']}",
    ]
    return "\n".join(lines)


def build_retriever(faiss_index_path: str, bm25_path: str) -> HybridRetriever:
    return HybridRetriever(faiss_index_path, bm25_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SHL retrieval quality.")
    parser.add_argument("--eval-set", default=DEFAULT_EVAL_SET)
    parser.add_argument("--faiss-index", default=DEFAULT_FAISS)
    parser.add_argument("--bm25-pkl", default=DEFAULT_BM25)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text")
    args = parser.parse_args()

    cases = load_eval_cases(args.eval_set)
    retriever = build_retriever(args.faiss_index, args.bm25_pkl)
    summary = evaluate_cases(cases, retriever)

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(format_summary(summary))


if __name__ == "__main__":
    main()