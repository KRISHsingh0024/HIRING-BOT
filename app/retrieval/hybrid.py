"""Hybrid retrieval: merge BM25 + vector search, then rerank."""
import pickle
import re
from typing import List

from app.services.nim_client import get_embedding, rerank
from app.retrieval.faiss_index import load_embeddings, load_index, search as faiss_search


class HybridRetriever:
    DEFAULT_RERANK_CANDIDATES = 30

    QUERY_EXPANSION_MAP = {
        "client-facing": ["communication", "stakeholder", "interpersonal"],
        "backend": ["api", "microservice", "database"],
        "frontend": ["ui", "javascript", "react"],
        "leadership": ["manager", "people management", "decision making"],
        "personality": ["behavioral", "culture fit", "interpersonal"],
    }

    def __init__(self, faiss_index_path: str, bm25_pkl_path: str):
        """
        Load FAISS index and BM25 retriever.
        
        Args:
            faiss_index_path: Path to FAISS index.
            bm25_pkl_path: Path to pickled BM25Retriever.
        """
        self.index = load_index(faiss_index_path)
        with open(bm25_pkl_path, "rb") as f:
            self.bm25 = pickle.load(f)
        self.entries, _ = load_embeddings()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9\-]+", (text or "").lower())

    def _expand_query(self, query: str) -> str:
        query_lower = query.lower()
        expansions = []
        for trigger, terms in self.QUERY_EXPANSION_MAP.items():
            if trigger in query_lower:
                expansions.extend(terms)
        if not expansions:
            return query
        deduped = []
        seen = set()
        for term in expansions:
            if term not in seen:
                deduped.append(term)
                seen.add(term)
        return f"{query} {' '.join(deduped)}"

    def _metadata_boost(self, query: str, entry: dict) -> float:
        query_tokens = set(self._tokenize(query))
        meta = entry.get("meta") or {}

        skills = [str(skill).lower() for skill in (meta.get("skills") or [])]
        skills_set = set(skills)
        overlap = len(query_tokens & skills_set)
        skill_boost = min(overlap * 0.4, 1.0)

        role_boost = 0.0
        title_text = str(entry.get("title") or "").lower()
        if "leadership" in query_tokens and ("lead" in title_text or "manager" in title_text):
            role_boost += 0.2
        if "personality" in query_tokens and "personality" in title_text:
            role_boost += 0.2

        return min(skill_boost + role_boost, 1.0)

    @staticmethod
    def _resolve_weights(alpha: float | None, metadata_weight: float | None) -> tuple[float, float, float]:
        if metadata_weight is None:
            metadata_weight = 0.1

        metadata_weight = max(0.0, min(1.0, metadata_weight))
        remaining = 1.0 - metadata_weight

        if alpha is None:
            semantic_weight = 0.6
            bm25_weight = 0.3
            scale = remaining / (semantic_weight + bm25_weight)
            semantic_weight *= scale
            bm25_weight *= scale
            return semantic_weight, bm25_weight, metadata_weight

        alpha = max(0.0, min(1.0, alpha))
        semantic_weight = remaining * alpha
        bm25_weight = remaining * (1.0 - alpha)
        return semantic_weight, bm25_weight, metadata_weight
    
    def retrieve(self, query: str, top_k: int = 5, alpha: float | None = None, metadata_weight: float | None = 0.1) -> List[dict]:
        """
        Hybrid retrieval: BM25 + vector search merged by weighted score.
        
        Args:
            query: User query.
            top_k: Number of results to return.
            alpha: Weight for vector score (1 - alpha for BM25).
        
        Returns:
            List of retrieved items with scores and provenance.
        """
        expanded_query = self._expand_query(query)
        semantic_weight, bm25_weight, metadata_boost_weight = self._resolve_weights(alpha, metadata_weight)

        # Vector retrieval
        query_emb = get_embedding(query, input_type="query")
        vec_distances, vec_indices = faiss_search(self.index, query_emb, k=top_k * 2)
        
        # Normalize vector distances to scores (lower distance = higher score)
        vec_scores = {idx: 1.0 / (1.0 + d) for d, idx in zip(vec_distances, vec_indices)}
        
        # BM25 retrieval
        bm25_indices, bm25_scores = self.bm25.search(expanded_query, k=top_k * 2)
        
        # Normalize BM25 scores
        max_bm25 = max(bm25_scores) if bm25_scores else 0.0
        if max_bm25 > 0:
            bm25_scores_norm = {idx: s / max_bm25 for idx, s in zip(bm25_indices, bm25_scores)}
        else:
            bm25_scores_norm = {idx: 0.0 for idx in bm25_indices}
        
        # Merge scores
        all_indices = set(vec_indices) | set(bm25_indices)
        merged_scores = {}
        for idx in all_indices:
            vec_score = vec_scores.get(idx, 0.0)
            bm25_score = bm25_scores_norm.get(idx, 0.0)
            entry = self.entries[idx] if idx < len(self.entries) else {}
            metadata_score = self._metadata_boost(expanded_query, entry)
            merged_scores[idx] = (
                semantic_weight * vec_score
                + bm25_weight * bm25_score
                + metadata_boost_weight * metadata_score
            )
        
        # Sort and get top-k
        sorted_indices = sorted(merged_scores.keys(), key=lambda x: merged_scores[x], reverse=True)[:top_k]
        
        # Build result list
        results = []
        for rank, idx in enumerate(sorted_indices):
            if idx < len(self.entries):
                entry = self.entries[idx]
                results.append({
                    "rank": rank + 1,
                    "id": entry.get("id"),
                    "title": entry.get("title"),
                    "meta": entry.get("meta"),
                    "hybrid_score": merged_scores[idx],
                    "vector_score": vec_scores.get(idx, 0.0),
                    "bm25_score": bm25_scores_norm.get(idx, 0.0),
                    "metadata_score": self._metadata_boost(expanded_query, entry)
                })
        
        return results
    
    def retrieve_and_rerank(
        self,
        query: str,
        top_k: int = 5,
        alpha: float | None = None,
        metadata_weight: float | None = 0.1,
        rerank_candidate_k: int = DEFAULT_RERANK_CANDIDATES,
    ) -> List[dict]:
        """
        Hybrid retrieval + reranking with NVIDIA reranker.
        
        Args:
            query: User query.
            top_k: Number of final results.
            alpha: Weight for vector score in hybrid merge.
        
        Returns:
            Reranked list of retrieved items.
        """
        # Get a wider hybrid candidate set before deep reranking.
        candidate_budget = max(top_k, rerank_candidate_k)
        candidates = self.retrieve(
            query,
            top_k=candidate_budget,
            alpha=alpha,
            metadata_weight=metadata_weight,
        )
        
        # Extract richer doc context for reranking so the scorer can compare title,
        # description, and skills instead of description-only text.
        doc_texts = []
        for candidate in candidates:
            meta = candidate.get("meta") or {}
            doc_texts.append(
                " ".join(
                    part
                    for part in [
                        candidate.get("title", ""),
                        meta.get("description", ""),
                        " ".join(meta.get("skills") or []),
                        meta.get("seniority", ""),
                    ]
                    if part
                )
            )
        
        # Rerank using NVIDIA reranker
        expanded_query = self._expand_query(query)
        rerank_scores = rerank(expanded_query, doc_texts)
        
        # Update candidates with reranker scores
        for i, candidate in enumerate(candidates):
            candidate["rerank_score"] = rerank_scores[i] if i < len(rerank_scores) else 0.0
        
        # Re-sort by rerank score
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
        
        for rank, item in enumerate(reranked):
            item["final_rank"] = rank + 1
        
        return reranked
