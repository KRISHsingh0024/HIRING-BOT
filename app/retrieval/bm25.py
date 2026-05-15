"""BM25 keyword-based retrieval for hybrid search."""
from typing import List, Tuple
from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, documents: List[str]):
        """
        Initialize BM25 retriever.
        
        Args:
            documents: List of text documents (e.g., catalog descriptions).
        """
        self.documents = documents
        tokenized = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)
    
    def search(self, query: str, k: int = 5) -> Tuple[List[int], List[float]]:
        """
        Search for top-k documents by BM25 score.
        
        Returns:
            (indices, scores) — indices into original documents, and their BM25 scores.
        """
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices sorted by score
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        top_scores = [scores[i] for i in top_indices]
        
        return top_indices, top_scores
