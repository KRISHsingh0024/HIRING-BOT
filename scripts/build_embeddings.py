"""Script to build embeddings using NVIDIA NV-Embed and save FAISS index."""
import os
import json
import pickle

from app.services.nim_client import get_embedding
from app.retrieval.faiss_index import build_index, save_index
from app.retrieval.bm25 import BM25Retriever

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CATALOG = os.path.join(DATA_DIR, "catalog.json")
EMBED_PKL = os.path.join(DATA_DIR, "embeddings.pkl")
INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
BM25_PKL = os.path.join(DATA_DIR, "bm25_retriever.pkl")


def main():
    with open(CATALOG, "r", encoding="utf-8") as f:
        items = json.load(f)

    entries = []
    embeddings = []
    doc_texts = []
    
    for it in items:
        text = it.get("title", "") + "\n" + it.get("description", "")
        doc_texts.append(text)
        
        # Get embedding from NVIDIA NV-Embed
        print(f"Embedding: {it.get('title')}")
        emb = get_embedding(text, input_type="passage")
        entries.append({
            "id": it.get("id"),
            "title": it.get("title"),
            "embedding": emb,
            "meta": it
        })
        embeddings.append(emb)

    # Save embeddings
    with open(EMBED_PKL, "wb") as f:
        pickle.dump(entries, f)
    print(f"Saved embeddings to {EMBED_PKL}")

    # Build and save FAISS index
    index = build_index(embeddings)
    save_index(index, INDEX_PATH)
    print(f"Built FAISS index and saved to {INDEX_PATH}")

    # Build and save BM25 retriever
    bm25 = BM25Retriever(doc_texts)
    with open(BM25_PKL, "wb") as f:
        pickle.dump(bm25, f)
    print(f"Built BM25 retriever and saved to {BM25_PKL}")


if __name__ == "__main__":
    main()

