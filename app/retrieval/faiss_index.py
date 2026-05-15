import os
import pickle
from typing import Any, List, Tuple

import numpy as np

try:
    import faiss
except ImportError:  # pragma: no cover - exercised when faiss is unavailable
    faiss = None

EMBEDDINGS_PKL = os.getenv("EMBEDDINGS_PKL", "data/embeddings.pkl")


class NumpyIndex:
    def __init__(self, vectors: List[List[float]]):
        self.vectors = np.array(vectors, dtype="float32")
        if self.vectors.ndim != 2:
            raise ValueError("vectors must be a 2D array")

    def search(self, query: np.ndarray, k: int):
        q = np.asarray(query, dtype="float32")
        if q.ndim == 1:
            q = q[None, :]
        if q.ndim != 2:
            raise ValueError("query must be a 1D or 2D array")
        if q.shape[1] != self.vectors.shape[1]:
            raise ValueError("query dimension does not match index dimension")

        diff = self.vectors[None, :, :] - q[:, None, :]
        distances = np.sum(diff * diff, axis=2)
        top_indices = np.argsort(distances, axis=1)[:, :k]
        top_distances = np.take_along_axis(distances, top_indices, axis=1)
        return top_distances, top_indices


def build_index(embeddings_list: List[List[float]]):
    if faiss is not None:
        vecs = np.array(embeddings_list).astype("float32")
        dim = vecs.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(vecs)
        return index

    return NumpyIndex(embeddings_list)


def save_index(index: Any, path: str):
    if faiss is not None and isinstance(index, faiss.Index):
        faiss.write_index(index, path)
        return

    with open(path, "wb") as f:
        pickle.dump({"backend": "numpy", "vectors": getattr(index, "vectors", None)}, f)


def load_index(path: str):
    if faiss is not None:
        try:
            return faiss.read_index(path)
        except Exception:
            pass

    with open(path, "rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, dict) and payload.get("backend") == "numpy":
        return NumpyIndex(payload["vectors"])

    if isinstance(payload, NumpyIndex):
        return payload

    raise ValueError(f"Unsupported index format in {path}")


def load_embeddings() -> Tuple[List[dict], List[List[float]]]:
    with open(EMBEDDINGS_PKL, "rb") as f:
        entries = pickle.load(f)
    embeddings = [e["embedding"] for e in entries]
    return entries, embeddings


def search(index: Any, query_vector: List[float], k: int = 5):
    q = np.array([query_vector]).astype("float32")
    distances, indices = index.search(q, k)
    return distances[0].tolist(), indices[0].tolist()
