import os
import pickle
from typing import Any, List, Tuple

import numpy as np

EMBEDDINGS_PKL = os.getenv("EMBEDDINGS_PKL", "data/embeddings.pkl")

try:
    import faiss  # type: ignore

    _FAISS_AVAILABLE = True
except Exception:  # pragma: no cover
    faiss = None  # type: ignore
    _FAISS_AVAILABLE = False


class NumpyFlatL2Index:
    def __init__(self, dim: int):
        self.dim = dim
        self.vectors: np.ndarray = np.zeros((0, dim), dtype=np.float32)

    def add(self, vecs: np.ndarray) -> None:
        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[1] != self.dim:
            raise ValueError(f"Expected shape (n, {self.dim}), got {vecs.shape}")
        if self.vectors.size == 0:
            self.vectors = vecs.copy()
        else:
            self.vectors = np.vstack([self.vectors, vecs])

    def search(self, q: np.ndarray, k: int):
        q = np.asarray(q, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        if q.ndim != 2 or q.shape[1] != self.dim:
            raise ValueError(f"Expected query shape (batch, {self.dim}), got {q.shape}")

        if self.vectors.size == 0:
            return (
                np.full((q.shape[0], k), np.inf, dtype=np.float32),
                np.full((q.shape[0], k), -1, dtype=np.int64),
            )

        diffs = self.vectors[None, :, :] - q[:, None, :]
        dists = np.sum(diffs * diffs, axis=-1)  # (batch, n)
        idx = np.argsort(dists, axis=1)[:, :k]
        dist_sorted = np.take_along_axis(dists, idx, axis=1)
        return dist_sorted.astype(np.float32), idx.astype(np.int64)


def build_index(embeddings_list: List[List[float]]):
    vecs = np.asarray(embeddings_list, dtype=np.float32)
    if vecs.ndim != 2:
        raise ValueError("embeddings_list must be a 2D list/array")
    dim = int(vecs.shape[1])

    if _FAISS_AVAILABLE:
        index = faiss.IndexFlatL2(dim)
        index.add(vecs)
        return index

    index = NumpyFlatL2Index(dim)
    index.add(vecs)
    return index


def save_index(index: Any, path: str) -> None:
    if _FAISS_AVAILABLE:
        try:
            faiss.write_index(index, path)
            return
        except Exception:
            pass

    # Stable pickle payload for environments without faiss wheels
    payload = {
        "backend": "numpy_flat_l2",
        "dim": getattr(index, "dim", None),
        "vectors": getattr(index, "vectors", None),
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_index(path: str):
    if _FAISS_AVAILABLE:
        try:
            return faiss.read_index(path)
        except Exception:
            pass

    with open(path, "rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, dict) and payload.get("backend") == "numpy_flat_l2":
        dim = int(payload["dim"])
        idx = NumpyFlatL2Index(dim)
        vectors = payload.get("vectors")
        if isinstance(vectors, np.ndarray) and vectors.size:
            idx.add(vectors)
        return idx

    raise ValueError(f"Unsupported index format in {path}")


def load_embeddings() -> Tuple[List[dict], List[List[float]]]:
    with open(EMBEDDINGS_PKL, "rb") as f:
        entries = pickle.load(f)
    embeddings = [e["embedding"] for e in entries]
    return entries, embeddings


def search(index: Any, query_vector: List[float], k: int = 5):
    q = np.asarray([query_vector], dtype=np.float32)
    D, I = index.search(q, k)
    return D[0].tolist(), I[0].tolist()
