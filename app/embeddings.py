
import hashlib
import re

import numpy as np

from app import config


class SentenceTransformerEmbedder:
    """Local, free embeddings. all-MiniLM-L6-v2: 384 dims, ~90 MB, fast on CPU."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


class HashEmbedder:
    """Bag-of-words feature hashing. No model download, NO real semantics.

    Only for unit tests / CI or machines that cannot download a model.
    """

    dim = 512

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", t.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        if config.EMBEDDING_BACKEND == "hash":
            _embedder = HashEmbedder()
        else:
            _embedder = SentenceTransformerEmbedder(config.EMBEDDING_MODEL)
    return _embedder
