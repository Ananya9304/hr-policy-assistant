
import re

import numpy as np
from rank_bm25 import BM25Okapi

from app import config
from app.embeddings import get_embedder
from app.models import Chunk, RetrievedChunk
from app.store import Store

STOPWORDS = set("""a an the is are was were be been do does did can could i my me we our you your it its
of to in on for at by with from as and or but if what which who how when where why there this that
these those any all much many have has had will would should may might must about into than then so""".split())
RRF_K = 60 


def _stem(token: str) -> str:
    """Tiny plural stemmer: 'passwords'->'password', 'policies'->'policy'. Enough for policy text,
    and predictable (a full Porter stemmer turns 'leave' into 'leav', which hurts readability of debug output)."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in re.findall(r"[a-z0-9]+(?:\.[0-9]+)*", text.lower()) if t not in STOPWORDS]


class Retriever:
    def __init__(self, store: Store):
        self.store = store
        self._bm25_for = None   
        self._bm25 = None

    def _bm25_index(self, chunks: list[Chunk]) -> BM25Okapi:
        if self._bm25_for is not chunks:      
            self._bm25 = BM25Okapi([tokenize(c.embed_text()) for c in chunks])
            self._bm25_for = chunks
        return self._bm25

    def search(self, question: str, top_k: int = config.TOP_K) -> list[RetrievedChunk]:
        chunks, matrix = self.store.all_chunks()
        if not chunks:
            return []


        q_vec = get_embedder().embed([question])[0]
        dense = matrix @ q_vec
        dense_rank = np.argsort(-dense)[: config.CANDIDATES]

        
        q_tokens = tokenize(question)
        bm25 = self._bm25_index(chunks).get_scores(q_tokens) if q_tokens else np.zeros(len(chunks))
        bm25_rank = [i for i in np.argsort(-bm25)[: config.CANDIDATES] if bm25[i] > 0]

        fused: dict[int, float] = {}
        for ranking in (dense_rank, bm25_rank):
            for rank, idx in enumerate(ranking):
                fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (RRF_K + rank + 1)

        best = sorted(fused, key=fused.get, reverse=True)[:top_k]
        return [RetrievedChunk(chunk=chunks[i], dense_score=float(dense[i]), fused_score=fused[i]) for i in best]
