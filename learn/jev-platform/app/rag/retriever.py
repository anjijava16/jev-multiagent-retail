"""
Hybrid retrieval: BM25 + kNN, fused with Reciprocal Rank Fusion.

    rrf(d) = sum over legs of 1 / (rrf_k + rank_leg(d))

`expand=True` (used by JEV on a retry after weak retrieval) asks the router model
for alternative phrasings and fuses across all of them. That's a cheap fix for
the most common RAG failure: the user's words don't match the document's words.
"""
from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel, Field

from app.config import Settings
from app.llm.gateway import LLMGateway
from app.observability.tracing import span
from app.rag.chunking import chunk_document
from app.rag.embeddings import Embedder
from app.rag.stores import Hit, VectorStore

log = logging.getLogger(__name__)


class QueryRewrites(BaseModel):
    queries: list[str] = Field(description="2-3 alternative search queries using likely document vocabulary")


REWRITE_SYSTEM = (
    "Rewrite the user's question into 2-3 short search queries for a company knowledge base. "
    "Use the vocabulary a policy or product document would use. No answers, only queries."
)


class HybridRetriever:
    def __init__(self, store: VectorStore, embedder: Embedder, gateway: LLMGateway, s: Settings):
        self.store, self.emb, self.gw, self.s = store, embedder, gateway, s

    async def _legs(self, q: str, n: int, filters: dict | None) -> list[list[Hit]]:
        vec = await self.emb.embed_query(q)
        kw, vs = await asyncio.gather(
            self.store.keyword_search(q, n, filters),
            self.store.vector_search(vec, n, filters),
        )
        return [kw, vs]

    def _rrf(self, legs: list[list[Hit]]) -> list[Hit]:
        fused: dict[str, Hit] = {}
        for leg in legs:
            for rank, h in enumerate(leg, start=1):
                cid = h["chunk_id"]
                if cid not in fused:
                    fused[cid] = {**h, "score": 0.0, "legs": 0}
                fused[cid]["score"] += 1.0 / (self.s.rrf_k + rank)
                fused[cid]["legs"] += 1
        return sorted(fused.values(), key=lambda h: h["score"], reverse=True)

    async def retrieve(self, query: str, k: int | None = None, filters: dict | None = None,
                       expand: bool = False) -> list[Hit]:
        k = k or self.s.retrieval_k
        n = self.s.retrieval_candidates
        with span("rag.retrieve", k=k, expand=expand) as attrs:
            queries = [query]
            if expand:
                try:
                    rw = await self.gw.structured("router", REWRITE_SYSTEM, query, QueryRewrites)
                    queries += [q for q in rw.queries if q.strip()][:3]
                except Exception as e:
                    log.warning("query rewrite failed: %s", e)
            all_legs: list[list[Hit]] = []
            for legs in await asyncio.gather(*(self._legs(q, n, filters) for q in queries)):
                all_legs.extend(legs)
            hits = [h for h in self._rrf(all_legs) if h["score"] >= self.s.min_retrieval_score][:k]
            attrs.update(queries=len(queries), hits=len(hits),
                         top_score=round(hits[0]["score"], 4) if hits else 0.0)
            return hits


class IngestService:
    def __init__(self, store: VectorStore, embedder: Embedder, s: Settings):
        self.store, self.emb, self.s = store, embedder, s

    async def ingest_text(self, text: str, source: str, title: str | None = None,
                          metadata: dict | None = None) -> int:
        chunks = chunk_document(text, source, title, metadata, self.s.chunk_size, self.s.chunk_overlap)
        if not chunks:
            return 0
        vectors: list[list[float]] = []
        for i in range(0, len(chunks), 64):  # batch embeddings
            vectors += await self.emb.embed_documents([c.embed_text for c in chunks[i:i + 64]])
        with span("rag.ingest", source=source, chunks=len(chunks)):
            return await self.store.upsert(chunks, vectors)
