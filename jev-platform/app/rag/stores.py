"""
Two retrieval legs per store: keyword (BM25) and vector (kNN). Fusion happens one
level up in the retriever with RRF, so it works on any OpenSearch version and
on the in-memory store the same way.

(OpenSearch 2.10+ also has a native `hybrid` query + normalization search pipeline.
RRF in Python was chosen here because it needs no cluster-side pipeline, scores
from BM25 and cosine are on different scales anyway, and it's easy to debug.)
"""
from __future__ import annotations

import asyncio
import logging
import math
import re
from collections import Counter
from typing import Any, Protocol

from app.config import Settings
from app.rag.chunking import Chunk

log = logging.getLogger(__name__)

Hit = dict[str, Any]  # chunk_id, text, source, title, section, metadata, score


class VectorStore(Protocol):
    async def ensure_index(self) -> None: ...
    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int: ...
    async def keyword_search(self, query: str, n: int, filters: dict | None) -> list[Hit]: ...
    async def vector_search(self, vector: list[float], n: int, filters: dict | None) -> list[Hit]: ...
    async def count(self) -> int: ...
    async def ping(self) -> bool: ...


# =========================================================================== #
# OpenSearch
# =========================================================================== #
class OpenSearchStore:
    def __init__(self, s: Settings, dim: int):
        from opensearchpy import OpenSearch

        auth = (s.opensearch_user, s.opensearch_password) if s.opensearch_user else None
        self.client = OpenSearch(
            hosts=[s.opensearch_url],
            http_auth=auth,
            use_ssl=s.opensearch_url.startswith("https"),
            verify_certs=s.opensearch_verify_certs,
            ssl_show_warn=False,
            timeout=30,
        )
        self.index = s.opensearch_index
        self.dim = dim

    def _mapping(self) -> dict:
        return {
            "settings": {"index": {"knn": True, "number_of_shards": 1, "number_of_replicas": 0}},
            "mappings": {
                "dynamic_templates": [
                    {"meta_strings": {"path_match": "metadata.*", "match_mapping_type": "string",
                                      "mapping": {"type": "keyword"}}}
                ],
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    "source": {"type": "keyword"},
                    "title": {"type": "text", "fields": {"raw": {"type": "keyword"}}},
                    "section": {"type": "text"},
                    "text": {"type": "text", "analyzer": "english"},
                    "metadata": {"type": "object", "dynamic": True},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": self.dim,
                        "method": {"name": "hnsw", "space_type": "cosinesimil", "engine": "lucene",
                                   "parameters": {"m": 16, "ef_construction": 128}},
                    },
                },
            },
        }

    async def ensure_index(self) -> None:
        def _go():
            if not self.client.indices.exists(index=self.index):
                self.client.indices.create(index=self.index, body=self._mapping())
                log.info("created index %s (dim=%d)", self.index, self.dim)
        await asyncio.to_thread(_go)

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        from opensearchpy import helpers

        actions = [{
            "_index": self.index, "_id": c.chunk_id,
            "_source": {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "source": c.source, "title": c.title,
                        "section": c.section, "text": c.text, "metadata": c.metadata, "embedding": v},
        } for c, v in zip(chunks, vectors)]

        def _go():
            ok, _ = helpers.bulk(self.client, actions, refresh="wait_for")
            return ok
        return await asyncio.to_thread(_go)

    @staticmethod
    def _filter_clause(filters: dict | None) -> list[dict]:
        if not filters:
            return []
        out = []
        for k, v in filters.items():
            field = k if k in ("source", "doc_id") else f"metadata.{k}"
            out.append({"terms": {field: v}} if isinstance(v, list) else {"term": {field: v}})
        return out

    @staticmethod
    def _hits(resp: dict) -> list[Hit]:
        return [{**h["_source"], "score": h["_score"]} for h in resp["hits"]["hits"]]

    async def keyword_search(self, query: str, n: int, filters: dict | None) -> list[Hit]:
        body = {
            "size": n,
            "_source": {"excludes": ["embedding"]},
            "query": {"bool": {
                "must": [{"multi_match": {"query": query, "fields": ["text", "title^2", "section^1.5"]}}],
                "filter": self._filter_clause(filters),
            }},
        }
        resp = await asyncio.to_thread(self.client.search, index=self.index, body=body)
        return self._hits(resp)

    async def vector_search(self, vector: list[float], n: int, filters: dict | None) -> list[Hit]:
        knn: dict[str, Any] = {"vector": vector, "k": n}
        fc = self._filter_clause(filters)
        if fc:
            knn["filter"] = {"bool": {"filter": fc}}  # lucene engine: efficient pre-filtering
        body = {"size": n, "_source": {"excludes": ["embedding"]}, "query": {"knn": {"embedding": knn}}}
        resp = await asyncio.to_thread(self.client.search, index=self.index, body=body)
        return self._hits(resp)

    async def count(self) -> int:
        def _go():
            if not self.client.indices.exists(index=self.index):
                return 0
            return self.client.count(index=self.index)["count"]
        return await asyncio.to_thread(_go)

    async def ping(self) -> bool:
        try:
            return await asyncio.to_thread(self.client.ping)
        except Exception:
            return False


# =========================================================================== #
# In-memory (dev / tests)
# =========================================================================== #
_TOK = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "is", "are", "of", "to", "and", "or", "in", "on", "for", "what", "how", "do", "i", "my", "our", "it", "be", "can", "does"}


def _toks(t: str) -> list[str]:
    return [w for w in _TOK.findall(t.lower()) if w not in _STOP]


class MemoryStore:
    def __init__(self):
        self.rows: list[tuple[Chunk, list[float]]] = []

    async def ensure_index(self) -> None:
        return None

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        ids = {c.chunk_id for c in chunks}
        self.rows = [r for r in self.rows if r[0].chunk_id not in ids] + list(zip(chunks, vectors))
        return len(chunks)

    def _match(self, c: Chunk, filters: dict | None) -> bool:
        for k, v in (filters or {}).items():
            have = getattr(c, k, None) if k in ("source", "doc_id") else c.metadata.get(k)
            if (isinstance(v, list) and have not in v) or (not isinstance(v, list) and have != v):
                return False
        return True

    @staticmethod
    def _hit(c: Chunk, score: float) -> Hit:
        return {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "source": c.source, "title": c.title,
                "section": c.section, "text": c.text, "metadata": c.metadata, "score": score}

    async def keyword_search(self, query: str, n: int, filters: dict | None) -> list[Hit]:
        rows = [r for r in self.rows if self._match(r[0], filters)]
        if not rows:
            return []
        docs = [Counter(_toks(c.embed_text)) for c, _ in rows]
        df = Counter(t for d in docs for t in d)
        N, avg = len(docs), sum(sum(d.values()) for d in docs) / len(docs)
        q = _toks(query)
        scored = []
        for (c, _), d in zip(rows, docs):
            dl = sum(d.values())
            s = 0.0
            for t in q:
                if t in d:
                    idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
                    s += idf * d[t] * 2.2 / (d[t] + 1.2 * (0.25 + 0.75 * dl / avg))
            if s > 0:
                scored.append(self._hit(c, s))
        return sorted(scored, key=lambda h: h["score"], reverse=True)[:n]

    async def vector_search(self, vector: list[float], n: int, filters: dict | None) -> list[Hit]:
        scored = [self._hit(c, sum(a * b for a, b in zip(vector, v)))
                  for c, v in self.rows if self._match(c, filters)]
        return sorted(scored, key=lambda h: h["score"], reverse=True)[:n]

    async def count(self) -> int:
        return len(self.rows)

    async def ping(self) -> bool:
        return True


def build_store(s: Settings, dim: int) -> VectorStore:
    return OpenSearchStore(s, dim) if s.vector_backend == "opensearch" else MemoryStore()
