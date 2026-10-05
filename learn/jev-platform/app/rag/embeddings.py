"""Embedding providers. OpenAI for real use, a hashing embedder for offline dev and tests."""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from app.config import Settings


class Embedder(Protocol):
    dim: int

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...


class OpenAIEmbedder:
    def __init__(self, s: Settings):
        from langchain_openai import OpenAIEmbeddings

        self.dim = s.embedding_dim
        self._e = OpenAIEmbeddings(model=s.openai_embedding_model, api_key=s.openai_api_key)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._e.aembed_documents(texts)

    async def embed_query(self, text: str) -> list[float]:
        return await self._e.aembed_query(text)


class HashEmbedder:
    """Feature-hashing bag of words + bigrams, L2-normalised.

    Not semantic, but lexically sensible and fully deterministic, which is exactly
    what you want in unit tests and when you're working on a plane.
    """

    _tok = re.compile(r"[a-z0-9]+")

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        toks = self._tok.findall(text.lower())
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        v = [0.0] * self.dim
        for f in feats:
            h = int(hashlib.md5(f.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 8) & 1 else -1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def build_embedder(s: Settings) -> Embedder:
    if s.embedding_provider == "openai":
        return OpenAIEmbedder(s)
    return HashEmbedder(384)
