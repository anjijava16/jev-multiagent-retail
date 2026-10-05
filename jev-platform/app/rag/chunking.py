"""
Heading-aware chunking.

Split on markdown headings first so a chunk never straddles two sections, then
pack paragraphs up to chunk_size with overlap. Each chunk carries its section
path ("Returns > Electronics") which gets prepended at embed time. That one
trick helps retrieval more than most chunk-size tuning does.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.M)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    title: str
    section: str
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def embed_text(self) -> str:
        return f"{self.title} | {self.section}\n{self.text}" if self.section else f"{self.title}\n{self.text}"


def _sections(text: str) -> list[tuple[str, str]]:
    """Return [(section_path, body)]."""
    out, stack, last, last_path = [], [], 0, ""
    for m in _HEADING.finditer(text):
        body = text[last:m.start()].strip()
        if body:
            out.append((last_path, body))
        level, name = len(m.group(1)), m.group(2).strip()
        stack = [s for s in stack if s[0] < level] + [(level, name)]
        last_path = " > ".join(n for _, n in stack)
        last = m.end()
    tail = text[last:].strip()
    if tail:
        out.append((last_path, tail))
    return out


def _pack(paras: list[str], size: int, overlap: int) -> list[str]:
    chunks, cur = [], ""
    for p in paras:
        if len(p) > size:  # very long paragraph: hard split
            for i in range(0, len(p), size - overlap):
                chunks.append(p[i:i + size])
            continue
        if len(cur) + len(p) + 2 <= size:
            cur = f"{cur}\n\n{p}" if cur else p
        else:
            if cur:
                chunks.append(cur)
            cur = (cur[-overlap:] + "\n\n" + p) if cur and overlap else p
    if cur:
        chunks.append(cur)
    return chunks


def chunk_document(text: str, source: str, title: str | None = None, metadata: dict | None = None,
                   size: int = 900, overlap: int = 150) -> list[Chunk]:
    doc_id = hashlib.sha1(source.encode()).hexdigest()[:12]
    title = title or source.rsplit("/", 1)[-1]
    out: list[Chunk] = []
    for section, body in _sections(text) or [("", text)]:
        paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        for piece in _pack(paras, size, overlap):
            idx = len(out)
            out.append(Chunk(
                chunk_id=f"{doc_id}-{idx:04d}", doc_id=doc_id, source=source, title=title,
                section=section, text=piece, metadata=dict(metadata or {}),
            ))
    return out
