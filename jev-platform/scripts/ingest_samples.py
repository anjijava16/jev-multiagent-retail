"""Load sample_data/*.md into the configured vector store (OpenSearch by default).

    python -m scripts.ingest_samples                  # uses .env
    python -m scripts.ingest_samples --dir ./my_docs  # your own docs
"""
import argparse
import asyncio
from pathlib import Path

from app.config import get_settings
from app.container import build_container


async def main(folder: str) -> None:
    c = build_container(get_settings())
    await c.store.ensure_index()
    total = 0
    for p in sorted(Path(folder).glob("*.md")):
        n = await c.ingest.ingest_text(p.read_text(), source=p.name, metadata={"department": "support"})
        print(f"  {p.name:<28} {n} chunks")
        total += n
    print(f"done: {total} chunks, index now has {await c.store.count()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="sample_data")
    asyncio.run(main(ap.parse_args().dir))
