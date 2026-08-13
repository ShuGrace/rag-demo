# RAG Demo

A small retrieval-augmented generation pipeline: dense retrieval via [Qdrant](https://qdrant.tech/), a dense+BM25 hybrid retriever, and a faithfulness checker for evaluating whether generated answers are actually supported by the retrieved context.

This is a public snapshot of an ongoing project. The full corpus (scraped from third-party educational sources) isn't included here for copyright reasons — see `data/sample/` for a small, self-authored sample corpus you can run the pipeline against out of the box.

## Setup

```bash
uv sync
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v ./qdrant_storage:/qdrant/storage qdrant/qdrant
```

## Quick start (sample corpus)

```python
from retriever import QdrantRetriever

r = QdrantRetriever(corpus_path="data/sample/sample_chunks.jsonl")
r.retrieve("How does formative assessment work in project-based learning?", top_k=3)
```

`QdrantRetriever` encodes the corpus with `BAAI/bge-m3` and writes it into a Qdrant collection the first time it runs; subsequent runs reuse the existing collection if the corpus hasn't changed.

## Structure

- `src/retriever.py` — `InMemoryRetriever` (baseline, numpy cosine similarity) and `QdrantRetriever` (same interface, backed by Qdrant)
- `src/hybrid_retriever.py` — dense (bge-m3) + sparse (BM25) fusion retriever
- `src/faithfulness_checker.py` — decomposes a generated answer into claims and checks each against the retrieved context
- `src/ingestion/` — scrapers used to build the original corpus (not included)
- `data/sample/` — small self-authored sample corpus for trying the pipeline without the original data
