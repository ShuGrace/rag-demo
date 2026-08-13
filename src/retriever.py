"""
In-memory retriever using sentence-transformers embeddings.
Temporary substitute for Qdrant until Docker is available.
"""
import os
import uuid
from sentence_transformers import SentenceTransformer
import numpy as np
import json

class InMemoryRetriever:
    def __init__(self, corpus_path, model_name="BAAI/bge-m3"):
        self.model = SentenceTransformer(model_name)
        self.corpus = self._load_jsonl(corpus_path)
        self.corpus_texts = [c["text"] for c in self.corpus]
        self.corpus_ids = [c["id"] for c in self.corpus]
        print(f"Encoding {len(self.corpus)} chunks with {model_name} ...")
        self.corpus_emb = self.model.encode(
            self.corpus_texts, normalize_embeddings=True, show_progress_bar=True,
            batch_size=8,
        )

    @staticmethod
    def _load_jsonl(path):
        with open(path, encoding="utf-8-sig") as f:
            return [json.loads(line) for line in f if line.strip()]

    def retrieve(self, query, top_k=5):
        q_emb = self.model.encode([query], normalize_embeddings=True)
        sims = self.corpus_emb @ q_emb.T
        ranked_idx = np.argsort(-sims.flatten())[:top_k]
        results = []
        for idx in ranked_idx:
            results.append({
                "id": self.corpus_ids[idx],
                "text": self.corpus_texts[idx],
                "source": self.corpus[idx]["source"],
                "url": self.corpus[idx].get("url", ""),
                "score": float(sims.flatten()[idx]),
            })
        return results


class QdrantRetriever:
    """
    Dense retriever backed by a Qdrant collection instead of an in-memory
    numpy matrix. Same constructor signature and retrieve() output shape as
    InMemoryRetriever, so it's a drop-in replacement.

    Point IDs are deterministic UUIDs derived from each chunk's own "id"
    field (uuid5), so re-running ingestion against the same corpus is
    idempotent (upsert, not duplicate).
    """

    def __init__(self, corpus_path, model_name="BAAI/bge-m3",
                 collection_name=None, url=None, force_reindex=False):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct

        self.model = SentenceTransformer(model_name)
        self.corpus = self._load_jsonl(corpus_path)
        self.corpus_texts = [c["text"] for c in self.corpus]
        self.corpus_ids = [c["id"] for c in self.corpus]

        self.collection_name = collection_name or _default_collection_name(model_name)
        self.client = QdrantClient(url=url or os.environ.get("QDRANT_URL", "http://localhost:6333"))

        vector_size = self.model.get_embedding_dimension()
        exists = self.client.collection_exists(self.collection_name)
        needs_index = force_reindex or not exists
        if not needs_index:
            count = self.client.count(self.collection_name, exact=True).count
            needs_index = count != len(self.corpus)

        if needs_index:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            print(f"Encoding {len(self.corpus)} chunks with {model_name} for Qdrant collection "
                  f"'{self.collection_name}' ...")
            embeddings = self.model.encode(
                self.corpus_texts, normalize_embeddings=True, show_progress_bar=True,
                batch_size=8,
            )
            points = [
                PointStruct(
                    id=_point_id(chunk["id"]),
                    vector=embeddings[i].tolist(),
                    payload=chunk,
                )
                for i, chunk in enumerate(self.corpus)
            ]
            batch_size = 256
            for start in range(0, len(points), batch_size):
                self.client.upsert(self.collection_name, points=points[start:start + batch_size])

    @staticmethod
    def _load_jsonl(path):
        with open(path, encoding="utf-8-sig") as f:
            return [json.loads(line) for line in f if line.strip()]

    def retrieve(self, query, top_k=5):
        q_emb = self.model.encode([query], normalize_embeddings=True)[0]
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=q_emb.tolist(),
            limit=top_k,
        ).points
        results = []
        for hit in hits:
            payload = hit.payload
            results.append({
                "id": payload["id"],
                "text": payload["text"],
                "source": payload["source"],
                "url": payload.get("url", ""),
                "score": float(hit.score),
            })
        return results


def _default_collection_name(model_name):
    return "rag_" + model_name.replace("/", "_").replace("-", "_").lower()


def _point_id(chunk_id):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))