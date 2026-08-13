"""
Compare embedding models on the PBE-related retrieval benchmark.
Metrics: Recall@5, MRR.
"""
from sentence_transformers import SentenceTransformer
import numpy as np
import json

MODELS = {
    "bge-large-en": "BAAI/bge-large-en-v1.5",
    "multilingual-e5-large": "intfloat/multilingual-e5-large",
    "bge-m3": "BAAI/bge-m3",
}

def load_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8-sig") if line.strip()]

def evaluate_model(model_name, corpus, benchmark):
    print(f"\nLoading model: {model_name} ...")
    model = SentenceTransformer(model_name)

    corpus_texts = [c["text"] for c in corpus]
    corpus_ids = [c["id"] for c in corpus]
    corpus_emb = model.encode(corpus_texts, normalize_embeddings=True, show_progress_bar=True)

    recalls_at_5, mrrs = [], []
    for item in benchmark:
        q_emb = model.encode([item["query"]], normalize_embeddings=True)
        sims = corpus_emb @ q_emb.T
        ranked_idx = np.argsort(-sims.flatten())
        ranked_ids = [corpus_ids[i] for i in ranked_idx]

        gold = set(item["relevant_passage_ids"])
        top5 = set(ranked_ids[:5])
        recalls_at_5.append(len(gold & top5) / len(gold) if gold else 0)

        rr = 0
        for rank, cid in enumerate(ranked_ids, start=1):
            if cid in gold:
                rr = 1 / rank
                break
        mrrs.append(rr)

    return {
        "Recall@5": round(np.mean(recalls_at_5), 4),
        "MRR": round(np.mean(mrrs), 4),
    }

if __name__ == "__main__":
    corpus = load_jsonl("data/processed/chunks.jsonl")
    benchmark = load_jsonl("benchmark/retrieval_benchmark.jsonl")

    print(f"Corpus size: {len(corpus)} chunks")
    print(f"Benchmark size: {len(benchmark)} queries")

    results = {}
    for name, path in MODELS.items():
        results[name] = evaluate_model(path, corpus, benchmark)

    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    for name, metrics in results.items():
        print(f"{name}: {metrics}")