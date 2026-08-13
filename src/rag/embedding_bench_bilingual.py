"""
Compare embedding models on the bilingual PBL benchmark.
Reports Recall@5 and MRR separately for English and Chinese queries.
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

    results_by_lang = {"en": {"recall": [], "mrr": []}, "zh": {"recall": [], "mrr": []}}

    for item in benchmark:
        q_emb = model.encode([item["query"]], normalize_embeddings=True)
        sims = corpus_emb @ q_emb.T
        ranked_idx = np.argsort(-sims.flatten())
        ranked_ids = [corpus_ids[i] for i in ranked_idx]

        gold = set(item["relevant_passage_ids"])
        top5 = set(ranked_ids[:5])
        recall = len(gold & top5) / len(gold) if gold else 0

        rr = 0
        for rank, cid in enumerate(ranked_ids, start=1):
            if cid in gold:
                rr = 1 / rank
                break

        lang = item.get("lang", "en")
        results_by_lang[lang]["recall"].append(recall)
        results_by_lang[lang]["mrr"].append(rr)

    summary = {}
    for lang, data in results_by_lang.items():
        if data["recall"]:
            summary[lang] = {
                "Recall@5": round(np.mean(data["recall"]), 4),
                "MRR": round(np.mean(data["mrr"]), 4),
                "n_queries": len(data["recall"]),
            }
    return summary

if __name__ == "__main__":
    corpus = load_jsonl("data/processed/chunks.jsonl")
    benchmark = load_jsonl("benchmark/retrieval_benchmark.jsonl")

    print(f"Corpus size: {len(corpus)} chunks")
    print(f"Benchmark size: {len(benchmark)} queries "
          f"({sum(1 for b in benchmark if b.get('lang')=='en')} EN, "
          f"{sum(1 for b in benchmark if b.get('lang')=='zh')} ZH)")

    all_results = {}
    for name, path in MODELS.items():
        all_results[name] = evaluate_model(path, corpus, benchmark)

    print("\n" + "="*60)
    print("RESULTS BY LANGUAGE")
    print("="*60)
    for name, langs in all_results.items():
        print(f"\n{name}:")
        for lang, metrics in langs.items():
            print(f"  [{lang}] Recall@5={metrics['Recall@5']}  MRR={metrics['MRR']}  (n={metrics['n_queries']})")