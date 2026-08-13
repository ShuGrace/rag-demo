"""
Quantitative evaluation: Recall@K and MRR for Dense-only vs Hybrid retrieval,
using a small hand-labeled ground-truth benchmark.

Pure local computation — no Claude API calls, no token cost.
"""
import pathlib
import json
from retriever import QdrantRetriever
from hybrid_retriever import HybridRetriever

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
CORPUS_PATH = str(PROJECT_ROOT / "data" / "processed" / "chunks.jsonl")
BENCHMARK_PATH = str(PROJECT_ROOT / "benchmark" / "hybrid_eval_benchmark.jsonl")

TOP_K = 5


def load_benchmark(path):
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def recall_at_k(retrieved_ids, relevant_ids, k):
    """Fraction of relevant_ids that appear in the top-k retrieved_ids."""
    top_k_ids = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    if not relevant_set:
        return None
    hit = len(top_k_ids & relevant_set)
    return hit / len(relevant_set)


def reciprocal_rank(retrieved_ids, relevant_ids):
    """1 / rank of the first relevant doc found; 0 if none found in the list."""
    relevant_set = set(relevant_ids)
    for i, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / i
    return 0.0


def evaluate(retriever_name, retriever, benchmark, top_k=TOP_K):
    recalls = []
    rr_scores = []
    per_query_results = []

    for item in benchmark:
        query = item["query"]
        relevant_ids = item["relevant_ids"]

        results = retriever.retrieve(query, top_k=top_k)
        retrieved_ids = [r["id"] for r in results]

        r_at_k = recall_at_k(retrieved_ids, relevant_ids, top_k)
        rr = reciprocal_rank(retrieved_ids, relevant_ids)

        recalls.append(r_at_k)
        rr_scores.append(rr)

        hit_status = "HIT" if rr > 0 else "MISS"
        per_query_results.append(
            f"  [{hit_status}] Recall@{top_k}={r_at_k:.2f}  RR={rr:.2f}  | {query[:60]}"
        )

    mean_recall = sum(recalls) / len(recalls)
    mean_rr = sum(rr_scores) / len(rr_scores)

    print(f"\n{'='*70}")
    print(f"{retriever_name}")
    print('='*70)
    for line in per_query_results:
        print(line)
    print(f"\n  --> Mean Recall@{top_k}: {mean_recall:.4f}")
    print(f"  --> Mean Reciprocal Rank (MRR): {mean_rr:.4f}")

    return {"mean_recall": mean_recall, "mean_rr": mean_rr}


def main():
    benchmark = load_benchmark(BENCHMARK_PATH)
    print(f"Loaded {len(benchmark)} labeled queries from benchmark.")

    print("\nLoading dense-only retriever (bge-m3) ...")
    dense_retriever = QdrantRetriever(corpus_path=CORPUS_PATH)

    print("\nLoading hybrid retriever (bge-m3 + BM25) ...")
    hybrid_retriever = HybridRetriever(corpus_path=CORPUS_PATH, dense_weight=0.5)

    dense_metrics = evaluate("DENSE ONLY (bge-m3)", dense_retriever, benchmark)
    hybrid_metrics = evaluate("HYBRID (dense + BM25, 50/50)", hybrid_retriever, benchmark)

    print(f"\n{'='*70}")
    print("SUMMARY COMPARISON")
    print('='*70)
    print(f"{'Metric':<20}{'Dense-only':<15}{'Hybrid':<15}{'Delta':<10}")
    print(f"{'Recall@5':<20}{dense_metrics['mean_recall']:<15.4f}{hybrid_metrics['mean_recall']:<15.4f}"
          f"{hybrid_metrics['mean_recall'] - dense_metrics['mean_recall']:+.4f}")
    print(f"{'MRR':<20}{dense_metrics['mean_rr']:<15.4f}{hybrid_metrics['mean_rr']:<15.4f}"
          f"{hybrid_metrics['mean_rr'] - dense_metrics['mean_rr']:+.4f}")


if __name__ == "__main__":
    main()
