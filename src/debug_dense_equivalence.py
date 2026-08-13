"""
Diagnostic script: HybridRetriever with dense_weight=1.0 should theoretically
produce IDENTICAL rankings to InMemoryRetriever (pure dense, cosine similarity),
since combined_score = 1.0*dense_norm + 0.0*sparse_norm = dense_norm, and
Min-Max normalization is a monotonic (order-preserving) transform.

But the grid search showed different Recall@5/MRR between the two. This script
pinpoints exactly where and why they diverge, by comparing raw dense scores,
normalized scores, and final rankings side by side for the same corpus and
the same set of benchmark queries.
"""
import pathlib
import json
import numpy as np
from retriever import InMemoryRetriever
from hybrid_retriever import HybridRetriever

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
CORPUS_PATH = str(PROJECT_ROOT / "data" / "processed" / "chunks.jsonl")
BENCHMARK_PATH = str(PROJECT_ROOT / "benchmark" / "hybrid_eval_benchmark.jsonl")


def load_benchmark(path):
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    print("Loading InMemoryRetriever (pure dense baseline)...")
    dense_retriever = InMemoryRetriever(corpus_path=CORPUS_PATH)

    print("\nLoading HybridRetriever with dense_weight=1.0...")
    hybrid_retriever = HybridRetriever(corpus_path=CORPUS_PATH, dense_weight=1.0)

    # --- Step 1: Check whether the underlying corpus order/content is identical ---
    print(f"\n{'='*70}")
    print("STEP 1: Corpus consistency check")
    print('='*70)
    print(f"  InMemoryRetriever corpus size: {len(dense_retriever.corpus_ids)}")
    print(f"  HybridRetriever corpus size:   {len(hybrid_retriever.corpus_ids)}")
    ids_match = dense_retriever.corpus_ids == hybrid_retriever.corpus_ids
    print(f"  Corpus IDs in identical order: {ids_match}")
    if not ids_match:
        print("  [!!] Corpus ordering differs between the two retrievers.")
        print("       This alone could cause tie-breaking differences in argsort.")

    # --- Step 2: Check whether raw dense embeddings are numerically identical ---
    print(f"\n{'='*70}")
    print("STEP 2: Embedding consistency check")
    print('='*70)
    if ids_match:
        emb_diff = np.abs(dense_retriever.corpus_emb - hybrid_retriever.corpus_emb).max()
        print(f"  Max absolute difference between embeddings: {emb_diff:.10f}")
        if emb_diff > 1e-6:
            print("  [!!] Embeddings differ meaningfully between the two retrievers.")
        else:
            print("  Embeddings are numerically identical (as expected, same model).")
    else:
        print("  Skipped (corpus order differs, embeddings not directly comparable by index).")

    # --- Step 3: For each benchmark query, compare raw dense-only ranking vs
    #     hybrid(w=1.0) ranking directly, id by id ---
    print(f"\n{'='*70}")
    print("STEP 3: Per-query ranking comparison (Top-10)")
    print('='*70)

    benchmark = load_benchmark(BENCHMARK_PATH)
    n_mismatches = 0

    for item in benchmark:
        query = item["query"]

        dense_results = dense_retriever.retrieve(query, top_k=10)
        hybrid_results = hybrid_retriever.retrieve(query, top_k=10)

        dense_ids = [r["id"] for r in dense_results]
        hybrid_ids = [r["id"] for r in hybrid_results]

        if dense_ids != hybrid_ids:
            n_mismatches += 1
            print(f"\n  [MISMATCH] Query: {query[:60]}")
            print(f"    Dense-only Top-10:      {dense_ids}")
            print(f"    Hybrid(w=1.0) Top-10:   {hybrid_ids}")

            # Show raw scores for the first few docs from each to see where
            # the actual numeric divergence starts
            print(f"    Dense-only scores (top 3): "
                  f"{[round(r['score'], 6) for r in dense_results[:3]]}")
            print(f"    Hybrid(w=1.0) combined_scores (top 3): "
                  f"{[round(r['combined_score'], 6) for r in hybrid_results[:3]]}")
            print(f"    Hybrid(w=1.0) dense_scores (top 3, raw pre-normalization): "
                  f"{[round(r['dense_score'], 6) for r in hybrid_results[:3]]}")

    print(f"\n{'='*70}")
    print(f"SUMMARY: {n_mismatches} / {len(benchmark)} queries had different Top-10 rankings")
    print('='*70)


if __name__ == "__main__":
    main()