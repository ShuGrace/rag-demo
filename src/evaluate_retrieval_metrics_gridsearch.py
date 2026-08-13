"""
CORRECTED grid search: fixes the bug where dense_weight passed to
HybridRetriever's constructor was silently overridden by query-routing logic
for all non-Chinese queries (which always used a hardcoded 0.9), meaning
prior grid searches only ever tested weight variation on the Chinese query
subset (~23% of the benchmark), not the full corpus.

This script now runs THREE things in a single execution, so one run gives
the complete picture:

  A) Dense-only baseline (InMemoryRetriever)
  B) TRUE global grid search: dense_weight is explicitly passed on every
     retrieve() call, overriding query routing entirely, so the SAME weight
     applies to ALL queries (Chinese and English alike). This answers
     "what is the best single global weight, ignoring language routing?"
  C) Current PRODUCTION behavior check: calls retrieve() with NO explicit
     weight (exactly as generate.py does), so query routing applies as
     designed (Chinese queries get the base weight, English queries are
     hardcoded to 0.9). This tells us what the system actually does today,
     and lets us test different BASE weights specifically for the Chinese
     subset under the real routing logic.

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
WEIGHT_GRID = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]


def load_benchmark(path):
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def recall_at_k(retrieved_ids, relevant_ids, k):
    top_k_ids = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    if not relevant_set:
        return None
    hit = len(top_k_ids & relevant_set)
    return hit / len(relevant_set)


def reciprocal_rank(retrieved_ids, relevant_ids):
    relevant_set = set(relevant_ids)
    for i, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / i
    return 0.0


def evaluate(retriever, benchmark, top_k=TOP_K, explicit_weight=None):
    """
    If explicit_weight is given, it is passed to retrieve() on every call,
    overriding any query-routing logic (true global weight test).
    If explicit_weight is None, retrieve() is called with no weight arg,
    letting the retriever's own internal routing logic decide (production
    behavior test).
    """
    recalls, rr_scores = [], []
    for item in benchmark:
        query = item["query"]
        relevant_ids = item["relevant_ids"]

        if explicit_weight is not None:
            results = retriever.retrieve(query, top_k=top_k, dense_weight=explicit_weight)
        else:
            results = retriever.retrieve(query, top_k=top_k)

        retrieved_ids = [r["id"] for r in results]
        recalls.append(recall_at_k(retrieved_ids, relevant_ids, top_k))
        rr_scores.append(reciprocal_rank(retrieved_ids, relevant_ids))

    return {
        "mean_recall": sum(recalls) / len(recalls),
        "mean_rr": sum(rr_scores) / len(rr_scores),
    }


def main():
    benchmark = load_benchmark(BENCHMARK_PATH)
    n_chinese = sum(1 for item in benchmark
                     if any('\u4e00' <= c <= '\u9fff' for c in item["query"]))
    print(f"Loaded {len(benchmark)} labeled queries "
          f"({n_chinese} Chinese-language, {len(benchmark)-n_chinese} non-Chinese).")

    print("\nLoading InMemoryRetriever (pure dense baseline) ...")
    dense_retriever = QdrantRetriever(corpus_path=CORPUS_PATH)
    dense_metrics = evaluate(dense_retriever, benchmark)

    print("\nLoading HybridRetriever (used for both Part A and Part B below) ...")
    # dense_weight passed here is just the initial base value; it will be
    # overridden per-call in Part A via explicit_weight, and used as the
    # Chinese-query base weight in Part B (production routing behavior).
    hybrid_retriever = HybridRetriever(corpus_path=CORPUS_PATH, dense_weight=0.5)

    # --- PART A: TRUE global grid search (bypasses query routing entirely) ---
    print(f"\n{'#'*70}")
    print("# PART A: TRUE GLOBAL GRID SEARCH")
    print("# (dense_weight explicitly applied to ALL queries, routing bypassed)")
    print(f"{'#'*70}")
    part_a_results = {}
    for w in WEIGHT_GRID:
        m = evaluate(hybrid_retriever, benchmark, explicit_weight=w)
        part_a_results[w] = m
        print(f"  w={w:<5}  Recall@5={m['mean_recall']:.4f}  MRR={m['mean_rr']:.4f}")

    # --- PART B: current PRODUCTION routing behavior, varying the BASE weight ---
    # (base weight only ever affects the Chinese-query subset; English queries
    # are always hardcoded to 0.9 by the routing logic regardless of base weight)
    print(f"\n{'#'*70}")
    print("# PART B: PRODUCTION ROUTING BEHAVIOR")
    print("# (no explicit weight passed -- exactly how generate.py calls retrieve();")
    print("#  base weight only affects the Chinese-query subset)")
    print(f"{'#'*70}")
    part_b_results = {}
    for w in WEIGHT_GRID:
        hybrid_retriever_b = HybridRetriever(corpus_path=CORPUS_PATH, dense_weight=w)
        m = evaluate(hybrid_retriever_b, benchmark, explicit_weight=None)
        part_b_results[w] = m
        print(f"  base_weight={w:<5}  Recall@5={m['mean_recall']:.4f}  MRR={m['mean_rr']:.4f}")

    # --- FINAL SUMMARY ---
    print(f"\n\n{'='*70}")
    print("FINAL SUMMARY")
    print('='*70)
    print(f"Dense-only baseline:       Recall@5={dense_metrics['mean_recall']:.4f}  "
          f"MRR={dense_metrics['mean_rr']:.4f}")

    print(f"\n{'Weight':<10}{'A: Global Recall@5':<22}{'A: Global MRR':<18}"
          f"{'B: Routed Recall@5':<22}{'B: Routed MRR':<18}")
    for w in WEIGHT_GRID:
        a, b = part_a_results[w], part_b_results[w]
        print(f"{w:<10}{a['mean_recall']:<22.4f}{a['mean_rr']:<18.4f}"
              f"{b['mean_recall']:<22.4f}{b['mean_rr']:<18.4f}")

    best_a_w = max(WEIGHT_GRID, key=lambda w: part_a_results[w]['mean_recall'])
    best_b_w = max(WEIGHT_GRID, key=lambda w: part_b_results[w]['mean_recall'])
    print(f"\nBest global weight (Part A, by Recall@5): w={best_a_w} "
          f"(Recall@5={part_a_results[best_a_w]['mean_recall']:.4f})")
    print(f"Best base weight under production routing (Part B, by Recall@5): "
          f"w={best_b_w} (Recall@5={part_b_results[best_b_w]['mean_recall']:.4f})")


if __name__ == "__main__":
    main()