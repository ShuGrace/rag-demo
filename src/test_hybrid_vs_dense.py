"""
Compare dense-only retrieval vs hybrid (dense+BM25) retrieval,
specifically on the History-topic queries that previously showed
topic misalignment issues.
"""
import pathlib
from retriever import QdrantRetriever
from hybrid_retriever import HybridRetriever

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
CORPUS_PATH = str(PROJECT_ROOT / "data" / "processed" / "chunks.jsonl")

TEST_QUERIES = [
    "Design a PBL unit for high school history students using a driving question about civic engagement.",
    "How can mathematics teachers design authentic project-based learning tasks that connect to real-world data?",
]

def print_results(label, results):
    print(f"\n--- {label} ---")
    for r in results:
        score = r.get("combined_score", r.get("score"))
        print(f"  [{score:.4f}] {r['id']}  ({r['source']})")

def main():
    print("Loading dense-only retriever ...")
    dense_retriever = QdrantRetriever(corpus_path=CORPUS_PATH)

    print("\nLoading hybrid retriever ...")
    hybrid_retriever = HybridRetriever(corpus_path=CORPUS_PATH, dense_weight=0.5)

    for query in TEST_QUERIES:
        print(f"\n{'='*70}")
        print(f"QUERY: {query}")
        print('='*70)

        dense_results = dense_retriever.retrieve(query, top_k=5)
        print_results("DENSE ONLY (bge-m3)", dense_results)

        hybrid_results = hybrid_retriever.retrieve(query, top_k=5)
        print_results("HYBRID (dense + BM25, 50/50)", hybrid_results)

if __name__ == "__main__":
    main()