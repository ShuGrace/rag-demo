"""
Debug script: inspect the full top-10 results for the persistently
under-performing "driving question" query, to determine whether the
second relevant chunk is just outside top-5 or genuinely far down.
"""
import pathlib
from hybrid_retriever import HybridRetriever

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
CORPUS_PATH = str(PROJECT_ROOT / "data" / "processed" / "chunks.jsonl")

QUERY = "How do you write an effective driving question for a PBL project?"
TARGET_IDS = ["edutopia_miller_driving_questions_0", "edutopia_search_driving_question_0"]

def main():
    retriever = HybridRetriever(corpus_path=CORPUS_PATH, dense_weight=0.5)
    results = retriever.retrieve(QUERY, top_k=10)

    print(f"\nQUERY: {QUERY}")
    print(f"TARGET IDS: {TARGET_IDS}\n")
    for rank, r in enumerate(results, start=1):
        marker = " <-- TARGET" if r["id"] in TARGET_IDS else ""
        print(f"  #{rank}  [{r['combined_score']:.4f}]  {r['id']}{marker}")

if __name__ == "__main__":
    main()