"""
Lightweight retrieval-only test: checks whether newly added higher-education
PBL content (Aalborg, Maastricht, McMaster, WPI, etc.) is being retrieved
correctly for relevant queries. No Claude API calls — pure local embedding
and similarity search.
"""
import pathlib
from retriever import QdrantRetriever

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]

TEST_QUERIES = [
    "What is the history of problem-based learning at McMaster University?",
    "How does Maastricht University's seven-jump tutorial method work?",
    "What is the Aalborg Model for project-based engineering education?",
    "What is the difference between problem-based learning and problem-oriented project learning?",
    "How does WPI's project-based curriculum work for undergraduates?",
    "What is Stanford d.school's design thinking process?",
    "How do Chinese secondary schools implement AI-based project learning?",
    "What are the Essential Design Elements of Gold Standard PBL?",
]

def main():
    print("Loading retriever and encoding corpus (local only, no API calls)...")
    retriever = QdrantRetriever(corpus_path=str(PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"))

    for query in TEST_QUERIES:
        print(f"\n{'='*70}")
        print(f"QUERY: {query}")
        print('='*70)
        results = retriever.retrieve(query, top_k=5)
        for r in results:
            print(f"  [{r['score']:.4f}] {r['id']}  ({r['source']})")

if __name__ == "__main__":
    main()