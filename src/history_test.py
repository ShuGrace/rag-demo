"""
Minimal-cost targeted test: 2 topics only (1 normal, 1 suspected weak coverage),
single generation each, human-readable inspection instead of full claim verification.
"""
import pathlib
from retriever import QdrantRetriever
from generate import generate_lesson_plan

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]

TEST_PROMPTS = [
    {"id": "normal_math", "prompt": "How can mathematics teachers design authentic project-based learning tasks that connect to real-world data?"},
    {"id": "weak_history", "prompt": "Design a PBL unit for high school history students using a driving question about civic engagement."},
]

def main():
    print("Loading retriever...")
    retriever = QdrantRetriever(corpus_path=str(PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"))

    for item in TEST_PROMPTS:
        print(f"\n{'='*70}")
        print(f"TEST: {item['id']}")
        print('='*70)

        result = generate_lesson_plan(item["prompt"], retriever, detail_level="concise")

        print("\n--- RETRIEVED SOURCES ---")
        for s in result["retrieved_sources"]:
            print(f"  [{s['id']}] score={s['score']:.4f}")
            print(f"    {s['text'][:150]}...")

        print("\n--- GENERATED TEXT ---")
        print(result["output"])

if __name__ == "__main__":
    main()