"""
Controlled comparison: test faithfulness scores for subjects with rich vs. sparse
coverage in the knowledge base, to verify whether coverage depth predicts faithfulness.
"""
import json
import pathlib
from retriever import QdrantRetriever
from generate import generate_lesson_plan
from faithfulness_checker import check_faithfulness

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]

TEST_PROMPTS = [
    # "Rich coverage" group: subjects with dedicated PBLWorks Spotlight + additional case studies
    {"id": "rich_math", "group": "rich", "prompt": "How can mathematics teachers design authentic project-based learning tasks that connect to real-world data?"},
    {"id": "rich_science", "group": "rich", "prompt": "How does PBL support the engineering design process and sustained inquiry in a science classroom?"},
    {"id": "rich_collaboration", "group": "rich", "prompt": "What does a high-quality collaboration rubric look like for middle school students?"},

    # "Sparse coverage" group: subjects with only 1 short Spotlight document each
    {"id": "sparse_pe", "group": "sparse", "prompt": "How can physical education teachers design a project-based learning experience using student fitness data?"},
    {"id": "sparse_arts", "group": "sparse", "prompt": "How can visual and performing arts teachers incorporate community partnerships into a PBL project?"},
    {"id": "sparse_special_ed", "group": "sparse", "prompt": "How can PBL be adapted to be inclusive of students with special needs, particularly around embedding IEP goals?"},
]

def run_coverage_test(output_path="data/eval/coverage_faithfulness_results.jsonl"):
    print("Loading retriever...")
    retriever = QdrantRetriever(corpus_path=str(PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"))

    output_full_path = PROJECT_ROOT / output_path
    output_full_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = []

    for item in TEST_PROMPTS:
        print(f"\n{'='*70}")
        print(f"[{item['group'].upper()}] Running: {item['id']}")
        print(f"Prompt: {item['prompt']}")
        print('='*70)

        result = generate_lesson_plan(item["prompt"], retriever)
        report = check_faithfulness(result["output"], result["retrieved_sources"], verbose=False)

        avg_retrieval_score = sum(s["score"] for s in result["retrieved_sources"]) / len(result["retrieved_sources"])

        summary = {
            "test_id": item["id"],
            "group": item["group"],
            "prompt": item["prompt"],
            "faithfulness_score": report["faithfulness_score"],
            "n_claims": report["n_claims"],
            "n_supported": report["n_supported"],
            "n_partially_supported": report["n_partially_supported"],
            "n_unsupported": report["n_unsupported"],
            "avg_retrieval_similarity": round(avg_retrieval_score, 4),
            "retrieved_source_ids": [s["id"] for s in result["retrieved_sources"]],
        }
        all_results.append(summary)

        print(f"  Faithfulness Score: {report['faithfulness_score']}")
        print(f"  Avg retrieval similarity: {summary['avg_retrieval_similarity']}")
        print(f"  Claims: {report['n_claims']} (supported={report['n_supported']}, "
              f"partial={report['n_partially_supported']}, unsupported={report['n_unsupported']})")

    with open(output_full_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'='*70}")
    print("GROUP COMPARISON")
    print('='*70)
    for group in ["rich", "sparse"]:
        group_results = [r for r in all_results if r["group"] == group]
        scores = [r["faithfulness_score"] for r in group_results if r["faithfulness_score"] is not None]
        sims = [r["avg_retrieval_similarity"] for r in group_results]
        print(f"\n[{group.upper()} coverage group] (n={len(group_results)})")
        print(f"  Mean faithfulness: {round(sum(scores)/len(scores), 4)}")
        print(f"  Mean retrieval similarity: {round(sum(sims)/len(sims), 4)}")

    print(f"\nResults saved to: {output_full_path}")

if __name__ == "__main__":
    run_coverage_test()