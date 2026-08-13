"""
Run faithfulness evaluation across multiple prompts (different subjects, EN/ZH)
to check whether the faithfulness score is stable across the system.
"""
import json
import pathlib
from retriever import QdrantRetriever
from generate import generate_lesson_plan
from faithfulness_checker import check_faithfulness

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]

TEST_PROMPTS = [
    {"id": "en_math", "prompt": "Design a project-based learning unit for a Grade 6 mathematics class using a driving question, with a public product and a plan for student reflection."},
    {"id": "zh_science", "prompt": "为初中科学课设计一个跨学科项目式学习方案，包含驱动性问题和评价量规。"},
    {"id": "en_collab_rubric", "prompt": "How should I structure a rubric to assess student collaboration during a project?"},
    {"id": "en_history", "prompt": "Design a PBL unit for high school history students using a driving question about civic engagement, with a public product presented to a real audience."},
    {"id": "zh_language_arts", "prompt": "为小学语文课设计一个项目式学习单元，包含驱动性问题、学生自主选择的空间和公开展示环节。"},
    {"id": "en_creativity_assess", "prompt": "How can teachers assess student creativity within a PBL project?"},
]

def run_batch_eval(output_path="data/eval/faithfulness_batch_results.jsonl"):
    print("Loading retriever...")
    retriever = QdrantRetriever(corpus_path=str(PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"))

    Path_output = PROJECT_ROOT / output_path
    Path_output.parent.mkdir(parents=True, exist_ok=True)

    all_results = []

    for item in TEST_PROMPTS:
        print(f"\n{'='*70}")
        print(f"Running: {item['id']}")
        print(f"Prompt: {item['prompt']}")
        print('='*70)

        result = generate_lesson_plan(item["prompt"], retriever)
        report = check_faithfulness(result["output"], result["retrieved_sources"], verbose=False)

        summary = {
            "test_id": item["id"],
            "prompt": item["prompt"],
            "faithfulness_score": report["faithfulness_score"],
            "n_claims": report["n_claims"],
            "n_supported": report["n_supported"],
            "n_partially_supported": report["n_partially_supported"],
            "n_unsupported": report["n_unsupported"],
            "retrieved_source_ids": [s["id"] for s in result["retrieved_sources"]],
        }
        all_results.append(summary)

        print(f"  Faithfulness Score: {report['faithfulness_score']}")
        print(f"  Claims: {report['n_claims']} (supported={report['n_supported']}, "
              f"partial={report['n_partially_supported']}, unsupported={report['n_unsupported']})")

    with open(Path_output, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    scores = [r["faithfulness_score"] for r in all_results if r["faithfulness_score"] is not None]
    print(f"\n{'='*70}")
    print("BATCH SUMMARY")
    print('='*70)
    print(f"Tests run: {len(all_results)}")
    print(f"Mean faithfulness score: {round(sum(scores)/len(scores), 4)}")
    print(f"Min: {min(scores)}  Max: {max(scores)}")
    print(f"Results saved to: {Path_output}")

if __name__ == "__main__":
    from pathlib import Path
    run_batch_eval()