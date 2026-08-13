"""
Controlled experiment: same prompts, two generation conditions (concise vs detailed),
to test whether output length/detail drives faithfulness score down.
"""
import json
import pathlib
from retriever import QdrantRetriever
from generate import generate_lesson_plan
from faithfulness_checker import check_faithfulness

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]

TEST_PROMPTS = [
    {"id": "math", "prompt": "How can mathematics teachers design authentic project-based learning tasks that connect to real-world data?"},
    {"id": "collaboration", "prompt": "What does a high-quality collaboration rubric look like for middle school students?"},
    {"id": "history", "prompt": "Design a PBL unit for high school history students using a driving question about civic engagement."},
    {"id": "creativity", "prompt": "How can teachers assess student creativity within a PBL project?"},
]

def run_length_experiment(output_path="data/eval/length_control_results.jsonl"):
    print("Loading retriever...")
    retriever = QdrantRetriever(corpus_path=str(PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"))

    output_full_path = PROJECT_ROOT / output_path
    output_full_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = []

    for item in TEST_PROMPTS:
        for condition in ["concise", "detailed"]:
            print(f"\n{'='*70}")
            print(f"[{condition.upper()}] {item['id']}")
            print('='*70)

            result = generate_lesson_plan(item["prompt"], retriever, detail_level=condition)
            report = check_faithfulness(result["output"], result["retrieved_sources"], verbose=False)

            word_count = len(result["output"].split())

            summary = {
                "test_id": item["id"],
                "condition": condition,
                "word_count": word_count,
                "faithfulness_score": report["faithfulness_score"],
                "n_claims": report["n_claims"],
                "n_supported": report["n_supported"],
                "n_partially_supported": report["n_partially_supported"],
                "n_unsupported": report["n_unsupported"],
            }
            all_results.append(summary)

            print(f"  Word count: {word_count}")
            print(f"  Faithfulness Score: {report['faithfulness_score']}")
            print(f"  Claims: {report['n_claims']} (supported={report['n_supported']}, "
                  f"partial={report['n_partially_supported']}, unsupported={report['n_unsupported']})")

    with open(output_full_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'='*70}")
    print("PAIRED COMPARISON")
    print('='*70)
    for item in TEST_PROMPTS:
        concise = next(r for r in all_results if r["test_id"] == item["id"] and r["condition"] == "concise")
        detailed = next(r for r in all_results if r["test_id"] == item["id"] and r["condition"] == "detailed")
        print(f"\n[{item['id']}]")
        print(f"  Concise:  words={concise['word_count']:4d}  claims={concise['n_claims']:2d}  score={concise['faithfulness_score']}")
        print(f"  Detailed: words={detailed['word_count']:4d}  claims={detailed['n_claims']:2d}  score={detailed['faithfulness_score']}")

    concise_scores = [r["faithfulness_score"] for r in all_results if r["condition"] == "concise"]
    detailed_scores = [r["faithfulness_score"] for r in all_results if r["condition"] == "detailed"]
    print(f"\n{'='*70}")
    print("OVERALL")
    print('='*70)
    print(f"Mean concise score:  {round(sum(concise_scores)/len(concise_scores), 4)}")
    print(f"Mean detailed score: {round(sum(detailed_scores)/len(detailed_scores), 4)}")
    print(f"\nResults saved to: {output_full_path}")

if __name__ == "__main__":
    run_length_experiment()