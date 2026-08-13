"""
Quick inspection of scraped chunks: count per source, preview content, flag empty/junk entries.
"""
import json
from collections import Counter

def inspect(path="data/processed/chunks.jsonl"):
    records = [json.loads(line) for line in open(path, encoding="utf-8-sig") if line.strip()]

    print(f"Total chunks: {len(records)}\n")

    source_counts = Counter(r["source"] for r in records)
    print("Chunks per source:")
    for source, count in source_counts.items():
        print(f"  {source}: {count}")

    print("\n--- Sample chunks (first 200 chars each) ---\n")
    seen_sources = set()
    for r in records:
        if r["source"] not in seen_sources:
            seen_sources.add(r["source"])
            print(f"[{r['source']}] {r['url']}")
            print(f"  {r['text'][:200]}...\n")

if __name__ == "__main__":
    inspect()