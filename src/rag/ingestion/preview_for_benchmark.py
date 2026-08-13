"""
Preview chunk content to help draft benchmark queries.
"""
import json

records = [json.loads(line) for line in open("data/raw/scraped_chunks.jsonl", encoding="utf-8")]

for r in records:
    preview = r["text"][:150].replace("\n", " ")
    print(f"[{r['id']}] {preview}...")
    print()