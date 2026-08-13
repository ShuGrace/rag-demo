import json

seen_ids = set()
deduped = []

with open("data/processed/chunks.jsonl", encoding="utf-8-sig") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec["id"] in seen_ids:
            continue
        seen_ids.add(rec["id"])
        deduped.append(rec)

with open("data/processed/chunks.jsonl", "w", encoding="utf-8") as f:
    for r in deduped:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Deduplicated. Total unique chunks: {len(deduped)}")