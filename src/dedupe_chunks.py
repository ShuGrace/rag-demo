import json
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"

seen_ids = set()
deduped = []

with open(CHUNKS_PATH, encoding="utf-8-sig") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec["id"] in seen_ids:
            continue
        seen_ids.add(rec["id"])
        deduped.append(rec)

with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
    for r in deduped:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Deduplicated. Total unique chunks: {len(deduped)}")
