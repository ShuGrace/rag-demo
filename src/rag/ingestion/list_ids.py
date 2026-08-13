"""
Print all chunk ids and a short preview, to help construct an accurate benchmark.
Output file is written explicitly as UTF-8 with BOM so Windows/VS Code display it correctly.
"""
import json
import codecs

records = [json.loads(line) for line in open("data/processed/chunks.jsonl", encoding="utf-8-sig") if line.strip()]

with codecs.open("id_list.txt", "w", encoding="utf-8-sig") as f:
    for r in records:
        preview = r["text"][:80].replace("\n", " ")
        f.write(f"{r['id']}\t{preview}\n")

print(f"Done. Wrote {len(records)} entries to id_list.txt")