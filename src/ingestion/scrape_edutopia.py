"""
Scrape curated, high-value Edutopia articles on project-based learning (PBL).
Edutopia (George Lucas Educational Foundation) publishes freely accessible,
no-login-required articles on PBL theory and practice.
"""
import trafilatura
import json
import time
from pathlib import Path

TARGET_URLS = [
    {
        "url": "https://www.edutopia.org/blog/what-heck-project-based-learning-heather-wolpert-gawron",
        "source_name": "edutopia_pbl_defined",
        "description": "What the Heck Is Project-Based Learning? - core PBL definition and elements",
    },
    {
        "url": "https://www.edutopia.org/project-based-learning-getting-started-resources",
        "source_name": "edutopia_getting_started",
        "description": "Resources for Getting Started With Project-Based Learning",
    },
    {
        "url": "https://www.edutopia.org/project-based-learning",
        "source_name": "edutopia_pbl_hub",
        "description": "Edutopia Project-Based Learning hub page",
    },
]

def chunk_text(text, chunk_size=350, overlap=40):
    """Word-based chunking, suitable for English text."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

def scrape_target(item, output_path="data/raw/edutopia_chunks.jsonl"):
    url = item["url"]
    source_name = item["source_name"]
    print(f"\nFetching: {item['description']}")
    print(f"URL: {url}")

    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            print("  [error] could not download page")
            return 0

        text = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
        if not text or len(text.strip()) < 100:
            print("  [warn] extracted text too short or empty")
            return 0

        chunks = chunk_text(text)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "a", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks):
                record = {
                    "id": f"{source_name}_{i}",
                    "text": chunk,
                    "source": source_name,
                    "url": url,
                    "description": item["description"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"  [ok] {len(chunks)} chunks written")
        return len(chunks)

    except Exception as e:
        print(f"  [error] {e}")
        return 0

if __name__ == "__main__":
    total_chunks = 0
    for item in TARGET_URLS:
        count = scrape_target(item)
        total_chunks += count
        time.sleep(6)

    print(f"\n{'='*60}")
    print(f"All targets done. Total chunks written: {total_chunks}")
    print(f"{'='*60}")
    