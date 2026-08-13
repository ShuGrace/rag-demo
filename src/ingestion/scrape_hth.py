"""
Scrape High Tech High's project library (English, real PBL project examples
across subjects and grade levels). Publicly accessible, no login required.
"""
import requests
import trafilatura
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json
import time
from pathlib import Path

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Research Bot; PhD Project - University of Auckland; contact: shan636@aucklanduni.ac.nz)",
}

LIST_PAGE = "https://www.hightechhigh.org/student-work/projects/"
MAX_PROJECTS = 20  # start with a modest batch to verify quality

def discover_project_links(list_url, max_projects=20):
    """Collect individual /project/xxx/ links from the listing page(s)."""
    links = []
    page = 1
    while len(links) < max_projects and page <= 3:
        url = list_url if page == 1 else f"{list_url}?page={page}"
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            break
        soup = BeautifulSoup(resp.content, "lxml")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/project/" in href and href not in links:
                links.append(href)
        page += 1
        time.sleep(5)
    return links[:max_projects]

def chunk_text(text, chunk_size=350, overlap=40):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

def scrape_project(url, idx, output_path="data/raw/hth_projects_chunks.jsonl"):
    print(f"[{idx}] Fetching: {url}")
    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        print("  [error] could not download")
        return 0

    text = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
    if not text or len(text.strip()) < 100:
        print("  [warn] too short, skipped")
        return 0

    chunks = chunk_text(text)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            record = {
                "id": f"hth_project_{idx}_{i}",
                "text": chunk,
                "source": "high_tech_high_project_library",
                "url": url,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"  [ok] {len(chunks)} chunks")
    return len(chunks)

if __name__ == "__main__":
    print("Discovering project links...")
    links = discover_project_links(LIST_PAGE, max_projects=MAX_PROJECTS)
    print(f"Found {len(links)} project links.\n")

    total = 0
    for i, link in enumerate(links, start=1):
        total += scrape_project(link, i)
        time.sleep(6)

    print(f"\nDone. Total chunks written: {total}")