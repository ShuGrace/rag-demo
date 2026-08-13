"""
Multi-site scraper for Chinese government/education websites related to
project-based learning (PBL) and interdisciplinary curriculum standards.
No robots.txt rules were found for most sites except shqp.gov.cn (explicitly
allows full crawl). Conservative delays are used for all sites as a default
politeness measure.
"""
import requests
import trafilatura
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import time
from pathlib import Path

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Research Bot; PhD Project - University of Auckland; contact: shan636@aucklanduni.ac.nz)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

SKIP_EXTENSIONS = [
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".mp3", ".mp4",
    ".wav", ".zip", ".doc", ".docx", ".xls", ".xlsx", ".css", ".js",
]

LOW_VALUE_KEYWORDS = [
    "login", "denglu", "search", "sitemap", "wechat", "weixin",
    "/zt/", "index_1", "banquan", "copyright",
]

def is_path_excluded(url, excluded_paths):
    path = urlparse(url).path.lower()
    if any(path.endswith(ext) for ext in SKIP_EXTENSIONS):
        return True
    if any(kw in path for kw in LOW_VALUE_KEYWORDS):
        return True
    for pattern in excluded_paths:
        if pattern.lower() in path:
            return True
    return False

def discover_links(start_url, max_pages=15, excluded_paths=None, crawl_delay=10):
    excluded_paths = excluded_paths or []
    visited = set()
    to_visit = [start_url]
    domain = urlparse(start_url).netloc
    found_urls = []

    while to_visit and len(found_urls) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            print(f"  [fetch] {url} -> status {resp.status_code}")
            if resp.status_code != 200:
                continue

            if not is_path_excluded(url, excluded_paths):
                found_urls.append(url)

            soup = BeautifulSoup(resp.content, "lxml")
            for link in soup.find_all("a", href=True):
                full_url = urljoin(url, link["href"]).split("#")[0]
                if urlparse(full_url).netloc == domain and full_url not in visited:
                    if not is_path_excluded(full_url, excluded_paths):
                        to_visit.append(full_url)

            time.sleep(crawl_delay)
        except Exception as e:
            print(f"  [error] {url}: {e}")
            continue

    return found_urls

def extract_content(url):
    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        return None
    return trafilatura.extract(downloaded, include_comments=False, include_tables=True)

def chunk_text(text, chunk_size=500, overlap=50):
    words = list(text)  # character-based chunking, more suitable for Chinese text
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append("".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

def scrape_site(start_url, source_name, max_pages=15, excluded_paths=None,
                 crawl_delay=10, output_path="data/raw/cn_scraped_chunks.jsonl"):
    print(f"\nDiscovering links from {start_url}...")
    urls = discover_links(start_url, max_pages=max_pages,
                           excluded_paths=excluded_paths, crawl_delay=crawl_delay)
    print(f"Found {len(urls)} pages to scrape.")

    if not urls:
        print(f"WARNING: no pages found for {source_name}.")
        return 0

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    chunk_count = 0

    with open(output_path, "a", encoding="utf-8") as f:
        for i, url in enumerate(urls):
            print(f"[{i+1}/{len(urls)}] Extracting: {url}")
            text = extract_content(url)
            if not text or len(text.strip()) < 80:
                print("  -> skipped (empty or too short)")
                continue

            chunks = chunk_text(text)
            for j, chunk in enumerate(chunks):
                record = {
                    "id": f"{source_name}_{i}_{j}",
                    "text": chunk,
                    "source": source_name,
                    "url": url,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                chunk_count += 1

            print(f"  -> {len(chunks)} chunks")
            time.sleep(crawl_delay)

    print(f"Done with {source_name}. {chunk_count} chunks written.")
    return chunk_count

if __name__ == "__main__":
    # Confirmed-accessible Chinese government/education sites.
    # cnaes.edu.cn excluded for now due to unstable DNS resolution.
    SITES = [
        {
            "start_url": "http://www.moe.gov.cn/",
            "source_name": "moe",
            "max_pages": 15,
            "crawl_delay": 10,
            "excluded_paths": [],
        },
        {
            "start_url": "https://hudong.moe.gov.cn/",
            "source_name": "moe_hudong",
            "max_pages": 15,
            "crawl_delay": 10,
            "excluded_paths": [],
        },
        {
            "start_url": "https://www.shqp.gov.cn/",
            "source_name": "shqp",
            "max_pages": 15,
            "crawl_delay": 5,  # robots.txt explicitly allows full crawl
            "excluded_paths": [],
        },
        {
            "start_url": "https://gdae.gdedu.gov.cn/",
            "source_name": "gdae",
            "max_pages": 15,
            "crawl_delay": 10,
            "excluded_paths": [],
        },
    ]

    total_chunks = 0
    for site in SITES:
        print(f"\n{'='*60}")
        print(f"Starting scrape: {site['source_name']}")
        print(f"{'='*60}")
        count = scrape_site(
            start_url=site["start_url"],
            source_name=site["source_name"],
            max_pages=site["max_pages"],
            excluded_paths=site["excluded_paths"],
            crawl_delay=site["crawl_delay"],
        )
        total_chunks += count

    print(f"\n{'='*60}")
    print(f"All sites done. Total chunks written: {total_chunks}")
    print(f"{'='*60}")