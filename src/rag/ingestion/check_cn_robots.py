"""
Check robots.txt for all target Chinese education websites before scraping.
No content is fetched, this only inspects crawling rules.
"""
import requests
from urllib.parse import urljoin

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Research Bot; PhD Project - University of Auckland; contact: shan636@aucklanduni.ac.nz)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

SITES = [
    "http://www.moe.gov.cn/",
    "https://hudong.moe.gov.cn/",
    "https://www.cnaes.edu.cn/",
    "https://gdae.gdedu.gov.cn/",
    "https://www.shanghai.gov.cn/",
    "https://www.shcm.gov.cn/",
    "https://www.shpt.gov.cn/",
    "https://www.shqp.gov.cn/",
]

def check_robots(base_url):
    robots_url = urljoin(base_url, "/robots.txt")
    try:
        resp = requests.get(robots_url, headers=HEADERS, timeout=10)
        print(f"\n{'='*70}")
        print(f"SITE: {base_url}")
        print(f"robots.txt URL: {robots_url}")
        print(f"Status code: {resp.status_code}")
        print(f"{'-'*70}")
        print(resp.text[:2000] if resp.text else "(empty response)")
        print(f"{'='*70}")
    except Exception as e:
        print(f"\nERROR fetching robots.txt for {base_url}: {e}")

if __name__ == "__main__":
    for site in SITES:
        check_robots(site)