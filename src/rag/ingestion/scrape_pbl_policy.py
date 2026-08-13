"""
Scrape specific, pre-identified official Chinese policy documents related to
project-based learning (PBL) and interdisciplinary thematic learning.
URLs are curated from search results rather than discovered via crawling,
since these are known high-value pages on government/education sites.
"""
import requests
import trafilatura
import json
import time
from pathlib import Path

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Research Bot; PhD Project - University of Auckland; contact: shan636@aucklanduni.ac.nz)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

# Curated list of confirmed, high-value URLs directly related to PBL /
# interdisciplinary thematic learning policy in China.
TARGET_URLS = [
    {
        "url": "http://www.moe.gov.cn/srcsite/A06/jcys_jyzb/202511/t20251111_1419878.html",
        "source_name": "moe_tech_education_opinion_2025",
        "description": "教育部等七部门关于加强中小学科技教育的意见 (教基〔2025〕7号)",
    },
    {
        "url": "http://www.moe.gov.cn/jyb_xwfb/gzdt_gzdt/s5987/202511/t20251112_1419962.html",
        "source_name": "moe_tech_education_deployment_2025",
        "description": "教育部等七部门部署推进中小学科技教育",
    },
    {
        "url": "http://www.moe.gov.cn/jyb_xwfb/s271/202511/t20251112_1419960.html",
        "source_name": "moe_tech_education_qa_2025",
        "description": "教育部基础教育司负责人就《关于加强中小学科技教育的意见》答记者问",
    },
    {
        "url": "https://www.shanghai.gov.cn/rkywjy3/20240514/1b318b5839df41d8a234432a4fd40557.html",
        "source_name": "shanghai_hongkou_pbl_plan_2024",
        "description": "关于印发《虹口区关于全面推进义务教育项目化学习的实施方案》的通知",
    },
    {
        "url": "https://www.shpt.gov.cn/zhengwu/ywjy-jyjjcjy/2024/242/191466.html",
        "source_name": "shanghai_putuo_pbl_plan_2024",
        "description": "关于印发《普陀区实施项目化学习推动义务教育育人方式改革的工作方案》的通知",
    },
    {
        "url": "https://www.shcm.gov.cn/govxxgk/qjyj/2024-04-01/f5ac251e-97f8-45e5-b8e4-689e797c0758.html",
        "source_name": "shanghai_chongming_pbl_plan_2024",
        "description": "崇明区教育局关于实施项目化学习推动义务教育育人方式改革的行动计划",
    },
    {
        "url": "https://www.shqp.gov.cn/edu/eduzwgk/lm/jy/ky/cg/20201110/802184.html",
        "source_name": "shanghai_qingpu_pbl_awards_2020",
        "description": "第二届学习素养·项目化学习全国案例征集与评选获奖名单",
    },
    {
        "url": "https://gdae.gdedu.gov.cn/gdjyyjy/tzgg/202107/d4dd99fd081c4c2d8e033e33ebe6997d.shtml",
        "source_name": "guangdong_pbl_case_call_2021",
        "description": "关于征集广东省中小学项目式学习案例的函",
    },
    {
        "url": "https://www.gov.cn/zhengce/zhengceku/202401/content_6925017.htm",
        "source_name": "gov_experimental_zones_notice",
        "description": "教育部办公厅关于推荐义务教育教学改革实验区和实验校的通知",
    },
    {
        "url": "https://www.ictdedu.cn/sknews/jcjyck/neirong/n20240912_85266.shtml",
        "source_name": "ictedu_pbl_action_path",
        "description": "新课程视域下项目式学习行动路径的建构",
    },
    {
        "url": "https://aic-fe.bnu.edu.cn/gnhz/gnhzepbl/index.html",
        "source_name": "bnu_epbl_intro",
        "description": "北京师范大学未来教育高精尖创新中心 EPBL项目介绍",
    },
]

def chunk_text(text, chunk_size=500, overlap=50):
    """Character-based chunking, suitable for Chinese text."""
    chars = list(text)
    chunks = []
    start = 0
    while start < len(chars):
        end = start + chunk_size
        chunks.append("".join(chars[start:end]))
        start += chunk_size - overlap
    return chunks

def scrape_target(item, output_path="data/raw/cn_pbl_policy_chunks.jsonl"):
    url = item["url"]
    source_name = item["source_name"]
    print(f"\nFetching: {item['description']}")
    print(f"URL: {url}")

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()

        # Try to detect encoding: many older Chinese sites use GBK/GB2312
        raw_bytes = resp.content
        html = None
        for encoding in ["utf-8", "gbk", "gb2312", "gb18030"]:
            try:
                html = raw_bytes.decode(encoding)
                break
            except UnicodeDecodeError:
                continue

        if html is None:
            print("  [error] could not decode with any known encoding")
            return 0

        text = trafilatura.extract(html, include_comments=False, include_tables=True)
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
        time.sleep(8)  # polite delay between requests across different domains

    print(f"\n{'='*60}")
    print(f"All targets done. Total chunks written: {total_chunks}")
    print(f"{'='*60}")