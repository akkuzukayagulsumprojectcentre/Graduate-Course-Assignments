"""Collect human-written CTI excerpts from CISA (public domain) and split them into chunks."""
import hashlib
import json
import random
import re
import subprocess
import time
from datetime import date, timedelta

import pandas as pd
from bs4 import BeautifulSoup

from src.settings import CACHE as CACHE_ROOT, DATA

SITEMAP = "https://www.cisa.gov/default/sitemap.xml"
UA = "Mozilla/5.0 (MSc course project; text collection for research)"
CACHE = CACHE_ROOT / "html"
RAW = DATA / "raw"

URL_KINDS = {
    "advisory": re.compile(r"/cybersecurity-advisories/aa\d{2}-"),
    "analysis": re.compile(r"/analysis-reports/(ar|mar)\d{2}-"),
    "alert": re.compile(r"/alerts/(201[5-9]|202[0-2])/"),
}

# Short vendor-patch notices and catalog updates carry no threat narrative.
SKIP_TITLES = re.compile(
    r"releases? security update|add(s|ed)? .*known exploited|security updates? for|"
    r"releases? .*advisor(y|ies)|end of life|webinar|reminder",
    re.I,
)

BOILERPLATE = re.compile(
    r'provided "as is"|TLP:|Download the PDF|downloadable copy|Disclosure is not limited|'
    r"report incidents|Contact Information|does not endorse|@|https?://|www\.",
    re.I,
)
MIN_PARA_WORDS = 20
FOOTNOTE = re.compile(r"\s*\[\s*\d+(\s*[-,]\s*\d+)*\s*\]")

TOPICS = [
    ("ransomware", r"ransom"),
    ("phishing", r"phish|spearphish|business email compromise"),
    ("apt", r"advanced persistent|state-sponsored|apt\d+|nation-state|hidden cobra|lazarus"),
    ("vulnerability", r"cve-\d{4}-\d+|vulnerabilit|exploit"),
    ("malware", r"malware|trojan|backdoor|botnet|implant|loader"),
]


def fetch(url: str, delay: float = 1.0) -> str:
    # cisa.gov rejects Python HTTP clients by TLS fingerprint, so downloads go through curl.
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / (hashlib.sha1(url.encode()).hexdigest() + ".html")
    if path.exists():
        return path.read_text(encoding="utf-8")
    r = subprocess.run(
        ["curl", "-s", "-L", "--fail", "--max-time", "60", "-A", UA, url],
        capture_output=True,
    )
    if r.returncode != 0:
        raise FetchError(f"curl exit {r.returncode}")
    text = r.stdout.decode("utf-8", errors="replace")
    path.write_text(text, encoding="utf-8")
    time.sleep(delay)
    return text


class FetchError(RuntimeError):
    pass


def candidate_urls() -> list[tuple[str, str]]:
    xml = fetch(SITEMAP)
    urls = re.findall(r"<loc>([^<]+)</loc>", xml)
    out = []
    for u in urls:
        for kind, pat in URL_KINDS.items():
            if pat.search(u):
                out.append((kind, u))
                break
    return out


def release_date(url: str) -> str | None:
    """Pages only show 'Last Revised'; the original date is encoded in the URL or alert code."""
    m = re.search(r"/alerts/(\d{4})/(\d{2})/(\d{2})/", url)
    if m:
        return "-".join(m.groups())
    m = re.search(r"/(?:aa|ar|mar)(\d{2})-(\d{3})", url)
    if m:
        d = date(2000 + int(m.group(1)), 1, 1) + timedelta(days=int(m.group(2)) - 1)
        return d.isoformat()
    return None


def is_narrative(text: str) -> bool:
    words = text.split()
    if len(words) < MIN_PARA_WORDS or BOILERPLATE.search(text):
        return False
    alpha = sum(w.isalpha() for w in words) / len(words)
    return alpha > 0.7


def parse(html: str, url: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.find("h1")
    updated = soup.select_one(".c-field--name-field-last-updated time")
    main = soup.select_one(".l-full__main") or soup
    for tag in main.find_all(["table", "pre", "code"]):
        tag.decompose()
    # Only <p>: list items read as fragments and would give the detector a formatting cue.
    paras = [" ".join(p.get_text(" ", strip=True).split()) for p in main.find_all("p")]
    # Footnote markers like "[ 1 ]" only occur in the human pages and would be a free cue for detectors.
    paras = [" ".join(FOOTNOTE.sub(" ", p).split()) for p in paras]
    return {
        "title": title.get_text(strip=True) if title else "",
        "released": release_date(url),
        "updated": updated["datetime"][:10] if updated else None,
        "paragraphs": [p for p in paras if is_narrative(p)],
    }


def chunk(paragraphs: list[str], lo: int, hi: int, max_chunks: int) -> list[str]:
    """Join consecutive paragraphs into excerpts of lo..hi words; a paragraph longer than hi is skipped."""
    chunks, cur = [], []
    for p in paragraphs + [None]:
        n = sum(len(x.split()) for x in cur)
        if p is None or n + len(p.split()) > hi:
            if lo <= n <= hi:
                chunks.append(" ".join(cur))
            cur = []
        if p is not None and len(p.split()) <= hi:
            cur.append(p)
    return chunks[:max_chunks]


def topic(text: str) -> str:
    low = text.lower()
    for name, pat in TOPICS:
        if re.search(pat, low):
            return name
    return "other"


def run(cfg: dict) -> None:
    c = cfg["collect"]
    lo, hi = c["chunk_words"]
    rng = random.Random(cfg["seed"])
    RAW.mkdir(parents=True, exist_ok=True)

    sources, chunks, blocked = [], [], []
    for kind, url in candidate_urls():
        try:
            doc = parse(fetch(url), url)
        except FetchError as e:
            print(f"skip {url}: {e}")
            continue
        except OSError:
            # Some malware analysis pages embed samples; Windows Defender blocks reading them.
            blocked.append(url)
            continue
        if not doc["released"] or doc["released"] > c["cutoff_date"]:
            continue
        # A revision after the cutoff could carry post-ChatGPT edits.
        if doc["updated"] and doc["updated"] > c["cutoff_date"]:
            continue
        if SKIP_TITLES.search(doc["title"]):
            continue
        parts = chunk(doc["paragraphs"], lo, hi, c["max_chunks_per_source"])
        if not parts:
            continue
        sources.append({
            "slug": url.rstrip("/").split("/")[-1], "origin": "cisa", "kind": kind, "url": url,
            "title": doc["title"], "released": doc["released"], "updated": doc["updated"],
            "topic": topic(doc["title"] + " " + " ".join(parts)), "n_chunks": len(parts), "parts": parts,
        })

    # Alerts can reuse a slug on different dates; only those get the date appended.
    seen = pd.Series([s["slug"] for s in sources]).value_counts()
    for s in sources:
        s["source_id"] = "cisa-" + s["slug"] + ("-" + s["released"].replace("-", "") if seen[s["slug"]] > 1 else "")
        for i, text in enumerate(s.pop("parts")):
            chunks.append({"source_id": s["source_id"], "chunk_id": f"{s['source_id']}#{i}", "text": text})
        del s["slug"]

    df = pd.DataFrame(sources)
    # Keep every advisory and analysis report, fill the rest with alerts.
    core = df[df.kind != "alert"]
    alerts = df[df.kind == "alert"]
    n_alerts = max(0, c["target_sources"] - len(core))
    picked = alerts.sample(n=min(n_alerts, len(alerts)), random_state=rng.randint(0, 2**31))
    df = pd.concat([core, picked]).sort_values("source_id")
    keep = set(df.source_id)

    df.to_csv(DATA / "sources.csv", index=False)
    with open(RAW / "human_chunks.jsonl", "w", encoding="utf-8") as f:
        for ch in chunks:
            if ch["source_id"] in keep:
                f.write(json.dumps(ch, ensure_ascii=False) + "\n")
    (DATA / "blocked_by_antivirus.txt").write_text("\n".join(blocked), encoding="utf-8")
    print(f"{len(df)} sources, {sum(ch['source_id'] in keep for ch in chunks)} chunks, "
          f"{len(blocked)} pages blocked by antivirus")
    print(df.groupby(["kind", "topic"]).size().to_string())
