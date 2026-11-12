"""Collect human CTI from publishers other than CISA, used only as an independent test set.

The text is copyrighted, so excerpts stay out of the repository; sources.csv and chunk hashes are committed.
"""
import hashlib
import json
import random
import re
import time
from datetime import datetime

import pandas as pd
from bs4 import BeautifulSoup

from src.collect import FOOTNOTE, FetchError, chunk, fetch, is_narrative, topic
from src.settings import DATA

OUT = DATA / "external"
LOC = re.compile(r"<loc>([^<]+)</loc>")
CTI = re.compile(
    r"threat|attack|ransom|malware|campaign|phish|backdoor|exploit|vulnerab|botnet|trojan|\bapt\d*\b|espionage|"
    r"nobelium|hafnium|actor|intrusion|compromise|hack|worm|implant|cobalt|lazarus|fin\d+|unc\d+|dev-\d+|"
    r"credential|supply.chain|zero.day|cve",
    re.I,
)
NOT_CTI = re.compile(
    r"flare.?on|flare-|challenge|webinar|podcast|announc|introducing|customer|best.practice|how.to|gartner|forrester|"
    r"\bleader\b|\bsummit\b|\bconference\b|rsa-|black.hat|zero.trust|\bcareer|women|hiring|compliance|gdpr|"
    r"defender-for|sentinel-|purview|\bsecurity.unlocked\b|\bpartner\b|year.in.review|m-trends|predictions|"
    r"threat.model",
    re.I,
)
# Mandiant's threat-intelligence feed also carries tooling and technique posts, and its tags do not tell them apart.
TOOLING = re.compile(
    r"\bcapa\b|flare|\bida\b|ghidra|speakeasy|fakenet|red.team|script series|emulat|forensic|for dummies|"
    r"hardware hacking|tips and tricks|\bvm\b|python|bytecode|telemetry|ontology|machine learning|obfuscation and detection|"
    r"configuring|\bpart (one|two|three|four)\b|approach to",
    re.I,
)
# Notes added after publication and calls to action; several of these are dated after the cutoff.
POST_NOTES = re.compile(
    r"\b(202[3-9]) update\b|new threat actor naming|to learn more|bookmark the security blog|follow us on|"
    r"written by:|stop attacks, reduce risk|subscribe|for more information|this blog (post )?was (updated|originally)|"
    r"\(updated|learn more about|sign up|contact us|read the (full|complete) report",
    re.I,
)
MONTH_DATE = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|November|December) (\d{1,2}), (\d{4})")


def microsoft_urls(index: str) -> list[str]:
    urls = []
    for sm in LOC.findall(fetch(index)):
        if "post-sitemap" in sm:
            urls += LOC.findall(fetch(sm))
    return [u for u in urls if re.search(r"/blog/\d{4}/\d{2}/\d{2}/", u)]


def fetch_retry(url: str, tries: int = 3, delay: float = 0.1) -> str | None:
    """Google's date-range sitemaps time out now and then; one of them must not stop the whole collection."""
    for attempt in range(tries):
        try:
            return fetch(url, delay=delay)
        except FetchError:
            time.sleep(5 * (attempt + 1))
    return None


def mandiant_urls(index: str) -> list[str]:
    urls, missed = [], []
    for sm in LOC.findall(fetch(index, delay=0.1)):
        if "/cloudblog/en/" in sm:
            xml = fetch_retry(sm)
            if xml is None:
                missed.append(sm)
                continue
            urls += [u for u in LOC.findall(xml) if "/blog/topics/threat-intelligence/" in u]
    print(f"[mandiant] {len(missed)} sitemaps could not be read", flush=True)
    (OUT / "missed_sitemaps.txt").write_text("\n".join(missed), encoding="utf-8")
    return sorted(set(urls))


def microsoft_date(url: str, soup: BeautifulSoup) -> str | None:
    m = re.search(r"/blog/(\d{4})/(\d{2})/(\d{2})/", url)
    return "-".join(m.groups()) if m else None


def mandiant_date(url: str, soup: BeautifulSoup) -> str | None:
    """Posts moved from mandiant.com keep their original date only in the visible byline, not in the metadata."""
    found = MONTH_DATE.search(str(soup))
    if found:
        return datetime.strptime(" ".join(found.groups()), "%B %d %Y").date().isoformat()
    return None


def microsoft_report(doc: dict, url: str) -> bool:
    """The blog labels its own posts; product news and best-practice posts are not threat reports."""
    return {"content-type/research", "topic/threat-intelligence"} <= doc["labels"] and bool(CTI.search(doc["title"]))


def mandiant_report(doc: dict, url: str) -> bool:
    """Every URL sits under /threat-intelligence/, so only the title and slug say what the post is about."""
    slug = url.rstrip("/").split("/")[-1]
    return bool(CTI.search(doc["title"] + " " + slug)) and not TOOLING.search(doc["title"])


PUBLISHERS = {
    "microsoft": {"urls": microsoft_urls, "date": microsoft_date, "body": ".entry-content", "keep": microsoft_report},
    "mandiant": {"urls": mandiant_urls, "date": mandiant_date, "body": "main article, article, main", "keep": mandiant_report},
}
LABEL = re.compile(r"/security/blog/((?:content-type|topic)/[^/]+)/")


def parse(html: str, url: str, pub: dict) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.find("h1")
    released = pub["date"](url, soup)
    labels = {m.group(1) for a in soup.find_all("a", href=True) if (m := LABEL.search(a["href"]))}
    body = soup.select_one(pub["body"]) or soup
    for tag in body.find_all(["table", "pre", "code"]):
        tag.decompose()
    paras = [" ".join(FOOTNOTE.sub(" ", p.get_text(" ", strip=True)).split()) for p in body.find_all("p")]
    return {"title": title.get_text(strip=True) if title else "", "released": released, "labels": labels,
            "paragraphs": [p for p in paras if is_narrative(p) and not POST_NOTES.search(p)]}


def run(cfg: dict) -> None:
    c, e = cfg["collect"], cfg["external"]
    lo, hi = c["chunk_words"]
    OUT.mkdir(parents=True, exist_ok=True)
    sources, chunks, blocked = [], [], []
    for name, spec in e["publishers"].items():
        pub = PUBLISHERS[name]
        urls = pub["urls"](spec["index"])
        cands = [u for u in urls if not NOT_CTI.search(u) and (name == "mandiant" or CTI.search(u))]
        print(f"[{name}] {len(urls)} urls, {len(cands)} candidates", flush=True)
        kept = []
        for url in cands:
            # Microsoft URLs carry the date, so later posts need no download.
            if name == "microsoft" and pub["date"](url, None) > c["cutoff_date"]:
                continue
            try:
                doc = parse(fetch(url), url, pub)
            except FetchError as err:
                print(f"skip {url}: {err}")
                continue
            except OSError:
                blocked.append(url)  # Windows Defender blocks pages with embedded samples; left as is.
                continue
            if not doc["released"] or doc["released"] > c["cutoff_date"]:
                continue
            if NOT_CTI.search(doc["title"]) or not pub["keep"](doc, url):
                continue
            parts = chunk(doc["paragraphs"], lo, hi, c["max_chunks_per_source"])
            if parts:
                slug = url.rstrip("/").split("/")[-1][:80]
                kept.append({"source_id": f"{name}-{slug}", "publisher": name, "url": url, "title": doc["title"],
                             "released": doc["released"], "parts": parts})
        random.Random(cfg["seed"]).shuffle(kept)
        print(f"[{name}] {len(kept)} usable sources before sampling", flush=True)
        for s in kept[: e["sources_per_publisher"]]:
            parts = s.pop("parts")
            s.update(topic=topic(s["title"] + " " + " ".join(parts)), n_chunks=len(parts))
            sources.append(s)
            chunks += [{"source_id": s["source_id"], "chunk_id": f"{s['source_id']}#{i}", "publisher": name, "text": t}
                       for i, t in enumerate(parts)]
    df = pd.DataFrame(sources).sort_values("source_id")
    df.to_csv(OUT / "sources.csv", index=False)
    with open(OUT / "human_chunks.jsonl", "w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")
    pd.DataFrame([{"chunk_id": ch["chunk_id"], "sha1": hashlib.sha1(ch["text"].encode()).hexdigest()} for ch in chunks]) \
        .to_csv(OUT / "chunk_hashes.csv", index=False)
    (OUT / "blocked_by_antivirus.txt").write_text("\n".join(blocked), encoding="utf-8")
    print(df.groupby("publisher").agg(n=("source_id", "size"), chunks=("n_chunks", "sum"),
                                      first=("released", "min"), last=("released", "max")).to_string())
    print(f"{len(blocked)} pages blocked by antivirus")
