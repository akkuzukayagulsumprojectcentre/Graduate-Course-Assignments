"""Merge human and generated texts, clean them and make source-level splits."""
import json
import random
import re
import unicodedata

import numpy as np
import pandas as pd

from src.dedup import dup_groups, purge
from src.settings import DATA

RAW = DATA / "raw" / "human_chunks.jsonl"
GEN = DATA / "generated"
PROCESSED = DATA / "processed"
EXTERNAL = DATA / "external"
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
PREAMBLE = re.compile(r"^\W*(sure|certainly|of course|here(['’]s| is| are)|below is)\b[^\n]*:\W*$", re.I | re.M)
# Memo headers (Subject:, To:, From: ...) are layout, not content: drop the whole line.
MEMO = re.compile(r"^\W*(subject|to|from|date|cc|classification|distribution|tlp|priority|severity)\W*:[^\n]*$",
                  re.I | re.M)
# Inline section labels ("**Summary:** Our team ...") keep their text, lose the label.
LABEL = re.compile(r"^\s*\*\*[^*\n]{1,40}:\*\*\s*", re.M)
HEADER = re.compile(r"^\s*(#{1,6}\s+.*|\*\*[^*\n]{1,200}\*\*:?|[A-Z][A-Za-z ]{2,60}:)\s*$", re.M)
BULLET = re.compile(r"^\s*([-*•]|\d+[.)])\s+", re.M)
TRAILER = re.compile(r"\n\s*(\(?word count|note:|\*?\(approx)[^\n]*$", re.I)
MD = re.compile(r"(\*\*|__|`)")

# Every human text comes from CISA, so agency names act as a label proxy. The masking ablation replaces them.
AGENCIES = re.compile(
    r"\b(CISA|FBI|NSA|DHS|NCCIC|US-CERT|MS-ISAC|CNMF|USCYBERCOM|DoD|Treasury|NCSC|ACSC|CCCS|"
    r"Cybersecurity and Infrastructure Security Agency|Federal Bureau of Investigation|"
    r"National Security Agency|Department of Homeland Security)\b"
)

# Typographic variants that some generators prefer (e.g. U+2011 non-breaking hyphen).
ASCII_MAP = str.maketrans({
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u00a0": " ", "\u202f": " ",
    "\u2026": "...",
})


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(ASCII_MAP)
    return " ".join(text.split())


def drop_title(text: str) -> str:
    """A short first line without end punctuation is a title, not prose."""
    first, _, rest = text.partition("\n")
    plain = MD.sub("", first).strip()
    if rest.strip() and len(plain.split()) <= 20 and not plain.endswith((".", "!", "?")):
        return rest
    return text


def clean(text: str) -> str:
    """Remove LLM wrapping (reasoning, preambles, titles, markdown, word-count notes); keep the prose."""
    text = THINK.sub("", text).strip()
    text = PREAMBLE.sub("", text).strip()
    text = MEMO.sub("", text).strip()
    text = LABEL.sub("", text)
    text = drop_title(text)
    text = TRAILER.sub("", text)
    text = HEADER.sub("", text)
    text = BULLET.sub("", text)
    text = MD.sub("", text)
    return normalize(text)


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load(sources: pd.DataFrame) -> pd.DataFrame:
    human = pd.DataFrame(read_jsonl(RAW)).assign(
        label=0, generator="human", mode="human", template="human", temperature=np.nan)
    parts = [human]
    for path in sorted(GEN.glob("gen_*.jsonl")):
        g = pd.DataFrame(read_jsonl(path))
        parts.append(g[["source_id", "chunk_id", "text", "generator", "mode", "template", "temperature"]]
                     .assign(label=1))
    df = pd.concat(parts, ignore_index=True)
    df = df.merge(sources[["source_id", "kind", "topic", "released"]], on="source_id", how="left")
    df["id"] = df.chunk_id + "|" + df.generator + "|" + df["mode"]
    return df


def sentences(text: str) -> list[str]:
    return SENT.split(text)


def mix(human: str, llm: str, share: float, key: str) -> tuple[str, float]:
    """A human excerpt with about `share` of its words replaced by sentences of its LLM version.

    The LLM block goes first or last at random, since RoBERTa only reads the first 256 tokens.
    """
    target = len(human.split())

    def take(sents: list[str], n: float) -> list[str]:
        out, words = [], 0
        for s in sents:
            if words >= n:
                break
            out.append(s)
            words += len(s.split())
        return out

    a, b = take(sentences(llm), share * target), take(sentences(human), (1 - share) * target)
    n_a, n_b = sum(len(s.split()) for s in a), sum(len(s.split()) for s in b)
    parts = a + b if random.Random(key).random() < 0.5 else b + a
    return " ".join(parts), n_a / max(n_a + n_b, 1)


def load_extra(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Texts for testing only: rewritten and partly generated versions of test sources, and other publishers."""
    splits = json.loads((PROCESSED / "splits.json").read_text(encoding="utf-8"))
    tests = set().union(*(set(s["test"]) for s in splits.values()))
    rows = []
    path = GEN / "variants.jsonl"
    if path.exists():
        v = pd.DataFrame(read_jsonl(path))
        rows.append(v[["source_id", "chunk_id", "text", "generator", "temperature"]].assign(
            mode=v.condition, template=v.condition, label=1, group="variant", publisher="cisa"))
    for g in cfg["train_generators"]:
        # The Writer's draft before the Editor pass is already on disk.
        w = pd.DataFrame(read_jsonl(GEN / f"gen_{g}.jsonl"))
        w = w[(w["mode"] == "multi") & w.source_id.isin(tests)]
        rows.append(pd.DataFrame({
            "source_id": w.source_id, "chunk_id": w.chunk_id, "generator": g, "temperature": w.temperature,
            "text": [m[0]["output"] for m in w.messages], "mode": "writer_only", "template": w.template,
            "label": 1, "group": "variant", "publisher": "cisa"}))
    if rows:
        extra = pd.concat(rows, ignore_index=True)
        extra["text"] = extra.text.map(clean)
    else:
        extra = pd.DataFrame()

    human = df[df.label == 0].set_index("chunk_id")
    base = df[(df["mode"] == "multi") & df.source_id.isin(tests) & df.generator.isin(cfg["train_generators"])]
    mixed = []
    for share in cfg["clean"]["mix_shares"]:
        for r in base.itertuples():
            text, real = mix(human.text[r.chunk_id], r.text, share, f"{r.id}|{share}")
            mixed.append({"source_id": r.source_id, "chunk_id": r.chunk_id, "generator": r.generator,
                          "temperature": r.temperature, "text": text, "mode": f"mix{int(share * 100)}",
                          "template": "mix", "label": 1, "group": "variant", "publisher": "cisa", "llm_share": real})
    extra = pd.concat([extra, pd.DataFrame(mixed)], ignore_index=True)

    raw = EXTERNAL / "human_chunks.jsonl"
    if raw.exists():
        h = pd.DataFrame(read_jsonl(raw))
        pub = dict(zip(h.chunk_id, h.publisher))
        parts = [h.assign(generator="human", mode="human", template="human", temperature=np.nan, label=0,
                          group="external")]
        for path in sorted(EXTERNAL.glob("gen_*.jsonl")):
            g = pd.DataFrame(read_jsonl(path))
            parts.append(g[["source_id", "chunk_id", "text", "generator", "mode", "template", "temperature"]].assign(
                label=1, group="external", publisher=g.chunk_id.map(pub)))
        e = pd.concat(parts, ignore_index=True)
        e["text"] = e.text.map(clean)
        extra = pd.concat([extra, e], ignore_index=True)
        human_len = e[e.label == 0].set_index("chunk_id").text.str.split().str.len()
    else:
        human_len = pd.Series(dtype=float)
    src = pd.concat([pd.read_csv(DATA / "sources.csv")[["source_id", "topic"]],
                     pd.read_csv(EXTERNAL / "sources.csv")[["source_id", "topic"]] if raw.exists() else None])
    extra = extra.merge(src, on="source_id", how="left")
    extra["id"] = extra.chunk_id + "|" + extra.generator + "|" + extra["mode"]
    extra["text_raw"] = extra.text
    extra["text_masked"] = extra.text.map(lambda t: AGENCIES.sub("AGENCY", t))
    extra["n_words"] = extra.text.str.split().str.len()
    extra = extra[extra.n_words >= 50].reset_index(drop=True)
    reference = pd.concat([human.n_words, human_len])
    extra["len_ratio"] = extra.n_words / extra.chunk_id.map(reference)
    return extra


def make_splits(source_ids: list[str], seeds: list[int], frac: dict) -> dict:
    """Split by source report, so no report contributes text to more than one split."""
    ids = np.array(sorted(set(source_ids)))
    out = {}
    for seed in seeds:
        perm = np.random.default_rng(seed).permutation(ids)
        n_test = round(len(ids) * frac["test"])
        n_val = round(len(ids) * frac["val"])
        out[str(seed)] = {
            "test": sorted(perm[:n_test].tolist()),
            "val": sorted(perm[n_test:n_test + n_val].tolist()),
            "train": sorted(perm[n_test + n_val:].tolist()),
        }
    return out


def run(cfg: dict) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    sources = pd.read_csv(DATA / "sources.csv")
    df = load(sources)
    df["text_raw"] = df.text.map(lambda t: " ".join(t.split()))  # for the no-cleaning ablation
    df["text"] = df.text.map(clean)
    df["text_masked"] = df.text.map(lambda t: AGENCIES.sub("AGENCY", t))
    df["n_words"] = df.text.str.split().str.len()
    df = df[df.n_words >= 50].reset_index(drop=True)

    human_len = df[df.label == 0].set_index("chunk_id").n_words
    df["len_ratio"] = df.n_words / df.chunk_id.map(human_len)

    df.to_parquet(PROCESSED / "dataset.parquet", index=False)
    # The plain source split is kept next to the purged one, so the leakage stage can show what purging removed.
    plain = make_splits(df.source_id.tolist(), cfg["seeds"], cfg["split"])
    (PROCESSED / "splits_unpurged.json").write_text(json.dumps(plain, indent=1), encoding="utf-8")
    groups = dup_groups(df[df.label == 0], cfg["leakage"]["shingle"], cfg["leakage"]["dup"])
    splits, log = purge(plain, groups)
    (PROCESSED / "splits.json").write_text(json.dumps(splits, indent=1), encoding="utf-8")
    pd.DataFrame(log).to_csv(PROCESSED / "purged.csv", index=False)
    print(f"purged: {len(log)} source moves or removals over all seeds (data/processed/purged.csv)")

    print(df.groupby(["generator", "mode"]).agg(n=("id", "size"), words=("n_words", "median")).to_string())
    s = splits[str(cfg["seeds"][0])]
    print({k: len(v) for k, v in s.items()}, "sources per split (first seed)")

    extra = load_extra(df, cfg)
    extra.to_parquet(PROCESSED / "extra.parquet", index=False)
    print(extra.groupby(["group", "publisher", "mode", "generator"]).agg(n=("id", "size"), words=("n_words", "median"))
          .to_string())
