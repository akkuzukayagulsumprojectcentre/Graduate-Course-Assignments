"""Stylometric features and the Human vs LLM linguistic comparison (analyze stage)."""
import re
from collections import Counter

import numpy as np
import pandas as pd
import spacy
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

from src import plots
from src.settings import DATA, RESULTS

PROCESSED = DATA / "processed"
POS_TAGS = ["NOUN", "PROPN", "VERB", "AUX", "ADJ", "ADV", "PRON", "DET", "ADP", "NUM", "CCONJ", "SCONJ"]
PUNCT = {"comma": ",", "period": ".", "semicolon": ";", "colon": ":", "paren": "(", "quote": '"', "hyphen": "-"}
WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")

# Feature groups are used by the stylometry detector and by the feature-group ablation.
GROUPS = {
    "length": ["n_words", "n_sents", "sent_len_mean", "sent_len_std", "word_len_mean"],
    "lexical": ["ttr", "mtld", "hapax_ratio", "stopword_ratio", "digit_ratio", "upper_ratio"],
    "punctuation": [f"p_{k}" for k in PUNCT],
    "repetition": ["rep_unigram", "rep_bigram"],
    "pos": [f"pos_{t}" for t in POS_TAGS],
}
ALL = [f for g in GROUPS.values() for f in g]


def mtld(tokens: list[str], threshold: float = 0.72) -> float:
    """Measure of Textual Lexical Diversity (McCarthy & Jarvis, 2010), mean of both directions."""
    def one_pass(seq):
        factors, types, count = 0.0, set(), 0
        for tok in seq:
            count += 1
            types.add(tok)
            if len(types) / count <= threshold:
                factors += 1
                types, count = set(), 0
        if count:
            factors += (1 - len(types) / count) / (1 - threshold)
        return len(seq) / factors if factors else float(len(seq))
    if len(tokens) < 10:
        return float("nan")
    return (one_pass(tokens) + one_pass(tokens[::-1])) / 2


def text_features(doc) -> dict:
    words = [t.text.lower() for t in doc if WORD.fullmatch(t.text)]
    n = max(len(words), 1)
    sent_lens = [sum(1 for t in s if not t.is_punct) for s in doc.sents]
    counts = Counter(words)
    bigrams = Counter(zip(words, words[1:]))
    pos = Counter(t.pos_ for t in doc if not t.is_space)
    n_tok = max(sum(pos.values()), 1)
    raw = doc.text
    f = {
        "n_words": len(words),
        "n_sents": len(sent_lens),
        "sent_len_mean": float(np.mean(sent_lens)) if sent_lens else 0.0,
        "sent_len_std": float(np.std(sent_lens)) if sent_lens else 0.0,
        "word_len_mean": float(np.mean([len(w) for w in words])) if words else 0.0,
        "ttr": len(counts) / n,
        "mtld": mtld(words),
        "hapax_ratio": sum(1 for c in counts.values() if c == 1) / n,
        "stopword_ratio": sum(t.is_stop for t in doc) / n_tok,
        "digit_ratio": sum(t.like_num for t in doc) / n_tok,
        "upper_ratio": sum(t.text.isupper() and len(t.text) > 1 for t in doc) / n_tok,
        "rep_unigram": 1 - len(counts) / n,
        "rep_bigram": sum(c - 1 for c in bigrams.values() if c > 1) / max(len(words) - 1, 1),
    }
    for k, ch in PUNCT.items():
        f[f"p_{k}"] = 100 * raw.count(ch) / n
    for t in POS_TAGS:
        f[f"pos_{t}"] = pos.get(t, 0) / n_tok
    return f


def extract(texts: list[str], batch_size: int = 64) -> pd.DataFrame:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    return pd.DataFrame([text_features(d) for d in nlp.pipe(texts, batch_size=batch_size)])


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    u = mannwhitneyu(x, y).statistic
    return 2 * u / (len(x) * len(y)) - 1


def compare(feats: pd.DataFrame) -> pd.DataFrame:
    """Each generator against human, per feature: Mann-Whitney U, Cliff's delta, BH-corrected p."""
    rows = []
    human = feats[feats.generator == "human"]
    for gen in sorted(set(feats.generator) - {"human"}):
        llm = feats[feats.generator == gen]
        for f in ALL:
            x, y = llm[f].dropna().to_numpy(), human[f].dropna().to_numpy()
            rows.append({"generator": gen, "feature": f, "llm_median": np.median(x),
                         "human_median": np.median(y), "cliffs_delta": cliffs_delta(x, y),
                         "p": mannwhitneyu(x, y).pvalue})
    out = pd.DataFrame(rows)
    ok = out.p.notna()
    out.loc[ok, "p_bh"] = multipletests(out.p[ok], method="fdr_bh")[1]
    return out


def log_odds(a: list[str], b: list[str], top: int = 25, ngram: tuple = (1, 2)) -> pd.DataFrame:
    """Weighted log-odds with an informative Dirichlet prior (Monroe et al., 2008); positive = more in a."""
    def grams(texts):
        c = Counter()
        for t in texts:
            w = [x.lower() for x in WORD.findall(t)]
            for n in range(ngram[0], ngram[1] + 1):
                c.update(" ".join(w[i:i + n]) for i in range(len(w) - n + 1))
        return c
    ca, cb = grams(a), grams(b)
    prior = ca + cb
    na, nb, a0 = sum(ca.values()), sum(cb.values()), sum(prior.values())
    rows = []
    for w, pw in prior.items():
        if pw < 10:
            continue
        ya, yb = ca[w], cb[w]
        la = np.log((ya + pw) / (na + a0 - ya - pw))
        lb = np.log((yb + pw) / (nb + a0 - yb - pw))
        var = 1 / (ya + pw) + 1 / (yb + pw)
        rows.append({"ngram": w, "z": (la - lb) / np.sqrt(var), "count_a": ya, "count_b": yb})
    df = pd.DataFrame(rows).sort_values("z")
    return pd.concat([df.tail(top).iloc[::-1].assign(side="a"), df.head(top).assign(side="b")])


def publisher_tests(feats: pd.DataFrame, efeats: pd.DataFrame) -> None:
    """Human excerpts of each other publisher against CISA, to size the institution shift."""
    human = feats[feats.generator == "human"]
    rows = []
    for pub, g in efeats[(efeats.group == "external") & (efeats.label == 0)].groupby("publisher"):
        for f in ALL:
            x, y = g[f].dropna().to_numpy(), human[f].dropna().to_numpy()
            rows.append({"publisher": pub, "feature": f, "pub_median": np.median(x), "cisa_median": np.median(y),
                         "cliffs_delta": cliffs_delta(x, y), "p": mannwhitneyu(x, y).pvalue})
    if rows:
        out = pd.DataFrame(rows)
        out["p_bh"] = multipletests(out.p, method="fdr_bh")[1]
        out.to_csv(RESULTS / "publisher_tests.csv", index=False)


def run(cfg: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    df = pd.read_parquet(PROCESSED / "dataset.parquet")
    feats = pd.concat([df[["id", "generator", "mode", "label"]].reset_index(drop=True),
                       extract(df.text.tolist())], axis=1)
    feats.to_parquet(PROCESSED / "features.parquet", index=False)

    multi = feats[feats["mode"].isin(["human", "multi"])]
    tests = compare(multi)
    tests.to_csv(RESULTS / "linguistic_tests.csv", index=False)

    texts = df[df["mode"].isin(["human", "multi"])]
    ngrams = [log_odds(texts[texts.generator == g].text.tolist(), texts[texts.generator == "human"].text.tolist())
              .assign(generator=g) for g in sorted(set(texts.generator) - {"human"})]
    pd.concat(ngrams).to_csv(RESULTS / "top_ngrams.csv", index=False)

    plots.feature_boxes(multi, ["n_words", "sent_len_mean", "mtld", "p_comma", "rep_bigram", "stopword_ratio"])
    plots.pos_bars(multi, POS_TAGS)

    extra = pd.read_parquet(PROCESSED / "extra.parquet")
    efeats = pd.concat([extra[["id", "generator", "mode", "label", "publisher", "group"]].reset_index(drop=True),
                        extract(extra.text.tolist())], axis=1)
    efeats.to_parquet(PROCESSED / "extra_features.parquet", index=False)
    publisher_tests(feats, efeats)

    strong = tests[(tests.p_bh < 0.05) & (tests.cliffs_delta.abs() >= 0.474)]
    print(f"{len(strong)} generator-feature pairs with large effect (|delta| >= 0.474, BH p < 0.05)")
    print(strong.sort_values("cliffs_delta", key=abs, ascending=False).head(15).to_string(index=False))
