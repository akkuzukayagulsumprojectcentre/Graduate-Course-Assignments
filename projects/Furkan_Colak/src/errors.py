"""Error analysis: what misclassified CTI texts share, why the unseen large model is missed and vendor text flagged."""
import json
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

from src import plots
from src.agents import read_jsonl
from src.settings import DATA, RESULTS

PROCESSED = DATA / "processed"
PREDS = RESULTS / "preds"
PREDS_EXT = RESULTS / "preds_ext"
OUT = RESULTS / "errors"
NAME = re.compile(r"main__train-(?P<train>[A-C]+)__seed-(?P<seed>\d+)")
DETS = ["lexical_lr", "transformer", "coordinator"]
FIRST = re.compile(r"\b(we|our|us|ours)\b", re.I)
AGENCY = re.compile(r"\b(CISA|FBI|NSA|DHS|US-CERT|NCCIC|MS-ISAC|NCSC|ACSC|CCCS)\b")
ADVICE = re.compile(r"\b(should|recommend\w*|advis\w*|mitigat\w*|encourag\w*|urge\w*)\b", re.I)
# Identifiers a reader could check against the source: file names, CVEs, IPs, hex values, ports, ATT&CK ids.
TECH = re.compile(r"\b[\w-]+\.(?:exe|dll|zip|docx?|xlsx?|pdf|ps1|bat|js|vbs|ini|dat|tmp)\b|\bCVE-\d{4}-\d+\b|"
                  r"\b\d{1,3}(?:\.\d{1,3}){3}\b|\b0x[0-9a-fA-F]+\b|\bport \d+\b|\bT\d{4}(?:\.\d{3})?\b", re.I)
FEATS = ["n_words", "mtld", "ttr", "pos_PROPN", "pos_PRON"]
PROFILE = FEATS + ["first_person", "first_rate", "agency", "advice", "invented", "cues_llm", "cues_human"]
EFFECTS = {"n_words": "words (log)", "mtld": "MTLD", "pos_PROPN": "proper nouns"}
FP_EFFECTS = {"first_rate": "first person per 100 words", "mtld": "MTLD", "pos_PROPN": "proper nouns",
              "n_words": "words (log)"}
FP_PANELS = {f: FP_EFFECTS[f] for f in ["first_rate", "mtld", "pos_PROPN"]}


def missed(scores: pd.Series) -> pd.Series:
    return scores < 0.5


def flagged(scores: pd.Series) -> pd.Series:
    return scores >= 0.5


def invented(text: str, facts: str | None) -> int:
    """Technical identifiers in the text that the Analyst's facts do not contain, a rough measure of fact drift."""
    if not facts:
        return 0
    return sum(m.lower() not in facts.lower() for m in set(TECH.findall(text)))


def cue_rates(text: pd.Series) -> pd.DataFrame:
    """Occurrences per 100 words of the word n-grams that Lexical-LR weights most, on its LLM and human side."""
    terms = pd.read_csv(RESULTS / "lr_top_terms.csv")
    terms = terms[terms.term.str.startswith("word:")]
    low, n = text.str.lower(), text.str.split().str.len().clip(lower=1)
    out = {}
    for side, part in terms.groupby("side"):
        pats = [re.compile(r"\b" + re.escape(t[5:]) + r"\b") for t in part.term]
        out[f"cues_{side}"] = [100 * sum(len(p.findall(t)) for p in pats) / k for t, k in zip(low, n)]
    return pd.DataFrame(out, index=text.index)


def describe(df: pd.DataFrame, facts: dict) -> pd.DataFrame:
    """Adds the text properties used by the categories, the profile table and the error models."""
    df = df.copy()
    hits = df.text.map(lambda t: len(FIRST.findall(t)))
    df["first_person"] = (hits > 0).astype(float)
    df["first_rate"] = 100 * hits / df.text.str.split().str.len().clip(lower=1)
    df["agency"] = df.text.map(lambda t: bool(AGENCY.search(t))).astype(float)
    df["advice"] = df.text.map(lambda t: len(ADVICE.findall(t)))
    df["invented"] = [invented(t, facts.get(c)) if g != "human" else 0
                      for t, c, g in zip(df.text, df.chunk_id, df.generator)]
    return df.join(cue_rates(df.text))


def corpus() -> pd.DataFrame:
    cols = ["id", "chunk_id", "source_id", "text", "generator", "mode", "topic", "temperature", "len_ratio"]
    df = pd.read_parquet(PROCESSED / "dataset.parquet")[cols]
    df = df.merge(pd.read_parquet(PROCESSED / "features.parquet")[["id"] + FEATS], on="id")
    return describe(df, {r["chunk_id"]: r["output"] for r in read_jsonl(DATA / "generated" / "facts.jsonl")})


def vendor() -> pd.DataFrame:
    """Human excerpts of the other publishers, with the same properties as the corpus."""
    e = pd.read_parquet(PROCESSED / "extra.parquet")
    e = e[(e.group == "external") & (e.label == 0)]
    e = e[["id", "chunk_id", "source_id", "text", "generator", "mode", "topic", "publisher"]]
    return describe(e.merge(pd.read_parquet(PROCESSED / "extra_features.parquet")[["id"] + FEATS], on="id"), {})


def predictions(texts: pd.DataFrame, folder=PREDS) -> pd.DataFrame:
    rows = []
    for path in sorted(folder.glob("main__*.parquet")):
        m = NAME.match(path.stem)
        rows.append(pd.read_parquet(path).assign(train=m["train"], seed=int(m["seed"])))
    keep = ["id", "train", "seed", "test", "label", "template"] + DETS
    return pd.concat(rows)[keep].merge(texts.drop(columns=["generator", "mode"]), on="id")


def fn_label(r) -> str:
    """Mutually exclusive categories of missed LLM text; the first rule that applies wins."""
    if r.first_person:
        return "first person, still missed"
    if r.agency:
        return "cites an agency"
    if r.invented > 0:
        return "unsupported technical detail"
    if r.len_ratio >= 0.8:
        return "impersonal, human length"
    return "impersonal, short"


def fp_label(r) -> str:
    """Mutually exclusive categories of human text flagged as LLM."""
    if r.first_person:
        return "first person"
    if r.advice >= 4:
        return "advice-heavy"
    if r.pos_PROPN < 0.08:
        return "few proper nouns"
    if r.topic == "vulnerability":
        return "product vulnerability notice"
    return "other"


def shares(part: pd.DataFrame, det: str, wrong, label, **tags) -> list[dict]:
    """Category counts among the errors, with the share of each category among all texts of the scope.

    A category explains errors only if it is more common among them than among the texts as a whole.
    """
    cat = part.apply(label, axis=1)
    bad = cat[wrong(part[det])]
    return [{**tags, "detector": det, "category": k, "n": int((bad == k).sum()),
             "share": float((bad == k).mean()) if len(bad) else 0.0, "base": float(v),
             "rate": float(wrong(part[det])[cat == k].mean())}
            for k, v in cat.value_counts(normalize=True).items()]


def categories(single: pd.DataFrame, v: pd.DataFrame) -> pd.DataFrame:
    llm = single[single.label == 1]
    # A human text is tested once per test generator; count each text once per training run and seed.
    human = single[single.label == 0].drop_duplicates(["id", "train", "seed"])
    scopes = [("FN", "D", llm[llm.test == "D"], missed, fn_label),
              ("FN", "other generators", llm[llm.test != "D"], missed, fn_label),
              ("FP", "CISA", human, flagged, fp_label), ("FP", "vendors", v, flagged, fp_label)]
    return pd.DataFrame([row for det in DETS for error, scope, part, wrong, label in scopes
                         for row in shares(part, det, wrong, label, error=error, scope=scope)])


def contrast(d: pd.DataFrame, feats: list[str], wrong) -> pd.DataFrame:
    """Wrongly against correctly classified texts, per feature: Mann-Whitney U, Cliff's delta, Benjamini-Hochberg."""
    rows = []
    for det in DETS:
        # Texts are averaged over runs first, so a text missed by every run is not counted nine times.
        g = d.groupby("id").agg(err=(det, lambda x: wrong(x).mean()), **{f: (f, "first") for f in feats})
        bad, good = g[g.err >= 0.5], g[g.err < 0.5]
        for f in feats:
            if len(bad) and len(good):
                u = mannwhitneyu(bad[f], good[f])
                rows.append({"detector": det, "feature": f, "n_wrong": len(bad), "n_right": len(good),
                             "wrong_mean": bad[f].mean(), "right_mean": good[f].mean(),
                             "cliffs_delta": 2 * u.statistic / (len(bad) * len(good)) - 1, "p": u.pvalue})
    out = pd.DataFrame(rows)
    out["p_bh"] = multipletests(out.p, method="fdr_bh")[1]
    return out


def error_model(d: pd.DataFrame, effects: dict, wrong) -> pd.DataFrame:
    """Logistic model of an error on standardized text properties, standard errors clustered by source."""
    rows = []
    d = d.copy()
    d["n_words"] = np.log(d.n_words)
    for f in effects:
        d[f] = (d[f] - d[f].mean()) / d[f].std()
    for det in DETS:
        y = wrong(d[det]).astype(int)
        if y.nunique() < 2:
            continue
        res = sm.GLM(y, sm.add_constant(d[list(effects)]), family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(d.source_id)[0]})
        ci = res.conf_int()
        for f in effects:
            rows.append({"detector": det, "feature": f, "odds_ratio": float(np.exp(res.params[f])),
                         "or_low": float(np.exp(ci.loc[f, 0])), "or_high": float(np.exp(ci.loc[f, 1])),
                         "p": float(res.pvalues[f]), "n": len(d), "n_errors": int(y.sum())})
    out = pd.DataFrame(rows)
    out["p_bh"] = multipletests(out.p, method="fdr_bh")[1]
    return out


def rate_by_bin(part: pd.DataFrame, feats, wrong, group: str) -> list[dict]:
    """Error rate by tertile of a feature; ranks break ties, so every tertile holds a third of the rows."""
    rows = []
    for f in feats:
        bins = pd.qcut(part[f].rank(method="first"), 3, labels=["low", "mid", "high"])
        for det in DETS:
            m = part.assign(b=bins, err=wrong(part[det])).groupby(["seed", "b"], observed=True).err.mean()
            for b, g in m.groupby(level="b", observed=True):
                rows.append({"group": group, "feature": f, "detector": det, "bin": b, "rate": g.mean(), "std": g.std()})
    return rows


def d_breakdown(d: pd.DataFrame) -> pd.DataFrame:
    """Miss rate on D by Writer template, sampling temperature and topic: mean and std over seeds."""
    rows = []
    for col in ["template", "temperature", "topic"]:
        n = d.groupby(col).id.nunique()
        for det in DETS:
            m = d.assign(err=missed(d[det])).groupby(["seed", col]).err.mean().groupby(level=col).agg(["mean", "std"])
            rows += [{"by": col, "value": k, "detector": det, "miss": r["mean"], "std": r["std"], "n": int(n[k])}
                     for k, r in m.iterrows()]
    return pd.DataFrame(rows)


def examples(single: pd.DataFrame, per_cat: int = 2) -> pd.DataFrame:
    """Confident errors with the most runs in agreement. Corpus texts here are CISA (public domain) or LLM output."""
    rows = []
    for det in DETS:
        wrong = single[(single[det] >= 0.5).astype(int) != single.label].copy()
        wrong["cat"] = [fn_label(r) if r.label == 1 else fp_label(r) for r in wrong.itertuples()]
        g = wrong.groupby(["label", "cat", "id"]).agg(runs=("id", "size"), score=(det, "mean"), text=("text", "first"),
                                                      gen=("test", "first"), words=("n_words", "first")).reset_index()
        for _, part in g.groupby(["label", "cat"]):
            for r in part.sort_values("runs", ascending=False).head(per_cat).itertuples():
                rows.append({"detector": det, "label": r.label, "category": r.cat, "id": r.id, "runs": r.runs,
                             "generator": r.gen if r.label == 1 else "human", "words": int(r.words),
                             "mean_score": r.score, "excerpt": " ".join(r.text.split()[:60])})
    return pd.DataFrame(rows)


def run(cfg: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    texts, ext = corpus(), vendor()
    multi = texts[texts["mode"].isin(["human", "multi"])]
    prof = pd.concat([multi.groupby("generator")[PROFILE].mean(), ext.groupby("publisher")[PROFILE].mean()])
    prof.rename_axis("generator").reset_index().to_csv(OUT / "profile_by_generator.csv", index=False)

    p = predictions(texts)
    single = p[p.train.str.len() == 1]
    llm = single[single.label == 1]
    d, rest = llm[llm.test == "D"], llm[llm.test != "D"]
    # Vendor false alarms come from the pooled models, the ones that recover D.
    v = predictions(ext, PREDS_EXT)
    v = v[v.test.str.endswith(":human") & (v.train.str.len() == 3)]

    categories(single, v).to_csv(OUT / "categories.csv", index=False)
    contrast(d, PROFILE, missed).to_csv(OUT / "fn_vs_tp.csv", index=False)
    contrast(v, [f for f in PROFILE if f != "invented"], flagged).to_csv(OUT / "fp_vs_tn.csv", index=False)
    error_model(d, EFFECTS, missed).to_csv(OUT / "error_model.csv", index=False)
    error_model(v, FP_EFFECTS, flagged).to_csv(OUT / "fp_model.csv", index=False)
    bins = pd.DataFrame(rate_by_bin(d, EFFECTS, missed, "D") + rate_by_bin(rest, EFFECTS, missed, "A-C, E")
                        + rate_by_bin(v, FP_PANELS, flagged, "vendors"))
    bins.to_csv(OUT / "rate_by_bin.csv", index=False)
    d_breakdown(d).to_csv(OUT / "d_breakdown.csv", index=False)
    examples(single).to_csv(OUT / "examples.csv", index=False)
    plots.error_bins(bins, EFFECTS, FP_PANELS)
    plots.score_ecdf(p[p.train == "A"], ["lexical_lr", "transformer"])
    (OUT / "codebook.json").write_text(json.dumps({"tech_pattern": TECH.pattern}, indent=1), encoding="utf-8")
    print(prof.round(3).to_string())
    print(pd.read_csv(OUT / "categories.csv").round(3).to_string(index=False))
    print(pd.read_csv(OUT / "error_model.csv").round(3).to_string(index=False))
    print(pd.read_csv(OUT / "fp_model.csv").round(3).to_string(index=False))
