"""Rewriting robustness, prompt strategies, single vs. multi-agent generation and publisher shift (robust stage)."""
import re

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from src import plots, stats
from src.settings import DATA, RESULTS

PREDS = RESULTS / "preds"
PREDS_EXT = RESULTS / "preds_ext"
OUT = RESULTS / "robust"
NAME = re.compile(r"(?P<exp>[^_]+)__train-(?P<train>[A-C]+)__seed-(?P<seed>\d+)")
DETS = ["lexical_lr", "stylometry_xgb", "transformer", "coordinator", "fast_detectgpt", "binoculars"]
REWRITES = ["writer_only", "paraphrase_self", "paraphrase_cross", "paraphrase_twice", "backtranslate", "humanize"]
PROMPTS = ["prompt_style", "prompt_fewshot"]
MIXES = ["mix25", "mix50", "mix75"]


def setting(train: str, gen: str) -> str | None:
    if len(train) == 3:
        return "pooled"
    if len(train) == 1:
        return "same" if train == gen else "other"
    return None


def load(folder, exp: str) -> list[dict]:
    out = []
    for path in sorted(folder.glob(f"{exp}__*.parquet")):
        m = NAME.match(path.stem)
        out.append({"exp": m["exp"], "train": m["train"], "seed": int(m["seed"]), "d": pd.read_parquet(path)})
    return out


def score(g: pd.DataFrame, det: str) -> dict:
    y, p = g.label.to_numpy(), g[det].to_numpy()
    return {"recall": float((p[y == 1] >= 0.5).mean()), "f1": f1_score(y, p >= 0.5, zero_division=0),
            "roc_auc": roc_auc_score(y, p), "n_llm": int((y == 1).sum()), "n_human": int((y == 0).sum())}


def pairs_for(ext_runs: list[dict], main_runs: list[dict]):
    """Yields (run info, condition, generator, condition frame, baseline frame on the same excerpts)."""
    mains = {(r["train"], r["seed"]): r["d"] for r in main_runs}
    for r in ext_runs:
        for test, g in r["d"].groupby("test"):
            if not test.startswith("var:") or g.label.nunique() < 2:
                continue
            _, cond, gen = test.split(":")
            if cond == "multi_same":
                continue
            sett = setting(r["train"], gen)
            if sett is None:
                continue
            base = mains[(r["train"], r["seed"])]
            base = base[base.test == gen]
            chunks = g.id.str.split("|").str[0]
            base = base[base.id.str.split("|").str[0].isin(chunks)]
            yield r, cond, gen, sett, g, base


def robustness(ext_runs, main_runs) -> pd.DataFrame:
    rows = []
    for r, cond, gen, sett, g, base in pairs_for(ext_runs, main_runs):
        for det in DETS:
            if det not in g:
                continue
            a, b = score(g, det), score(base, det)
            rows.append({"train": r["train"], "seed": r["seed"], "condition": cond, "generator": gen, "setting": sett,
                         "detector": det, **a, "base_recall": b["recall"], "base_f1": b["f1"], "base_auc": b["roc_auc"]})
    return pd.DataFrame(rows)


def recall_tests(ext_runs, main_runs) -> pd.DataFrame:
    """Recall under each condition minus recall on the Writer + Editor text of the same excerpts."""
    groups: dict[tuple, list] = {}
    for r, cond, gen, sett, g, base in pairs_for(ext_runs, main_runs):
        if sett == "pooled":
            for det in DETS:
                if det in g:
                    groups.setdefault((cond, det), []).append((g, det, base, det))
    rows = [{"condition": c, "detector": det, **stats.paired_diff(v, "recall", seed=i)}
            for i, ((c, det), v) in enumerate(sorted(groups.items()))]
    return pd.concat([stats.corrected(p) for _, p in pd.DataFrame(rows).groupby("detector")], ignore_index=True)


def human_only(d: pd.DataFrame, det: str) -> float:
    return float((d[det].to_numpy() >= 0.5).mean())


def publishers(ext_runs: list[dict], main_runs: list[dict], aug_runs: list[dict]) -> pd.DataFrame:
    """False positives on human text and recall on LLM versions, per publisher, CISA test set as the reference."""
    rows = []
    for r in [x for x in ext_runs if len(x["train"]) == 3] + aug_runs:
        for test, g in r["d"].groupby("test"):
            if not test.startswith("ext:"):
                continue
            _, pub, gen = test.split(":")
            for det in DETS:
                if det not in g:
                    continue
                base = {"exp": r["exp"], "train": r["train"], "seed": r["seed"], "publisher": pub, "generator": gen,
                        "detector": det}
                if gen == "human":
                    rows.append({**base, "fpr": human_only(g, det), "n": len(g)})
                else:
                    rows.append({**base, **score(g, det)})
    cisa = [x for x in main_runs if len(x["train"]) == 3]
    for r in cisa:
        for gen in "ABC":
            g = r["d"][r["d"].test == gen]
            for det in DETS:
                if det in g:
                    rows.append({"exp": "main", "train": r["train"], "seed": r["seed"], "publisher": "cisa", "generator": gen,
                                 "detector": det, **score(g, det)})
        h = r["d"][(r["d"].test == "A") & (r["d"].label == 0)]
        for det in DETS:
            if det in h:
                rows.append({"exp": "main", "train": r["train"], "seed": r["seed"], "publisher": "cisa",
                             "generator": "human", "detector": det, "fpr": human_only(h, det), "n": len(h)})
    return pd.DataFrame(rows)


def fpr_intervals(ext_runs: list[dict], main_runs: list[dict]) -> pd.DataFrame:
    """FPR gap of each publisher over CISA, unpaired because they are different sources; Holm per detector."""
    rows = []
    pool = [x for x in ext_runs if len(x["train"]) == 3]
    cisa = {r["seed"]: r["d"][(r["d"].test == "A") & (r["d"].label == 0)]
            for r in main_runs if len(r["train"]) == 3}
    pubs = sorted({t.split(":")[1] for r in pool for t in r["d"].test.unique() if t.startswith("ext:")})
    for det in DETS:
        for i, pub in enumerate(pubs):
            runs = [(r["d"][r["d"].test == f"ext:{pub}:human"], det) for r in pool if det in r["d"]]
            if not runs:
                continue
            rows.append({"publisher": pub, "detector": det, **stats.interval(runs, "fpr", seed=i),
                         "cisa": np.mean([human_only(cisa[r["seed"]], det) for r in pool])})
    return pd.DataFrame(rows)


def fpr_by_pool(ext_runs: list[dict], main_runs: list[dict]) -> pd.DataFrame:
    """False positive rate on human text by the number of generators in training.

    Pooling generators moves the boundary towards human text: it recovers D and costs false alarms elsewhere.
    """
    rows = []
    for r in ext_runs:
        for test, g in r["d"].groupby("test"):
            if test.startswith("ext:") and test.endswith(":human"):
                rows += [{"k": len(r["train"]), "seed": r["seed"], "publisher": test.split(":")[1], "detector": det,
                          "fpr": human_only(g, det)} for det in DETS if det in g]
    for r in main_runs:
        h = r["d"][r["d"].label == 0].drop_duplicates("id")
        rows += [{"k": len(r["train"]), "seed": r["seed"], "publisher": "cisa", "detector": det,
                  "fpr": human_only(h, det)} for det in DETS if det in h]
    per = pd.DataFrame(rows).groupby(["seed", "k", "publisher", "detector"]).fpr.mean().reset_index()
    return per.groupby(["k", "publisher", "detector"]).fpr.agg(["mean", "std"]).reset_index()


def mix_position(ext_runs: list[dict]) -> pd.DataFrame:
    """Recall on partly generated texts by where the LLM block sits; RoBERTa reads only the first 256 tokens."""
    e = pd.read_parquet(DATA / "processed" / "extra.parquet")
    e = e[e["mode"].isin(MIXES)]
    ds = pd.read_parquet(DATA / "processed" / "dataset.parquet")
    human = ds[ds.label == 0].set_index("chunk_id").text
    # The build stage puts the block first or last at random and does not store which; the opening tells.
    first = pd.Series([not t.startswith(human[c][:40]) for t, c in zip(e.text, e.chunk_id)], index=e.id.to_numpy())
    rows = []
    for r in [x for x in ext_runs if len(x["train"]) == 3]:
        d = r["d"]
        d = d[d.test.str.startswith("var:mix") & (d.label == 1)]
        d = d.assign(condition=d.test.str.split(":").str[1], llm_first=d.id.map(first))
        for (cond, pos), g in d.groupby(["condition", "llm_first"]):
            rows += [{"seed": r["seed"], "condition": cond, "llm_first": bool(pos), "detector": det,
                      "recall": float((g[det] >= 0.5).mean()), "n": len(g)} for det in DETS if det in g]
    return pd.DataFrame(rows).groupby(["condition", "llm_first", "detector"]).agg(
        recall_mean=("recall", "mean"), recall_std=("recall", "std"), n=("n", "sum")).reset_index()


def trigrams(text: str) -> set:
    w = re.findall(r"\w+", text.lower())
    return set(zip(w, w[1:], w[2:]))


def source_overlap(ext_runs: list[dict], local: list[str]) -> pd.DataFrame:
    """Share of an LLM text's word trigrams found in its human excerpt, single- against multi-agent, same excerpts."""
    ds = pd.read_parquet(DATA / "processed" / "dataset.parquet")
    human = {c: trigrams(t) for c, t in zip(ds[ds.label == 0].chunk_id, ds[ds.label == 0].text)}
    llm = ds[(ds.label == 1) & ds.generator.isin(local) & ds.chunk_id.isin(ds[ds["mode"] == "single"].chunk_id)]
    grams = [trigrams(t) for t in llm.text]
    llm = llm.assign(overlap=[len(g & human[c]) / max(len(g), 1) for g, c in zip(grams, llm.chunk_id)])
    out = llm.groupby(["generator", "mode"]).overlap.agg(["mean", "std", "size"]).reset_index()
    tested = pd.concat([r["d"] for r in ext_runs])
    tested = tested[tested.test.str.startswith("var:single:") & (tested.label == 1)]
    n = tested.groupby(tested.test.str.split(":").str[2]).id.nunique()
    out["n_test"] = [int(n.get(g, 0)) if m == "single" else np.nan for g, m in zip(out.generator, out["mode"])]
    return out


def run(cfg: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    main_runs = load(PREDS, "main")
    ext_runs = load(PREDS_EXT, "main")
    aug_runs = [r for p in sorted({x.stem.split("__")[0] for x in PREDS_EXT.glob("aug-*")}) for r in load(PREDS_EXT, p)]

    long = robustness(ext_runs, main_runs)
    long.to_csv(OUT / "long.csv", index=False)
    key = ["setting", "condition", "detector"]
    per_seed = long.groupby(["seed"] + key)[["recall", "f1", "roc_auc", "base_recall", "base_f1", "base_auc"]].mean().reset_index()
    s = per_seed.groupby(key)[["recall", "f1", "roc_auc", "base_recall", "base_f1", "base_auc"]].agg(["mean", "std"])
    s.columns = ["_".join(c) for c in s.columns]
    s = s.reset_index()
    s.to_csv(OUT / "summary.csv", index=False)
    tests = recall_tests(ext_runs, main_runs)
    tests.to_csv(OUT / "recall_tests.csv", index=False)
    plots.robustness(s[s.setting == "pooled"], REWRITES + PROMPTS + ["single"], [d for d in DETS if d != "stylometry_xgb"])
    plots.mix_curve(s[s.setting == "pooled"], MIXES, [d for d in DETS if d != "stylometry_xgb"])

    pub = publishers(ext_runs, main_runs, aug_runs)
    pub.to_csv(OUT / "publishers_long.csv", index=False)
    per = pub.groupby(["seed", "exp", "publisher", "generator", "detector"])[["fpr", "recall", "f1", "roc_auc"]].mean().reset_index()
    p = per.groupby(["exp", "publisher", "generator", "detector"])[["fpr", "recall", "f1", "roc_auc"]].agg(["mean", "std"])
    p.columns = ["_".join(c) for c in p.columns]
    p.reset_index().to_csv(OUT / "publishers.csv", index=False)
    fpr_intervals(ext_runs, main_runs).to_csv(OUT / "publisher_fpr.csv", index=False)
    plots.publisher_shift(p.reset_index(), [d for d in DETS if d != "stylometry_xgb"])
    fpr_by_pool(ext_runs, main_runs).to_csv(OUT / "fpr_by_pool.csv", index=False)
    mix_position(ext_runs).to_csv(OUT / "mix_position.csv", index=False)
    source_overlap(ext_runs, cfg["train_generators"]).to_csv(OUT / "source_overlap.csv", index=False)

    # Rewrite-augmented training against the ABC baseline, on the conditions and generators it was meant to help.
    aug = [r for r in aug_runs if r["exp"] == "aug-rewrite"]
    if aug:
        rows = []
        for r in aug:
            for test, g in r["d"].groupby("test"):
                if test.startswith("val:") or test.startswith("ext:") or g.label.nunique() < 2:
                    continue
                for det in DETS:
                    if det in g:
                        rows.append({"exp": "aug-rewrite", "seed": r["seed"], "test": test, "detector": det, **score(g, det)})
        for r in [x for x in ext_runs + main_runs if len(x["train"]) == 3]:
            for test, g in r["d"].groupby("test"):
                if test.startswith("val:") or test.startswith("ext:") or g.label.nunique() < 2:
                    continue
                for det in DETS:
                    if det in g:
                        rows.append({"exp": "main", "seed": r["seed"], "test": test, "detector": det, **score(g, det)})
        a = pd.DataFrame(rows)
        a["group"] = [t.split(":")[1] if t.startswith("var:") else t for t in a.test]
        a = a.groupby(["seed", "exp", "group", "detector"])[["recall", "f1", "roc_auc"]].mean().reset_index()
        a = a.groupby(["exp", "group", "detector"])[["recall", "f1", "roc_auc"]].agg(["mean", "std"])
        a.columns = ["_".join(c) for c in a.columns]
        a.reset_index().to_csv(OUT / "augmented.csv", index=False)

    print(s[(s.setting == "pooled") & (s.detector == "transformer")][["condition", "recall_mean", "base_recall_mean", "roc_auc_mean"]]
          .round(3).to_string(index=False))
    print(p.reset_index().query("exp == 'main' and generator == 'human'")[["publisher", "detector", "fpr_mean"]].round(3).to_string(index=False))
