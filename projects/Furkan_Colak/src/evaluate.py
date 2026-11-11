"""Metrics, statistics, tables and figures from results/preds (evaluate stage)."""
import re

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from statsmodels.stats.contingency_tables import mcnemar

from src import plots, stats
from src.settings import RESULTS

PREDS = RESULTS / "preds"
META = ["id", "source_id", "generator", "mode", "template", "topic", "label", "len_ratio", "test"]
NAME = re.compile(r"(?P<exp>.+)__train-(?P<train>[A-C]+)__seed-(?P<seed>\d+)")
MAIN_DETECTORS = ["lexical_lr", "lexical_svm", "stylometry_xgb", "transformer", "llm_judge", "fast_detectgpt",
                  "binoculars", "coordinator"]
BOOT = stats.BOOT


def metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict:
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    two = len(set(y)) == 2
    return {
        "accuracy": accuracy_score(y, yhat), "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0), "f1": f1_score(y, yhat, zero_division=0),
        "macro_f1": f1_score(y, yhat, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y, p) if two else np.nan, "pr_auc": average_precision_score(y, p) if two else np.nan,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def load_preds() -> list[tuple[dict, pd.DataFrame]]:
    out = []
    for path in sorted(PREDS.glob("*.parquet")):
        m = NAME.match(path.stem)
        out.append(({"exp": m["exp"], "train": m["train"], "seed": int(m["seed"])}, pd.read_parquet(path)))
    return out


def all_metrics(preds) -> pd.DataFrame:
    rows = []
    for meta, d in preds:
        dets = [c for c in d.columns if c not in META]
        for test, g in d.groupby("test"):
            for det in dets:
                rows.append({**meta, "test": test, "detector": det, "n": len(g),
                             **metrics(g.label.to_numpy(), g[det].to_numpy())})
    return pd.DataFrame(rows)


def summarize(m: pd.DataFrame) -> pd.DataFrame:
    cols = ["accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc", "pr_auc"]
    g = m.groupby(["exp", "train", "test", "detector"])[cols]
    s = g.mean().add_suffix("_mean").join(g.std().add_suffix("_std")).reset_index()
    return s


def setting(row, train_gens: str = "ABC") -> str:
    if row.test not in train_gens:
        return "E4 external (unseen API model)"
    if row.train == row.test:
        return "E1 in-generator"
    if len(row.train) == 3:
        return "E1 pooled"
    if len(row.train) == 1:
        return "E2 cross-generator"
    return "E3 leave-one-out" if row.test not in row.train else "E1 partial-pool"


def fmt(mean: float, std: float) -> str:
    return f"{mean:.3f} ± {std:.3f}"


def bootstrap_h1(preds, detectors: list[str]) -> pd.DataFrame:
    """In-generator F1 minus cross-generator F1, resampling test sources (paired: same sources in both)."""
    def counts(g: pd.DataFrame, det: str, sources: np.ndarray) -> np.ndarray:
        """Per-source [tp, fp, fn], so a bootstrap sample is a weighted sum of rows."""
        yhat = g[det] >= 0.5
        c = pd.DataFrame({"s": g.source_id, "tp": yhat & (g.label == 1), "fp": yhat & (g.label == 0),
                          "fn": ~yhat & (g.label == 1)}).groupby("s").sum()
        return c.reindex(sources, fill_value=0).to_numpy(dtype=float)

    def f1(c: np.ndarray) -> np.ndarray:
        tp, fp, fn = c[..., 0], c[..., 1], c[..., 2]
        return np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)

    rows = []
    for meta, d in preds:
        if meta["exp"] != "main" or len(meta["train"]) != 1:
            continue
        x = meta["train"]
        rng = np.random.default_rng(meta["seed"])
        sources = np.sort(d.source_id.unique())
        w = np.stack([np.bincount(rng.integers(0, len(sources), len(sources)), minlength=len(sources))
                      for _ in range(BOOT)]).astype(float)
        for y in sorted(set(d.test) - {x}):
            for det in detectors:
                cx, cy = counts(d[d.test == x], det, sources), counts(d[d.test == y], det, sources)
                diffs = f1(w @ cx) - f1(w @ cy)
                in_f1, cross_f1 = float(f1(cx.sum(0))), float(f1(cy.sum(0)))
                lo, hi = np.percentile(diffs, [2.5, 97.5])
                p = min(1.0, 2 * (min((diffs <= 0).sum(), (diffs >= 0).sum()) + 1) / (BOOT + 1))
                rows.append({"train": x, "test": y, "seed": meta["seed"], "detector": det,
                             "f1_in": in_f1, "f1_cross": cross_f1, "diff": in_f1 - cross_f1,
                             "ci_low": lo, "ci_high": hi, "significant": lo > 0 or hi < 0, "p": p})
    return stats.corrected(pd.DataFrame(rows))


def kind(train: str, test: str, local: str) -> str:
    if test == train:
        return "in"
    return "cross" if test in local else test


def pooled_h1(preds, detectors: list[str], local: str) -> pd.DataFrame:
    """In-generator minus other-generator F1 per detector and target, pooled over runs.

    The per-run tests above form a family too large for Holm at this bootstrap resolution (smallest p is 1/2001);
    pooling the runs leaves one test per detector and target.
    """
    groups: dict[tuple, list] = {}
    for meta, d in preds:
        if meta["exp"] != "main" or len(meta["train"]) != 1:
            continue
        x = meta["train"]
        for y in sorted(set(d.test) - {x}):
            for det in detectors:
                groups.setdefault((det, kind(x, y, local)), []).append((d[d.test == x], det, d[d.test == y], det))
    rows = [{"detector": det, "target": target, "n_runs": len(v), **stats.paired_diff(v, "f1", seed=i)}
            for i, ((det, target), v) in enumerate(sorted(groups.items()))]
    return stats.corrected(pd.DataFrame(rows))


def f1_intervals(preds, detectors: list[str], local: str) -> pd.DataFrame:
    """Source-level 95% interval of the F1 in each setting, single-generator trainings pooled over runs."""
    groups: dict[tuple, list] = {}
    for meta, d in preds:
        if meta["exp"] != "main" or len(meta["train"]) != 1:
            continue
        for test, g in d.groupby("test"):
            for det in detectors:
                groups.setdefault((det, kind(meta["train"], test, local)), []).append((g, det))
    return pd.DataFrame([{"detector": det, "setting": s, "n_runs": len(v), **stats.interval(v, "f1", seed=i)}
                         for i, ((det, s), v) in enumerate(sorted(groups.items()))])


def mcnemar_tests(preds, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for meta, d in preds:
        if meta["exp"] != "main":
            continue
        for test, g in d.groupby("test"):
            for a, b in pairs:
                ca, cb = (g[a] >= 0.5) == g.label, (g[b] >= 0.5) == g.label
                table = [[(ca & cb).sum(), (ca & ~cb).sum()], [(~ca & cb).sum(), (~ca & ~cb).sum()]]
                rows.append({**meta, "test": test, "a": a, "b": b, "a_only": table[0][1], "b_only": table[1][0],
                             "p": mcnemar(table, exact=True).pvalue})
    # One family per detector pair: the number of test sets differs from the number of pairs of interest.
    return pd.concat([stats.corrected(g) for _, g in pd.DataFrame(rows).groupby(["a", "b"])], ignore_index=True)


def by_group(preds, col: str, local: str, detector: str = "coordinator") -> pd.DataFrame:
    """Recall on LLM texts by topic (or another column), main single-generator trainings.

    The unseen API generators are kept apart from the local cross-generator case: D alone would set the average.
    """
    rows = []
    for meta, d in preds:
        if meta["exp"] != "main" or len(meta["train"]) != 1:
            continue
        llm = d[d.label == 1]
        for (test, val), g in llm.groupby(["test", col]):
            rows.append({**meta, "setting": kind(meta["train"], test, local), col: val, "n": len(g),
                         "recall": (g[detector] >= 0.5).mean()})
    return pd.DataFrame(rows).groupby(["setting", col]).agg(n=("n", "sum"), recall=("recall", "mean"),
                                                            recall_std=("recall", "std")).reset_index()


def length_matched(preds, lo=0.8, hi=1.2) -> pd.DataFrame:
    rows = []
    for meta, d in preds:
        if meta["exp"] != "main" or len(meta["train"]) != 1:
            continue
        for test, g in d.groupby("test"):
            # Keep LLM texts close in length to their human source, and only those human sources (paired).
            chunk = g.id.str.split("|").str[0]
            llm = g[(g.label == 1) & g.len_ratio.between(lo, hi)]
            g = g[((g.label == 0) & chunk.isin(chunk[llm.index])) | g.index.isin(llm.index)]
            if g.label.nunique() < 2:
                continue
            for det in MAIN_DETECTORS:
                if det in g:
                    rows.append({**meta, "test": test, "detector": det, "n_llm": int(g.label.sum()),
                                 **metrics(g.label.to_numpy(), g[det].to_numpy())})
    return pd.DataFrame(rows)


def run(cfg: dict) -> None:
    preds = load_preds()
    local = "".join(cfg["train_generators"])
    m = all_metrics(preds)
    m.to_csv(RESULTS / "metrics_all.csv", index=False)
    s = summarize(m)
    s["setting"] = s.apply(setting, axis=1)
    s.to_csv(RESULTS / "summary.csv", index=False)

    main = s[(s.exp == "main") & s.detector.isin(MAIN_DETECTORS)].copy()
    for col in ["f1", "macro_f1", "roc_auc", "pr_auc", "precision", "recall", "accuracy"]:
        main[col] = [fmt(a, b) for a, b in zip(main[f"{col}_mean"], main[f"{col}_std"])]
    view = ["setting", "train", "test", "detector", "accuracy", "precision", "recall", "f1", "macro_f1",
            "roc_auc", "pr_auc"]
    main.sort_values(["setting", "train", "test", "detector"])[view].to_csv(RESULTS / "table_main.csv", index=False)

    dets = [d for d in MAIN_DETECTORS if any(d in x for _, x in preds)]
    h1 = bootstrap_h1(preds, dets)
    h1.to_csv(RESULTS / "h1_bootstrap.csv", index=False)
    pooled = pooled_h1(preds, dets, local)
    pooled.to_csv(RESULTS / "h1_pooled.csv", index=False)
    f1_intervals(preds, dets, local).to_csv(RESULTS / "f1_intervals.csv", index=False)
    mcnemar_tests(preds, [("coordinator", "lexical_lr"), ("transformer", "lexical_lr"),
                          ("coordinator", "transformer")]).to_csv(RESULTS / "mcnemar.csv", index=False)
    by_group(preds, "topic", local).to_csv(RESULTS / "recall_by_topic.csv", index=False)
    by_group(preds, "template", local).to_csv(RESULTS / "recall_by_template.csv", index=False)
    lm = length_matched(preds)
    # A length-matched subset can hold a handful of texts; the size goes into the file next to the scores.
    size = lm.groupby(["train", "test", "detector"]).n_llm.agg(n_llm="mean", n_runs="size").reset_index()
    summarize(lm.assign(exp="length-matched")).merge(size, on=["train", "test", "detector"]) \
        .to_csv(RESULTS / "length_matched.csv", index=False)

    ref = next(d for meta, d in preds if meta["exp"] == "main" and meta["train"] == "A"
               and meta["seed"] == cfg["seeds"][0])
    plots.cross_heatmaps(s[s.exp == "main"], ["lexical_lr", "stylometry_xgb", "transformer", "coordinator"])
    plots.curves(ref, "transformer", "train A")
    plots.confusions(ref, "coordinator")

    print(pooled.round(3).to_string(index=False))
    print(f"per-run in-vs-cross gaps with a 95% interval off zero: {h1.significant.sum()} / {len(h1)}, "
          f"after Benjamini-Hochberg: {(h1.p_bh < 0.05).sum()}")
