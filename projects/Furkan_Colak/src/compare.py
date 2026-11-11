"""Coordinator against the best single detector, and variants of the Coordinator (compare stage)."""
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from src import plots, stats
from src.calibration import logit, sigmoid
from src.settings import RESULTS

PREDS = RESULTS / "preds"
PREDS_EXT = RESULTS / "preds_ext"
OUT = RESULTS / "compare"
NAME = re.compile(r"main__train-(?P<train>[A-C]+)__seed-(?P<seed>\d+)")
MEMBERS = ["lexical_lr", "lexical_svm", "stylometry_xgb", "transformer", "llm_judge"]
ZS = ["fast_detectgpt", "binoculars"]
SINGLES = MEMBERS + ZS


def stack(val: pd.DataFrame, cols: list[str], seed: int) -> LogisticRegression:
    return LogisticRegression(max_iter=1000, random_state=seed).fit(logit(val[cols].to_numpy()), val.label)


def scenario(train: str, test: str) -> str | None:
    if test in ("D", "E"):
        return f"{test}, train single" if len(train) == 1 else (f"{test}, train ABC" if len(train) == 3 else None)
    if test.startswith("ext:"):
        parts = test.split(":")
        if parts[2] == "human":
            return None
        return f"{parts[1]}, train ABC" if len(train) == 3 else None
    if test.startswith("var:"):
        cond = test.split(":")[1]
        return f"{cond}, train ABC" if len(train) == 3 else None
    if test in train:
        return "in-generator" if len(train) == 1 else None
    return "cross, train single" if len(train) == 1 else "leave-one-out"


def add_variants(frames: pd.DataFrame, vals: dict, train: str, seed: int) -> pd.DataFrame:
    """Scores for each Coordinator variant; anything fitted here sees validation frames only."""
    val = pd.concat([vals[g] for g in train]).drop_duplicates("id")
    out = frames.copy()
    out["coord_mean"] = sigmoid(logit(out[MEMBERS].to_numpy()).mean(axis=1))
    out["coord_zero_shot"] = stack(val, MEMBERS + ZS, seed).predict_proba(logit(out[MEMBERS + ZS].to_numpy()))[:, 1]
    # Best single member by validation F1, so the choice is made without looking at any test set.
    f1s = {m: f1_score(val.label, val[m] >= 0.5) for m in MEMBERS}
    best = max(f1s, key=f1s.get)
    out["best_on_val"] = out[best]
    # Stacker fitted on generators absent from training, never on the generator under test.
    out["coord_unseen"] = np.nan
    if len(train) < 3:
        for test, idx in out.groupby("test").groups.items():
            target = test.split(":")[-1]
            others = [g for g in vals if g not in train and g != target]
            if others:
                o = pd.concat([vals[g] for g in others]).drop_duplicates("id")
                x = logit(out.loc[idx, MEMBERS].to_numpy())
                out.loc[idx, "coord_unseen"] = stack(o, MEMBERS, seed).predict_proba(x)[:, 1]
    return out.assign(best_name=best)


def load_runs() -> list[dict]:
    runs = []
    for path in sorted(PREDS.glob("main__*.parquet")):
        m = NAME.match(path.stem)
        train, seed = m["train"], int(m["seed"])
        main = pd.read_parquet(path)
        ext = pd.read_parquet(PREDS_EXT / path.name)
        vals = {t[4:]: g for t, g in ext.groupby("test") if t.startswith("val:")}
        tests = pd.concat([main, ext[~ext.test.str.startswith("val:")]], ignore_index=True)
        runs.append({"train": train, "seed": seed, "frames": add_variants(tests, vals, train, seed)})
    return runs


def scores_table(runs: list[dict]) -> pd.DataFrame:
    dets = SINGLES + ["coordinator", "coord_mean", "coord_zero_shot", "best_on_val", "coord_unseen"]
    rows = []
    for r in runs:
        for test, g in r["frames"].groupby("test"):
            sc = scenario(r["train"], test)
            if sc is None or g.label.nunique() < 2:
                continue
            for det in dets:
                if g[det].isna().any():
                    continue
                y, p = g.label.to_numpy(), g[det].to_numpy()
                rows.append({"train": r["train"], "seed": r["seed"], "test": test, "scenario": sc, "detector": det,
                             "f1": f1_score(y, p >= 0.5, zero_division=0), "roc_auc": roc_auc_score(y, p),
                             "pr_auc": average_precision_score(y, p), "best_name": g.best_name.iloc[0]})
    return pd.DataFrame(rows)


def gaps(runs: list[dict], pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Paired F1 gap per scenario with a bootstrap interval; one Holm family per detector pair."""
    groups: dict[str, list] = {}
    for r in runs:
        for test, g in r["frames"].groupby("test"):
            sc = scenario(r["train"], test)
            if sc is not None and g.label.nunique() == 2:
                groups.setdefault(sc, []).append(g)
    rows = []
    for a, b in pairs:
        part = []
        for i, (sc, frames) in enumerate(sorted(groups.items())):
            usable = [(g, a, g, b) for g in frames if not g[a].isna().any()]
            if usable:
                part.append({"scenario": sc, "a": a, "b": b, "n_runs": len(usable),
                             **stats.paired_diff(usable, "f1", seed=i)})
        rows.append(stats.corrected(pd.DataFrame(part)))
    return pd.concat(rows, ignore_index=True)


def weights() -> pd.DataFrame:
    w = pd.read_csv(RESULTS / "coord_weights.csv")
    w = w[w.exp == "main"]
    return w.groupby(["train", "member"]).coef.agg(["mean", "std"]).reset_index()


def member_spread(runs: list[dict]) -> pd.DataFrame:
    """Member logits on D against the training generators: which member moves the stacked score most under shift."""
    rows = []
    for r in runs:
        d = r["frames"]
        if len(r["train"]) != 1:
            continue
        d = d[d.label == 1]
        coef = pd.read_csv(RESULTS / "coord_weights.csv").query(
            "exp == 'main' and train == @r['train'] and seed == @r['seed']").set_index("member").coef
        for m in MEMBERS:
            on_d = logit(d[d.test == "D"][m].to_numpy()).mean()
            on_tr = logit(d[d.test == r["train"]][m].to_numpy()).mean()
            rows.append({"train": r["train"], "seed": r["seed"], "member": m, "logit_train": on_tr, "logit_D": on_d,
                         "weighted_drop": coef[m] * (on_tr - on_d)})
    return pd.DataFrame(rows)


def run(cfg: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    s = scores_table(runs)
    s.to_csv(OUT / "scores.csv", index=False)
    per_seed = s.groupby(["seed", "scenario", "detector"])[["f1", "roc_auc", "pr_auc"]].mean().reset_index()
    summ = per_seed.groupby(["scenario", "detector"])[["f1", "roc_auc", "pr_auc"]].agg(["mean", "std"])
    summ.columns = ["_".join(c) for c in summ.columns]
    summ = summ.reset_index()
    summ.to_csv(OUT / "summary.csv", index=False)

    g = gaps(runs, [("coordinator", "transformer"), ("coordinator", "best_on_val"), ("coord_mean", "transformer"),
                    ("coord_unseen", "transformer"), ("coord_zero_shot", "coordinator"),
                    ("transformer", "fast_detectgpt"), ("transformer", "binoculars")])
    g.to_csv(OUT / "gaps.csv", index=False)
    weights().to_csv(OUT / "weights.csv", index=False)
    spread = member_spread(runs)
    spread.groupby("member")[["logit_train", "logit_D", "weighted_drop"]].mean().reset_index() \
        .to_csv(OUT / "member_spread.csv", index=False)
    s[["train", "seed", "best_name"]].drop_duplicates().to_csv(OUT / "best_on_val.csv", index=False)
    plots.coordinator_gap(g[(g.a == "coordinator") & (g.b == "transformer")])
    print(g[g.a == "coordinator"].round(3).to_string(index=False))
