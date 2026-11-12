"""Probability calibration and decision thresholds (calibrate stage).

Calibrators and thresholds are fitted on validation frames only. Test frames, including D, E and the external
publishers, are only scored. The Coordinator was itself fitted on validation data, so its calibrators use
out-of-fold stacked scores from a source-grouped split of that data.
"""
import re

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold

from src import plots
from src.settings import RESULTS

PREDS = RESULTS / "preds"
PREDS_EXT = RESULTS / "preds_ext"
OUT = RESULTS / "calibration"
NAME = re.compile(r"main__train-(?P<train>[A-C]+)__seed-(?P<seed>\d+)")
SINGLE = ["lexical_lr", "lexical_svm", "stylometry_xgb", "transformer", "llm_judge", "fast_detectgpt", "binoculars"]
MEMBERS = ["lexical_lr", "lexical_svm", "stylometry_xgb", "transformer", "llm_judge"]
EPS = 1e-6
BINS = 15


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def ece(y: np.ndarray, p: np.ndarray, bins: int = BINS) -> float:
    """Expected calibration error with equal-width bins (Naeini et al., 2015)."""
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(y[which == b].mean() - p[which == b].mean()) * (which == b).mean()
                     for b in range(bins) if (which == b).any()))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def nll(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_platt(y, p):
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
    return lambda q: lr.predict_proba(logit(q).reshape(-1, 1))[:, 1]


def fit_isotonic(y, p):
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p, y)
    return lambda q: iso.predict(q)


def fit_temperature(y, p):
    z = logit(p)
    t = minimize_scalar(lambda T: nll(y, sigmoid(z / T)), bounds=(0.05, 50), method="bounded").x
    return lambda q: sigmoid(logit(q) / t)


CALIBRATORS = {"raw": None, "platt": fit_platt, "isotonic": fit_isotonic, "temperature": fit_temperature}


def thr_f1(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold with the best validation F1, at the midpoint of the widest tied plateau of scores."""
    order = np.argsort(-p)
    s, hits = p[order], y[order].astype(float)
    tp = np.cumsum(hits)
    f1 = 2 * tp / (np.arange(1, len(s) + 1) + y.sum())
    ends = np.append(np.flatnonzero(s[1:] != s[:-1]), len(s) - 1)  # a threshold cannot split tied scores
    best = np.flatnonzero(f1[ends] >= f1[ends].max() - 1e-12)
    # Among equally good cuts take the middle one: near-separable validation data allow a wide range of thresholds.
    i = ends[best[len(best) // 2]]
    return float((s[i] + s[i + 1]) / 2) if i + 1 < len(s) else float(s[i])


def thr_fpr(y: np.ndarray, p: np.ndarray, fpr: float) -> float:
    """Threshold from the human class alone: the (1 - fpr) quantile of validation human scores.

    It needs no LLM text, so it does not assume anything about the generator seen at test time.
    """
    return float(np.quantile(p[y == 0], 1 - fpr))


def thresholds(val_y, val_p) -> dict[str, float]:
    return {"fixed_0.5": 0.5, "val_f1": thr_f1(val_y, val_p), "human_fpr_1": thr_fpr(val_y, val_p, 0.01),
            "human_fpr_5": thr_fpr(val_y, val_p, 0.05)}


def stack_oof(val: pd.DataFrame, members: list[str], seed: int) -> np.ndarray:
    """Out-of-fold Coordinator scores on validation data, grouped by source."""
    X, y, groups = logit(val[members].to_numpy()), val.label.to_numpy(), val.source_id.to_numpy()
    p = np.zeros(len(val))
    for tr, te in GroupKFold(n_splits=5, shuffle=True, random_state=seed).split(X, y, groups):
        p[te] = LogisticRegression(max_iter=1000, random_state=seed).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def frame_kind(train: str, test: str) -> str:
    if test.startswith("ext:"):
        return "external"
    if test.startswith("var:"):
        return "variant"
    if test in ("D", "E"):
        return test
    return "in" if test in train and len(train) == 1 else ("pooled" if test in train else "cross")


def load_run(train: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    main = pd.read_parquet(PREDS / f"main__train-{train}__seed-{seed}.parquet")
    ext = pd.read_parquet(PREDS_EXT / f"main__train-{train}__seed-{seed}.parquet")
    return main, ext


def evaluate_one(train: str, seed: int, curves: list) -> list[dict]:
    main, ext = load_run(train, seed)
    vals = {t[4:]: g for t, g in ext.groupby("test") if t.startswith("val:")}
    val = pd.concat([vals[g] for g in train]).drop_duplicates("id")
    tests = pd.concat([main, ext[~ext.test.str.startswith("val:")]])
    rows = []
    detectors = [d for d in SINGLE + ["coordinator"] if d in tests]
    for det in detectors:
        y_val = val.label.to_numpy()
        # Coordinator: out-of-fold scores, because the stacker has already seen these validation rows.
        p_val = stack_oof(val, MEMBERS, seed) if det == "coordinator" else val[det].to_numpy()
        for method, make in CALIBRATORS.items():
            cal = make(y_val, p_val) if make else (lambda q: q)
            common = thresholds(y_val, cal(p_val))
            pseudo: dict[tuple, float] = {}
            for test, g in tests.groupby("test"):
                y, p = g.label.to_numpy(), cal(g[det].to_numpy())
                # Pseudo-unseen validation: generators absent from training, and never the one under test.
                others = tuple(k for k in vals if k not in train and k != test.split(":")[-1])
                th = dict(common)
                if others:
                    if others not in pseudo:
                        o = pd.concat([vals[k] for k in others]).drop_duplicates("id")
                        pseudo[others] = thr_f1(o.label.to_numpy(), cal(o[det].to_numpy()))
                    th["pseudo_unseen_f1"] = pseudo[others]
                base = {"train": train, "seed": seed, "test": test, "kind": frame_kind(train, test), "detector": det,
                        "calibrator": method, "n": len(g)}
                both = len(set(y)) == 2
                cal_row = {**base, "threshold": "none", "brier": brier(y, p), "ece": ece(y, p), "nll": nll(y, p)}
                rows.append(cal_row)
                for name, t in th.items():
                    yhat = p >= t
                    fp, tn = (yhat & (y == 0)).sum(), (~yhat & (y == 0)).sum()
                    rows.append({**base, "threshold": name, "thr": t,
                                 "precision": precision_score(y, yhat, zero_division=0) if both else np.nan,
                                 "recall": recall_score(y, yhat, zero_division=0) if both else np.nan,
                                 "f1": f1_score(y, yhat, zero_division=0) if both else np.nan,
                                 "fpr": fp / max(fp + tn, 1)})
                if method in ("raw", "platt") and det in ("transformer", "lexical_lr", "coordinator") and both:
                    curves.append({**base, "y": y, "p": p})
    return rows


def run(cfg: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, curves = [], []
    runs = [m.groupdict() for p in sorted(PREDS.glob("main__*.parquet")) if (m := NAME.match(p.stem))]
    for r in runs:
        rows += evaluate_one(r["train"], int(r["seed"]), curves)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "metrics.csv", index=False)
    plots.reliability(curves)
    plots.threshold_sweep(curves)
    plots.pr_curves(curves)

    key = ["kind", "detector", "calibrator", "threshold"]
    per_seed = res.groupby(["seed"] + key)[["brier", "ece", "nll", "precision", "recall", "f1", "fpr"]].mean().reset_index()
    s = per_seed.groupby(key)[["brier", "ece", "nll", "precision", "recall", "f1", "fpr"]].agg(["mean", "std"])
    s.columns = ["_".join(c) for c in s.columns]
    s.reset_index().to_csv(OUT / "summary.csv", index=False)
    print(s.reset_index().query("kind == 'D' and calibrator == 'raw' and threshold != 'none'")
          [["detector", "threshold", "f1_mean", "recall_mean", "fpr_mean"]].round(3).to_string(index=False))
