"""Source-level bootstrap and multiple-comparison helpers shared by the analysis stages."""
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

BOOT = 2000
ALPHA = 0.05


def counts(d: pd.DataFrame, det: str, sources: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """Per-source [tp, fp, fn, tn], so that a bootstrap sample is a weighted sum of rows."""
    yhat = d[det].to_numpy() >= thr
    y = d.label.to_numpy() == 1
    c = pd.DataFrame({"s": d.source_id.to_numpy(), "tp": yhat & y, "fp": yhat & ~y, "fn": ~yhat & y, "tn": ~yhat & ~y})
    return c.groupby("s").sum().reindex(sources, fill_value=0).to_numpy(dtype=float)


def metric(c: np.ndarray, name: str) -> np.ndarray:
    tp, fp, fn, tn = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    num, den = {"f1": (2 * tp, 2 * tp + fp + fn), "recall": (tp, tp + fn), "precision": (tp, tp + fp),
                "fpr": (fp, fp + tn)}[name]
    return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)


def paired_diff(runs: list[tuple[pd.DataFrame, str, pd.DataFrame, str]], name: str = "f1", seed: int = 0) -> dict:
    """Bootstrap of metric(a) - metric(b), averaged over runs; each run resamples its own test sources.

    A run is (frame_a, detector_a, frame_b, detector_b). Sharing one resample between a and b keeps it paired.
    """
    rng = np.random.default_rng(seed)
    diffs, point = [], []
    for a, det_a, b, det_b in runs:
        sources = np.sort(pd.unique(np.concatenate([a.source_id.to_numpy(), b.source_id.to_numpy()])))
        ca, cb = counts(a, det_a, sources), counts(b, det_b, sources)
        point.append(metric(ca.sum(0), name) - metric(cb.sum(0), name))
        w = np.stack([np.bincount(rng.integers(0, len(sources), len(sources)), minlength=len(sources))
                      for _ in range(BOOT)]).astype(float)
        diffs.append(metric(w @ ca, name) - metric(w @ cb, name))
    mean = np.nanmean(np.stack(diffs), axis=0)
    lo, hi = np.nanpercentile(mean, [2.5, 97.5])
    # Two-sided percentile p-value, with +1 so it is never reported as exactly zero.
    p = min(1.0, 2 * (min((mean <= 0).sum(), (mean >= 0).sum()) + 1) / (len(mean) + 1))
    return {"diff": float(np.nanmean(point)), "ci_low": float(lo), "ci_high": float(hi), "p": float(p)}


def interval(runs: list[tuple[pd.DataFrame, str]], name: str = "f1", seed: int = 0) -> dict:
    """Percentile 95% interval of a metric averaged over runs, resampling test sources."""
    rng = np.random.default_rng(seed)
    samples, point = [], []
    for d, det in runs:
        sources = np.sort(d.source_id.unique())
        c = counts(d, det, sources)
        point.append(metric(c.sum(0), name))
        w = np.stack([np.bincount(rng.integers(0, len(sources), len(sources)), minlength=len(sources))
                      for _ in range(BOOT)]).astype(float)
        samples.append(metric(w @ c, name))
    mean = np.nanmean(np.stack(samples), axis=0)
    lo, hi = np.nanpercentile(mean, [2.5, 97.5])
    return {"value": float(np.nanmean(point)), "ci_low": float(lo), "ci_high": float(hi)}


def corrected(df: pd.DataFrame, col: str = "p") -> pd.DataFrame:
    """Holm (family-wise) and Benjamini-Hochberg (false discovery) adjusted p-values for one family of tests."""
    ok = df[col].notna()
    out = df.copy()
    out["p_holm"], out["p_bh"] = np.nan, np.nan
    if ok.any():
        out.loc[ok, "p_holm"] = multipletests(out.loc[ok, col], alpha=ALPHA, method="holm")[1]
        out.loc[ok, "p_bh"] = multipletests(out.loc[ok, col], alpha=ALPHA, method="fdr_bh")[1]
    return out
