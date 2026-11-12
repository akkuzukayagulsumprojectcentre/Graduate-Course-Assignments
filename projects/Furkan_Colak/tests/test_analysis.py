import numpy as np
import pandas as pd
import pytest
import torch

from src.calibration import brier, ece, fit_isotonic, fit_platt, fit_temperature, frame_kind, nll, stack_oof, thr_f1, thr_fpr
from src.dataset import mix
from src.errors import DETS, flagged, fn_label, fp_label, invented, missed, rate_by_bin, shares
from src.evaluate import kind
from src.experiments import subset
from src.robust import trigrams
from src.stats import corrected, counts, metric, paired_diff
from src.zeroshot import curvature


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    return y, np.clip(0.5 + (y - 0.5) * 0.5 + rng.normal(0, 0.2, n), 0.01, 0.99)


def test_ece_is_zero_when_perfectly_calibrated_and_large_when_not():
    y = np.array([0] * 50 + [1] * 50)
    assert ece(y, np.full(100, 0.5)) == pytest.approx(0)
    assert ece(y, y.astype(float) * 0 + 0.99) == pytest.approx(0.49)


def test_brier_and_nll_prefer_better_scores():
    y, p = _toy()
    assert brier(y, y.astype(float) * 0.98 + 0.01) < brier(y, p)
    assert nll(y, y.astype(float) * 0.98 + 0.01) < nll(y, p)


@pytest.mark.parametrize("make", [fit_platt, fit_temperature, fit_isotonic])
def test_calibrators_improve_nll_of_overconfident_scores(make):
    y, p = _toy()
    over = np.clip(np.where(p > 0.5, p ** 0.1, p ** 6), 1e-4, 1 - 1e-4)
    cal = make(y[:200], over[:200])
    assert nll(y[200:], np.clip(cal(over[200:]), 1e-3, 1 - 1e-3)) < nll(y[200:], over[200:])


def test_platt_and_temperature_keep_ranking():
    y, p = _toy()
    for make in (fit_platt, fit_temperature):
        q = make(y, p)(p)
        assert (np.argsort(q) == np.argsort(p)).mean() > 0.99


def test_threshold_from_human_scores_ignores_llm_scores():
    y = np.array([0] * 100 + [1] * 3)
    p = np.concatenate([np.linspace(0, 0.5, 100), [0.01, 0.02, 0.03]])
    t = thr_fpr(y, p, 0.05)
    assert t == pytest.approx(np.quantile(p[y == 0], 0.95))
    assert (p[y == 0] >= t).mean() == pytest.approx(0.05, abs=0.011)


def test_threshold_f1_separates_a_separable_set_and_never_splits_ties():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    t = thr_f1(y, p)
    assert 0.3 < t <= 0.7
    tied = np.array([0.5] * 6)
    assert ((tied >= thr_f1(y, tied)) == (tied >= 0.5)).all()


def test_coordinator_oof_scores_never_use_their_own_source():
    # A source whose label is opposite to its feature: an in-sample fit would learn it, out-of-fold cannot.
    n = 60
    val = pd.DataFrame({"m": np.r_[np.full(n, 0.9), np.full(n, 0.1)], "label": np.r_[np.ones(n), np.zeros(n)].astype(int),
                        "source_id": np.arange(2 * n) % 10})
    p = stack_oof(val, ["m"], 0)
    assert p.shape == (2 * n,) and p[val.label == 1].mean() > p[val.label == 0].mean()


def test_frame_kinds():
    assert frame_kind("A", "A") == "in"
    assert frame_kind("A", "B") == "cross"
    assert frame_kind("ABC", "A") == "pooled"
    assert frame_kind("A", "D") == "D"
    assert frame_kind("ABC", "ext:microsoft:A") == "external"
    assert frame_kind("ABC", "var:humanize:B") == "variant"


def test_mix_share_and_both_sources_present():
    human = " ".join(f"Human sentence number {i} is here." for i in range(40))
    llm = " ".join(f"Generated line number {i} goes on." for i in range(40))
    text, share = mix(human, llm, 0.5, "k")
    assert "Human sentence" in text and "Generated line" in text
    assert share == pytest.approx(0.5, abs=0.1)
    assert mix(human, llm, 0.5, "k") == (text, share)


def test_paired_bootstrap_detects_a_real_gap_and_not_a_null_one():
    rng = np.random.default_rng(0)
    n = 200
    d = pd.DataFrame({"source_id": np.arange(n) // 2, "label": np.arange(n) % 2})
    d["good"] = np.where(d.label == 1, 0.9, 0.1)
    d["bad"] = rng.random(n)
    gap = paired_diff([(d, "good", d, "bad")], "f1")
    assert gap["diff"] > 0.3 and gap["ci_low"] > 0 and gap["p"] < 0.01
    same = paired_diff([(d, "good", d, "good")], "f1")
    assert same["diff"] == 0 and same["ci_low"] == 0 and same["ci_high"] == 0


def test_counts_and_metric():
    d = pd.DataFrame({"source_id": ["a", "a", "b", "b"], "label": [1, 0, 1, 0], "m": [0.9, 0.9, 0.1, 0.1]})
    c = counts(d, "m", np.array(["a", "b"]))
    assert c.tolist() == [[1, 1, 0, 0], [0, 0, 1, 1]]
    assert metric(c.sum(0), "f1") == pytest.approx(0.5)
    assert metric(c.sum(0), "fpr") == pytest.approx(0.5)


def test_holm_is_at_least_as_strict_as_bh():
    out = corrected(pd.DataFrame({"p": [0.001, 0.02, 0.04, 0.5]}))
    assert (out.p_holm >= out.p_bh - 1e-12).all()


def test_curvature_ignores_padding_and_ranks_likely_text_higher():
    logits = torch.zeros(1, 3, 4)
    logits[0, :, 2] = 5.0
    likely, unlikely = torch.tensor([[2, 2, 2]]), torch.tensor([[0, 0, 0]])
    mask = torch.ones(1, 3)
    assert curvature(logits, likely, mask) > curvature(logits, unlikely, mask)
    masked = torch.tensor([[1.0, 1.0, 0.0]])
    swapped = torch.tensor([[2, 2, 0]])
    assert curvature(logits, swapped, masked) == pytest.approx(curvature(logits, likely, masked))


def test_invented_counts_identifiers_missing_from_the_facts():
    assert invented('The dropper "Invoice_2024.zip" abused CVE-2021-44228.', '{"cves": ["CVE-2021-44228"]}') == 1
    assert invented("The actor sent phishing emails.", "{}") == 0


def test_error_categories_are_decided_in_order():
    row = lambda **k: pd.Series({"first_person": 0.0, "agency": 0.0, "invented": 0, "len_ratio": 1.0, "advice": 0,
                                 "pos_PROPN": 0.12, "topic": "apt", **k})
    assert fn_label(row(first_person=1.0, agency=1.0)) == "first person, still missed"
    assert fn_label(row(agency=1.0, invented=2)) == "cites an agency"
    assert fn_label(row(invented=2)) == "unsupported technical detail"
    assert fn_label(row()) == "impersonal, human length"
    assert fn_label(row(len_ratio=0.5)) == "impersonal, short"
    assert fp_label(row(advice=5)) == "advice-heavy"
    assert fp_label(row(pos_PROPN=0.05)) == "few proper nouns"


def test_category_shares_come_with_the_base_rate_and_the_error_rate():
    part = pd.DataFrame({"m": [0.9, 0.9, 0.1, 0.1], "c": ["x", "y", "y", "y"]})
    rows = {r["category"]: r for r in shares(part, "m", missed, lambda r: r.c, error="FN", scope="t")}
    assert rows["y"]["base"] == pytest.approx(0.75) and rows["y"]["share"] == pytest.approx(1.0)
    assert rows["y"]["rate"] == pytest.approx(2 / 3)
    assert rows["x"]["n"] == 0 and rows["x"]["rate"] == 0


def test_error_rate_by_tertile_splits_ties_into_equal_bins():
    scores = [0.1] * 6 + [0.9] * 3
    part = pd.DataFrame({"f": [0] * 6 + [1, 2, 3], "seed": 1, **{d: scores for d in DETS}})
    rates = {r["bin"]: r["rate"] for r in rate_by_bin(part, ["f"], flagged, "g") if r["detector"] == DETS[0]}
    assert rates == {"low": 0.0, "mid": 0.0, "high": 1.0}


def test_unseen_generators_are_kept_apart_from_local_cross_generator_tests():
    assert [kind("A", t, "ABC") for t in "ABDE"] == ["in", "cross", "D", "E"]


def test_trigrams_ignore_case_and_punctuation():
    assert trigrams("The actor sent, phishing emails.") == trigrams("the actor sent phishing emails")
    assert len(trigrams("one two")) == 0


def test_subset_keeps_only_paired_human_texts():
    df = pd.DataFrame({"source_id": ["s", "s", "s", "t"], "chunk_id": ["c1", "c1", "c2", "c3"],
                       "label": [0, 1, 0, 0], "generator": ["human", "A", "human", "human"],
                       "mode": ["human", "multi", "human", "human"]})
    out = subset(df, ["s"], ["A"])
    assert sorted(out.chunk_id) == ["c1", "c1"]
