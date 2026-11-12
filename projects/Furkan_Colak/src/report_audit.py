"""Tables and macros of the leakage checks and of the joint error and calibration view on D and E (report stage)."""
import json

import pandas as pd

from src.report import DETECTORS, macro, table
from src.report_ext import FIVE, NAMES_X, THREE, cpm, name
from src.settings import RESULTS

LEAK = RESULTS / "leakage"
SUPERVISED = ["lexical_lr", "lexical_svm", "stylometry_xgb", "transformer", "coordinator"]
SETTINGS = {"In": "In-generator", "Cross": "Cross (A--C)", "D": "D, one generator", "E": "E, one generator",
            "D, ABC": "D, ABC", "E, ABC": "E, ABC"}
KINDS = {"source": "By report (used)", "group": "Near-duplicate reports joined", "chunk": "By excerpt (leaky)",
         "shuffled": "By report, labels shuffled"}


def leakage_tables(out: list[str]) -> None:
    facts = json.loads((LEAK / "facts.json").read_text(encoding="utf-8"))
    for k, v in facts.items():
        # Thousands separator for large counts, as in the rest of the paper (2{,}000).
        v = f"{v:,}".replace(",", "{,}") if isinstance(v, int) and v >= 10000 else v
        out.append(macro(name("Leak", k), float(v) if isinstance(v, float) else v))
    per = pd.read_csv(LEAK / "per_seed.csv")
    for col in ["test_chunks", "human_dup", "chunks_dup", "chunks_strict"]:
        out += [macro(name("Leak", col, "min"), int(per[col].min())), macro(name("Leak", col, "max"), int(per[col].max()))]
    out.append(macro("LeakLlmDupTotal", int(per.llm_dup.sum())))
    share = per.chunks_dup / per.test_chunks
    out += [macro("LeakDupShareMin", f"{100 * share.min():.0f}\\%"), macro("LeakDupShareMax", f"{100 * share.max():.0f}\\%")]

    s = pd.read_csv(LEAK / "sensitivity_summary.csv").set_index(["setting", "detector", "subset"])
    sup = s.loc[s.index.get_level_values("detector").isin(SUPERVISED)]
    for sub in ["clean", "strict"]:
        for v in ["f1", "auc"]:
            d = sup.xs(sub, level="subset")[f"{v}_mean"] - sup.xs("all", level="subset")[f"{v}_mean"]
            out += [macro(name("Leak", "max_delta", sub, v), float(d.abs().max())),
                    macro(name("Leak", "n_up", sub, v), int((d > 0.0005).sum())),
                    macro(name("Leak", "n_down", sub, v), int((d < -0.0005).sum())),
                    macro(name("Leak", "n_comp", sub, v), int(len(d)))]
    zs = s.loc[s.index.get_level_values("detector").isin(["fast_detectgpt", "binoculars"])]
    d = (zs.xs("clean", level="subset").f1_mean - zs.xs("all", level="subset").f1_mean).abs()
    out.append(macro("LeakMaxDeltaZeroFone", float(d.max())))
    rows = []
    for sett, label in SETTINGS.items():
        for det in THREE:
            cells = [cpm(s.loc[(sett, det, sub), "f1_mean"], s.loc[(sett, det, sub), "f1_std"])
                     for sub in ["all", "clean", "strict"]]
            cells += [f"{s.loc[(sett, det, sub), 'auc_mean']:.3f}" for sub in ["all", "clean", "strict"]]
            for sub in ["all", "clean", "strict"]:
                out.append(macro(name("Leak", sett, DETECTORS[det], sub), float(s.loc[(sett, det, sub), "f1_mean"])))
            rows.append(f"{label} & {NAMES_X[det]} & " + " & ".join(cells))
    n = {sub: s.xs(("In", "lexical_lr")).loc[sub, "n_mean"] for sub in ["all", "clean", "strict"]}
    table("leakage.tex",
          "Setting & Detector & \\multicolumn{3}{c}{F1} & \\multicolumn{3}{c}{ROC-AUC} \\\\\n"
          "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\n"
          " & & All & No dup. & Strict & All & No dup. & Strict",
          rows, f"Main results on all test excerpts and without those whose human or LLM text has a near-duplicate in "
          f"the training or validation split of the same seed (containment of word 5-grams $\\geq$ {facts['dup']}; "
          f"strict: $\\geq$ {facts['strict']}). Mean test-set size: {n['all']:.0f} / {n['clean']:.0f} / {n['strict']:.0f} "
          f"texts. F1 is mean $\\pm$ std over three seeds.", "tab:leakage", wide=True,
          spec="llcccccc", size="\\scriptsize")

    c = pd.read_csv(LEAK / "controls_summary.csv").set_index(["kind", "setting", "detector"])
    rows = []
    for kind, label in KINDS.items():
        for det in ["lexical_lr", "stylometry_xgb"]:
            cells = []
            for sett in ["In", "Cross", "D", "E"]:
                r = c.loc[(kind, sett, det)]
                cells.append(f"{r.f1_mean:.3f}")
                out += [macro(name("Ctrl", kind, sett, DETECTORS[det]), float(r.f1_mean)),
                        macro(name("CtrlAuc", kind, sett, DETECTORS[det]), float(r.auc_mean))]
            rows.append(f"{label} & {NAMES_X[det]} & " + " & ".join(cells))
    shuf = c.xs("shuffled", level="kind").auc_mean
    out += [macro("CtrlShuffledAucMin", float(shuf.min())), macro("CtrlShuffledAucMax", float(shuf.max()))]
    table("controls.tex", "Split & Detector & In & Cross & D & E", rows,
          "Control trainings, F1 at threshold 0.5 (mean over three seeds and the training generators A, B, C). Under "
          "the two alternative splits the test sets differ from the main ones, and D and E cover fewer of their "
          "excerpts.", "tab:controls", spec="llcccc", size="\\scriptsize\\setlength{\\tabcolsep}{3pt}")


def newgen_table(out: list[str]) -> None:
    """F1, error rates, ranking and probability quality side by side for the unseen generators."""
    m = pd.read_csv(RESULTS / "metrics_all.csv")
    m = m[m.exp == "main"].copy()
    m["fpr"] = m.fp / (m.fp + m.tn)
    m["fnr"] = m.fn / (m.fn + m.tp)
    cal = pd.read_csv(RESULTS / "calibration" / "metrics.csv", low_memory=False)
    raw = cal[(cal.calibrator == "raw") & (cal.threshold == "none")][["train", "seed", "test", "detector", "ece", "brier"]]
    h5 = cal[(cal.calibrator == "raw") & (cal.threshold == "human_fpr_5")][["train", "seed", "test", "detector", "fpr", "recall"]]
    h5 = h5.rename(columns={"fpr": "fpr5", "recall": "rec5"}).assign(fnr5=lambda d: 1 - d.rec5)
    m = m.merge(raw, on=["train", "seed", "test", "detector"], how="left") \
         .merge(h5, on=["train", "seed", "test", "detector"], how="left")
    m["setting"] = ["In" if (len(t) == 1 and t == x) else (f"{x}, ABC" if t == "ABC" and x in "DE" else
                    (x if len(t) == 1 and x in "DE" else "")) for t, x in zip(m.train, m.test)]
    cols = ["f1", "fpr", "fnr", "roc_auc", "ece", "brier", "fpr5", "fnr5"]
    per = m[m.setting != ""].groupby(["seed", "setting", "detector"])[cols].mean().reset_index()
    s = per.groupby(["setting", "detector"])[cols].agg(["mean", "std"])
    rows = []
    for sett, label in [("In", "In-generator"), ("D", "D, one generator"), ("D, ABC", "D, ABC"),
                        ("E", "E, one generator"), ("E, ABC", "E, ABC")]:
        for det in FIVE:
            r = s.loc[(sett, det)]
            rows.append(f"{label} & {NAMES_X[det]} & " + " & ".join(cpm(r[(v, 'mean')], r[(v, 'std')]) for v in cols))
            for v in cols:
                out.append(macro(name("Ng", sett, DETECTORS[det], v), float(r[(v, "mean")])))
    table("newgen.tex",
          " & & \\multicolumn{6}{c}{Threshold 0.5, raw scores} & \\multicolumn{2}{c}{Human 5\\% rule} \\\\\n"
          "\\cmidrule(lr){3-8}\\cmidrule(lr){9-10}\n"
          "Test & Detector & F1 & FPR & FNR & ROC-AUC & ECE & Brier & FPR & FNR",
          rows, "Errors, ranking and probability quality on the unseen generators D and E, next to the in-generator "
          "reference. FPR is measured on the CISA excerpts of each test set and FNR on the LLM texts. ECE and Brier "
          "use the uncalibrated scores. The last two columns use the threshold at the 95th percentile of validation "
          "human scores. Mean $\\pm$ std over three seeds.", "tab:newgen", wide=True, spec="@{}llcccccccc@{}",
          size="\\scriptsize\\setlength{\\tabcolsep}{2.5pt}")
