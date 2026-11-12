"""Tables and macros of the calibration, comparison, robustness and cost stages (part of the report stage)."""
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.report import DETECTORS, NAMES, macro, pm, table
from src.settings import DATA, RESULTS

NAMES_X = {**NAMES, "fast_detectgpt": "Fast-DetectGPT", "binoculars": "Binoculars"}
THREE = ["lexical_lr", "transformer", "coordinator"]
FIVE = ["lexical_lr", "transformer", "coordinator", "fast_detectgpt", "binoculars"]
THR = {"fixed_0.5": ("Fixed 0.5", "Fixed"), "val_f1": ("Validation F1", "ValF"),
       "pseudo_unseen_f1": ("Pseudo-unseen F1", "Pseudo"), "human_fpr_5": ("Human 5\\% FPR", "HumFive"),
       "human_fpr_1": ("Human 1\\% FPR", "HumOne")}
CONDS = {"writer_only": "Writer only, no Editor", "paraphrase_self": "Paraphrase, same model",
         "paraphrase_cross": "Paraphrase, other model", "paraphrase_twice": "Paraphrase twice",
         "backtranslate": "Back-translation", "humanize": "``Humanize'' prompt", "prompt_style": "Advisory-style prompt",
         "prompt_fewshot": "Few-shot human examples", "single": "Single agent reading the source",
         "mix25": "25\\% LLM sentences", "mix50": "50\\% LLM sentences", "mix75": "75\\% LLM sentences"}
SCEN = {"in-generator": "In-generator", "cross, train single": "Cross, local", "leave-one-out": "Leave-one-out",
        "D, train single": "D, one generator", "D, train ABC": "D, ABC", "E, train single": "E, one generator",
        "E, train ABC": "E, ABC", "microsoft, train ABC": "Microsoft, ABC", "mandiant, train ABC": "Mandiant, ABC",
        "writer_only, train ABC": "Writer only, ABC", "paraphrase_self, train ABC": "Paraphrase, ABC",
        "paraphrase_cross, train ABC": "Paraphrase, other model, ABC",
        "paraphrase_twice, train ABC": "Paraphrase twice, ABC", "backtranslate, train ABC": "Back-translation, ABC",
        "humanize, train ABC": "Humanize, ABC", "prompt_style, train ABC": "Advisory-style prompt, ABC",
        "prompt_fewshot, train ABC": "Few-shot prompt, ABC", "multi_same, train ABC": "Multi-agent, same sources, ABC",
        "single, train ABC": "Single-agent, ABC", "mix25, train ABC": "25\\% LLM, ABC",
        "mix50, train ABC": "50\\% LLM, ABC", "mix75, train ABC": "75\\% LLM, ABC"}
DIGITS = dict(zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]))
POOL = {1: "One", 2: "Two", 3: "Three"}


def name(*parts) -> str:
    """Macro name: letters only, since LaTeX control words cannot hold digits or punctuation."""
    s = "".join(str(p).replace("_", " ").title().replace(" ", "") for p in parts)
    return re.sub(r"[^A-Za-z]", "", "".join(DIGITS.get(c, c) for c in s))


def cpm(mean: float, std: float) -> str:
    """Mean and std without the spaces, for tables with many columns."""
    return pm(mean, std).replace(" $\\pm$ ", "$\\pm$")


def pct(v: float) -> str:
    return f"{100 * v:.1f}\\%"


def sci(v: float) -> str:
    m, e = f"{v:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def data_table(out: list[str]) -> None:
    ds = pd.read_parquet(DATA / "processed" / "dataset.parquet")
    src = pd.read_csv(DATA / "sources.csv")
    ext = pd.read_parquet(DATA / "processed" / "extra.parquet")
    var = ext[(ext.group == "variant") & (ext.label == 1)]
    out += [macro("NVarTexts", len(var)), macro("NVarChunks", var.chunk_id.nunique())]
    ext = ext[(ext.group == "external") & (ext.label == 0)]
    meta = pd.read_csv(DATA / "external" / "sources.csv")
    h = ds[ds.label == 0]
    # Word counts come from the stylometric features (alphabetic tokens), the same count as in the corpus table.
    feats = pd.concat([pd.read_parquet(DATA / "processed" / "features.parquet")[["id", "n_words"]],
                       pd.read_parquet(DATA / "processed" / "extra_features.parquet")[["id", "n_words"]]])
    words = feats.drop_duplicates("id").set_index("id").n_words

    def spread(ids) -> str:
        w = words[ids]
        return f"{w.median():.0f} & {w.quantile(.25):.0f}--{w.quantile(.75):.0f}"

    rows = [f"CISA & {h.source_id.nunique()} & {len(h)} & {src.released.min()}--{src.released.max()} & "
            f"{spread(h.id)} & Train, val., test"]
    for pub, label in [("microsoft", "Microsoft"), ("mandiant", "Mandiant")]:
        g, m = ext[ext.publisher == pub], meta[meta.publisher == pub]
        m = m[m.source_id.isin(g.source_id)]
        rows.append(f"{label} & {g.source_id.nunique()} & {len(g)} & {m.released.min()}--{m.released.max()} & "
                    f"{spread(g.id)} & Independent test")
        k = pub.capitalize()
        out += [macro(f"NExtSources{k}", g.source_id.nunique()), macro(f"NExtChunks{k}", len(g)),
                macro(f"ExtFirst{k}", m.released.min()), macro(f"ExtLast{k}", m.released.max()),
                macro(f"ExtWords{k}", f"{words[g.id].median():.0f}")]
    table("external.tex", "Publisher & Sources & Excerpts & Released & Median words & IQR words & Role", rows,
          "Human-written CTI by publisher. All excerpts are from documents released on or before 30 November 2022.",
          "tab:external", wide=True, spec="lrrcrrl")
    tests = pd.read_csv(RESULTS / "publisher_tests.csv")
    large = tests[(tests.p_bh < 0.05) & (tests.cliffs_delta.abs() >= 0.474)]
    out += [macro("NPubLarge", len(large)), macro("NPubTests", len(tests))]
    for pub in ["microsoft", "mandiant"]:
        for f in ["n_words", "mtld", "pos_PROPN", "pos_PRON"]:
            r = tests[(tests.publisher == pub) & (tests.feature == f)].iloc[0]
            out.append(macro(name("Pub", pub, f, "Delta"), float(r.cliffs_delta)))


def publisher_table(out: list[str]) -> None:
    p = pd.read_csv(RESULTS / "robust" / "publishers.csv")
    fpr = pd.read_csv(RESULTS / "robust" / "publisher_fpr.csv")
    rows = []
    for pub, label in [("cisa", "CISA"), ("microsoft", "Microsoft"), ("mandiant", "Mandiant")]:
        for det in FIVE:
            part = p[(p.exp == "main") & (p.publisher == pub) & (p.detector == det)]
            h = part[part.generator == "human"]
            l = part[part.generator != "human"]
            if h.empty:
                continue
            f, r, f1, a = h.fpr_mean.iloc[0], l.recall_mean.mean(), l.f1_mean.mean(), l.roc_auc_mean.mean()
            ci = fpr[(fpr.publisher == pub) & (fpr.detector == det)]
            band = f"[{ci.ci_low.iloc[0]:.3f}, {ci.ci_high.iloc[0]:.3f}]" if len(ci) else "--"
            # Training-free detectors are not refitted, so adding a publisher to training cannot change them.
            aug = p[(p.exp != "main") & p.exp.str.startswith("aug-") & (p.exp != "aug-rewrite") & (p.exp != f"aug-{pub}")
                    & (p.publisher == pub) & (p.detector == det) & (p.generator == "human")]
            a_f = cpm(aug.fpr_mean.mean(), aug.fpr_std.mean()) if pub != "cisa" and len(aug) and det in THREE else "--"
            rows.append(f"{label} & {NAMES_X[det]} & {cpm(f, h.fpr_std.iloc[0])} & {band} & {r:.3f} & "
                        f"{cpm(f1, l.f1_std.mean())} & {a:.3f} & {a_f}")
            k = name(pub, DETECTORS.get(det, det))
            out += [macro(f"Pub{k}Fpr", float(f)), macro(f"Pub{k}Rec", float(r)), macro(f"Pub{k}Fone", float(f1)),
                    macro(f"Pub{k}Auc", float(a))]
            if a_f != "--":
                out.append(macro(f"Pub{k}FprAug", float(aug.fpr_mean.mean())))
    table("publishers.tex",
          "Publisher & Detector & FPR & 95\\% CI & Recall & F1 & ROC-AUC & FPR, other publisher in training",
          rows, "Models trained on CISA + ABC, threshold 0.5. FPR on all human excerpts of the publisher; recall, F1 and AUC "
          "on human excerpts paired with their LLM versions from A, B and C, averaged. FPR and F1 are mean $\\pm$ std over "
          "three seeds; the interval resamples sources. The last column adds the other external publisher to training.",
          "tab:publishers", wide=True, spec="llccrcrc")


def calibration_tables(out: list[str]) -> None:
    m = pd.read_csv(RESULTS / "calibration" / "metrics.csv", low_memory=False)
    m = m[m.train.str.len() == 1]
    c = m[m.threshold == "none"]
    per = c.groupby(["seed", "kind", "detector", "calibrator"])[["brier", "ece", "nll"]].mean().reset_index()
    s = per.groupby(["kind", "detector", "calibrator"])[["brier", "ece", "nll"]].agg(["mean", "std"])
    rows = []
    for det in THREE:
        for cal, label in [("raw", "Raw"), ("platt", "Platt"), ("temperature", "Temperature"), ("isotonic", "Isotonic")]:
            cells = []
            for kind in ["in", "D", "E"]:
                r = s.loc[(kind, det, cal)]
                for v in ["brier", "ece", "nll"]:
                    cells.append(cpm(r[(v, "mean")], r[(v, "std")]))
                    out.append(macro(name("Cal", DETECTORS[det], cal, kind, v), float(r[(v, "mean")])))
            rows.append(f"{NAMES_X[det]} & {label} & " + " & ".join(cells))
    table("calibration.tex",
          "Detector & Calibrator & \\multicolumn{3}{c}{In-generator} & \\multicolumn{3}{c}{D: gpt-oss-120b} & "
          "\\multicolumn{3}{c}{E: Gemini} \\\\\n & & Brier & ECE & NLL & Brier & ECE & NLL & Brier & ECE & NLL",
          rows, "Probability quality on test data, training on one of A, B, C (mean $\\pm$ std over three seeds). "
          "Calibrators are fitted on validation data only.", "tab:calibration", wide=True, spec="@{}ll@{\\ }ccccccccc@{}",
          size="\\scriptsize\\setlength{\\tabcolsep}{2.5pt}")

    t = m[(m.threshold != "none") & (m.calibrator == "raw")]
    per = t.groupby(["seed", "kind", "detector", "threshold"])[["f1", "recall", "fpr", "precision"]].mean().reset_index()
    s = per.groupby(["kind", "detector", "threshold"])[["f1", "recall", "fpr", "precision"]].agg(["mean", "std"])
    rows = []
    for det in THREE:
        for th, (label, key) in THR.items():
            cells = []
            for kind in ["in", "cross", "D", "E", "external"]:
                if (kind, det, th) not in s.index:
                    cells += ["--", "--"] if kind in ("D", "external") else ["--"]
                    continue
                r = s.loc[(kind, det, th)]
                cells.append(pm(r[("f1", "mean")], r[("f1", "std")]).replace(" $\\pm$ ", "$\\pm$"))
                if kind in ("D", "external"):
                    cells.append(f"{r[('recall' if kind == 'D' else 'fpr', 'mean')]:.3f}")
                for v in ["f1", "recall", "fpr", "precision"]:
                    out.append(macro(name("Thr", kind, DETECTORS[det], key, v), float(r[(v, "mean")])))
            rows.append(f"{NAMES_X[det]} & {label} & " + " & ".join(cells))
    table("thresholds.tex",
          "Detector & Threshold rule & F1, in & F1, cross & F1, D & Recall, D & F1, E & F1, ext. & FPR, ext.",
          rows, "Effect of the decision rule, training on one of A, B, C. Every threshold is set on validation data. "
          "Pseudo-unseen uses validation texts of local generators that are neither in training nor the test generator. "
          "Human 1\\% and 5\\% use validation human scores alone.", "tab:thresholds", wide=True, spec="llccccccc")

    # Does calibration, not only a new threshold, change F1 on an unseen generator?
    t = m[(m.threshold == "fixed_0.5")]
    per = t.groupby(["seed", "kind", "detector", "calibrator"]).f1.mean().reset_index()
    s = per.groupby(["kind", "detector", "calibrator"]).f1.mean()
    for det in THREE:
        for cal in ["raw", "platt", "temperature", "isotonic"]:
            for kind in ["in", "D", "E"]:
                out.append(macro(name("CalFone", DETECTORS[det], cal, kind), float(s[(kind, det, cal)])))


def coordinator_table(out: list[str]) -> None:
    g = pd.read_csv(RESULTS / "compare" / "gaps.csv")
    s = pd.read_csv(RESULTS / "compare" / "summary.csv")

    def f1(sc, det):
        r = s[(s.scenario == sc) & (s.detector == det)]
        return f"{r.f1_mean.iloc[0]:.3f}" if len(r) else "--"

    def f1_pm(sc, det):
        r = s[(s.scenario == sc) & (s.detector == det)]
        return cpm(r.f1_mean.iloc[0], r.f1_std.iloc[0]) if len(r) else "--"

    rows = []
    cg = g[(g.a == "coordinator") & (g.b == "transformer")].set_index("scenario")
    for sc, label in SCEN.items():
        if sc not in cg.index:
            continue
        r = cg.loc[sc]
        star = "*" if r.p_holm < 0.05 else ""
        rows.append(f"{label} & {f1_pm(sc, 'transformer')} & {f1_pm(sc, 'coordinator')} & {r['diff']:+.3f}{star} & "
                    f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}] & {f1(sc, 'best_on_val')} & {f1(sc, 'coord_mean')} & "
                    f"{f1(sc, 'coord_unseen')} & {f1(sc, 'coord_zero_shot')}")
        k = name(label)
        out += [macro(f"Gap{k}", float(r["diff"])), macro(f"GapLo{k}", float(r.ci_low)), macro(f"GapHi{k}", float(r.ci_high))]
        for det, tag in [("transformer", "Trans"), ("coordinator", "Coord"), ("coord_mean", "CoordMean"),
                         ("coord_unseen", "CoordUnseen"), ("coord_zero_shot", "CoordZero"), ("best_on_val", "BestVal")]:
            v = f1(sc, det)
            if v != "--":
                out.append(macro(f"Sc{k}{tag}", v))
    table("coordinator.tex",
          "Scenario & RoBERTa & Coord. & Gap & 95\\% CI & Best on val. & Mean logit & Unseen stack & + zero-shot",
          rows, "F1 of RoBERTa, the Coordinator and its variants at threshold 0.5 in all scenarios (mean over runs; $\\pm$ std over "
          "seeds for the first two). Gap is Coordinator minus RoBERTa, with a paired source-level bootstrap interval; * marks "
          "Holm-adjusted p $<$ 0.05. ``Unseen stack'' fits the Coordinator on validation texts of generators absent from training. "
          "``Multi-agent, same sources'' is the reference for the single-agent row.",
          "tab:coordinator", wide=True, spec="@{}lcccccccc@{}",
          size="\\footnotesize\\setlength{\\tabcolsep}{3pt}")
    out += [macro("GapWins", int(((cg.p_holm < 0.05) & (cg["diff"] > 0)).sum())),
            macro("GapLosses", int(((cg.p_holm < 0.05) & (cg["diff"] < 0)).sum())), macro("GapN", len(cg))]
    w = pd.read_csv(RESULTS / "compare" / "weights.csv")
    for m in ["lexical_lr", "lexical_svm", "stylometry_xgb", "transformer", "llm_judge"]:
        out.append(macro(name("Wgt", DETECTORS[m]), float(w[(w.member == m) & (w.train.str.len() == 1)]["mean"].mean())))
    sp = pd.read_csv(RESULTS / "compare" / "member_spread.csv").set_index("member")
    for m in sp.index:
        out.append(macro(name("Drop", DETECTORS[m]), float(sp.weighted_drop[m])))
    best = pd.read_csv(RESULTS / "compare" / "best_on_val.csv")
    out.append(macro("BestValTop", NAMES_X[best.best_name.value_counts().index[0]]))


def robust_tables(out: list[str]) -> None:
    s = pd.read_csv(RESULTS / "robust" / "summary.csv")
    s = s[s.setting == "pooled"]
    tests = pd.read_csv(RESULTS / "robust" / "recall_tests.csv")
    rows = []
    for cond, label in CONDS.items():
        cells = []
        for det in FIVE:
            r = s[(s.condition == cond) & (s.detector == det)]
            if r.empty:
                cells.append("--")
                continue
            t = tests[(tests.condition == cond) & (tests.detector == det)]
            star = "*" if len(t) and t.p_holm.iloc[0] < 0.05 else ""
            cells.append(f"{cpm(r.recall_mean.iloc[0], r.recall_std.iloc[0])}{star}")
            k = name(cond, DETECTORS.get(det, det))
            out += [macro(f"Rec{k}", float(r.recall_mean.iloc[0])), macro(f"RecBase{k}", float(r.base_recall_mean.iloc[0])),
                    macro(f"RecAuc{k}", float(r.roc_auc_mean.iloc[0]))]
        rows.append(f"{label} & " + " & ".join(cells))
    table("robustness.tex", "Condition & " + " & ".join(NAMES_X[d] for d in FIVE), rows,
          "Recall on LLM texts at threshold 0.5 under rewriting, prompt strategy and partial generation, trained on CISA + ABC "
          "(mean $\\pm$ std over three seeds). * marks a recall change from the same excerpts before the change with "
          "Holm-adjusted p $<$ 0.05 (per detector).",
          "tab:robustness", wide=True, spec="lccccc")

    a = pd.read_csv(RESULTS / "robust" / "augmented.csv")
    rows = []
    for group, label in [("A", "A"), ("D", "D"), ("E", "E"), ("paraphrase_self", "Paraphrase, same model"),
                         ("paraphrase_twice", "Paraphrase twice"), ("humanize", "``Humanize'' prompt"), ("single", "Single agent")]:
        cells = []
        for det in THREE:
            for exp in ["main", "aug-rewrite"]:
                r = a[(a.exp == exp) & (a.group == group) & (a.detector == det)]
                cells.append(cpm(r.recall_mean.iloc[0], r.recall_std.iloc[0]) if len(r) else "--")
                if len(r):
                    out.append(macro(name("Aug", exp, group, DETECTORS[det]), float(r.recall_mean.iloc[0])))
        rows.append(f"{label} & " + " & ".join(cells))
    table("augmented.tex",
          "Test & \\multicolumn{2}{c}{Lexical-LR} & \\multicolumn{2}{c}{RoBERTa} & \\multicolumn{2}{c}{Coordinator} \\\\\n"
          " & base & aug. & base & aug. & base & aug.",
          rows, "Recall at threshold 0.5 (mean $\\pm$ std over three seeds), ABC training (base) against ABC plus single-agent "
          "and paraphrased texts of training sources (aug.).",
          "tab:augmented", wide=True, spec="lcccccc")

    # Where the LLM block sits in a partly generated text, pooling against false alarms, and copying from the source.
    root = RESULTS / "robust"
    for r in pd.read_csv(root / "mix_position.csv").itertuples():
        if r.detector in THREE:
            out.append(macro(name("Pos", r.condition, "First" if r.llm_first else "Last", DETECTORS[r.detector]),
                             float(r.recall_mean)))
    for r in pd.read_csv(root / "fpr_by_pool.csv").itertuples():
        if r.detector in THREE:
            out.append(macro(name("PoolFpr", POOL[r.k], r.publisher, DETECTORS[r.detector]), float(r.mean)))
    ov = pd.read_csv(root / "source_overlap.csv")
    for mode in ["single", "multi"]:
        part = ov[ov["mode"] == mode]["mean"]
        out += [macro(f"Overlap{mode.capitalize()}Min", float(part.min())), macro(f"Overlap{mode.capitalize()}Max", float(part.max()))]
    out.append(macro("NSingleTest", int(ov.n_test.sum())))


def cost_table(out: list[str]) -> None:
    tr = pd.read_csv(RESULTS / "zeroshot_truncation.csv")
    out += [macro("NBinoCheck", len(tr)), macro("BinoMaxDiff", sci(tr.abs_diff.max())),
            macro("BinoMedDiff", sci(tr.abs_diff.median())), macro("BinoMedScore", f"{tr.exact.median():.3f}"),
            macro("BinoRho", float(spearmanr(tr.exact, tr.top_k).correlation)),
            macro("BinoHigher", int((tr.top_k > tr.exact).sum()))]
    c = pd.read_csv(RESULTS / "cost" / "inference.csv").set_index("detector")
    fit = pd.read_csv(RESULTS / "cost" / "fit.csv")
    fit = fit[fit.train.str.len() == 1].groupby("detector")["mean"].mean()
    s = pd.read_csv(RESULTS / "compare" / "summary.csv")

    def v(sc, det, col="f1_mean"):
        r = s[(s.scenario == sc) & (s.detector == det)]
        return float(r[col].iloc[0]) if len(r) else np.nan

    rows = []
    for det, label in [("lexical_lr", "Lexical-LR"), ("lexical_svm", "Lexical-SVM"), ("stylometry_xgb", "Stylometry"),
                       ("transformer", "RoBERTa"), ("llm_judge", "LLM Judge"), ("fast_detectgpt", "Fast-DetectGPT"),
                       ("binoculars", "Binoculars"), ("coordinator", "Coordinator")]:
        r = c.loc[det]
        gpu = f"{r.gpu_mib:.0f}" if pd.notna(r.gpu_mib) and r.gpu_mib > 0 else "--"
        ft = f"{fit[det]:.0f}" if det in fit.index else "--"
        rows.append(f"{label} & {r.ms_per_text:.1f} & {r.relative_to_transformer:.2f} & {gpu} & {ft} & "
                    f"{v('in-generator', det):.3f} & {v('D, train single', det):.3f} & {v('D, train single', det, 'roc_auc_mean'):.3f} & "
                    f"{v('E, train single', det):.3f}")
        k = DETECTORS.get(det, name(det))
        out += [macro(f"Ms{k}", f"{r.ms_per_text:.1f}"), macro(f"Rel{k}", f"{r.relative_to_transformer:.1f}")]
    out.append(macro("MsCoordNoJudge", f"{c.ms_per_text['coord_no_judge']:.1f}"))
    table("cost.tex", "Detector & ms/text & Rel. & GPU MiB & Fit s & F1, in & F1, D & AUC, D & F1, E", rows,
          "Cost and performance, one training generator. Inference on one RTX 4060 laptop GPU, three repetitions over a fixed sample. "
          "The Coordinator cost is the sum of its members run one after another; ``Fit'' excludes the LLM judge, which is not trained.",
          "tab:cost", wide=True, spec="lrrrrcccc")
