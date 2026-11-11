"""Paper numbers and LaTeX tables from results/ (report stage). Every number in the paper comes from here."""
import re

import numpy as np
import pandas as pd

from src.report_tr import TR_EXT
from src.settings import DATA, RESULTS, ROOT, load_config

PAPER = ROOT / "paper"
TABLES = PAPER / "tables"
DETECTORS = {"lexical_lr": "LexLR", "lexical_svm": "LexSVM", "stylometry_xgb": "Stylo",
             "transformer": "Trans", "llm_judge": "Judge", "fast_detectgpt": "Fast", "binoculars": "Bino",
             "coordinator": "Coord"}
NAMES = {"lexical_lr": "Lexical-LR", "lexical_svm": "Lexical-SVM", "stylometry_xgb": "Stylometry",
         "transformer": "RoBERTa", "llm_judge": "LLM Judge", "fast_detectgpt": "Fast-DetectGPT",
         "binoculars": "Binoculars", "coordinator": "Coordinator"}
MODELS = {"A": "Llama 3.1 8B", "B": "Qwen 2.5 7B", "C": "Mistral 7B", "D": "gpt-oss-120b", "E": "Gemini 3.1 Flash-Lite"}
TRAIN = "ABC"


def group(train: str, test: str) -> str:
    if test not in TRAIN:
        return test
    return "In" if train == test else "Cross"


def by_seed(m: pd.DataFrame, metric: str = "f1") -> pd.DataFrame:
    """Average over generators within each seed first, then mean and std over seeds."""
    m = m.assign(g=[group(a, b) for a, b in zip(m.train, m.test)])
    per_seed = m.groupby(["seed", "detector", "g"])[metric].mean().reset_index()
    return per_seed.groupby(["detector", "g"])[metric].agg(["mean", "std"]).reset_index()


def pm(mean: float, std: float) -> str:
    return f"{mean:.3f} $\\pm$ {std:.3f}"


def macro(name: str, value) -> str:
    if isinstance(value, float):
        value = f"{value:.3f}"
    return f"\\newcommand{{\\{name}}}{{{value}}}"


# Turkish strings for the tables of paper/tr. Longer keys are replaced first.
TR = {
    "Corpus per source. Medians over multi-agent texts (human excerpts for the first row).":
        "Kaynağa göre derlem. Medyanlar çok ajanlı metinler üzerinden (ilk satırda insan parçaları).",
    "Models trained on one generator (A, B or C). F1 is mean $\\pm$ std over three seeds of the average over "
    "training generators; AUC is ROC-AUC.":
        "Tek bir üreticiyle (A, B veya C) eğitilen modeller. F1, eğitim üreticileri ortalamasının üç seed "
        "üzerinden ortalama $\\pm$ standart sapmasıdır; AUC, ROC-AUC'dir.",
    "Training on Human + A, testing on Human + each generator (mean $\\pm$ std over seeds).":
        "İnsan + A ile eğitim, İnsan + her üretici ile test (seed'ler üzerinden ortalama $\\pm$ std).",
    "Ablations, F1 (mean $\\pm$ std over seeds; training on one of A, B, C).":
        "Ablasyonlar, F1 (seed'ler üzerinden ortalama $\\pm$ std; A, B veya C'den biriyle eğitim).",
    "Source & Multi & Single & Words & MTLD & Rep.\\ bigram & PROPN":
        "Kaynak & Çok ajan & Tek ajan & Kelime & MTLD & Tekrar bigram & PROPN",
    "Test & Detector & Acc & Precision & Recall & F1 & Macro-F1 & ROC-AUC & PR-AUC":
        "Test & Dedektör & Doğruluk & Kesinlik & Duyarlılık & F1 & Makro-F1 & ROC-AUC & PR-AUC",
    "TF-IDF + stylometry (LR)": "TF-IDF + stilometri (LR)",
    "TF-IDF only (LR)": "Yalnız TF-IDF (LR)",
    "Stylometry only (XGB)": "Yalnız stilometri (XGB)",
    "LR, no output cleaning": "LR, çıktı temizliği kapalı",
    "agency names masked": "kurum adları maskeli",
    "Stylometry: length only": "Stilometri: yalnız uzunluk",
    "Stylometry: lexical only": "Stilometri: yalnız sözcük çeşitliliği",
    "Stylometry: punctuation only": "Stilometri: yalnız noktalama",
    "Stylometry: repetition only": "Stilometri: yalnız tekrar",
    "Stylometry: POS only": "Stilometri: yalnız POS",
    "Coordinator, all agents": "Koordinatör, tüm ajanlar",
    "Human (CISA)": "İnsan (CISA)",
    "In-generator": "Aynı üretici",
    "Cross (A--C)": "Çapraz (A--C)",
    "Condition": "Koşul",
    "Detector": "Dedektör",
    "Stylometry": "Stilometri",
    "LLM Judge": "LLM Yargıç",
    "Coordinator": "Koordinatör",
    **TR_EXT,
}


def turkish(s: str) -> str:
    s = re.sub(r"Coordinator without ([^&]+?) &", r"Koordinatör, \1 olmadan &", s)
    for en in sorted(TR, key=len, reverse=True):
        s = s.replace(en, TR[en])
    return s


def table(name: str, header: str, rows: list[str], caption: str, label: str, wide: bool = False,
          spec: str = "", size: str = "\\footnotesize") -> None:
    """Writes paper/tables/en/<name> and a Turkish copy to paper/tables/tr/<name>."""
    env = "table*" if wide else "table"
    lines = [f"\\begin{{{env}}}[t]", "\\centering", size, f"\\caption{{{caption}}}",
             f"\\label{{{label}}}", f"\\begin{{tabular}}{{{spec}}}", "\\toprule", header + " \\\\", "\\midrule",
             *[r + " \\\\" for r in rows], "\\bottomrule", "\\end{tabular}", f"\\end{{{env}}}"]
    text = "\n".join(lines) + "\n"
    for lang, body in [("en", text), ("tr", turkish(text))]:
        (TABLES / lang).mkdir(parents=True, exist_ok=True)
        (TABLES / lang / name).write_text(body, encoding="utf-8")


def run(cfg: dict) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    gcfg = load_config("configs/generation.yaml")
    m = pd.read_csv(RESULTS / "metrics_all.csv")
    main = m[(m.exp == "main") & (m.train.str.len() == 1)]
    ds = pd.read_parquet(DATA / "processed" / "dataset.parquet")
    feats = pd.read_parquet(DATA / "processed" / "features.parquet")
    sources = pd.read_csv(DATA / "sources.csv")
    blocked = [u for u in (DATA / "blocked_by_antivirus.txt").read_text(encoding="utf-8").splitlines() if u]
    ling = pd.read_csv(RESULTS / "linguistic_tests.csv")
    out = []

    # ---- data ----
    human = ds[ds.label == 0]
    out += [macro("NSources", len(sources)), macro("NHumanChunks", len(human)),
            macro("MaxChunks", gcfg["collect"]["max_chunks_per_source"]),
            macro("SingleShare", f"{int(gcfg['sampling']['single_agent_share'] * 100)}\\%"),
            macro("NBlocked", len(blocked)), macro("NTexts", len(ds)),
            macro("NYearMin", sources.released.min()[:4]), macro("NYearMax", sources.released.max()[:4])]
    for g in "DE":
        cov = ds[(ds.generator == g) & (ds["mode"] == "multi")].chunk_id.nunique() / len(human)
        out.append(macro(f"NCover{g}", f"{100 * cov:.0f}\\%"))
    multi = feats[feats["mode"].isin(["human", "multi"])].merge(ds[["id", "n_words"]], on="id", suffixes=("", "_ds"))
    rows = []
    for g in ["human", *MODELS]:
        f = multi[multi.generator == g]
        n_single = int(((ds.generator == g) & (ds["mode"] == "single")).sum())
        name = "Human (CISA)" if g == "human" else f"{g}: {MODELS[g]}"
        rows.append(f"{name} & {len(f)} & {n_single} & {f.n_words.median():.0f} & {f.mtld.median():.0f} & "
                    f"{f.rep_bigram.median():.3f} & {f.pos_PROPN.median():.3f}")
        key = "Human" if g == "human" else g
        out.append(macro(f"MedWords{key}", f"{f.n_words.median():.0f}"))
        out.append(macro(f"MedMtld{key}", f"{f.mtld.median():.0f}"))
    table("data.tex",
          "Source & Multi & Single & Words & MTLD & Rep.\\ bigram & PROPN",
          rows, "Corpus per source. Medians over multi-agent texts (human excerpts for the first row).",
          "tab:data", wide=True, spec="lrrrrrr")

    strong = ling[(ling.p_bh < 0.05) & (ling.cliffs_delta.abs() >= 0.474)]
    hapax = ling[ling.feature == "hapax_ratio"].cliffs_delta
    out += [macro("NLargeEffects", len(strong)), macro("NTests", len(ling)),
            macro("HapaxDeltaMin", float(hapax.min())), macro("HapaxDeltaMax", float(hapax.max()))]

    # ---- main results ----
    f1, auc = by_seed(main, "f1"), by_seed(main, "roc_auc")
    rows = []
    for det, key in DETECTORS.items():
        cells = []
        for g in ["In", "Cross", "D", "E"]:
            r = f1[(f1.detector == det) & (f1.g == g)].iloc[0]
            a = auc[(auc.detector == det) & (auc.g == g)].iloc[0]
            cells.append(f"{pm(r['mean'], r['std'])} & {a['mean']:.3f}")
            out.append(macro(f"Fone{g}{key}", float(r["mean"])))
            out.append(macro(f"Auc{g}{key}", float(a["mean"])))
        rows.append(f"{NAMES[det]} & " + " & ".join(cells))
    table("main.tex",
          "Detector & \\multicolumn{2}{c}{In-generator} & \\multicolumn{2}{c}{Cross (A--C)} & "
          "\\multicolumn{2}{c}{D: gpt-oss-120b} & \\multicolumn{2}{c}{E: Gemini} \\\\\n"
          " & F1 & AUC & F1 & AUC & F1 & AUC & F1 & AUC",
          rows, "Models trained on one generator (A, B or C). F1 is mean $\\pm$ std over three seeds of the "
          "average over training generators; AUC is ROC-AUC.", "tab:main", wide=True, spec="lcccccccc")

    # Required setting: train on A, every metric.
    cols = ["accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc", "pr_auc"]
    a = main[(main.train == "A") & main.detector.isin(["lexical_lr", "transformer", "coordinator"])]
    s = a.groupby(["test", "detector"])[cols].agg(["mean", "std"])
    rows = []
    for test in [t for t in "ABCDE" if t in s.index.get_level_values(0)]:
        for det in ["lexical_lr", "transformer", "coordinator"]:
            r = s.loc[(test, det)]
            rows.append(f"{test} & {NAMES[det]} & " + " & ".join(pm(r[(c, 'mean')], r[(c, 'std')]) for c in cols))
    table("required.tex",
          "Test & Detector & Acc & Precision & Recall & F1 & Macro-F1 & ROC-AUC & PR-AUC", rows,
          "Training on Human + A, testing on Human + each generator (mean $\\pm$ std over seeds).",
          "tab:required", wide=True, spec="llccccccc")

    # Pooled and leave-one-out training.
    pool = m[(m.exp == "main") & (m.train == "ABC")]
    loo = m[(m.exp == "main") & (m.train.str.len() == 2)]
    loo = loo[[t not in tr and t in TRAIN for t, tr in zip(loo.test, loo.train)]]
    pool_d = pool[pool.test == "D"].groupby("detector").f1.mean()
    for det, key in DETECTORS.items():
        out.append(macro(f"PoolD{key}", float(pool_d.get(det, np.nan))))
        out.append(macro(f"Loo{key}", float(loo[loo.detector == det].f1.mean())))
    on_d = main[main.test == "D"].groupby("detector").f1.mean()
    out.append(macro("CoordNoTransD", float(on_d["coord_minus_transformer"])))
    others = [d for d in on_d.index if d.startswith("coord_minus") and d != "coord_minus_transformer"]
    out.append(macro("CoordDropMaxOther", float((on_d[others] - on_d["coordinator"]).abs().max())))

    # ---- statistics ----
    # Per-run tests are counted after Benjamini-Hochberg; the Holm family is the pooled test per detector and target.
    h1 = pd.read_csv(RESULTS / "h1_bootstrap.csv")
    h1 = h1[h1.detector != "llm_judge"]
    for name, part in [("Cross", h1[h1.test.isin(list(TRAIN))]), ("Ext", h1[~h1.test.isin(list(TRAIN))])]:
        out += [macro(f"HoneSig{name}", int((part.p_bh < 0.05).sum())), macro(f"HoneN{name}", len(part)),
                macro(f"HoneDiff{name}", float(part["diff"].mean()))]
    hp = pd.read_csv(RESULTS / "h1_pooled.csv")
    for r in hp.itertuples():
        k = f"{r.target.capitalize()}{DETECTORS[r.detector]}"
        out += [macro(f"Hp{k}", float(r.diff)), macro(f"HpLo{k}", float(r.ci_low)), macro(f"HpHi{k}", float(r.ci_high))]
    for target in ["cross", "D", "E"]:
        part = hp[(hp.target == target) & (hp.detector != "llm_judge")]
        out += [macro(f"HpSig{target.capitalize()}", int((part.p_holm < 0.05).sum())),
                macro(f"HpN{target.capitalize()}", len(part))]
    ci = pd.read_csv(RESULTS / "f1_intervals.csv")
    for r in ci[ci.setting == "D"].itertuples():
        out += [macro(f"FoneDLo{DETECTORS[r.detector]}", float(r.ci_low)),
                macro(f"FoneDHi{DETECTORS[r.detector]}", float(r.ci_high))]
    mc = pd.read_csv(RESULTS / "mcnemar.csv")
    for a_, b_, key in [("coordinator", "lexical_lr", "CoordLex"), ("transformer", "lexical_lr", "TransLex"),
                        ("coordinator", "transformer", "CoordTrans")]:
        part = mc[(mc.a == a_) & (mc.b == b_)]
        out += [macro(f"Mc{key}Sig", int((part.p_holm < 0.05).sum())), macro(f"Mc{key}N", len(part))]
    judge = main[main.detector == "llm_judge"]
    out.append(macro("JudgeHumanFlag", f"{100 * judge.fp.sum() / (judge.fp.sum() + judge.tn.sum()):.0f}\\%"))
    two = m[(m.exp == "main") & (m.train.str.len() == 2) & (m.test == "D")].groupby("detector").f1.mean()
    for det, key in DETECTORS.items():
        out.append(macro(f"PoolTwoD{key}", float(two.get(det, np.nan))))

    ref = main[(main.train == "A") & (main.detector == "coordinator") & (main.seed == cfg["seeds"][0])]
    for test in ["A", "D"]:
        r = ref[ref.test == test].iloc[0]
        for c in ["tp", "fp", "fn", "tn"]:
            out.append(macro(f"Cm{test}{c.upper()}", int(r[c])))

    # ---- ablations ----
    abl = m[m.exp == "abl-features"]
    ab = by_seed(abl, "f1")
    labels = {"tfidf": "TF-IDF only (LR)", "stylometry": "Stylometry only (XGB)", "both": "TF-IDF + stylometry (LR)",
              "lr_no_clean": "LR, no output cleaning", "lr_masked": "LR, agency names masked",
              "transformer_masked": "RoBERTa, agency names masked",
              **{f"group_{g}": f"Stylometry: {g.upper() if g == 'pos' else g} only" for g in ["length", "lexical", "punctuation", "repetition", "pos"]}}
    rows = []
    for det, label in labels.items():
        cells = []
        for g in ["In", "Cross", "D", "E"]:
            r = ab[(ab.detector == det) & (ab.g == g)]
            cells.append(pm(r["mean"].iloc[0], r["std"].iloc[0]) if len(r) else "--")
        rows.append(f"{label} & " + " & ".join(cells))
        key = "".join(w.capitalize() for w in det.split("_"))
        for g in ["In", "Cross", "D"]:
            r = ab[(ab.detector == det) & (ab.g == g)]
            if len(r):
                out.append(macro(f"Abl{key}{g}", float(r["mean"].iloc[0])))
    for det in ["coordinator"] + sorted(c for c in main.detector.unique() if c.startswith("coord_minus")):
        r = by_seed(main[main.detector == det], "f1")
        cells = [pm(*r[r.g == g][["mean", "std"]].iloc[0]) if (r.g == g).any() else "--" for g in ["In", "Cross", "D", "E"]]
        label = "Coordinator, all agents" if det == "coordinator" else f"Coordinator without {NAMES[det.split('_minus_')[1]]}"
        rows.append(f"{label} & " + " & ".join(cells))
    table("ablation.tex", "Condition & In-generator & Cross (A--C) & D & E", rows,
          "Ablations, F1 (mean $\\pm$ std over seeds; training on one of A, B, C).", "tab:ablation",
          wide=True, spec="lcccc")

    single = m[(m.exp == "abl-single")]
    sd = single.assign(g=[group(a, b) for a, b in zip(single.train, single.test)])
    for det, key in [("lexical_lr", "LexLR"), ("stylometry_xgb", "Stylo")]:
        for g in ["In", "Cross"]:
            out.append(macro(f"Single{g}{key}", float(sd[(sd.detector == det) & (sd.g == g)].f1.mean())))
    tpl = m[m.exp.str.startswith("abl-template") & (m.detector == "lexical_lr")]
    rec = tpl.groupby(["exp", "train"]).recall.mean()
    out += [macro("TplRecallMin", float(rec.min())), macro("TplRecallMax", float(rec.max())),
            macro("TplAucMin", float(tpl.roc_auc.min()))]
    # Length-matched subsets, with their size: generator A almost never writes as long as the human excerpt.
    lm = pd.read_csv(RESULTS / "length_matched.csv")
    lm = lm[lm.train == lm.test]
    same = main[main.train == main.test].groupby(["train", "detector"]).f1.mean()
    for g in TRAIN:
        size = lm[lm.train == g]
        out.append(macro(f"Len{g}N", f"{size.n_llm.iloc[0]:.0f}" if len(size) else "0"))
        for det, key in DETECTORS.items():
            r = lm[(lm.detector == det) & (lm.train == g)]
            if len(r):
                out += [macro(f"Len{g}{key}", float(r.f1_mean.iloc[0])), macro(f"Len{g}Auc{key}", float(r.roc_auc_mean.iloc[0])),
                        macro(f"Same{g}{key}", float(same[(g, det)]))]
    topic = pd.read_csv(RESULTS / "recall_by_topic.csv")
    for sett, tag in [("cross", "Cross"), ("D", "D")]:
        part = topic[topic.setting == sett].set_index("topic")
        for t, r in part.iterrows():
            out.append(macro(f"Topic{tag}{t.capitalize()}", float(r.recall)))
        out += [macro(f"Topic{tag}Min", float(part.recall.min())), macro(f"Topic{tag}Max", float(part.recall.max()))]
    imp = pd.read_csv(RESULTS / "feature_importance.csv")
    out.append(macro("TopFeature", imp.feature.iloc[0].replace("_", "\\_")))

    # Imported here because these modules reuse the helpers above.
    from src import report_err, report_ext
    report_ext.data_table(out)
    report_ext.publisher_table(out)
    report_ext.calibration_tables(out)
    report_ext.coordinator_table(out)
    report_ext.robust_tables(out)
    report_ext.cost_table(out)
    report_err.profile_table(out)
    report_err.category_table(out)
    report_err.contrast_table(out)
    report_err.example_table(out)

    (PAPER / "numbers.tex").write_text("% Written by `python -m src.cli report`. Do not edit by hand.\n"
                                       + "\n".join(out) + "\n", encoding="utf-8")
    print(f"{len(out)} macros, tables: {sorted(p.name for p in (TABLES / 'en').glob('*.tex'))}")
