"""Tables and macros of the error analysis (part of the report stage)."""
import pandas as pd

from src.report import DETECTORS, MODELS, macro, table
from src.report_ext import NAMES_X, THREE, name
from src.settings import RESULTS

ROOT = RESULTS / "errors"
TEX = str.maketrans({"\\": "\\textbackslash{}", "&": "\\&", "%": "\\%", "$": "\\$", "#": "\\#", "_": "\\_",
                     "{": "\\{", "}": "\\}", "~": "\\textasciitilde{}", "^": "\\textasciicircum{}"})
SOURCES = {"human": "Human (CISA)", **{g: f"{g}: {m}" for g, m in MODELS.items()},
           "microsoft": "Human (Microsoft)", "mandiant": "Human (Mandiant)"}
SCOPES = {"D": "FN, D", "other generators": "FN, other generators", "CISA": "FP, CISA", "vendors": "FP, vendor blogs"}
TEMPLATES = {"t1": "Soc", "t2": "Blog", "t3": "Advisory", "t4": "Incident"}
# Property, label and number format of the rows in the contrast table.
ROWS = {"n_words": ("Words", "{:.0f}"), "mtld": ("MTLD", "{:.0f}"), "ttr": ("Type-token ratio", "{:.3f}"),
        "pos_PROPN": ("Proper-noun share", "{:.3f}"), "pos_PRON": ("Pronoun share", "{:.3f}"),
        "first_person": ("Uses first person (share of texts)", "{:.2f}"),
        "first_rate": ("First person per 100 words", "{:.2f}"), "advice": ("Advice terms per text", "{:.2f}"),
        "cues_llm": ("LLM-side cues per 100 words", "{:.2f}"), "cues_human": ("Human-side cues per 100 words", "{:.2f}"),
        "invented": ("Identifiers not in the facts", "{:.2f}")}
MISSED = ["n_words", "mtld", "ttr", "pos_PROPN", "first_person", "cues_llm", "cues_human", "invented"]
FLAGGED = ["n_words", "mtld", "ttr", "pos_PROPN", "pos_PRON", "first_rate", "cues_llm", "cues_human"]


def profile_table(out: list[str]) -> None:
    prof = pd.read_csv(ROOT / "profile_by_generator.csv").set_index("generator")
    rows = []
    for g, label in SOURCES.items():
        r = prof.loc[g]
        rows.append(f"{label} & {r.n_words:.0f} & {r.mtld:.0f} & {r.pos_PROPN:.3f} & {100 * r.first_person:.0f}\\% & "
                    f"{100 * r.agency:.0f}\\% & {r.cues_llm:.2f} & {r.cues_human:.2f}")
        for f in prof.columns:
            out.append(macro(name("Prof", "Human" if g == "human" else g, f), float(r[f])))
        out.append(macro(name("ProfPct", "Human" if g == "human" else g, "first"), f"{100 * r.first_person:.0f}\\%"))
    table("profile.tex",
          "Source & Words & MTLD & Proper nouns & First person & Agency named & LLM-side cues & Human-side cues", rows,
          "Mean text properties by source (multi-agent texts for A--E). First person and agency: share of texts with at "
          "least one first-person pronoun or agency name. Cues: occurrences per 100 words of the word n-grams with the "
          "largest Lexical-LR weights on each side (model trained on A).", "tab:profile", wide=True, spec="lrrrrrrr")
    terms = pd.read_csv(RESULTS / "lr_top_terms.csv")
    words = terms[terms.term.str.startswith("word:")]
    out += [macro("NCueLlm", int((words.side == "llm").sum())), macro("NCueHuman", int((words.side == "human").sum()))]
    for r in pd.read_csv(ROOT / "d_breakdown.csv").itertuples():
        if r.by == "template":
            out.append(macro(name("DMiss", TEMPLATES[r.value], DETECTORS[r.detector]), float(r.miss)))
        if r.by == "temperature":
            out.append(macro(name("DMissTemp", "Low" if float(r.value) < 0.9 else "High", DETECTORS[r.detector]), float(r.miss)))


def category_table(out: list[str]) -> None:
    cat = pd.read_csv(ROOT / "categories.csv")
    rows = []
    for scope, label in SCOPES.items():
        part = cat[cat.scope == scope]
        for c in part.groupby("category").n.sum().sort_values(ascending=False).index:
            sub = part[part.category == c].set_index("detector")
            err = sub.error.iloc[0]
            out.append(macro(name("ErrBase", err, scope, c), f"{100 * sub.base.iloc[0]:.0f}\\%"))
            for det in THREE:
                k = name(err, scope, c, DETECTORS[det])
                out += [macro(f"Err{k}", f"{100 * sub.share[det]:.0f}\\%"), macro(f"ErrN{k}", int(sub.n[det])),
                        macro(f"ErrRate{k}", float(sub.rate[det]))]
            if sub.n.sum() == 0:
                continue
            cells = [f"{int(sub.n[d])} ({100 * sub.share[d]:.0f}\\%)" if sub.n[d] else "0" for d in THREE]
            rows.append(f"{label} & {c} & {100 * sub.base.iloc[0]:.0f}\\% & " + " & ".join(cells))
        for det in THREE:
            out.append(macro(name("ErrTotal", part.error.iloc[0], scope, DETECTORS[det]), int(part[part.detector == det].n.sum())))
    table("errors.tex", "Error & Category & All texts & Lexical-LR & RoBERTa & Coordinator", rows,
          "Error categories over all seeds: count and share of the errors of each detector. Categories are mutually exclusive "
          "and assigned by the first rule that applies (Section~\\ref{sec:errors}); a text counts once per run in which it is "
          "misclassified. ``All texts'' is the share of the category among all texts of the group, errors or not, so a category "
          "is over-represented among the errors only when its share exceeds this. FN and CISA rows: models trained on one "
          "generator; vendor rows: models trained on ABC.", "tab:errors", wide=True, spec="llcccc")
    for path, tag in [("error_model.csv", "Or"), ("fp_model.csv", "FpOr")]:
        for r in pd.read_csv(ROOT / path).itertuples():
            k = name(DETECTORS[r.detector], r.feature)
            out += [macro(f"{tag}{k}", float(r.odds_ratio)), macro(f"{tag}Lo{k}", float(r.or_low)),
                    macro(f"{tag}Hi{k}", float(r.or_high))]


def contrast_table(out: list[str]) -> None:
    rows = []
    blocks = [("fn_vs_tp.csv", "D texts, missed / caught", MISSED, ("Fn", "Tp")),
              ("fp_vs_tn.csv", "Vendor excerpts, flagged / passed", FLAGGED, ("Fp", "Tn"))]
    for path, label, feats, (bad, good) in blocks:
        t = pd.read_csv(ROOT / path).set_index(["detector", "feature"])
        for r in t.reset_index().itertuples():
            k = name(DETECTORS[r.detector], r.feature)
            out += [macro(f"{bad}Mean{k}", float(r.wrong_mean)), macro(f"{good}Mean{k}", float(r.right_mean)),
                    macro(f"{bad}Delta{k}", float(r.cliffs_delta))]
        sizes = []
        for det in THREE:
            r = t.loc[(det, feats[0])]
            sizes += [f"{int(r.n_wrong)} / {int(r.n_right)}", ""]
            out += [macro(f"{bad}N{DETECTORS[det]}", int(r.n_wrong)), macro(f"{good}N{DETECTORS[det]}", int(r.n_right))]
        rows.append(f"{label} & Texts & " + " & ".join(sizes))
        for f in feats:
            text, fmt = ROWS[f]
            cells = []
            for det in THREE:
                r = t.loc[(det, f)]
                star = "*" if r.p_bh < 0.05 else ""
                cells += [f"{fmt.format(r.wrong_mean)} / {fmt.format(r.right_mean)}", f"{r.cliffs_delta:+.2f}{star}"]
            rows.append(f" & {text} & " + " & ".join(cells))
    table("contrast.tex",
          "Group & Property & \\multicolumn{2}{c}{Lexical-LR} & \\multicolumn{2}{c}{RoBERTa} & "
          "\\multicolumn{2}{c}{Coordinator} \\\\\n & & means & $\\delta$ & means & $\\delta$ & means & $\\delta$", rows,
          "Properties of misclassified against correctly classified texts: means of the two groups and Cliff's $\\delta$ "
          "(positive when the misclassified texts have more). A text belongs to the first group when at least half of the "
          "runs misclassify it. * marks a Mann-Whitney test with Benjamini-Hochberg adjusted p $<$ 0.05. D: models trained "
          "on one generator; vendor excerpts: models trained on ABC.", "tab:contrast", wide=True, spec="llcccccc")


def example_table(out: list[str]) -> None:
    ex = pd.read_csv(ROOT / "examples.csv")
    ex = ex[ex.id.str.startswith("cisa")]
    picks = [("coordinator", 1, "unsupported technical detail"), ("coordinator", 1, "impersonal, human length"),
             ("lexical_lr", 1, "cites an agency"), ("transformer", 1, "first person, still missed"),
             ("transformer", 0, "first person"), ("transformer", 0, "few proper nouns")]
    rows = []
    for det, label, cat in picks:
        r = ex[(ex.detector == det) & (ex.label == label) & (ex.category == cat)]
        if r.empty:
            continue
        r = r.iloc[0]
        kind, who = ("FN", f"Generator {r.generator}") if label == 1 else ("FP", "CISA")
        rows.append(f"{kind}, {NAMES_X[det]}: {cat}. {who}, {r.words} words, mean P(LLM) {r.mean_score:.2f} "
                    f"& {r.excerpt.translate(TEX)}\\ldots")
    if rows:
        table("examples.tex", "Error & Opening of the text", rows,
              "Examples of confident errors (CISA excerpts and LLM output), with the generator, the length and the mean "
              "score over the runs that misclassify the text.", "tab:examples", wide=True, spec="p{4.2cm}p{12.8cm}")
