"""Static figures for the paper. One palette, fixed order: human, A, B, C, D, E."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.settings import FIGURES  # noqa: E402

COLORS = {"human": "#2a78d6", "A": "#eb6834", "B": "#1baf7a", "C": "#eda100", "D": "#e87ba4", "E": "#008300"}
LABELS = {"human": "Human", "A": "A (Llama 3.1 8B)", "B": "B (Qwen 2.5 7B)", "C": "C (Mistral 7B)",
          "D": "D (gpt-oss-120b)", "E": "E (Gemini 3.1 Flash-Lite)"}
DETECTOR_NAMES = {"lexical_lr": "Lexical-LR", "lexical_svm": "Lexical-SVM", "stylometry_xgb": "Stylometry",
                  "transformer": "RoBERTa", "llm_judge": "LLM Judge", "coordinator": "Coordinator"}
INK, MUTED, GRID ="#1f1f1e", "#6b6a64", "#e4e3de"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


def order(groups) -> list[str]:
    return [g for g in COLORS if g in set(groups)]


def save(fig, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png")
    plt.close(fig)


def feature_boxes(feats: pd.DataFrame, features: list[str]) -> None:
    gens = order(feats.generator)
    cols = 3
    rows = int(np.ceil(len(features) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.2, 2.1 * rows))
    for ax, f in zip(axes.flat, features):
        data = [feats.loc[feats.generator == g, f].dropna() for g in gens]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False,
                        medianprops={"color": INK, "linewidth": 1})
        for patch, g in zip(bp["boxes"], gens):
            patch.set(facecolor=COLORS[g], edgecolor="white", linewidth=1.5, alpha=0.9)
        ax.set_xticks(range(1, len(gens) + 1), [g if g != "human" else "Human" for g in gens])
        ax.set_title(f, fontsize=8, color=INK)
        ax.grid(axis="x", visible=False)
    for ax in list(axes.flat)[len(features):]:
        ax.axis("off")
    fig.tight_layout()
    save(fig, "feature_boxes")


SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def cross_heatmaps(summary: pd.DataFrame, detectors: list[str], metric: str = "f1_mean") -> None:
    """Train-set (rows) x test generator (columns) grid per detector, mean over seeds."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ_BLUE)
    trains = ["A", "B", "C", "AB", "AC", "BC", "ABC"]
    tests = order(summary.test)
    fig, axes = plt.subplots(1, len(detectors), figsize=(7.2, 2.9), sharey=True)
    for ax, det in zip(axes, detectors):
        d = summary[summary.detector == det].pivot(index="train", columns="test", values=metric)
        d = d.reindex(index=[t for t in trains if t in d.index], columns=tests)
        im = ax.imshow(d.to_numpy(), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for (i, j), v in np.ndenumerate(d.to_numpy()):
            if not np.isnan(v):
                # ".97" instead of "0.97" so five columns fit in a quarter of the page width.
                label = "1" if v >= 0.995 else f"{v:.2f}"[1:]
                ax.text(j, i, label, ha="center", va="center", fontsize=5.5,
                        color="white" if v > 0.6 else INK)
        ax.set_xticks(range(len(tests)), tests)
        ax.set_yticks(range(len(d.index)), d.index)
        ax.set_title(DETECTOR_NAMES.get(det, det), fontsize=8, color=INK)
        ax.set_xlabel("test generator")
        ax.grid(False)
    axes[0].set_ylabel("trained on")
    fig.colorbar(im, ax=axes, shrink=0.8, label=metric.replace("_mean", "").upper())
    save(fig, "cross_generator_heatmap")


def curves(d: pd.DataFrame, detector: str, title: str) -> None:
    from sklearn.metrics import precision_recall_curve, roc_curve
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for test in order(d.test):
        g = d[d.test == test]
        fpr, tpr, _ = roc_curve(g.label, g[detector])
        pr, rc, _ = precision_recall_curve(g.label, g[detector])
        a1.plot(fpr, tpr, color=COLORS[test], lw=2, label=f"test {LABELS[test]}")
        a2.plot(rc, pr, color=COLORS[test], lw=2, label=f"test {LABELS[test]}")
    a1.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--")
    a1.set(xlabel="false positive rate", ylabel="true positive rate", title=f"ROC, {detector}, {title}")
    a2.set(xlabel="recall", ylabel="precision", title=f"Precision-recall, {detector}, {title}")
    for ax in (a1, a2):
        ax.title.set_fontsize(8)
    a2.legend(frameon=False, fontsize=6.5, loc="lower left")
    save(fig, f"roc_pr_{detector}")


def confusions(d: pd.DataFrame, detector: str) -> None:
    from sklearn.metrics import confusion_matrix
    tests = order(d.test)
    fig, axes = plt.subplots(1, len(tests), figsize=(7.2, 2.2))
    for ax, test in zip(axes, tests):
        g = d[d.test == test]
        cm = confusion_matrix(g.label, g[detector] >= 0.5, labels=[0, 1])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.sum())
        for (i, j), v in np.ndenumerate(cm):
            tag = [["TN", "FP"], ["FN", "TP"]][i][j]
            ax.text(j, i, f"{tag}\n{v}", ha="center", va="center", fontsize=7,
                    color="white" if v > cm.sum() / 2 else INK)
        ax.set_xticks([0, 1], ["Human", "LLM"])
        ax.set_yticks([0, 1], ["Human", "LLM"])
        ax.set_xlabel("predicted")
        ax.set_title(f"test {test}", fontsize=8)
        ax.grid(False)
    axes[0].set_ylabel("true")
    fig.tight_layout()
    save(fig, f"confusion_{detector}")


def pos_bars(feats: pd.DataFrame, tags: list[str]) -> None:
    gens = order(feats.generator)
    means = feats.groupby("generator")[[f"pos_{t}" for t in tags]].mean()
    x = np.arange(len(tags))
    w = 0.8 / len(gens)
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    for i, g in enumerate(gens):
        ax.bar(x + i * w - 0.4 + w / 2, means.loc[g], width=w * 0.9, color=COLORS[g], label=LABELS[g])
    ax.set_xticks(x, tags)
    ax.set_ylabel("share of tokens")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=len(gens), loc="upper right", fontsize=7)
    save(fig, "pos_distribution")


# ---- figures of the calibration, comparison, robustness, cost and error stages ----

DET_COLORS = {"lexical_lr": "#eb6834", "transformer": "#2a78d6", "coordinator": "#1baf7a", "fast_detectgpt": "#eda100",
              "binoculars": "#e87ba4", "stylometry_xgb": "#008300", "lexical_svm": "#6b6a64", "llm_judge": "#a39f94"}
DET_MARKS = {"lexical_lr": "s", "transformer": "o", "coordinator": "D", "fast_detectgpt": "^", "binoculars": "v",
             "stylometry_xgb": "P", "lexical_svm": "X", "llm_judge": "*"}
DET_NAMES = {**DETECTOR_NAMES, "fast_detectgpt": "Fast-DetectGPT", "binoculars": "Binoculars"}
TEST_COLORS = {"A": "#2a78d6", "D": "#eb6834", "E": "#1baf7a"}
THREE = ["lexical_lr", "transformer", "coordinator"]


def _curve_frame(curves: list[dict], train: str, detector: str, method: str, test: str) -> tuple[np.ndarray, np.ndarray]:
    parts = [c for c in curves if c["train"] == train and c["detector"] == detector and c["calibrator"] == method
             and c["test"] == test]
    return np.concatenate([c["y"] for c in parts]), np.concatenate([c["p"] for c in parts])


def reliability(curves: list[dict], train: str = "A", bins: int = 10) -> None:
    """Reliability diagrams: raw and Platt-scaled, one panel per detector, one line per test generator."""
    labels = {"A": "same generator (A)", "D": "D (gpt-oss-120b)", "E": "E (Gemini)"}
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), sharex=True, sharey=True)
    edges = np.linspace(0, 1, bins + 1)
    for i, method in enumerate(["raw", "platt"]):
        for j, det in enumerate(THREE):
            ax = axes[i, j]
            ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--")
            for test, label in labels.items():
                y, p = _curve_frame(curves, train, det, method, test)
                which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
                used = [b for b in range(bins) if (which == b).sum() >= 5]
                ax.plot([p[which == b].mean() for b in used], [y[which == b].mean() for b in used],
                        color=TEST_COLORS[test], lw=1.4, marker="o", ms=3, label=label)
            ax.set_title(f"{DET_NAMES[det]}, {'raw' if method == 'raw' else 'Platt'}", fontsize=8, color=INK)
            ax.set(xlim=(0, 1), ylim=(0, 1))
    for ax in axes[1]:
        ax.set_xlabel("mean predicted P(LLM)")
    for ax in axes[:, 0]:
        ax.set_ylabel("observed share of LLM")
    axes[0, 0].legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout()
    save(fig, "reliability")


def threshold_sweep(curves: list[dict], train: str = "A", test: str = "D") -> None:
    grid = np.linspace(0.001, 0.999, 200)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    series = {"precision": "#2a78d6", "recall": "#eb6834", "F1": "#1baf7a"}
    for ax, det in zip(axes, THREE):
        y, p = _curve_frame(curves, train, det, "raw", test)
        tp = np.array([((p >= t) & (y == 1)).sum() for t in grid], float)
        fp = np.array([((p >= t) & (y == 0)).sum() for t in grid], float)
        fn = (y == 1).sum() - tp
        vals = {"precision": np.divide(tp, tp + fp, out=np.ones_like(tp), where=tp + fp > 0), "recall": tp / (tp + fn),
                "F1": 2 * tp / np.maximum(2 * tp + fp + fn, 1)}
        for k, v in vals.items():
            ax.plot(grid, v, color=series[k], lw=1.5, label=k)
        ax.axvline(0.5, color=MUTED, lw=0.8, ls="--")
        ax.set_title(DET_NAMES[det], fontsize=8, color=INK)
        ax.set_xlabel("decision threshold")
        ax.set(ylim=(0, 1.02))
    axes[0].set_ylabel(f"test {test}, trained on {train}")
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    save(fig, "threshold_sweep")


def pr_curves(curves: list[dict], train: str = "A") -> None:
    from sklearn.metrics import average_precision_score, precision_recall_curve
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
    for ax, det in zip(axes, THREE):
        for test, col in TEST_COLORS.items():
            y, p = _curve_frame(curves, train, det, "raw", test)
            pr, rc, _ = precision_recall_curve(y, p)
            ax.plot(rc, pr, color=col, lw=1.5, label=f"{test}: AP {average_precision_score(y, p):.3f}")
        ax.set_title(DET_NAMES[det], fontsize=8, color=INK)
        ax.set_xlabel("recall")
        ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    axes[0].set_ylabel("precision")
    fig.tight_layout()
    save(fig, "pr_curves")


def coordinator_gap(g: pd.DataFrame) -> None:
    """F1 of the Coordinator minus RoBERTa per scenario, with the bootstrap interval."""
    g = g.sort_values("diff").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(3.5, 0.2 * len(g) + 0.9))
    ax.errorbar(g["diff"], range(len(g)), xerr=[g["diff"] - g.ci_low, g.ci_high - g["diff"]], fmt="o", ms=3.5,
                color=DET_COLORS["coordinator"], ecolor=MUTED, elinewidth=0.8, capsize=0)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(range(len(g)), [s.replace("_", " ") for s in g.scenario], fontsize=6.5)
    ax.set_xlabel("F1 gap, Coordinator minus RoBERTa (95% CI)")
    ax.grid(axis="y", visible=False)
    save(fig, "coordinator_gap")


def robustness(s: pd.DataFrame, conditions: list[str], dets: list[str]) -> None:
    """Recall per condition: one row per condition, one mark per detector, recall before the change as a tick."""
    conds = [c for c in conditions if c in set(s.condition)]
    fig, ax = plt.subplots(figsize=(7.2, 0.3 * len(conds) + 1.1))
    offs = np.linspace(-0.27, 0.27, len(dets))
    for k, det in enumerate(dets):
        part = s[s.detector == det].set_index("condition").reindex(conds)
        ax.scatter(part.recall_mean, np.arange(len(conds)) + offs[k], color=DET_COLORS[det], marker=DET_MARKS[det], s=20,
                   label=DET_NAMES[det], zorder=3, edgecolor="white", linewidth=0.4)
        ax.scatter(part.base_recall_mean, np.arange(len(conds)) + offs[k], color=DET_COLORS[det], marker="|", s=26, zorder=2)
    ax.set_yticks(range(len(conds)), [c.replace("_", " ") for c in conds])
    ax.invert_yaxis()
    ax.set(xlim=(-0.02, 1.02), xlabel="recall on LLM texts at threshold 0.5 (tick: same excerpts before the change)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=7, ncol=len(dets), loc="upper center", bbox_to_anchor=(0.5, 1.13))
    save(fig, "robustness")


def mix_curve(s: pd.DataFrame, mixes: list[str], dets: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    x = [int(m[3:]) for m in mixes]
    for det in dets:
        part = s[s.detector == det].set_index("condition").reindex(mixes)
        ax.plot(x, part.recall_mean, color=DET_COLORS[det], marker=DET_MARKS[det], ms=4, lw=1.4, label=DET_NAMES[det])
    ax.set(xlabel="target LLM share of the words (%)", ylabel="recall at threshold 0.5", ylim=(-0.02, 1.02))
    ax.set_xticks(x)
    ax.legend(frameon=False, fontsize=6.5)
    save(fig, "mix_recall")


def publisher_shift(p: pd.DataFrame, dets: list[str]) -> None:
    """False positive rate on human text of each publisher, models trained on CISA only."""
    m = p[(p.exp == "main") & (p.generator == "human")]
    pubs = [x for x in ["cisa", "microsoft", "mandiant"] if x in set(m.publisher)]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    w = 0.8 / len(dets)
    for k, det in enumerate(dets):
        part = m[m.detector == det].set_index("publisher").reindex(pubs)
        ax.bar(np.arange(len(pubs)) + k * w - 0.4 + w / 2, part.fpr_mean, width=w * 0.9, color=DET_COLORS[det],
               label=DET_NAMES[det])
    ax.set_xticks(range(len(pubs)), [x.capitalize() if x != "cisa" else "CISA" for x in pubs])
    ax.set_ylabel("false positive rate at 0.5")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=6.5)
    save(fig, "publisher_fpr")


def cost_scatter(out: pd.DataFrame) -> None:
    """Inference time per text on a log scale; the performance side is in the paper's cost table."""
    d = out.dropna(subset=["ms_per_text"]).sort_values("ms_per_text").reset_index(drop=True)
    names = {**DET_NAMES, "coord_no_judge": "Coordinator without judge"}
    fig, ax = plt.subplots(figsize=(3.5, 0.2 * len(d) + 0.8))
    ax.barh(range(len(d)), d.ms_per_text, color="#2a78d6", height=0.6)
    ax.set_yticks(range(len(d)), [names.get(x, x) for x in d.detector], fontsize=7)
    for i, v in enumerate(d.ms_per_text):
        ax.text(v * 1.1, i, f"{v:.3g}", va="center", fontsize=6.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("inference time per text (ms, log scale)")
    ax.grid(axis="y", visible=False)
    save(fig, "inference_time")


def error_bins(b: pd.DataFrame, missed: dict, flagged: dict) -> None:
    """Error rate by feature tertile: missed D texts (top row) and flagged vendor excerpts (bottom row)."""
    lines = [("D", missed, "share of D texts missed", "tertile within D"),
             ("vendors", flagged, "share of vendor excerpts flagged", "tertile within vendor excerpts")]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.3), sharey="row")
    tertiles = ["low", "mid", "high"]
    for row, (group, effects, ylabel, xlabel) in zip(axes, lines):
        d = b[b.group == group]
        for ax, (f, name) in zip(row, effects.items()):
            for det in THREE:
                part = d[(d.feature == f) & (d.detector == det)].set_index("bin").reindex(tertiles)
                ax.errorbar(tertiles, part.rate, yerr=part["std"], color=DET_COLORS[det], marker=DET_MARKS[det], ms=4,
                            lw=1.4, elinewidth=0.8, capsize=2, label=DET_NAMES[det])
            ax.set_title(name, fontsize=8, color=INK)
            ax.set_xlabel(xlabel)
        row[0].set_ylabel(ylabel)
    axes[0, 0].set(ylim=(-0.02, 1.02))
    axes[1, 0].set(ylim=(-0.01, None))
    axes[1, 0].legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout()
    save(fig, "error_by_bin")


def score_ecdf(p: pd.DataFrame, dets: list[str]) -> None:
    """Score distribution of LLM texts per generator: the D curve sits apart although ranking stays good."""
    fig, axes = plt.subplots(1, len(dets), figsize=(7.2, 2.4), sharey=True)
    for ax, det in zip(axes, dets):
        for test in [g for g in "ABCDE" if g in set(p.test)]:
            s = np.sort(p[(p.test == test) & (p.label == 1)][det].to_numpy())
            ax.plot(s, np.arange(1, len(s) + 1) / len(s), color=COLORS[test], lw=1.4, label=LABELS[test])
        h = np.sort(p[p.label == 0][det].to_numpy())
        ax.plot(h, np.arange(1, len(h) + 1) / len(h), color=COLORS["human"], lw=1.4, ls="--", label="Human")
        ax.axvline(0.5, color=MUTED, lw=0.8)
        ax.set_title(DET_NAMES[det], fontsize=8, color=INK)
        ax.set_xlabel("P(LLM), trained on A")
    axes[0].set_ylabel("cumulative share of texts")
    axes[0].legend(frameon=False, fontsize=6, loc="lower right")
    fig.tight_layout()
    save(fig, "score_ecdf")
