"""Consistency check (verify stage): the paper's numbers and tables against the committed results.

1. Every row of results/metrics_all.csv is recomputed from the saved per-text predictions, with code that does not
   share anything with the evaluate stage.
2. paper/numbers.tex and paper/tables are rebuilt from results/ by the report stage and compared with the files on
   disk, which are put back afterwards.
3. A SHA-256 manifest of results/, data/processed/, figures/ and the generated paper files is written, so that any
   copy of the repository (such as the course submission) can be checked against it. Only files git sees are
   listed, and text files are hashed with LF line endings, so a checkout on another system gives the same hashes.

Imports the report stage on purpose: the check is that this code, run on these results, gives this paper.
"""
import hashlib
import json
import re
import subprocess

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.settings import RESULTS, ROOT

PAPER = ROOT / "paper"
NAME = re.compile(r"(?P<exp>.+)__train-(?P<train>[A-C]+)__seed-(?P<seed>\d+)")
META = {"id", "source_id", "generator", "mode", "template", "topic", "label", "len_ratio", "test"}
COUNTS = ["tp", "fp", "fn", "tn"]
RATES = ["accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc", "pr_auc"]
SKIP = {"manifest.csv", "verify.json"}
TRACKED = ["results", "data/processed", "figures", "paper/numbers.tex", "paper/tables"]


def recompute(y: np.ndarray, p: np.ndarray) -> dict:
    yhat = p >= 0.5
    tp, fp = int((yhat & (y == 1)).sum()), int((yhat & (y == 0)).sum())
    fn, tn = int((~yhat & (y == 1)).sum()), int((~yhat & (y == 0)).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    f1_neg = 2 * tn / (2 * tn + fn + fp) if tn else 0.0
    two = len(set(y)) == 2
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "accuracy": (tp + tn) / len(y),
            "precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": f1, "macro_f1": (f1 + f1_neg) / 2,
            "roc_auc": roc_auc_score(y, p) if two else np.nan, "pr_auc": average_precision_score(y, p) if two else np.nan}


def check_metrics() -> dict:
    saved = pd.read_csv(RESULTS / "metrics_all.csv").set_index(["exp", "train", "seed", "test", "detector"])
    rows, worst, bad = 0, 0.0, []
    for path in sorted((RESULTS / "preds").glob("*.parquet")):
        m = NAME.match(path.stem)
        d = pd.read_parquet(path)
        for test, g in d.groupby("test"):
            for det in [c for c in d.columns if c not in META]:
                new = recompute(g.label.to_numpy(), g[det].to_numpy())
                old = saved.loc[(m["exp"], m["train"], int(m["seed"]), test, det)]
                diff = max([abs(new[k] - old[k]) for k in COUNTS] +
                           [abs(new[k] - old[k]) for k in RATES if not (np.isnan(new[k]) and np.isnan(old[k]))])
                rows += 1
                worst = max(worst, diff)
                if diff > 1e-9:
                    bad.append(f"{path.stem} {test} {det}: {diff:.2e}")
    return {"metric_rows": rows, "metric_rows_saved": len(saved), "metric_max_abs_diff": worst,
            "metric_mismatches": bad[:20], "n_metric_mismatches": len(bad)}


def generated() -> list:
    return [PAPER / "numbers.tex", *sorted((PAPER / "tables").rglob("*.tex"))]


def lf(b: bytes) -> bytes:
    """Text with LF line endings; binary files (with NUL bytes) unchanged."""
    return b if bytes(1) in b else b.replace(b"\r\n", b"\n")


def check_report(cfg: dict) -> dict:
    before = {p: p.read_bytes() for p in generated()}
    from src import report
    report.run(cfg)
    after = {p: p.read_bytes() for p in generated()}
    changed = [str(p.relative_to(ROOT)) for p in after if lf(before.get(p, b"")) != lf(after[p])]
    changed += [str(p.relative_to(ROOT)) for p in before if p not in after]
    for p, b in before.items():  # leave the paper exactly as it was
        p.write_bytes(b)
    macros = before[PAPER / "numbers.tex"].decode("utf-8").count("\\newcommand")
    return {"macros": macros, "tables": len(before) - 1, "report_identical": not changed, "report_changed": changed}


def manifest() -> int:
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", *TRACKED],
                         cwd=ROOT, capture_output=True, check=True).stdout.decode("utf-8")
    files = [ROOT / f for f in sorted(out.split("\0")) if f and (ROOT / f).is_file() and (ROOT / f).name not in SKIP]
    rows = [{"path": p.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(lf(p.read_bytes())).hexdigest()}
            for p in files]
    pd.DataFrame(rows).to_csv(RESULTS / "manifest.csv", index=False)
    return len(rows)


def run(cfg: dict) -> None:
    out = {**check_metrics(), **check_report(cfg)}
    out["manifest_files"] = manifest()
    (RESULTS / "verify.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    if out["n_metric_mismatches"] or not out["report_identical"]:
        raise SystemExit("verify: results and paper differ, see results/verify.json")
