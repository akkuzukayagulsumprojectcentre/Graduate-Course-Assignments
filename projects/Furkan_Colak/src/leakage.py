"""Leakage checks (leakage stage).

1. Audit of the saved predictions: no scored test row may come from a training or validation source of its seed.
2. Near-duplicate texts across splits by word-shingle containment, for the plain source split and for the purged
   split that the experiments use (src.dedup, build stage).
3. Main results of the purged run against the run on the plain source split (results/before_purge, same test
   sets), and of the purged run without the test excerpts that pass a stricter similarity cut.
4. Control trainings (CPU detectors): a chunk-level split that lets excerpts of one report fall on both sides, and
   the purged split with shuffled training labels.
"""
import json
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from src.dataset import make_splits
from src.dedup import containment, dup_groups, shingles
from src.detectors import LexicalLR, Stylometry
from src.features import ALL
from src.settings import DATA, RESULTS

PROCESSED = DATA / "processed"
OUT = RESULTS / "leakage"
BEFORE = RESULTS / "before_purge"  # main predictions of the run on the plain source split (commit b4185b7)
NAME = re.compile(r"(?P<exp>.+)__train-(?P<train>[A-C]+)__seed-(?P<seed>\d+)")
SKIP = {"id", "source_id", "generator", "mode", "template", "topic", "label", "len_ratio", "test"}


def audit(splits: dict) -> pd.DataFrame:
    """Rows of every saved prediction file that come from a source outside the test split of their seed."""
    rows = []
    for folder in ("preds", "preds_ext"):
        for path in sorted((RESULTS / folder).glob("*.parquet")):
            m = NAME.match(path.stem)
            s = splits[m["seed"]]
            d = pd.read_parquet(path, columns=["source_id", "test"])
            d = d[~d.test.str.startswith(("val:", "ext:"))]  # validation frames and other publishers are not CISA test
            rows.append({"folder": folder, **m.groupdict(), "rows": len(d), "sources": d.source_id.nunique(),
                         "in_train": int(d.source_id.isin(s["train"]).sum()),
                         "in_val": int(d.source_id.isin(s["val"]).sum()),
                         "not_test": int((~d.source_id.isin(s["test"])).sum())})
    return pd.DataFrame(rows)


def near_dups(df: pd.DataFrame, sh: list[set], splits: dict) -> pd.DataFrame:
    """Largest containment of every test text in one training or validation text of the same seed."""
    ids = df.id.to_numpy()
    parts = []
    for seed, s in splits.items():
        fit = np.flatnonzero(df.source_id.isin(set(s["train"]) | set(s["val"])).to_numpy())
        test = np.flatnonzero(df.source_id.isin(s["test"]).to_numpy())
        best, arg = containment([sh[i] for i in test], [sh[i] for i in fit])
        t = df.iloc[test]
        parts.append(pd.DataFrame({"seed": seed, "id": ids[test], "chunk_id": t.chunk_id.to_numpy(),
                                   "source_id": t.source_id.to_numpy(), "label": t.label.to_numpy(),
                                   "generator": t.generator.to_numpy(), "containment": best,
                                   "nearest": np.where(arg >= 0, ids[fit][np.maximum(arg, 0)], "")}))
    return pd.concat(parts, ignore_index=True)


def per_seed(nd: pd.DataFrame, cfg: dict, split: str) -> list[dict]:
    rows = []
    for seed, g in nd.groupby("seed"):
        h = g[g.label == 0]
        rows.append({"split": split, "seed": seed, "test_chunks": h.chunk_id.nunique(),
                     "human_median": float(h.containment.median()),
                     "human_dup": int((h.containment >= cfg["dup"]).sum()),
                     "llm_dup": int((g[g.label == 1].containment >= cfg["dup"]).sum()),
                     "chunks_dup": g.chunk_id[g.containment >= cfg["dup"]].nunique(),
                     "chunks_strict": g.chunk_id[g.containment >= cfg["strict"]].nunique(),
                     "max_containment": float(g.containment.max())})
    return rows


def siblings(human: pd.DataFrame, n: int) -> np.ndarray:
    """Containment of each human excerpt in the other excerpts of its own report."""
    out = []
    for _, g in human.groupby("source_id"):
        sh = [shingles(t, n) for t in g.text]
        out += [containment([sh[i]], sh[:i] + sh[i + 1:])[0][0] for i in range(len(sh)) if len(sh) > 1]
    return np.array(out)


def scores(y: np.ndarray, p: np.ndarray) -> dict:
    yhat = p >= 0.5
    pos, neg = y == 1, y == 0
    return {"n": len(y), "f1": f1_score(y, yhat, zero_division=0), "auc": roc_auc_score(y, p),
            "fpr": (yhat & neg).sum() / max(neg.sum(), 1), "fnr": (~yhat & pos).sum() / max(pos.sum(), 1)}


def setting(train: str, test: str) -> str:
    if len(train) > 1:
        return f"{test}, ABC" if train == "ABC" and test in "DE" else ""
    return "In" if test == train else ("Cross" if test in "ABC" else test)


def main_scores(folder, drops: dict[str, dict[str, set]]) -> pd.DataFrame:
    """Scores of every main run, on all test excerpts and without the chunks of each named subset (per seed)."""
    rows = []
    for path in sorted(folder.glob("main__*.parquet")):
        m = NAME.match(path.stem)
        d = pd.read_parquet(path)
        chunk = d.id.str.split("|").str[0]
        dets = [c for c in d.columns if c not in SKIP and not c.startswith("coord_minus")]
        for sub, drop in {"all": {}, **drops}.items():
            keep = d[~chunk.isin(drop.get(m["seed"], set()))]
            for test, g in keep.groupby("test"):
                if (sett := setting(m["train"], test)) and g.label.nunique() == 2:
                    rows += [{"train": m["train"], "seed": m["seed"], "test": test, "setting": sett, "subset": sub,
                              "detector": det, **scores(g.label.to_numpy(), g[det].to_numpy())} for det in dets]
    return pd.DataFrame(rows)


def subset(df: pd.DataFrame, units: list[str], gens: list[str]) -> pd.DataFrame:
    """As experiments.subset, on the split unit of a control (report or chunk)."""
    d = df[df.unit.isin(units)]
    llm = d[d.generator.isin(gens) & (d["mode"] == "multi")]
    return pd.concat([d[(d.label == 0) & d.chunk_id.isin(llm.chunk_id)], llm])


def controls(df: pd.DataFrame, splits: dict, cfg: dict, ecfg: dict) -> pd.DataFrame:
    gens = sorted(set(df.generator) - {"human"})
    kinds = {"source": (df.source_id, splits),
             "chunk": (df.chunk_id, make_splits(df.chunk_id.tolist(), ecfg["seeds"], ecfg["split"])),
             "shuffled": (df.source_id, splits)}
    p = ecfg["detectors"]
    rows = []
    for kind, (unit, sp) in kinds.items():
        d = df.assign(unit=unit.to_numpy())
        for seed in ecfg["seeds"]:
            s = sp[str(seed)]
            for g in ecfg["train_generators"]:
                tr = subset(d, s["train"], [g])
                if kind == "shuffled":
                    tr = tr.assign(label=np.random.default_rng(seed).permutation(tr.label.to_numpy()))
                agents = {"lexical_lr": LexicalLR(p["lexical_lr"], seed),
                          "stylometry_xgb": Stylometry(p["stylometry_xgb"], seed)}
                agents = {k: a.fit(tr) for k, a in agents.items() if k in cfg["detectors"]}
                for test in gens:
                    te = subset(d, s["test"], [test])
                    if te.label.nunique() < 2:
                        continue
                    rows += [{"kind": kind, "seed": seed, "train": g, "test": test, "setting": setting(g, test),
                              "detector": k, **scores(te.label.to_numpy(), a.predict(te))} for k, a in agents.items()]
        print(f"control {kind} done", flush=True)
    return pd.DataFrame(rows)


def by_seed(r: pd.DataFrame, keys: list[str], cols: list[str]) -> pd.DataFrame:
    """Mean over runs within a seed, then mean and std over seeds (as in the main tables)."""
    per = r.groupby(["seed"] + keys)[cols].mean().reset_index()
    s = per.groupby(keys)[cols].agg(["mean", "std"])
    s.columns = ["_".join(c) for c in s.columns]
    return s.reset_index()


def run(ecfg: dict) -> None:
    cfg = ecfg["leakage"]
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PROCESSED / "dataset.parquet")
    feats = pd.read_parquet(PROCESSED / "features.parquet")
    df = df.drop(columns=[c for c in ALL if c in df.columns]).merge(feats[["id"] + ALL], on="id")
    splits = json.loads((PROCESSED / "splits.json").read_text(encoding="utf-8"))
    plain = json.loads((PROCESSED / "splits_unpurged.json").read_text(encoding="utf-8"))
    purged = pd.read_csv(PROCESSED / "purged.csv")
    n = cfg["shingle"]

    a = audit(splits)
    a.to_csv(OUT / "audit.csv", index=False)
    seeds = {seed: {k: set(v) for k, v in s.items()} for seed, s in splits.items()}
    overlap = max(len(s["train"] & s["test"]) + len(s["val"] & s["test"]) + len(s["train"] & s["val"])
                  for s in seeds.values())

    human = df[df.label == 0].reset_index(drop=True)
    sh = [shingles(t, n) for t in df.text]
    nd_plain, nd = near_dups(df, sh, plain), near_dups(df, sh, splits)
    nd_plain.to_csv(OUT / "near_dup_unpurged.csv", index=False)
    nd.to_csv(OUT / "near_dup.csv", index=False)
    pd.DataFrame(per_seed(nd_plain, cfg, "unpurged") + per_seed(nd, cfg, "purged")).to_csv(OUT / "per_seed.csv",
                                                                                          index=False)
    sizes = Counter(dup_groups(human, n, cfg["dup"]).values())
    sib = siblings(human, n)
    ext = pd.read_parquet(PROCESSED / "extra.parquet")
    vendor = ext[(ext.group == "external") & (ext.label == 0)]
    vend, _ = containment([shingles(t, n) for t in vendor.text], [shingles(t, n) for t in human.text])
    size = {k: [len(s[k]) for s in splits.values()] for k in ("train", "val")}
    size["val_human"] = [int(human.source_id.isin(s["val"]).sum()) for s in splits.values()]
    removed = purged[purged.action == "removed"].groupby("seed").size().reindex(map(int, splits), fill_value=0)
    moved = purged[purged.action != "removed"].groupby("seed").size().reindex(map(int, splits), fill_value=0)

    facts = {"shingle": n, "dup": cfg["dup"], "strict": cfg["strict"], "split_overlap": overlap,
             "audit_files": len(a), "audit_rows": int(a.rows.sum()),
             "audit_bad": int(a[["in_train", "in_val", "not_test"]].sum().sum()),
             "dup_groups": sum(v > 1 for v in sizes.values()), "dup_sources": sum(v for v in sizes.values() if v > 1),
             "n_sources": int(human.source_id.nunique()),
             "removed_min": int(removed.min()), "removed_max": int(removed.max()),
             "moved_min": int(moved.min()), "moved_max": int(moved.max()),
             "train_min": min(size["train"]), "train_max": max(size["train"]),
             "val_min": min(size["val"]), "val_max": max(size["val"]),
             "val_human_min": min(size["val_human"]), "val_human_max": max(size["val_human"]),
             "sibling_median": float(np.median(sib)), "sibling_dup": int((sib >= cfg["dup"]).sum()),
             "n_siblings": len(sib), "vendor_max": float(vend.max()), "vendor_dup": int((vend >= cfg["dup"]).sum()),
             "n_vendor": len(vendor)}
    (OUT / "facts.json").write_text(json.dumps(facts, indent=1), encoding="utf-8")

    strict = {seed: set(g.chunk_id[g.containment >= cfg["strict"]]) for seed, g in nd.groupby("seed")}
    parts = [main_scores(RESULTS / "preds", {"strict": strict}).assign(run="purged")]
    if BEFORE.exists():
        parts.append(main_scores(BEFORE, {}).assign(run="unpurged"))
    sens = pd.concat(parts, ignore_index=True)
    sens.to_csv(OUT / "sensitivity.csv", index=False)
    by_seed(sens, ["run", "subset", "setting", "detector"], ["n", "f1", "auc", "fpr", "fnr"]) \
        .to_csv(OUT / "sensitivity_summary.csv", index=False)

    ctrl = controls(df, splits, cfg, ecfg)
    ctrl.to_csv(OUT / "controls.csv", index=False)
    by_seed(ctrl, ["kind", "setting", "detector"], ["n", "f1", "auc"]).to_csv(OUT / "controls_summary.csv", index=False)
    print(json.dumps(facts, indent=1))
    print(pd.read_csv(OUT / "per_seed.csv").to_string(index=False))
