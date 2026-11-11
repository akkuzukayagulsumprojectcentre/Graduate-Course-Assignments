"""Experiment runner (detect stage). Writes per-text probabilities to results/preds and results/preds_ext."""
import asyncio
import json
import time
from itertools import combinations

import httpx
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.detectors import (JUDGE, Combined, Coordinator, LexicalLR, LexicalSVM, Stylometry, Transformer, _judge)
from src.features import ALL, GROUPS
from src.settings import DATA, RESULTS, env

PROCESSED = DATA / "processed"
PREDS = RESULTS / "preds"
PREDS_EXT = RESULTS / "preds_ext"
ZEROSHOT = PROCESSED / "zeroshot.parquet"
ZS_SCORES = {"fast_detectgpt": 1.0, "binoculars": -1.0}  # sign makes larger mean more LLM-like
TRAIN_GENS: list[str] = []  # generators used for training (configs/experiments.yaml)
TEST_GENS: list[str] = []   # every generator present in the dataset, incl. test-only ones
KEEP = ["id", "source_id", "generator", "mode", "template", "topic", "label", "len_ratio"]
BASE: set[str] = set()  # dataset columns; anything else in a frame is a detector probability
PASS_THROUGH = {"llm_judge", *ZS_SCORES}
LOGS: dict[str, list[dict]] = {"coord_weights": [], "fit_times": []}


def subset(df: pd.DataFrame, sources: list[str], gens: list[str], mode: str = "multi") -> pd.DataFrame:
    """LLM texts of the given generators from these sources, plus the human excerpts they were made from.

    Pairing keeps classes balanced when a generator (e.g. a test-only API model) covers only part of the data.
    """
    d = df[df.source_id.isin(sources)]
    llm = d[d.generator.isin(gens) & (d["mode"] == mode)]
    return pd.concat([d[(d.label == 0) & d.chunk_id.isin(llm.chunk_id)], llm])


def platt(train_scores: np.ndarray, y: np.ndarray, scores: np.ndarray, seed: int) -> np.ndarray:
    """Zero-shot scores have no probability scale, so a one-feature logistic regression maps them on validation."""
    # Pooled validation holds three LLM texts per human one while every test set is balanced; unweighted, that prior
    # alone pushed all of Binoculars' human scores above 0.5.
    lr = LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced").fit(train_scores.reshape(-1, 1), y)
    return lr.predict_proba(scores.reshape(-1, 1))[:, 1]


def run_members(cfg: dict, train, val, tests: dict, seed: int, tag: dict) -> dict:
    p = cfg["detectors"]
    agents = {
        "lexical_lr": LexicalLR(p["lexical_lr"], seed),
        "lexical_svm": LexicalSVM({**p["lexical_lr"], **p["lexical_svm"]}, seed),
        "stylometry_xgb": Stylometry(p["stylometry_xgb"], seed),
        "transformer": Transformer(p["transformer"], seed),
    }
    out = {"val": val.copy(), **{k: v.copy() for k, v in tests.items()}}
    for name, agent in agents.items():
        t0 = time.perf_counter()
        agent.fit(train)
        LOGS["fit_times"].append({**tag, "seed": seed, "detector": name, "n_train": len(train),
                                  "fit_s": time.perf_counter() - t0})
        for k in out:
            out[k][name] = agent.predict(out[k])
        if isinstance(agent, Transformer):
            agent.free()
    members = [m for m in p["coordinator"]["members"] if m in out["val"]]
    coord = Coordinator(members, seed).fit(out["val"])
    LOGS["coord_weights"] += [{**tag, "seed": seed, "member": m, "coef": c, "intercept": coord.clf.intercept_[0]}
                              for m, c in zip(members, coord.clf.coef_[0])]
    # Ablation: rebuild the coordinator without each member in turn.
    drops = {f"coord_minus_{m}": Coordinator([x for x in members if x != m], seed).fit(out["val"]) for m in members}
    for k in tests:
        out[k]["coordinator"] = coord.predict(out[k])
        for name, c in drops.items():
            out[k][name] = c.predict(out[k])
    for z, sign in ZS_SCORES.items():
        if z in out["val"]:
            raw = (sign * out["val"][z]).to_numpy()
            for k in out:
                out[k][z] = platt(raw, out["val"].label.to_numpy(), (sign * out[k][z]).to_numpy(), seed)
    return out


def importance(df, splits, cfg, train_gen: str = "A") -> None:
    """Feature importance on one reference setting (train on human + generator A, first seed)."""
    seed = cfg["seeds"][0]
    p = cfg["detectors"]
    tr = subset(df, splits[str(seed)]["train"], [train_gen])
    xgb = Stylometry(p["stylometry_xgb"], seed).fit(tr)
    gain = xgb.clf.get_booster().get_score(importance_type="gain")
    pd.DataFrame({"feature": ALL, "gain": [gain.get(f, 0.0) for f in ALL]}) \
        .sort_values("gain", ascending=False).to_csv(RESULTS / "feature_importance.csv", index=False)

    lr = LexicalLR(p["lexical_lr"], seed).fit(tr)
    names = np.concatenate([[f"{n}:{t}" for t in v.get_feature_names_out()] for n, v in lr.vec.transformer_list])
    coef = lr.clf.coef_[0]
    top = np.argsort(coef)
    pd.concat([
        pd.DataFrame({"term": names[top[-30:]][::-1], "coef": coef[top[-30:]][::-1], "side": "llm"}),
        pd.DataFrame({"term": names[top[:30]], "coef": coef[top[:30]], "side": "human"}),
    ]).to_csv(RESULTS / "lr_top_terms.csv", index=False)


def save(frames: dict, exp: str, train_gens, seed: int, folder=PREDS) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for test_gen, d in frames.items():
        if test_gen == "val":
            continue
        probs = [c for c in d.columns if c not in BASE or c in PASS_THROUGH]
        rows.append(d[KEEP + probs].assign(test=test_gen))
    name = f"{exp}__train-{''.join(train_gens)}__seed-{seed}.parquet"
    pd.concat(rows).to_parquet(folder / name, index=False)


def extra_tests(df: pd.DataFrame, extra: pd.DataFrame, s: dict) -> dict:
    """Validation per generator, rewriting conditions and external publishers, keyed by a test name."""
    out = {f"val:{g}": subset(df, s["val"], [g]) for g in TRAIN_GENS}
    tests = set(s["test"])
    var = extra[(extra.group == "variant") & extra.source_id.isin(tests)]
    human = df[(df.label == 0) & df.source_id.isin(tests)]
    for (cond, g), llm in var.groupby(["mode", "generator"]):
        out[f"var:{cond}:{g}"] = pd.concat([human[human.chunk_id.isin(llm.chunk_id)], llm])
    # Paired comparison on the excerpts that have a single-agent version: same model, same sources, two procedures.
    for g in TRAIN_GENS:
        single = subset(df, s["test"], [g], mode="single")
        multi = subset(df, s["test"], [g])
        out[f"var:single:{g}"] = single
        out[f"var:multi_same:{g}"] = multi[multi.chunk_id.isin(single.chunk_id)]
    ext = extra[extra.group == "external"]
    for pub, d in ext.groupby("publisher"):
        h = d[d.label == 0]
        out[f"ext:{pub}:human"] = h
        for g, llm in d[d.label == 1].groupby("generator"):
            out[f"ext:{pub}:{g}"] = pd.concat([h[h.chunk_id.isin(llm.chunk_id)], llm])
    return out


def main_experiments(df, extra, splits, cfg):
    """One training per generator subset and seed; every subset is tested on every generator."""
    for k in range(1, len(TRAIN_GENS) + 1):
        for train_gens in combinations(TRAIN_GENS, k):
            for seed in cfg["seeds"]:
                s = splits[str(seed)]
                tests = {g: t for g in TEST_GENS if (t := subset(df, s["test"], [g])).label.nunique() == 2}
                more = extra_tests(df, extra, s)
                frames = run_members(cfg, subset(df, s["train"], train_gens), subset(df, s["val"], train_gens),
                                     {**tests, **more}, seed, {"exp": "main", "train": "".join(train_gens)})
                save({k: v for k, v in frames.items() if k in tests or k == "val"}, "main", train_gens, seed)
                save({k: v for k, v in frames.items() if k in more or k == "val"}, "main", train_gens, seed, PREDS_EXT)
                print(f"main train={''.join(train_gens)} seed={seed} done", flush=True)


def augmented_experiments(df, extra, splits, cfg):
    """Does more diverse training data help? Pooled ABC plus one publisher, or plus rewritten texts."""
    pubs = sorted(extra.loc[extra.group == "external", "publisher"].unique())
    ext = extra[extra.group == "external"]
    for seed in cfg["seeds"]:
        s = splits[str(seed)]
        tr, va = subset(df, s["train"], TRAIN_GENS), subset(df, s["val"], TRAIN_GENS)
        base = {g: t for g in TEST_GENS if (t := subset(df, s["test"], [g])).label.nunique() == 2}
        more = extra_tests(df, extra, s)
        for pub in pubs:
            # The held-in publisher's own frames are dropped from testing: its texts are now training data.
            tests = {**base, **{k: v for k, v in more.items() if not k.startswith(f"ext:{pub}:")}}
            frames = run_members(cfg, pd.concat([tr, ext[ext.publisher == pub]]), va, tests, seed,
                                 {"exp": f"aug-{pub}", "train": "ABC"})
            save(frames, f"aug-{pub}", TRAIN_GENS, seed, PREDS_EXT)
        # Texts rewritten by one agent or paraphrased, from training sources only.
        train_src = set(s["train"])
        para = extra[(extra.group == "variant") & (extra["mode"] == "paraphrase_self") & extra.source_id.isin(train_src)]
        single = df[df.source_id.isin(train_src) & (df["mode"] == "single") & df.generator.isin(TRAIN_GENS)]
        frames = run_members(cfg, pd.concat([tr, para, single]), va, {**base, **more}, seed,
                             {"exp": "aug-rewrite", "train": "ABC"})
        save(frames, "aug-rewrite", TRAIN_GENS, seed, PREDS_EXT)
        print(f"augmented seed={seed} done", flush=True)


def ablations(df, splits, cfg):
    p = cfg["detectors"]
    for seed in cfg["seeds"]:
        s = splits[str(seed)]
        for g in TRAIN_GENS:
            tr, te = subset(df, s["train"], [g]), {x: subset(df, s["test"], [x]) for x in TEST_GENS}
            # Feature groups: TF-IDF only (lexical_lr), stylometry only, both together, and each group alone.
            agents = {"tfidf": LexicalLR(p["lexical_lr"], seed), "stylometry": Stylometry(p["stylometry_xgb"], seed),
                      "both": Combined(p["lexical_lr"], seed),
                      "lr_no_clean": LexicalLR(p["lexical_lr"], seed, text_col="text_raw"),
                      "lr_masked": LexicalLR(p["lexical_lr"], seed, text_col="text_masked"),
                      "transformer_masked": Transformer(p["transformer"], seed, text_col="text_masked")}
            agents |= {f"group_{n}": Stylometry(p["stylometry_xgb"], seed, cols) for n, cols in GROUPS.items()}
            frames = {k: v.copy() for k, v in te.items()}
            for name, agent in agents.items():
                agent.fit(tr)
                for k in frames:
                    frames[k][name] = agent.predict(frames[k])
                if isinstance(agent, Transformer):
                    agent.free()
            save(frames, "abl-features", [g], seed)

            # Generation mode: train on multi-agent texts, test on single-agent texts of each generator.
            single = {x: subset(df, s["test"], [x], mode="single") for x in TEST_GENS}
            frames = {k: v.copy() for k, v in single.items() if v.label.nunique() == 2}  # D/E have no single-agent texts
            for name, agent in {"lexical_lr": LexicalLR(p["lexical_lr"], seed),
                                "stylometry_xgb": Stylometry(p["stylometry_xgb"], seed)}.items():
                agent.fit(tr)
                for k in frames:
                    frames[k][name] = agent.predict(frames[k])
            save(frames, "abl-single", [g], seed)

            # Unseen prompt template: hold one writer template out of training.
            for t in sorted(df.loc[df.template.str.startswith("t"), "template"].unique()):
                tr_t = tr[(tr.label == 0) | (tr.template != t)]
                held = te[g][te[g].template == t]
                te_t = te[g][((te[g].label == 0) & te[g].chunk_id.isin(held.chunk_id)) | (te[g].template == t)]
                te_t = te_t.drop(columns=[c for c in PASS_THROUGH if c in te_t]).copy()
                te_t["lexical_lr"] = LexicalLR(p["lexical_lr"], seed).fit(tr_t).predict(te_t)
                save({g: te_t}, f"abl-template-{t}", [g], seed)
        print(f"ablations seed={seed} done", flush=True)


def with_scores(d: pd.DataFrame, judge: pd.DataFrame, zs: pd.DataFrame | None) -> pd.DataFrame:
    d = d.merge(judge[["id", "llm_judge"]], on="id", how="left")
    d["llm_judge"] = d.llm_judge.fillna(0.5)
    if zs is not None:
        d = d.merge(zs, on="id", how="left")
    return d


def run(cfg: dict) -> None:
    df = pd.read_parquet(PROCESSED / "dataset.parquet")
    feats = pd.read_parquet(PROCESSED / "features.parquet")
    df = df.drop(columns=[c for c in ALL if c in df.columns]).merge(feats[["id"] + ALL], on="id")
    extra = pd.read_parquet(PROCESSED / "extra.parquet")
    efeats = pd.read_parquet(PROCESSED / "extra_features.parquet")
    extra = extra.drop(columns=[c for c in ALL if c in extra.columns]).merge(efeats[["id"] + ALL], on="id")
    splits = json.loads((PROCESSED / "splits.json").read_text(encoding="utf-8"))

    judge_spec = cfg["detectors"]["llm_judge"]
    asyncio.run(_judge(pd.concat([df[["id", "text"]], extra[["id", "text"]]]), judge_spec))
    # Unload the judge from VRAM; RoBERTa fine-tuning needs the 8 GB GPU next.
    httpx.post(f"{env('OLLAMA_HOST', 'http://localhost:11434')}/api/generate",
               json={"model": judge_spec["model"], "keep_alive": 0}, timeout=60)
    judge = pd.read_parquet(JUDGE).drop_duplicates("id")
    zs = pd.read_parquet(ZEROSHOT) if ZEROSHOT.exists() else None
    df, extra = with_scores(df, judge, zs), with_scores(extra, judge, zs)
    BASE.update(df.columns, extra.columns)
    TRAIN_GENS[:] = cfg["train_generators"]
    TEST_GENS[:] = sorted(set(df.generator) - {"human"})

    importance(df, splits, cfg)
    ablations(df, splits, cfg)
    main_experiments(df, extra, splits, cfg)
    augmented_experiments(df, extra, splits, cfg)
    for name, rows in LOGS.items():
        pd.DataFrame(rows).to_csv(RESULTS / f"{name}.csv", index=False)
