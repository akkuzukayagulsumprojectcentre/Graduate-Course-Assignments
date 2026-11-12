"""Inference time and memory per detector on one fixed sample (cost stage). Run with nothing else on the GPU."""
import asyncio
import json
import time

import httpx
import numpy as np
import pandas as pd
import torch

from src import plots, zeroshot
from src.detectors import JUDGE_PROMPT, LexicalLR, LexicalSVM, Stylometry, Transformer
from src.features import ALL, extract
from src.llm import make_client
from src.settings import DATA, RESULTS, env

PROCESSED = DATA / "processed"
OUT = RESULTS / "cost"


def timed(fn, repeats: int) -> dict:
    """Wall-clock over repeated calls, plus peak GPU memory; models stay loaded where loading is separate."""
    times = []
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
    return {"sec_mean": float(np.mean(times)), "sec_std": float(np.std(times, ddof=1)), "gpu_mib": peak}


def judge_call(texts: list[str], spec: dict) -> None:
    async def go():
        client = make_client({**spec, "concurrency": spec.get("concurrency", 3)})
        await asyncio.gather(*(client.chat("You are a careful analyst.", JUDGE_PROMPT.format(text=t), 0.0, 8) for t in texts))
        await client.close()
    asyncio.run(go())


def unload_judge(spec: dict) -> None:
    httpx.post(f"{env('OLLAMA_HOST', 'http://localhost:11434')}/api/generate",
               json={"model": spec["model"], "keep_alive": 0}, timeout=60)


def run(cfg: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    c = cfg["cost"]
    df = pd.read_parquet(PROCESSED / "dataset.parquet")
    feats = pd.read_parquet(PROCESSED / "features.parquet")
    df = df.drop(columns=[x for x in ALL if x in df.columns]).merge(feats[["id"] + ALL], on="id")
    splits = json.loads((PROCESSED / "splits.json").read_text(encoding="utf-8"))
    s = splits[str(cfg["seeds"][0])]
    pick = df[df["mode"].isin(["human", "multi"])]
    tr = pick[pick.source_id.isin(s["train"]) & pick.generator.isin(["human", c["train_generator"]])]
    te = pick[pick.source_id.isin(s["test"])].sample(c["n_texts"], random_state=cfg["seeds"][0])
    texts = te.text.tolist()
    p = cfg["detectors"]
    rows = []

    def add(name: str, **stats):
        rows.append({"detector": name, "n_texts": len(texts), **stats, "ms_per_text": 1000 * stats["sec_mean"] / len(texts)})
        print(rows[-1], flush=True)

    lr = LexicalLR(p["lexical_lr"], cfg["seeds"][0]).fit(tr)
    add("lexical_lr", **timed(lambda: lr.predict(te), c["repeats"]))
    svm = LexicalSVM({**p["lexical_lr"], **p["lexical_svm"]}, cfg["seeds"][0]).fit(tr)
    add("lexical_svm", **timed(lambda: svm.predict(te), c["repeats"]))
    xgb = Stylometry(p["stylometry_xgb"], cfg["seeds"][0]).fit(tr)
    # Stylometry cost includes spaCy parsing: features are not free at inference.
    add("stylometry_xgb", **timed(lambda: xgb.predict(extract(texts)), c["repeats"]))
    roberta = Transformer(p["transformer"], cfg["seeds"][0]).fit(tr)
    add("transformer", **timed(lambda: roberta.predict(te), c["repeats"]))
    roberta.free()

    spec = p["llm_judge"]
    judge_call(texts[:3], spec)  # the first call only loads the model, so it is not timed
    # A smaller sample: each call takes seconds, and judge time is linear in the number of texts.
    sub = texts[: c["n_judge"]]
    t = timed(lambda: judge_call(sub, spec), c["repeats"])
    rows.append({"detector": "llm_judge", "n_texts": len(sub), **t, "ms_per_text": 1000 * t["sec_mean"] / len(sub)})
    unload_judge(spec)

    z = cfg["zeroshot"]
    add("fast_detectgpt", **timed(lambda: zeroshot.fast_detectgpt(texts, z["fast_detectgpt"]), c["repeats"]))
    add("binoculars", **timed(lambda: zeroshot.binoculars(texts, {**z["binoculars"], "n_check": 0}), 1))

    out = pd.DataFrame(rows)
    by = out.set_index("detector").ms_per_text
    members = cfg["detectors"]["coordinator"]["members"]
    # Members run one after another on one GPU; stacking itself takes microseconds and is ignored.
    extra = [{"detector": "coordinator", "ms_per_text": by[members].sum()},
             {"detector": "coord_no_judge", "ms_per_text": by[[m for m in members if m != "llm_judge"]].sum()}]
    out = pd.concat([out, pd.DataFrame(extra)], ignore_index=True)
    out["relative_to_transformer"] = out.ms_per_text / by["transformer"]
    out.to_csv(OUT / "inference.csv", index=False)
    fit = pd.read_csv(RESULTS / "fit_times.csv")
    fit = fit[fit.exp == "main"].groupby(["train", "detector"]).fit_s.agg(["mean", "std"]).reset_index()
    fit.to_csv(OUT / "fit.csv", index=False)
    plots.cost_scatter(out)
    print(out.round(2).to_string(index=False))
