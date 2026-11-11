"""Training-free detectors: analytic Fast-DetectGPT and Binoculars. Scores are cached per text (zeroshot stage)."""
import gc
import json
import time

import httpx
import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.models.falcon import modeling_falcon as falcon

from src.settings import CACHE, DATA, RESULTS, env

PROCESSED = DATA / "processed"
OUT = PROCESSED / "zeroshot.parquet"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load(name: str, four_bit: bool = False):
    if DEV == "cuda":
        # Under the cap the allocator frees its cache and retries; at the device limit Windows fails outright.
        torch.cuda.set_per_process_memory_fraction(0.85)
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw: dict = {"dtype": torch.float16 if DEV == "cuda" else torch.float32}
    if four_bit:
        # transformers 5 builds Falcon from its own nn.Linear subclass, which bitsandbytes skips, so the model would
        # load in fp16 (14 GB) and spill into system RAM. The subclass computes exactly what nn.Linear does.
        falcon.FalconLinear = nn.Linear
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.float16)
        kw["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(name, **kw)
    if not four_bit:
        model = model.to(DEV)
    return tok, model.eval()


def free_gpu() -> None:
    """Callers must drop their own model references first, or nothing is freed."""
    gc.collect()
    torch.cuda.empty_cache()


def batches(texts: list[str], tok, max_len: int, size: int):
    """Length-sorted batches keep padding, and so memory, small. Yields original indices and the encoding."""
    enc = tok(texts, truncation=True, max_length=max_len, return_token_type_ids=False)
    order = np.argsort([len(x) for x in enc["input_ids"]])
    for i in range(0, len(order), size):
        j = order[i:i + size]
        # Every batch is a little longer than the last; cached blocks of the old sizes otherwise pile up until the
        # 8 GB card is full (seen after ~800 texts).
        if DEV == "cuda":
            torch.cuda.empty_cache()
        b = tok.pad({k: [enc[k][x] for x in j] for k in enc}, padding=True, return_tensors="pt")
        yield j, b.to(DEV)


def shifted(b) -> tuple[torch.Tensor, torch.Tensor]:
    return b["input_ids"][:, 1:], b["attention_mask"][:, 1:].float()


def curvature(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Conditional probability curvature in closed form (Bao et al., 2024), with the same model for both roles."""
    # One text at a time: float32 vocabulary-sized temporaries for a whole batch overflow 8 GB of VRAM.
    out = []
    for lg, lab, m in zip(logits, labels, mask):
        lp = torch.log_softmax(lg.float(), -1)
        p = lp.exp()
        ll = lp.gather(-1, lab.unsqueeze(-1)).squeeze(-1)
        mean = (p * lp).sum(-1)
        var = (p * lp.square()).sum(-1) - mean.square()
        out.append(((ll - mean) * m).sum() / (var * m).sum().clamp_min(1e-8).sqrt())
    return torch.stack(out)


@torch.no_grad()
def fast_detectgpt(texts: list[str], spec: dict) -> np.ndarray:
    tok, model = load(spec["model"])
    out = np.zeros(len(texts))
    for j, b in batches(texts, tok, spec["max_len"], spec["batch_size"]):
        logits = model(**b).logits[:, :-1]
        labels, mask = shifted(b)
        out[j] = curvature(logits, labels, mask).cpu().numpy()
    del model
    free_gpu()
    return out


@torch.no_grad()
def observer_pass(texts: list[str], spec: dict, tok, model, folder) -> tuple[np.memmap, np.memmap]:
    """Keeps the observer's top-k next-token distribution per position on disk."""
    L, K = spec["max_len"] - 1, spec["top_k"]
    idx = np.memmap(folder / "idx.dat", dtype=np.int32, mode="w+", shape=(len(texts), L, K))
    prob = np.memmap(folder / "prob.dat", dtype=np.float32, mode="w+", shape=(len(texts), L, K))
    for j, b in batches(texts, tok, spec["max_len"], spec["batch_size"]):
        logits = model(**b).logits[:, :-1]
        for r, lg in zip(j, logits):
            top = torch.softmax(lg.float(), -1).topk(K, dim=-1)
            n = top.indices.shape[0]
            idx[r, :n] = top.indices.cpu().numpy()
            prob[r, :n] = top.values.cpu().numpy()
    idx.flush(), prob.flush()
    return idx, prob


@torch.no_grad()
def performer_pass(texts: list[str], spec: dict, tok, model, idx, prob) -> np.ndarray:
    """Binoculars ratio: perplexity under the performer over cross-perplexity of observer against performer."""
    out = np.zeros(len(texts))
    for j, b in batches(texts, tok, spec["max_len"], spec["batch_size"]):
        logits = model(**b).logits[:, :-1]
        labels, mask = shifted(b)
        n = labels.shape[1]
        for r, lg, lab, m in zip(j, logits, labels, mask):
            lq = torch.log_softmax(lg.float(), -1)
            top_i = torch.as_tensor(np.asarray(idx[r, :n]), device=lq.device, dtype=torch.long)
            top_p = torch.as_tensor(np.asarray(prob[r, :n]), device=lq.device)
            top_p = top_p / top_p.sum(-1, keepdim=True).clamp_min(1e-8)
            ce = -lq.gather(-1, lab.unsqueeze(-1)).squeeze(-1)
            xce = -(top_p * lq.gather(-1, top_i)).sum(-1)
            out[r] = ((ce * m).sum() / m.sum() / ((xce * m).sum() / m.sum())).item()
    return out


@torch.no_grad()
def full_observer(texts: list[str], spec: dict, tok, model, folder) -> np.memmap:
    """Full next-token distributions for a few texts, to measure what the top-k truncation costs."""
    L, V = spec["max_len"] - 1, model.get_output_embeddings().out_features
    full = np.memmap(folder / "full.dat", dtype=np.float16, mode="w+", shape=(len(texts), L, V))
    for j, b in batches(texts, tok, spec["max_len"], 1):
        p = torch.softmax(model(**b).logits[:, :-1].float(), -1)
        full[j, :p.shape[1]] = p.cpu().numpy().astype(np.float16)
    full.flush()
    return full


@torch.no_grad()
def exact_scores(texts: list[str], spec: dict, tok, model, full) -> np.ndarray:
    out = np.zeros(len(texts))
    for j, b in batches(texts, tok, spec["max_len"], 1):
        lq = torch.log_softmax(model(**b).logits[:, :-1].float(), -1)
        labels, mask = shifted(b)
        po = torch.as_tensor(np.asarray(full[j, :labels.shape[1]]), device=lq.device).float()
        ce = -lq.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        xce = -(po * lq).sum(-1)
        out[j] = (((ce * mask).sum(-1) / mask.sum(-1)) / ((xce * mask).sum(-1) / mask.sum(-1))).cpu().numpy()
    return out


def binoculars(texts: list[str], spec: dict) -> np.ndarray:
    """Two 7B models do not fit in 8 GB of VRAM together, so they run in two passes over a sparse observer output."""
    folder = CACHE / "binoculars"
    folder.mkdir(parents=True, exist_ok=True)
    chk = np.random.default_rng(0).choice(len(texts), min(spec["n_check"], len(texts)), replace=False)
    chk_texts = [texts[i] for i in chk]
    tok, obs = load(spec["observer"], spec["load_4bit"])
    idx, prob = observer_pass(texts, spec, tok, obs, folder)
    full = full_observer(chk_texts, spec, tok, obs, folder) if len(chk) else None
    del obs
    free_gpu()
    tok, perf = load(spec["performer"], spec["load_4bit"])
    out = performer_pass(texts, spec, tok, perf, idx, prob)
    if len(chk):
        exact = exact_scores(chk_texts, spec, tok, perf, full)
        pd.DataFrame({"exact": exact, "top_k": out[chk], "abs_diff": np.abs(exact - out[chk])}) \
            .to_csv(RESULTS / "zeroshot_truncation.csv", index=False)
    del perf
    free_gpu()
    del idx, prob, full
    for f in folder.glob("*.dat"):
        f.unlink()
    return out


METHODS = {"fast_detectgpt": fast_detectgpt, "binoculars": binoculars}


def needed_texts() -> pd.DataFrame:
    """Only validation, test and external texts are scored; training-free methods never see training sources."""
    df = pd.read_parquet(PROCESSED / "dataset.parquet")[["id", "source_id", "text"]]
    extra = pd.read_parquet(PROCESSED / "extra.parquet")[["id", "source_id", "text"]]
    splits = json.loads((PROCESSED / "splits.json").read_text(encoding="utf-8"))
    used = set().union(*(set(s["val"]) | set(s["test"]) for s in splits.values()))
    return pd.concat([df[df.source_id.isin(used)], extra]).drop_duplicates("id").reset_index(drop=True)


def free_ollama() -> None:
    """Ollama keeps the last generation model in VRAM for minutes; these models need the whole GPU."""
    host = env("OLLAMA_HOST", "http://localhost:11434")
    try:
        for m in httpx.get(f"{host}/api/ps", timeout=10).json().get("models", []):
            httpx.post(f"{host}/api/generate", json={"model": m["name"], "keep_alive": 0}, timeout=60)
    except httpx.HTTPError:
        pass


def run(cfg: dict) -> None:
    free_ollama()
    texts = needed_texts()
    old = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame({"id": []})
    res = texts[["id"]].merge(old, on="id", how="left")
    for name, spec in cfg["zeroshot"].items():
        if name not in res:
            res[name] = np.nan
        todo = res[res[name].isna()].index
        print(f"[zeroshot] {name}: {len(todo)} texts", flush=True)
        t0 = time.perf_counter()
        # Saved chunk by chunk, so a crash hours in loses one chunk; the truncation check runs on the first only.
        for k, start in enumerate(range(0, len(todo), spec["chunk"])):
            part = todo[start:start + spec["chunk"]]
            run_spec = spec if k == 0 else {**spec, "n_check": 0}
            res.loc[part, name] = METHODS[name](texts.loc[part, "text"].tolist(), run_spec)
            res.to_parquet(OUT, index=False)
            print(f"[zeroshot] {name}: {start + len(part)}/{len(todo)}, {time.perf_counter() - t0:.0f} s", flush=True)
