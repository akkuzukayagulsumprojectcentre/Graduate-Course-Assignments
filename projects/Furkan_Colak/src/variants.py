"""Rewritten and prompt-varied LLM texts for the robustness tests, and LLM versions of external excerpts."""
import asyncio
import json
import random
from datetime import datetime, timezone

import pandas as pd

from src.agents import Agent, append, extract_facts, multi_agent, read_jsonl, test_sources
from src.llm import make_client
from src.settings import DATA

PROCESSED = DATA / "processed"
GEN = DATA / "generated"
EXTERNAL = DATA / "external"
SYSTEM = "You write cyber threat intelligence."
FIRST_ROUND = ["paraphrase_self", "paraphrase_cross", "humanize", "backtranslate", "prompt_style", "prompt_fewshot"]


async def rewrite(client, prompt: str, temp: float, cfg: dict) -> dict:
    agent = Agent("rewriter", SYSTEM, client, temp, cfg["sampling"]["max_tokens"])
    return await agent.run(prompt)


async def one_job(job: dict, client, cfg: dict, examples: list[str]) -> dict:
    """Runs one condition on one base text; `text` is the input text, `facts` the Analyst output."""
    v, s = cfg["variants"]["prompts"], cfg["sampling"]
    cond = job["condition"]
    rng = random.Random(f"{cfg['seed']}|{job['generator']}|{job['chunk_id']}|{cond}")
    temp = rng.choice(s["temperatures"])
    if cond in ("paraphrase_self", "paraphrase_cross", "paraphrase_twice"):
        msgs = [await rewrite(client, v["paraphrase"].format(text=job["text"]), temp, cfg)]
    elif cond == "humanize":
        msgs = [await rewrite(client, v["humanize"].format(text=job["text"]), temp, cfg)]
    elif cond == "backtranslate":
        lang = cfg["variants"]["pivot_language"]
        m1 = await rewrite(client, v["to_pivot"].format(language=lang, text=job["text"]), temp, cfg)
        msgs = [m1, await rewrite(client, v["from_pivot"].format(text=m1["output"]), temp, cfg)]
    else:  # prompt_style / prompt_fewshot: same Writer -> Editor flow, only the Writer prompt changes
        n = job["n_words"]
        if cond == "prompt_style":
            w = v["style_writer"].format(n_words=n, facts=job["facts"])
        else:
            shots = rng.sample(examples, cfg["variants"]["n_style_examples"])
            block = "\n\n".join(f"Example {i + 1}: {t}" for i, t in enumerate(shots))
            w = v["fewshot_writer"].format(examples=block, n_words=n, facts=job["facts"])
        writer = Agent("writer", SYSTEM, client, temp, s["max_tokens"])
        editor = Agent("editor", cfg["agents"]["editor"].format(n_words=n), client, temp, s["max_tokens"])
        m1 = await writer.run(w)
        msgs = [m1, await editor.run(f"Facts: {job['facts']}\n\nDraft:\n{m1['output']}")]
    return {"chunk_id": job["chunk_id"], "source_id": job["source_id"], "generator": job["generator"],
            "condition": cond, "executor": job["executor"], "model": job["model"], "temperature": temp,
            "messages": msgs, "text": msgs[-1]["output"],
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}


async def run_jobs(jobs: list[dict], spec: dict, path, cfg: dict, examples: list[str]) -> None:
    client = make_client(spec)
    lock = asyncio.Lock()
    slots = asyncio.Semaphore(spec["concurrency"])
    failed = 0

    async def one(job: dict) -> None:
        nonlocal failed
        try:
            async with slots:
                rec = await one_job(job, client, cfg, examples)
        except Exception as e:
            failed += 1
            print(f"[variants] {job['chunk_id']} {job['condition']}: {e}")
            return
        await append(path, rec, lock)

    await asyncio.gather(*(one(j) for j in jobs))
    await client.close()
    print(f"[variants] {spec['model']}: {len(jobs) - failed}/{len(jobs)} jobs done", flush=True)


def base_jobs(cfg: dict, done: set) -> list[dict]:
    """Every first-round job not on disk yet: excerpts from test sources x generators x conditions."""
    df = pd.read_parquet(PROCESSED / "dataset.parquet")
    tests = test_sources()
    base = df[df.generator.isin(cfg["variants"]["generators"]) & (df["mode"] == "multi") & df.source_id.isin(tests)]
    facts = {r["chunk_id"]: r["output"] for r in read_jsonl(GEN / "facts.jsonl")}
    human = df[df.label == 0].set_index("chunk_id").n_words
    jobs = []
    for r in base.itertuples():
        for cond in FIRST_ROUND:
            if (r.chunk_id, r.generator, cond) in done:
                continue
            ex = cfg["variants"]["cross_executor"][r.generator] if cond == "paraphrase_cross" else r.generator
            jobs.append({"chunk_id": r.chunk_id, "source_id": r.source_id, "generator": r.generator,
                         "condition": cond, "executor": ex, "model": cfg["generators"][ex]["model"],
                         "text": r.text, "facts": facts.get(r.chunk_id), "n_words": int(human[r.chunk_id])})
    return jobs


def style_examples() -> list[str]:
    """Human excerpts from sources in no seed's test split, so test text never reaches a Writer prompt."""
    df = pd.read_parquet(PROCESSED / "dataset.parquet")
    return df[(df.label == 0) & ~df.source_id.isin(test_sources())].text.tolist()


async def rewrite_all(cfg: dict) -> None:
    path = GEN / "variants.jsonl"
    examples = style_examples()

    def done() -> set:
        return {(r["chunk_id"], r["generator"], r["condition"]) for r in read_jsonl(path)}

    first = base_jobs(cfg, done())
    print(f"[variants] first round: {len(first)} jobs", flush=True)
    # One model at a time: two 7-8B models do not fit in 8 GB of VRAM together.
    for ex in cfg["variants"]["generators"]:
        part = [j for j in first if j["executor"] == ex]
        if part:
            await run_jobs(part, cfg["generators"][ex], path, cfg, examples)

    # Recursive paraphrase: the first paraphrase goes through the same model once more.
    have = done()
    second = [{"chunk_id": r["chunk_id"], "source_id": r["source_id"], "generator": r["generator"],
               "condition": "paraphrase_twice", "executor": r["executor"], "model": r["model"], "text": r["text"]}
              for r in read_jsonl(path) if r["condition"] == "paraphrase_self"
              and (r["chunk_id"], r["generator"], "paraphrase_twice") not in have]
    print(f"[variants] second round: {len(second)} jobs", flush=True)
    for ex in cfg["variants"]["generators"]:
        part = [j for j in second if j["executor"] == ex]
        if part:
            await run_jobs(part, cfg["generators"][ex], path, cfg, examples)


async def external_generation(cfg: dict) -> None:
    """Analyst, then Writer and Editor of each local generator on the excerpts of other publishers."""
    raw = EXTERNAL / "human_chunks.jsonl"
    if not raw.exists():
        print("[external] no excerpts yet, run collect_ext first")
        return
    chunks = read_jsonl(raw)
    facts_path = EXTERNAL / "facts.jsonl"
    await extract_facts(chunks, cfg, facts_path)
    facts = {r["chunk_id"]: r["output"] for r in read_jsonl(facts_path)}
    for gid in cfg["external"]["generators"]:
        spec = cfg["generators"][gid]
        path = EXTERNAL / f"gen_{gid}.jsonl"
        have = {r["chunk_id"] for r in read_jsonl(path)}
        todo = [c for c in chunks if c["chunk_id"] in facts and c["chunk_id"] not in have]
        print(f"[external] {gid}: {len(todo)} jobs left", flush=True)
        client = make_client(spec)
        lock, slots = asyncio.Lock(), asyncio.Semaphore(spec["concurrency"])

        async def one(chunk: dict) -> None:
            rng = random.Random(f"{cfg['seed']}|{gid}|{chunk['chunk_id']}|multi")
            try:
                async with slots:
                    rec = await multi_agent(chunk, facts[chunk["chunk_id"]], client, cfg, rng)
            except Exception as e:
                print(f"[external] {chunk['chunk_id']}: {e}")
                return
            rec.update({"chunk_id": chunk["chunk_id"], "source_id": chunk["source_id"], "generator": gid,
                        "provider": spec["provider"], "model": spec["model"],
                        "created": datetime.now(timezone.utc).isoformat(timespec="seconds")})
            await append(path, rec, lock)

        await asyncio.gather(*(one(c) for c in todo))
        await client.close()


def run(cfg: dict) -> None:
    asyncio.run(rewrite_all(cfg))
    asyncio.run(external_generation(cfg))
    print(json.dumps({"variants": len(read_jsonl(GEN / "variants.jsonl"))}))
