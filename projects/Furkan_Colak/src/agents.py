"""Multi-agent generation of LLM CTI texts.

Stage 1: one Analyst (local model) extracts facts from every human chunk -> facts.jsonl.
Stage 2: each generator runs its own Writer and Editor on those facts -> gen_<id>.jsonl.
All generators share the same facts, so they differ in writing style, not in content.
"""
import asyncio
import hashlib
import json
import random
import re
from datetime import datetime, timezone

from src.llm import DailyLimitReached, LLMClient, make_client
from src.settings import DATA

RAW = DATA / "raw" / "human_chunks.jsonl"
OUT = DATA / "generated"
FACTS = OUT / "facts.jsonl"

REFUSAL = re.compile(r"^\W*(i['’]m sorry|i am sorry|sorry,|i can['’]?not|i can['’]t|i won['’]t)", re.I)
MAX_REFUSALS = 3


class Agent:
    def __init__(self, name: str, role_prompt: str, client: LLMClient, temperature: float, max_tokens: int):
        self.name = name
        self.role_prompt = role_prompt
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def run(self, msg: str) -> dict:
        refusals = 0
        while True:
            out = (await self.client.chat(self.role_prompt, msg, self.temperature, self.max_tokens)).strip()
            if out and not REFUSAL.search(out):
                break
            refusals += 1
            if refusals >= MAX_REFUSALS:
                raise RuntimeError(f"{self.name}: empty or refused {refusals} times")
        return {"agent": self.name, "input": msg, "output": out, "refusals": refusals}


def is_single(source_id: str, share: float) -> bool:
    """Same sources get the single-agent condition for every generator."""
    h = int(hashlib.md5(source_id.encode()).hexdigest(), 16) % 1000
    return h < share * 1000


def read_jsonl(path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


async def append(path, rec: dict, lock: asyncio.Lock) -> None:
    line = json.dumps(rec, ensure_ascii=False) + "\n"
    async with lock:
        # OneDrive can hold a short lock on the file while it syncs; retry instead of losing the record.
        for attempt in range(10):
            try:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(line)
                return
            except OSError:
                await asyncio.sleep(0.5 * (attempt + 1))
        raise OSError(f"could not append to {path}")


async def extract_facts(chunks: list[dict], cfg: dict, path=FACTS) -> None:
    done = {r["chunk_id"] for r in read_jsonl(path)}
    todo = [c for c in chunks if c["chunk_id"] not in done]
    print(f"[analyst] {len(todo)} chunks left")
    client = make_client(cfg["analyst"])
    s = cfg["sampling"]
    analyst = Agent("analyst", cfg["agents"]["analyst"], client, s["analyst_temperature"], s["max_tokens"])
    lock = asyncio.Lock()

    async def one(chunk: dict) -> None:
        try:
            m = await analyst.run(chunk["text"])
        except Exception as e:
            print(f"[analyst] {chunk['chunk_id']}: {e}")
            return
        try:
            m["output"] = json.dumps(json.loads(m["output"]), ensure_ascii=False)
        except json.JSONDecodeError:
            print(f"[analyst] {chunk['chunk_id']}: output is not valid JSON, kept as text")
        await append(path, {"chunk_id": chunk["chunk_id"], "model": cfg["analyst"]["model"], **m}, lock)

    await asyncio.gather(*(one(c) for c in todo))
    await client.close()


async def multi_agent(chunk: dict, facts: str, client: LLMClient, cfg: dict, rng: random.Random) -> dict:
    s, a = cfg["sampling"], cfg["agents"]
    template = rng.choice(sorted(a["writer_templates"]))
    temp = rng.choice(s["temperatures"])
    n_words = len(chunk["text"].split())

    writer = Agent("writer", "You write cyber threat intelligence.", client, temp, s["max_tokens"])
    editor = Agent("editor", a["editor"].format(n_words=n_words), client, temp, s["max_tokens"])
    m1 = await writer.run(a["writer_templates"][template].format(n_words=n_words, facts=facts))
    m2 = await editor.run(f"Facts: {facts}\n\nDraft:\n{m1['output']}")
    return {"mode": "multi", "template": template, "temperature": temp, "n_words_target": n_words,
            "messages": [m1, m2], "text": m2["output"]}


async def single_agent(chunk: dict, client: LLMClient, cfg: dict, rng: random.Random) -> dict:
    s = cfg["sampling"]
    temp = rng.choice(s["temperatures"])
    n_words = len(chunk["text"].split())
    agent = Agent("single", "You write cyber threat intelligence.", client, temp, s["max_tokens"])
    m = await agent.run(cfg["agents"]["single"].format(n_words=n_words, text=chunk["text"]))
    return {"mode": "single", "template": "single", "temperature": temp, "n_words_target": n_words,
            "messages": [m], "text": m["output"]}


def test_sources() -> set[str]:
    """Union of the test splits over all seeds (written by the build stage); empty before the first build."""
    path = DATA / "processed" / "splits.json"
    if not path.exists():
        return set()
    return set().union(*(set(s["test"]) for s in json.loads(path.read_text(encoding="utf-8")).values()))


def pending_jobs(gid: str, spec: dict, chunks: list[dict], facts: dict, cfg: dict) -> list[tuple[dict, str]]:
    done = {(r["chunk_id"], r["mode"]) for r in read_jsonl(OUT / f"gen_{gid}.jsonl")}
    if spec.get("test_only") and (tests := test_sources()):
        # Test-only generators are scored on test splits only and have no single-agent ablation,
        # so free-tier quota goes to the chunks that are actually evaluated.
        chunks = [c for c in chunks if c["source_id"] in tests]
    jobs = [(c, "multi") for c in chunks if c["chunk_id"] in facts]
    if not spec.get("test_only"):
        share = cfg["sampling"]["single_agent_share"]
        jobs += [(c, "single") for c in chunks if is_single(c["source_id"], share)]
    return [(c, m) for c, m in jobs if (c["chunk_id"], m) not in done]


def pending_counts(cfg: dict) -> dict[str, int]:
    """Jobs left per generator; used by experiments/resume_generation.sh to know when to stop."""
    chunks = read_jsonl(RAW)
    facts = {r["chunk_id"]: r["output"] for r in read_jsonl(FACTS)}
    return {g: len(pending_jobs(g, s, chunks, facts, cfg)) for g, s in cfg["generators"].items()}


async def run_generator(gid: str, spec: dict, chunks: list[dict], facts: dict, cfg: dict) -> None:
    path = OUT / f"gen_{gid}.jsonl"
    jobs = pending_jobs(gid, spec, chunks, facts, cfg)
    print(f"[{gid}] {spec['provider']}/{spec['model']}: {len(jobs)} jobs left")

    client = make_client(spec)
    lock = asyncio.Lock()
    # Limit whole jobs, not single calls: otherwise every Writer call queues ahead of every Editor call
    # and no record is finished until all drafts exist.
    job_slots = asyncio.Semaphore(spec["concurrency"])

    async def one(chunk: dict, mode: str) -> None:
        # Per-job RNG keeps template/temperature choice stable across resumed runs.
        rng = random.Random(f"{cfg['seed']}|{gid}|{chunk['chunk_id']}|{mode}")
        try:
            async with job_slots:
                if client.exhausted:
                    return
                if mode == "multi":
                    rec = await multi_agent(chunk, facts[chunk["chunk_id"]], client, cfg, rng)
                else:
                    rec = await single_agent(chunk, client, cfg, rng)
        except DailyLimitReached:
            return
        except Exception as e:
            print(f"[{gid}] {chunk['chunk_id']} {mode}: {e}")
            return
        rec.update({
            "chunk_id": chunk["chunk_id"], "source_id": chunk["source_id"], "generator": gid,
            "provider": spec["provider"], "model": spec["model"],
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        await append(path, rec, lock)

    await asyncio.gather(*(one(c, m) for c, m in jobs))
    await client.close()
    print(f"[{gid}] {len(read_jsonl(path))} records on disk")


async def generate_all(cfg: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    chunks = read_jsonl(RAW)
    if cfg["sampling"].get("limit"):
        chunks = chunks[: cfg["sampling"]["limit"]]
    await extract_facts(chunks, cfg)
    facts = {r["chunk_id"]: r["output"] for r in read_jsonl(FACTS)}
    gens = cfg["generators"].items()

    async def local_in_turn():
        # Two 7-8B models do not fit in 8 GB of VRAM together; running them in turn avoids constant swapping.
        for g, spec in gens:
            if spec["provider"] == "ollama":
                await run_generator(g, spec, chunks, facts, cfg)

    remote = [run_generator(g, spec, chunks, facts, cfg) for g, spec in gens if spec["provider"] != "ollama"]
    await asyncio.gather(local_in_turn(), *remote)


def run(cfg: dict) -> None:
    asyncio.run(generate_all(cfg))
