"""Async chat clients for Ollama, Groq and Gemini with rate limiting and retries."""
import asyncio
import random
import re
import time

import httpx

from src.settings import env, secret


class RateLimiter:
    """Spaces requests to stay under a requests-per-minute and requests-per-day budget."""

    def __init__(self, rpm: int | None, rpd: int | None):
        self.interval = 60.0 / rpm if rpm else 0.0
        self.rpd = rpd
        self.used = 0
        self.next_at = 0.0
        self.lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self.lock:
            if self.rpd and self.used >= self.rpd:
                raise DailyLimitReached
            delay = self.next_at - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            self.next_at = time.monotonic() + self.interval
            self.used += 1


class DailyLimitReached(RuntimeError):
    pass


DAILY_QUOTA = re.compile(r"PerDay|per day|\bRPD\b|tokens per day", re.I)


class LLMClient:
    retries = 6

    def __init__(self, model: str, concurrency: int, rpm: int | None = None, rpd: int | None = None,
                 extra: dict | None = None):
        self.model = model
        self.extra = extra or {}
        self.sem = asyncio.Semaphore(concurrency)
        self.limiter = RateLimiter(rpm, rpd)
        self.http = httpx.AsyncClient(timeout=180)
        self.exhausted = False

    async def chat(self, system: str, user: str, temperature: float, max_tokens: int) -> str:
        async with self.sem:
            for attempt in range(self.retries):
                if self.exhausted:
                    raise DailyLimitReached
                await self.limiter.wait()
                try:
                    return await self._call(system, user, temperature, max_tokens)
                except httpx.HTTPStatusError as e:
                    code = e.response.status_code
                    # A daily quota will not recover by retrying; stop this client until the next run.
                    if code == 429 and DAILY_QUOTA.search(e.response.text):
                        self.exhausted = True
                        msg = " ".join(e.response.text.split())[:300]
                        print(f"[{self.provider}] daily quota reached for {self.model}: {msg}")
                        raise DailyLimitReached from None
                    if code not in (429, 500, 502, 503, 504) or attempt == self.retries - 1:
                        raise RuntimeError(f"{self.provider} {self.model}: HTTP {code}") from None
                except httpx.TransportError:
                    if attempt == self.retries - 1:
                        raise
                await asyncio.sleep(min(60, 2 ** attempt) + random.random())
        raise RuntimeError("unreachable")

    async def close(self) -> None:
        await self.http.aclose()


class Ollama(LLMClient):
    provider = "ollama"

    async def _call(self, system, user, temperature, max_tokens):
        r = await self.http.post(f"{env('OLLAMA_HOST', 'http://localhost:11434')}/api/chat", json={
            "model": self.model,
            "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": temperature, "num_predict": max_tokens},
            **self.extra,
        })
        r.raise_for_status()
        return r.json()["message"]["content"]


class Groq(LLMClient):
    provider = "groq"

    async def _call(self, system, user, temperature, max_tokens):
        r = await self.http.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {secret('GROQ_API_KEY')}"},
            json={
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                **self.extra,
            },
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


class Gemini(LLMClient):
    provider = "gemini"

    async def _call(self, system, user, temperature, max_tokens):
        # Key goes in a header so it never shows up in URLs or error messages.
        r = await self.http.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            headers={"x-goog-api-key": secret("GEMINI_API_KEY")},
            json={
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                    "thinkingConfig": {"thinkingBudget": 0},
                },
            },
        )
        r.raise_for_status()
        parts = r.json()["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)


PROVIDERS = {"ollama": Ollama, "groq": Groq, "gemini": Gemini}


def make_client(spec: dict) -> LLMClient:
    cls = PROVIDERS[spec["provider"]]
    return cls(spec["model"], spec["concurrency"], spec.get("rpm"), spec.get("rpd"), spec.get("extra"))
