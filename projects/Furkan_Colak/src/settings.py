"""Paths, config loading and API keys. The only module that touches .env."""
import logging
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
# Kept outside the repo: OneDrive sync locks and moves files that are read or written often.
CACHE = Path(os.getenv("LLMCTI_CACHE", Path.home() / ".cache" / "llm-cti"))

load_dotenv(ROOT / ".env")

# httpx logs full request URLs at INFO level; keep them out of the console.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class MissingSecret(RuntimeError):
    pass


def secret(name: str) -> str:
    """Return an API key from the environment without ever echoing it."""
    value = os.getenv(name, "").strip()
    if not value:
        raise MissingSecret(f"{name} is not set. Copy .env.example to .env and fill it in.")
    return value


def env(name: str, default: str) -> str:
    return os.getenv(name, default).strip() or default


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
