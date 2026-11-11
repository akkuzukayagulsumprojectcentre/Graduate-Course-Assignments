import re
import subprocess

from src.settings import ROOT

PATTERNS = [
    re.compile(r"gsk_[A-Za-z0-9]{20,}"),          # Groq
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),       # Google
    re.compile(r"(?<![A-Za-z0-9])sk-(ant-)?[A-Za-z0-9_\-]{20,}"),  # OpenAI / Anthropic ("ksk-rollover" is not a key)
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),    # GitHub
]


def repo_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    return [line for line in out.stdout.splitlines() if line]


def test_env_is_ignored():
    r = subprocess.run(["git", "check-ignore", "-q", ".env"], cwd=ROOT)
    assert r.returncode == 0, ".env must be listed in .gitignore"


def test_no_keys_in_repo():
    hits = []
    for rel in repo_files():
        path = ROOT / rel
        if path.suffix in {".pdf", ".png", ".parquet", ".pt"} or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pat in PATTERNS:
            if pat.search(text):
                hits.append(f"{rel}: {pat.pattern}")
    assert not hits, "possible secrets found:\n" + "\n".join(hits)
