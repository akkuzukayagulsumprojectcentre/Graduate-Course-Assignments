#!/usr/bin/env bash
# Re-runs the generate stage until no generator has jobs left.
# Free-tier daily token buckets refill gradually, so each pass picks up where the last one stopped.
# Usage: bash experiments/resume_generation.sh [python] [sleep_seconds]
set -u
PY=${1:-python}
WAIT=${2:-3600}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LOG=${LLMCTI_CACHE:-$HOME/.cache/llm-cti}/generate.log
cd "$ROOT"

while true; do
    left=$("$PY" -c "
from src.agents import pending_counts
from src.settings import load_config
c = pending_counts(load_config('configs/generation.yaml'))
print(' '.join(f'{g}={n}' for g, n in c.items()), sum(c.values()))
")
    echo "$(date '+%F %T') jobs left: ${left% *}"
    [ "${left##* }" -eq 0 ] && { echo "all generators complete"; break; }
    PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 "$PY" -m src.cli generate --config configs/generation.yaml >> "$LOG" 2>&1
    sleep "$WAIT"
done
