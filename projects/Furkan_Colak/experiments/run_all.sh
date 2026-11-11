#!/usr/bin/env bash
# Runs every stage in order. Generation needs Ollama and the API keys in .env. Excerpts of other publishers are
# copyrighted and not committed, so run collect_ext first; `--from build` then reproduces all results from data/.
# Usage: bash experiments/run_all.sh [--from <stage>] [python]
set -euo pipefail
FROM=collect
if [ "${1:-}" = "--from" ]; then FROM=$2; shift 2; fi
PY=${1:-python}
cd "$(dirname "$0")/.."

stages=(collect collect_ext generate variants build analyze zeroshot detect evaluate calibrate compare robust errors cost report)
started=0
for st in "${stages[@]}"; do
    [ "$st" = "$FROM" ] && started=1
    [ "$started" -eq 0 ] && continue
    case $st in
        collect|collect_ext|generate|variants) cfg=configs/generation.yaml ;;
        *) cfg=configs/experiments.yaml ;;
    esac
    echo "=== $st ($(date '+%H:%M'))"
    PYTHONIOENCODING=utf-8 "$PY" -m src.cli "$st" --config "$cfg"
done
"$PY" -m pytest -q tests
