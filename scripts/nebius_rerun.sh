#!/usr/bin/env bash
# Re-run every eval whose numbers the docs quote, on Nebius Token Factory (or MODEL_PROVIDER=nvidia for dev).
# Needs NEBIUS_API_KEY in .env. Each result file records its provider; every call is in out/model_calls.jsonl.
#   scripts/nebius_rerun.sh            # ~20 min, a few $ at most
set -euo pipefail
cd "$(dirname "$0")/.."
export MODEL_PROVIDER="${MODEL_PROVIDER:-nebius}"
PY="uv run --python 3.12 --with-requirements requirements.lock --with matplotlib python"  # pinned, as in scripts/demo.sh
SCAN="${SCAN:-data/lidar/lidar-20261023-172353.zip}"

$PY -m agent.nebius | tail -6                    # the model ids: fix [nebius.nebius] in config.toml if one is MISSING
$PY eval/commands.py --runs keywords,ultra,lightning
[ -f "$SCAN" ] && $PY eval/vl_rooms.py "$SCAN" --ask view --refs 2
[ -f "$SCAN" ] && for s in 1 2 3 4; do $PY eval/lowview_test.py "$SCAN" --all-frames --seed $s; done
$PY -m pytest -q eval/adversarial/test_language.py
echo "done: out/commands/results.json, out/vl_rooms/, out/lowview/ (provider: $MODEL_PROVIDER)"
