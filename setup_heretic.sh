#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] uv not found. Install it: https://docs.astral.sh/uv/"
  exit 1
fi

cd heretic-2.1.0
uv sync
echo "[setup] Heretic ready."
