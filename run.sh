#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -d venv ]]; then
    echo "ERROR: venv not found. Run ./deploy.sh first." >&2
    exit 1
fi

# shellcheck disable=SC1091
source venv/bin/activate

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo "==> Starting voice agent on http://${HOST}:${PORT}"
exec uvicorn app.server:app --host "$HOST" --port "$PORT"
