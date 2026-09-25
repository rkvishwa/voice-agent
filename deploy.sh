#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

INSTALL_DIR="/opt/voice-agent"
IS_ROOT=false
if [[ $EUID -eq 0 ]]; then
    IS_ROOT=true
fi

# ---------------------------------------------------------------------------
# 1. Validate .env
# ---------------------------------------------------------------------------
if [[ ! -f .env ]]; then
    if [[ -f .env.example ]]; then
        echo "==> Creating .env from .env.example — fill in your Azure credentials"
        cp .env.example .env
    else
        echo "ERROR: No .env file found. Create one with Azure credentials." >&2
        exit 1
    fi
fi

# shellcheck disable=SC1091
source .env

MISSING=()
for var in AZURE_OPENAI_ENDPOINT AZURE_OPENAI_API_KEY AZURE_OPENAI_DEPLOYMENT; do
    if [[ -z "${!var:-}" ]]; then
        MISSING+=("$var")
    fi
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "ERROR: Missing required environment variables in .env:" >&2
    for v in "${MISSING[@]}"; do
        echo "  - $v" >&2
    done
    echo "Edit .env and re-run ./deploy.sh" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. System packages (when root)
# ---------------------------------------------------------------------------
if $IS_ROOT; then
    echo "==> Installing system packages"
    apt-get update -qq
    apt-get install -y -qq python3-venv python3-pip espeak-ng libsndfile1 curl
fi

# ---------------------------------------------------------------------------
# 3. Python version check
# ---------------------------------------------------------------------------
PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [[ "$major" -eq 3 && "$minor" -ge 11 && "$minor" -le 13 ]]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [[ -z "$PYTHON" ]]; then
    echo "ERROR: Python 3.11–3.13 required (kokoro-onnx constraint)." >&2
    exit 1
fi
echo "==> Using $PYTHON ($("$PYTHON" --version))"

# ---------------------------------------------------------------------------
# 4. Set MODELS_DIR in .env
# ---------------------------------------------------------------------------
if $IS_ROOT; then
    TARGET_MODELS_DIR="$INSTALL_DIR/models"
else
    TARGET_MODELS_DIR="$SCRIPT_DIR/models"
fi

if ! grep -q '^MODELS_DIR=' .env 2>/dev/null; then
    echo "MODELS_DIR=$TARGET_MODELS_DIR" >> .env
else
    sed -i "s|^MODELS_DIR=.*|MODELS_DIR=$TARGET_MODELS_DIR|" .env
fi
export MODELS_DIR="$TARGET_MODELS_DIR"

# ---------------------------------------------------------------------------
# 5. Virtual environment + pip install
# ---------------------------------------------------------------------------
echo "==> Setting up Python virtual environment"
if [[ ! -d venv ]]; then
    "$PYTHON" -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "==> Python dependencies installed"

# ---------------------------------------------------------------------------
# 6. Download models + prefetch Whisper
# ---------------------------------------------------------------------------
bash scripts/download_models.sh

echo "==> Prefetching Faster-Whisper ${ASR_MODEL:-small.en} model (first run may take a minute)"
PYTHONPATH="$SCRIPT_DIR" python "$SCRIPT_DIR/scripts/prefetch_asr.py"

# ---------------------------------------------------------------------------
# 7. Systemd install (root) or foreground start (user)
# ---------------------------------------------------------------------------
if $IS_ROOT; then
    echo "==> Installing to $INSTALL_DIR"
    mkdir -p "$INSTALL_DIR"
    rsync -a --delete \
        --exclude venv \
        --exclude .git \
        --exclude '__pycache__' \
        "$SCRIPT_DIR/" "$INSTALL_DIR/"

    if [[ ! -d "$INSTALL_DIR/venv" ]]; then
        "$PYTHON" -m venv "$INSTALL_DIR/venv"
    fi
    "$INSTALL_DIR/venv/bin/pip" install --upgrade pip -q
    "$INSTALL_DIR/venv/bin/pip" install -r "$INSTALL_DIR/requirements.txt" -q

    cp "$SCRIPT_DIR/.env" "$INSTALL_DIR/.env"
    sed -i "s|^MODELS_DIR=.*|MODELS_DIR=$INSTALL_DIR/models|" "$INSTALL_DIR/.env"

    MODELS_DIR="$INSTALL_DIR/models" bash "$INSTALL_DIR/scripts/download_models.sh"

    echo "==> Prefetching Whisper in install dir"
    PYTHONPATH="$INSTALL_DIR" "$INSTALL_DIR/venv/bin/python" "$INSTALL_DIR/scripts/prefetch_asr.py"

    bash "$INSTALL_DIR/scripts/setup_systemd.sh"

    echo ""
    echo "==> App deployed. For HTTPS with nginx:"
    echo "    sudo DOMAIN=voice.metl.run CERTBOT_EMAIL=you@example.com $INSTALL_DIR/scripts/setup_nginx.sh"
else
    chmod +x run.sh scripts/*.sh
    echo ""
    echo "==> Deploy complete. Starting server in foreground ..."
    echo "    Open http://localhost:8000 in your browser"
    echo "    For production: sudo ./deploy.sh"
    echo ""
    exec ./run.sh
fi
