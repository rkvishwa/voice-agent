#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODELS_DIR="${MODELS_DIR:-$PROJECT_ROOT/models}"
mkdir -p "$MODELS_DIR"

KOKORO_URL="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx"
VOICES_URL="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin"

download_if_missing() {
    local url="$1"
    local dest="$2"
    if [[ -f "$dest" ]]; then
        echo "  [skip] $(basename "$dest") already exists"
        return
    fi
    echo "  [download] $(basename "$dest") ..."
    curl -fL --progress-bar -o "$dest" "$url"
}

echo "==> Downloading Kokoro ONNX models to $MODELS_DIR"
download_if_missing "$KOKORO_URL" "$MODELS_DIR/kokoro-v0_19.onnx"
download_if_missing "$VOICES_URL" "$MODELS_DIR/voices.bin"
echo "==> Model download complete"
