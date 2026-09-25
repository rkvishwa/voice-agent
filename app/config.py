"""Environment configuration loaded via python-dotenv."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_REQUIRED_VARS = (
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_DEPLOYMENT",
    "AZURE_SPEECH_KEY",
    "AZURE_SPEECH_REGION",
)


def _clean(value: str) -> str:
    """Strip whitespace and inline comments accidentally copied into .env values."""
    cleaned = value.strip()
    if "#" in cleaned:
        cleaned = cleaned.split("#", 1)[0].strip()
    return cleaned


def _require(name: str) -> str:
    value = _clean(os.getenv(name, ""))
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Copy .env.example to .env and set your Azure credentials."
        )
    return value


def _optional(name: str, default: str = "") -> str:
    return _clean(os.getenv(name, default))


AZURE_OPENAI_ENDPOINT: str = _require("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY: str = _require("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT: str = _require("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION: str = _optional("AZURE_OPENAI_API_VERSION", "2024-06-01")

AZURE_SPEECH_KEY: str = _require("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION: str = _require("AZURE_SPEECH_REGION").lower()
AZURE_SPEECH_LANGUAGE: str = _optional("AZURE_SPEECH_LANGUAGE", "en-US")
AZURE_SPEECH_PHRASES: list[str] = [
    phrase.strip()
    for phrase in _optional("AZURE_SPEECH_PHRASES", "").split(",")
    if phrase.strip()
]

HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))
MODELS_DIR: Path = Path(os.getenv("MODELS_DIR", "/opt/voice-agent/models"))

KOKORO_MODEL_PATH: Path = MODELS_DIR / "kokoro-v0_19.onnx"
KOKORO_VOICES_PATH: Path = MODELS_DIR / "voices.bin"
KOKORO_VOICE: str = _optional("KOKORO_VOICE", "")
KOKORO_SPEED: float = float(_optional("KOKORO_SPEED", "1.05"))

# Audio / barge-in VAD
SAMPLE_RATE: int = 16000
FRAME_MS: int = int(os.getenv("FRAME_MS", "20"))
FRAME_SAMPLES: int = int(SAMPLE_RATE * FRAME_MS / 1000)
RMS_SPEECH_START_THRESHOLD: float = float(os.getenv("RMS_SPEECH_START_THRESHOLD", "0.035"))
BARGE_IN_SUSTAINED_FRAMES: int = int(os.getenv("BARGE_IN_SUSTAINED_FRAMES", "10"))
