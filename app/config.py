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


def _require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Copy .env.example to .env and set your Azure credentials."
        )
    return value


AZURE_OPENAI_ENDPOINT: str = _require("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY: str = _require("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT: str = _require("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")

AZURE_SPEECH_KEY: str = _require("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION: str = _require("AZURE_SPEECH_REGION")
AZURE_SPEECH_LANGUAGE: str = os.getenv("AZURE_SPEECH_LANGUAGE", "en-US")
AZURE_SPEECH_PHRASES: list[str] = [
    phrase.strip()
    for phrase in os.getenv("AZURE_SPEECH_PHRASES", "").split(",")
    if phrase.strip()
]

HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))
MODELS_DIR: Path = Path(os.getenv("MODELS_DIR", "/opt/voice-agent/models"))

KOKORO_MODEL_PATH: Path = MODELS_DIR / "kokoro-v0_19.onnx"
KOKORO_VOICES_PATH: Path = MODELS_DIR / "voices.bin"

# Audio / barge-in VAD
SAMPLE_RATE: int = 16000
FRAME_MS: int = int(os.getenv("FRAME_MS", "20"))
FRAME_SAMPLES: int = int(SAMPLE_RATE * FRAME_MS / 1000)
RMS_SPEECH_START_THRESHOLD: float = float(os.getenv("RMS_SPEECH_START_THRESHOLD", "0.035"))
