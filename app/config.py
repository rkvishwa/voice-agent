"""Environment configuration loaded via python-dotenv."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_REQUIRED_VARS = (
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_DEPLOYMENT",
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
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))
MODELS_DIR: Path = Path(os.getenv("MODELS_DIR", "/opt/voice-agent/models"))

KOKORO_MODEL_PATH: Path = MODELS_DIR / "kokoro-v0_19.onnx"
KOKORO_VOICES_PATH: Path = MODELS_DIR / "voices.bin"

SAMPLE_RATE: int = 16000
FRAME_SAMPLES: int = 1600  # 100 ms at 16 kHz
RMS_SPEECH_THRESHOLD: float = 0.035
SILENCE_FRAMES_FOR_END: int = 6
MIN_TURN_SECONDS: float = 0.4
