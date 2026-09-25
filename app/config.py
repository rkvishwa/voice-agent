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

# Audio / VAD
SAMPLE_RATE: int = 16000
FRAME_MS: int = int(os.getenv("FRAME_MS", "20"))
FRAME_SAMPLES: int = int(SAMPLE_RATE * FRAME_MS / 1000)  # 320 @ 20 ms
RMS_SPEECH_START_THRESHOLD: float = float(os.getenv("RMS_SPEECH_START_THRESHOLD", "0.035"))
RMS_SPEECH_CONTINUE_THRESHOLD: float = float(
    os.getenv("RMS_SPEECH_CONTINUE_THRESHOLD", "0.020")
)
PRE_ROLL_MS: int = int(os.getenv("PRE_ROLL_MS", "300"))
PRE_ROLL_SAMPLES: int = int(SAMPLE_RATE * PRE_ROLL_MS / 1000)
END_SILENCE_MS: int = int(os.getenv("END_SILENCE_MS", "650"))
SILENCE_FRAMES_FOR_END: int = max(1, int(END_SILENCE_MS / FRAME_MS))
MIN_TURN_SECONDS: float = float(os.getenv("MIN_TURN_SECONDS", "0.4"))

# ASR (Faster-Whisper)
ASR_MODEL: str = os.getenv("ASR_MODEL", "small.en")
ASR_BEAM_SIZE: int = int(os.getenv("ASR_BEAM_SIZE", "3"))
ASR_CPU_THREADS: int = int(os.getenv("ASR_CPU_THREADS", "2"))
ASR_COMPUTE_TYPE: str = os.getenv("ASR_COMPUTE_TYPE", "int8")
