"""Local Kokoro-82M ONNX text-to-speech."""

import asyncio
import inspect
import io
import logging
from typing import Optional

import soundfile as sf
from kokoro_onnx import Kokoro

from app.config import (
    KOKORO_MODEL_PATH,
    KOKORO_SPEED,
    KOKORO_VOICE,
    KOKORO_VOICES_PATH,
)

logger = logging.getLogger(__name__)

_kokoro: Optional[Kokoro] = None
_voice: Optional[str] = None
_create_kwargs: dict = {}


def pick_kokoro_voice(preferred: tuple[str, ...], available: set[str]) -> str:
    """Choose the first preferred voice present in the voices bundle."""
    for name in preferred:
        if name in available:
            return name
    if not available:
        raise RuntimeError("No Kokoro voices found in voices.bin")
    return sorted(available)[0]


def _init_kokoro() -> Kokoro:
    global _kokoro, _voice, _create_kwargs

    if _kokoro is not None:
        return _kokoro

    if not KOKORO_MODEL_PATH.is_file():
        raise RuntimeError(
            f"Kokoro model not found at {KOKORO_MODEL_PATH}. "
            "Run scripts/download_models.sh or deploy.sh."
        )
    if not KOKORO_VOICES_PATH.is_file():
        raise RuntimeError(
            f"Kokoro voices not found at {KOKORO_VOICES_PATH}. "
            "Run scripts/download_models.sh or deploy.sh."
        )

    _kokoro = Kokoro(str(KOKORO_MODEL_PATH), str(KOKORO_VOICES_PATH))
    available = set(_kokoro.get_voices())

    if KOKORO_VOICE:
        if KOKORO_VOICE not in available:
            raise RuntimeError(
                f"KOKORO_VOICE={KOKORO_VOICE!r} not in voices.bin "
                f"(available: {', '.join(sorted(available)[:12])}…)"
            )
        _voice = KOKORO_VOICE
    else:
        _voice = pick_kokoro_voice(("af_heart", "af_bella", "am_adam"), available)

    sig = inspect.signature(_kokoro.create)
    if "lang" in sig.parameters:
        _create_kwargs["lang"] = "en-us"

    logger.info("Kokoro TTS voice=%s speed=%s", _voice, KOKORO_SPEED)
    return _kokoro


def _synth_sync(text: str) -> bytes:
    kokoro = _init_kokoro()
    samples, sample_rate = kokoro.create(
        text,
        voice=_voice,
        speed=KOKORO_SPEED,
        **_create_kwargs,
    )
    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format="WAV")
    return buffer.getvalue()


async def synth_chunk(text: str) -> bytes:
    """Synthesize text into in-memory WAV bytes."""
    cleaned = text.strip()
    if not cleaned:
        return b""
    return await asyncio.to_thread(_synth_sync, cleaned)
