"""Kokoro-82M ONNX text-to-speech wrapper."""

import asyncio
import inspect
import io
from typing import Optional

import soundfile as sf
from kokoro_onnx import Kokoro

from app.config import KOKORO_MODEL_PATH, KOKORO_VOICES_PATH

_kokoro: Optional[Kokoro] = None
_voice: Optional[str] = None
_create_kwargs: dict = {}


def _init_kokoro() -> Kokoro:
    global _kokoro, _voice, _create_kwargs

    if _kokoro is not None:
        return _kokoro

    _kokoro = Kokoro(str(KOKORO_MODEL_PATH), str(KOKORO_VOICES_PATH))

    preferred = ("af_heart", "am_adam")
    available = set(_kokoro.get_voices())
    _voice = next((v for v in preferred if v in available), None)
    if _voice is None and available:
        _voice = sorted(available)[0]
    if _voice is None:
        raise RuntimeError("No Kokoro voices found in voices.bin")

    sig = inspect.signature(_kokoro.create)
    if "lang" in sig.parameters:
        _create_kwargs["lang"] = "en-us"

    return _kokoro


def _synth_sync(text: str) -> bytes:
    kokoro = _init_kokoro()
    samples, sample_rate = kokoro.create(
        text,
        voice=_voice,
        speed=1.05,
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
