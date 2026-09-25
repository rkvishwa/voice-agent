"""Faster-Whisper speech-to-text wrapper."""

import asyncio
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel

_model: Optional[WhisperModel] = None


def get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(
            "base.en",
            device="cpu",
            compute_type="int8",
            cpu_threads=2,
        )
    return _model


def _transcribe_sync(audio: np.ndarray) -> str:
    model = get_model()
    segments, _ = model.transcribe(
        audio,
        beam_size=1,
        language="en",
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


async def transcribe(audio: np.ndarray) -> str:
    """Transcribe a float32 audio array asynchronously."""
    return await asyncio.to_thread(_transcribe_sync, audio)
