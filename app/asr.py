"""Faster-Whisper speech-to-text wrapper."""

import asyncio
import logging
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel

from app.config import (
    ASR_BEAM_SIZE,
    ASR_COMPUTE_TYPE,
    ASR_CPU_THREADS,
    ASR_MODEL,
)

logger = logging.getLogger(__name__)

_model: Optional[WhisperModel] = None


def get_model() -> WhisperModel:
    global _model
    if _model is None:
        logger.info(
            "loading ASR model=%s compute=%s threads=%d beam=%d",
            ASR_MODEL,
            ASR_COMPUTE_TYPE,
            ASR_CPU_THREADS,
            ASR_BEAM_SIZE,
        )
        _model = WhisperModel(
            ASR_MODEL,
            device="cpu",
            compute_type=ASR_COMPUTE_TYPE,
            cpu_threads=ASR_CPU_THREADS,
        )
    return _model


def _transcribe_sync(audio: np.ndarray) -> str:
    model = get_model()
    segments, _ = model.transcribe(
        audio,
        beam_size=ASR_BEAM_SIZE,
        language="en",
        condition_on_previous_text=False,
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


async def transcribe(audio: np.ndarray) -> str:
    """Transcribe a float32 audio array asynchronously."""
    return await asyncio.to_thread(_transcribe_sync, audio)
