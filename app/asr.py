"""Faster-Whisper speech-to-text with serialized preview and final decoders."""

import asyncio
import logging
import time
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel

from app.config import (
    ASR_COMPUTE_TYPE,
    ASR_CPU_THREADS,
    ASR_FINAL_BEAM_SIZE,
    ASR_FINAL_MODEL,
    ASR_INITIAL_PROMPT,
    ASR_PREVIEW_BEAM_SIZE,
    ASR_PREVIEW_MODEL,
    SAMPLE_RATE,
)

logger = logging.getLogger(__name__)

_preview_model: Optional[WhisperModel] = None
_final_model: Optional[WhisperModel] = None


class ASREngine:
    """Serialize CPU inference; final decode takes priority over preview."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._final_active = False

    def _get_preview_model(self) -> WhisperModel:
        global _preview_model
        if _preview_model is None:
            logger.info(
                "loading ASR preview model=%s compute=%s threads=%d",
                ASR_PREVIEW_MODEL,
                ASR_COMPUTE_TYPE,
                ASR_CPU_THREADS,
            )
            _preview_model = WhisperModel(
                ASR_PREVIEW_MODEL,
                device="cpu",
                compute_type=ASR_COMPUTE_TYPE,
                cpu_threads=ASR_CPU_THREADS,
            )
        return _preview_model

    def _get_final_model(self) -> WhisperModel:
        global _final_model
        if _final_model is None:
            logger.info(
                "loading ASR final model=%s compute=%s threads=%d",
                ASR_FINAL_MODEL,
                ASR_COMPUTE_TYPE,
                ASR_CPU_THREADS,
            )
            _final_model = WhisperModel(
                ASR_FINAL_MODEL,
                device="cpu",
                compute_type=ASR_COMPUTE_TYPE,
                cpu_threads=ASR_CPU_THREADS,
            )
        return _final_model

    def _transcribe_sync(
        self,
        audio: np.ndarray,
        model: WhisperModel,
        beam_size: int,
        label: str,
    ) -> str:
        audio_duration = audio.size / SAMPLE_RATE if audio.size else 0.0
        start = time.monotonic()

        kwargs: dict = {
            "beam_size": beam_size,
            "language": "en",
            "condition_on_previous_text": False,
        }
        if ASR_INITIAL_PROMPT:
            kwargs["initial_prompt"] = ASR_INITIAL_PROMPT

        segments, _ = model.transcribe(audio, **kwargs)
        text = " ".join(segment.text.strip() for segment in segments).strip()

        elapsed = time.monotonic() - start
        rtf = elapsed / audio_duration if audio_duration > 0 else 0.0
        logger.info(
            "%s decode duration=%.2fs audio=%.2fs rtf=%.2f chars=%d",
            label,
            elapsed,
            audio_duration,
            rtf,
            len(text),
        )
        return text

    async def transcribe_preview(self, audio: np.ndarray) -> str:
        """Fast partial transcription for live captions."""
        if self._final_active or audio.size == 0:
            return ""
        async with self._lock:
            if self._final_active:
                return ""
            return await asyncio.to_thread(
                self._transcribe_sync,
                audio,
                self._get_preview_model(),
                ASR_PREVIEW_BEAM_SIZE,
                "preview",
            )

    async def transcribe_final(self, audio: np.ndarray) -> str:
        """Higher-accuracy transcription after end-of-turn."""
        if audio.size == 0:
            return ""
        self._final_active = True
        try:
            async with self._lock:
                return await asyncio.to_thread(
                    self._transcribe_sync,
                    audio,
                    self._get_final_model(),
                    ASR_FINAL_BEAM_SIZE,
                    "final",
                )
        finally:
            self._final_active = False


_engine = ASREngine()


async def transcribe_preview(audio: np.ndarray) -> str:
    return await _engine.transcribe_preview(audio)


async def transcribe_final(audio: np.ndarray) -> str:
    return await _engine.transcribe_final(audio)


# Backward-compatible alias
async def transcribe(audio: np.ndarray) -> str:
    return await transcribe_final(audio)
