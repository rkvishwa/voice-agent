"""Azure neural text-to-speech synthesis."""

import asyncio
import logging
from typing import Any

import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.speech import ResultReason

from app.config import AZURE_SPEECH_KEY, AZURE_SPEECH_REGION, AZURE_SPEECH_VOICE

logger = logging.getLogger(__name__)


def audio_from_synthesis_result(result: Any) -> bytes:
    """Extract WAV bytes from a Speech SDK synthesis result or raise with details."""
    if result.reason == ResultReason.SynthesizingAudioCompleted:
        audio = result.audio_data
        if not audio:
            raise RuntimeError("Azure Speech TTS returned empty audio")
        return bytes(audio)

    if result.reason == ResultReason.Canceled:
        details = result.cancellation_details
        if details is not None:
            message = details.error_details or str(details.reason)
        else:
            message = "unknown cancellation"
        raise RuntimeError(f"Azure Speech TTS canceled: {message}")

    raise RuntimeError(f"Azure Speech TTS failed: {result.reason}")


def _synth_sync(text: str) -> bytes:
    speech_config = speechsdk.SpeechConfig(
        subscription=AZURE_SPEECH_KEY,
        region=AZURE_SPEECH_REGION,
    )
    speech_config.speech_synthesis_voice_name = AZURE_SPEECH_VOICE
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm,
    )

    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config,
        audio_config=None,
    )
    result = synthesizer.speak_text_async(text).get()
    audio = audio_from_synthesis_result(result)
    logger.debug("Azure TTS synthesized %d bytes for %r", len(audio), text[:80])
    return audio


async def synth_chunk(text: str) -> bytes:
    """Synthesize text into in-memory WAV bytes (24 kHz mono)."""
    cleaned = text.strip()
    if not cleaned:
        return b""
    return await asyncio.to_thread(_synth_sync, cleaned)
