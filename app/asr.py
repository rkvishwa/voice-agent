"""Azure AI Speech streaming speech-to-text."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Optional

import azure.cognitiveservices.speech as speechsdk

from app.config import (
    AZURE_SPEECH_KEY,
    AZURE_SPEECH_LANGUAGE,
    AZURE_SPEECH_PHRASES,
    AZURE_SPEECH_REGION,
    SAMPLE_RATE,
)

logger = logging.getLogger(__name__)

PartialCallback = Callable[[str], Awaitable[None]]
FinalCallback = Callable[[str], Awaitable[None]]
ErrorCallback = Callable[[str], Awaitable[None]]


def build_partial_event(text: str, turn_id: int) -> dict:
    """Build a WebSocket payload for a live partial transcript."""
    return {
        "type": "transcript_partial",
        "role": "user",
        "text": text,
        "turn_id": turn_id,
    }


def build_final_event(text: str, turn_id: int) -> dict:
    """Build a WebSocket payload for a finalized user transcript."""
    return {
        "type": "transcript",
        "role": "user",
        "text": text,
        "turn_id": turn_id,
        "final": True,
    }


class AzureSpeechSession:
    """Stream PCM16 audio to Azure Speech and emit partial/final transcripts."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        on_partial: PartialCallback,
        on_final: FinalCallback,
        on_error: ErrorCallback,
    ) -> None:
        self._loop = loop
        self._on_partial = on_partial
        self._on_final = on_final
        self._on_error = on_error
        self._closed = False
        self._recognizer: Optional[speechsdk.SpeechRecognizer] = None
        self._push_stream: Optional[speechsdk.audio.PushAudioInputStream] = None

    def _schedule(self, coro: Awaitable[None]) -> None:
        if self._closed:
            return
        asyncio.run_coroutine_threadsafe(coro, self._loop)

    def _start_sync(self) -> None:
        speech_config = speechsdk.SpeechConfig(
            subscription=AZURE_SPEECH_KEY,
            region=AZURE_SPEECH_REGION,
        )
        speech_config.speech_recognition_language = AZURE_SPEECH_LANGUAGE

        stream_format = speechsdk.audio.AudioStreamFormat(
            samples_per_second=SAMPLE_RATE,
            bits_per_sample=16,
            channels=1,
        )
        self._push_stream = speechsdk.audio.PushAudioInputStream(stream_format)
        audio_config = speechsdk.audio.AudioConfig(stream=self._push_stream)

        self._recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config,
        )

        if AZURE_SPEECH_PHRASES:
            phrase_list = speechsdk.PhraseListGrammar.from_recognizer(self._recognizer)
            for phrase in AZURE_SPEECH_PHRASES:
                phrase_list.addPhrase(phrase)

        self._recognizer.recognizing.connect(self._handle_recognizing)
        self._recognizer.recognized.connect(self._handle_recognized)
        self._recognizer.canceled.connect(self._handle_canceled)

        self._recognizer.start_continuous_recognition_async().get()
        logger.info(
            "Azure Speech session started region=%s language=%s",
            AZURE_SPEECH_REGION,
            AZURE_SPEECH_LANGUAGE,
        )

    async def start(self) -> None:
        await asyncio.to_thread(self._start_sync)

    def push_audio(self, pcm_bytes: bytes) -> None:
        if self._closed or not self._push_stream or not pcm_bytes:
            return
        self._push_stream.write(bytes(pcm_bytes))

    def _handle_recognizing(self, evt: speechsdk.SpeechRecognitionEventArgs) -> None:
        if self._closed:
            return
        if evt.result.reason != speechsdk.ResultReason.RecognizingSpeech:
            return
        text = evt.result.text.strip()
        if text:
            self._schedule(self._on_partial(text))

    def _handle_recognized(self, evt: speechsdk.SpeechRecognitionEventArgs) -> None:
        if self._closed:
            return
        if evt.result.reason != speechsdk.ResultReason.RecognizedSpeech:
            return
        text = evt.result.text.strip()
        if text:
            logger.info("Azure recognized: %s", text)
            self._schedule(self._on_final(text))

    def _handle_canceled(self, evt: speechsdk.SpeechRecognitionCanceledEventArgs) -> None:
        if self._closed:
            return
        reason = evt.reason
        if reason == speechsdk.CancellationReason.Error:
            message = evt.error_details or "Azure Speech recognition error"
            logger.error("Azure Speech canceled (error): %s", message)
            self._schedule(self._on_error(message))
            return
        reason_name = getattr(reason, "name", str(reason))
        message = f"Azure Speech recognition stopped: {reason_name}"
        if evt.error_details:
            message = f"{message} — {evt.error_details}"
        logger.warning("Azure Speech canceled: %s", message)
        self._schedule(self._on_error(message))

    def _close_sync(self) -> None:
        if self._recognizer is not None:
            try:
                self._recognizer.stop_continuous_recognition_async().get()
            except Exception:
                logger.exception("failed stopping Azure Speech recognizer")
        if self._push_stream is not None:
            try:
                self._push_stream.close()
            except Exception:
                logger.exception("failed closing Azure Speech push stream")

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await asyncio.to_thread(self._close_sync)
        logger.info("Azure Speech session closed")
