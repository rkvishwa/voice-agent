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
    STT_SEGMENTATION_SILENCE_MS,
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


def build_speech_ready_event() -> dict:
    """Build a WebSocket payload indicating Azure Speech is ready."""
    return {"type": "speech_ready"}


def build_startup_error_event(message: str) -> dict:
    """Build a WebSocket payload for Azure Speech startup failure."""
    return {"type": "error", "message": message}


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
        self._started = False
        self._recognizer: Optional[speechsdk.SpeechRecognizer] = None
        self._push_stream: Optional[speechsdk.audio.PushAudioInputStream] = None
        self._bytes_written = 0

    @property
    def is_started(self) -> bool:
        return self._started

    def _schedule(self, coro: Awaitable[None]) -> None:
        if self._closed:
            return
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)

        def _log_error(fut: asyncio.Future) -> None:
            try:
                fut.result()
            except Exception:
                logger.exception("Azure Speech callback failed")

        future.add_done_callback(_log_error)

    def _start_sync(self) -> None:
        logger.info(
            "starting Azure Speech region=%s language=%s",
            AZURE_SPEECH_REGION,
            AZURE_SPEECH_LANGUAGE,
        )
        speech_config = speechsdk.SpeechConfig(
            subscription=AZURE_SPEECH_KEY,
            region=AZURE_SPEECH_REGION,
        )
        speech_config.speech_recognition_language = AZURE_SPEECH_LANGUAGE
        speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs,
            "30000",
        )
        speech_config.set_property(
            speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs,
            str(STT_SEGMENTATION_SILENCE_MS),
        )

        stream_format = speechsdk.audio.AudioStreamFormat(
            samples_per_second=SAMPLE_RATE,
            bits_per_sample=16,
            channels=1,
        )
        self._push_stream = speechsdk.audio.PushAudioInputStream(
            stream_format=stream_format,
        )
        audio_config = speechsdk.audio.AudioConfig(stream=self._push_stream)

        self._recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config,
        )

        if AZURE_SPEECH_PHRASES:
            phrase_list = speechsdk.PhraseListGrammar.from_recognizer(self._recognizer)
            for phrase in AZURE_SPEECH_PHRASES:
                phrase_list.addPhrase(phrase)

        self._recognizer.session_started.connect(self._handle_session_started)
        self._recognizer.session_stopped.connect(self._handle_session_stopped)
        self._recognizer.speech_start_detected.connect(self._handle_speech_start)
        self._recognizer.speech_end_detected.connect(self._handle_speech_end)
        self._recognizer.recognizing.connect(self._handle_recognizing)
        self._recognizer.recognized.connect(self._handle_recognized)
        self._recognizer.canceled.connect(self._handle_canceled)

        self._recognizer.start_continuous_recognition_async().get()
        self._started = True
        logger.info(
            "Azure Speech session started region=%s language=%s",
            AZURE_SPEECH_REGION,
            AZURE_SPEECH_LANGUAGE,
        )

    async def start(self) -> None:
        if self._started:
            return
        await asyncio.to_thread(self._start_sync)

    def push_audio(self, pcm_bytes: bytes) -> None:
        if self._closed or not self._push_stream or not pcm_bytes:
            return
        payload = bytes(pcm_bytes)
        self._push_stream.write(payload)
        self._bytes_written += len(payload)

    def _handle_session_started(self, evt: speechsdk.SessionEventArgs) -> None:
        logger.info("Azure Speech session_started id=%s", evt.session_id)

    def _handle_session_stopped(self, evt: speechsdk.SessionEventArgs) -> None:
        logger.info(
            "Azure Speech session_stopped id=%s bytes_written=%d",
            evt.session_id,
            self._bytes_written,
        )

    def _handle_speech_start(self, evt: speechsdk.RecognitionEventArgs) -> None:
        logger.info("Azure Speech VAD: speech_start offset=%s", evt.offset)

    def _handle_speech_end(self, evt: speechsdk.RecognitionEventArgs) -> None:
        logger.info("Azure Speech VAD: speech_end offset=%s", evt.offset)

    def _handle_recognizing(self, evt: speechsdk.SpeechRecognitionEventArgs) -> None:
        if self._closed:
            return
        reason = evt.result.reason
        text = evt.result.text.strip()
        if reason == speechsdk.ResultReason.RecognizingSpeech and text:
            logger.info("Azure recognizing: %s", text)
            self._schedule(self._on_partial(text))
        elif reason != speechsdk.ResultReason.RecognizingSpeech:
            logger.debug("Azure recognizing skipped reason=%s", reason)

    def _handle_recognized(self, evt: speechsdk.SpeechRecognitionEventArgs) -> None:
        if self._closed:
            return
        reason = evt.result.reason
        text = evt.result.text.strip()
        if reason == speechsdk.ResultReason.RecognizedSpeech and text:
            logger.info("Azure recognized: %s", text)
            self._schedule(self._on_final(text))
            return
        if reason == speechsdk.ResultReason.NoMatch:
            logger.info("Azure no match (bytes_written=%d)", self._bytes_written)
        else:
            logger.info(
                "Azure recognized event reason=%s text=%r bytes_written=%d",
                reason,
                text,
                self._bytes_written,
            )

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
        if self._recognizer is not None and self._started:
            try:
                self._recognizer.stop_continuous_recognition_async().get()
            except Exception:
                logger.exception("failed stopping Azure Speech recognizer")
        if self._push_stream is not None:
            try:
                self._push_stream.close()
            except Exception:
                logger.exception("failed closing Azure Speech push stream")
        self._recognizer = None
        self._push_stream = None
        self._started = False

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await asyncio.to_thread(self._close_sync)
        logger.info("Azure Speech session closed bytes_written=%d", self._bytes_written)
