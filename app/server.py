"""FastAPI WebSocket orchestration for the voice agent."""

import asyncio
import json
import logging
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from app import llm, tts
from app.asr import (
    AzureSpeechSession,
    build_final_event,
    build_partial_event,
    build_speech_ready_event,
    build_startup_error_event,
)
from app.barge_in import BargeInGate
from app.config import (
    AZURE_SPEECH_LANGUAGE,
    AZURE_SPEECH_REGION,
    RMS_SPEECH_START_THRESHOLD,
    SAMPLE_RATE,
    SPECULATIVE_SILENCE_MS,
    TTS_BACKEND,
    TTS_SPEED,
)
from app.turn_text import pop_next_speech_chunk, strip_spoken_markup, utterances_match
from app.echo_guard import is_likely_agent_echo
from app.tts import (
    TtsBackend,
    default_voice_for_backend,
    get_tts_catalog,
    validate_tts_config,
)
from app.vad import amplify_pcm16, compute_rms, pcm16_to_float32

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

# ~1s of PCM16 silence at 16 kHz to flush Azure Speech utterance boundaries.
STT_FLUSH_SILENCE_BYTES = int(SAMPLE_RATE * 1.0) * 2

app = FastAPI(title="Voice Agent")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/tts")
async def tts_catalog():
    return get_tts_catalog()


class VoiceSession:
    """Per-connection state for Azure Speech streaming and barge-in."""

    def __init__(self, websocket: WebSocket, loop: asyncio.AbstractEventLoop):
        self.websocket = websocket
        self.history: list[dict] = []
        self.turn_id = 0
        self.active_task: asyncio.Task | None = None
        self.ai_busy = False
        self._closed = False
        self._speech_ready = False
        self._frames_received = 0
        self._last_audio_log_at = 0.0
        self._barge_in = BargeInGate()
        default_backend: TtsBackend = (
            TTS_BACKEND if TTS_BACKEND in ("kokoro", "azure") else "kokoro"
        )
        self.tts_backend: TtsBackend = default_backend
        self.tts_voice: str = default_voice_for_backend(default_backend)
        self.tts_speed: float = TTS_SPEED
        self._stt_muted: bool = False
        self._stt_ignore_finals_until: float = 0.0
        self._agent_audio_open: bool = False
        self._playback_epoch: int = 0
        self._recent_agent_text: str | None = None
        self._latest_partial_text: str = ""
        self._last_voice_at: float = 0.0
        self._turn_confirmed: bool = True
        self._speculative_user_text: str | None = None
        self._turn_started_at: float = 0.0
        self._logged_first_audio_for_turn: int | None = None
        self._last_barge_in_handled_at: float = 0.0
        self._discard_interrupt_audio: bool = False
        self.speech_session = AzureSpeechSession(
            loop=loop,
            on_partial=self._on_speech_partial,
            on_final=self._on_speech_final,
            on_error=self._on_speech_error,
        )

    async def start(self) -> None:
        logger.info(
            "Azure Speech startup region=%s language=%s",
            AZURE_SPEECH_REGION,
            AZURE_SPEECH_LANGUAGE,
        )
        await self.speech_session.start()
        self._speech_ready = True
        await self.send_json(build_speech_ready_event())

    async def send_json(self, payload: dict) -> None:
        if not self._closed:
            await self.websocket.send_json(payload)

    async def send_wav(self, wav_bytes: bytes) -> None:
        if wav_bytes and not self._closed:
            await self.websocket.send_bytes(wav_bytes)

    def _rollback_speculative_user_message(self) -> None:
        if (
            self.history
            and self.history[-1].get("role") == "user"
            and self._speculative_user_text
            and utterances_match(
                self.history[-1].get("content", ""),
                self._speculative_user_text,
            )
        ):
            self.history.pop()

    async def cancel_active_task(self, rollback_speculative: bool = False) -> None:
        if rollback_speculative:
            self._rollback_speculative_user_message()
            self._speculative_user_text = None
            self._turn_confirmed = True
        if self.active_task and not self.active_task.done():
            self.active_task.cancel()
            try:
                await self.active_task
            except asyncio.CancelledError:
                pass
        self.active_task = None
        self.ai_busy = False

    async def _cancel_speculative_turn(self) -> None:
        await self.cancel_active_task(rollback_speculative=True)

    def _stt_results_suppressed(self) -> bool:
        if self._stt_muted:
            return True
        return time.monotonic() < self._stt_ignore_finals_until

    def _flush_stt_silence(self) -> None:
        silence = bytes(STT_FLUSH_SILENCE_BYTES)
        self.speech_session.push_audio(silence)

    def mute_stt_for_playback(self) -> None:
        if not self._stt_muted:
            self._stt_muted = True
            self._flush_stt_silence()
            logger.info("STT muted for agent playback")

    def handle_playback_idle(self, epoch: int | None = None) -> None:
        if self._agent_audio_open:
            logger.debug(
                "ignored playback_idle while agent audio open epoch=%s", epoch
            )
            return
        if epoch is None or epoch != self._playback_epoch:
            logger.debug(
                "ignored playback_idle stale epoch=%s expected=%s",
                epoch,
                self._playback_epoch,
            )
            return
        self._barge_in.disarm()
        self._flush_stt_silence()
        self._stt_muted = False
        self._stt_ignore_finals_until = time.monotonic() + 0.4
        logger.info(
            "STT unmuted after playback_idle epoch=%s (400ms final grace)",
            epoch,
        )

    async def handle_barge_in(self) -> None:
        now = time.monotonic()
        if now - self._last_barge_in_handled_at < 0.45:
            return
        self._last_barge_in_handled_at = now
        self._agent_audio_open = False
        self._latest_partial_text = ""
        self._recent_agent_text = None
        self._discard_interrupt_audio = True
        self._barge_in.enter_discard_mode()
        self._stt_muted = True
        self._flush_stt_silence()
        self.turn_id += 1
        await self.cancel_active_task(rollback_speculative=True)
        await self.send_json({"type": "barge_in"})

    def apply_tts_config(
        self,
        backend: str,
        voice: str,
        speed: float | str | int | None = None,
    ) -> None:
        self.tts_backend, self.tts_voice, self.tts_speed = validate_tts_config(
            backend, voice, speed
        )
        logger.info(
            "TTS config backend=%s voice=%s speed=%s",
            self.tts_backend,
            self.tts_voice,
            self.tts_speed,
        )

    async def _on_speech_partial(self, text: str) -> None:
        if self._closed or self._stt_results_suppressed():
            return
        self._latest_partial_text = text
        self._last_voice_at = time.monotonic()
        if (
            self._speculative_user_text
            and not utterances_match(self._speculative_user_text, text)
        ):
            logger.info(
                "partial revised; cancel speculative turn was=%r now=%r",
                self._speculative_user_text[:80],
                text[:80],
            )
            await self._cancel_speculative_turn()
        await self.send_json(build_partial_event(text, self.turn_id))

    async def _on_speech_final(self, text: str) -> None:
        if self._closed or not text.strip() or self._stt_results_suppressed():
            if text.strip() and self._stt_results_suppressed():
                logger.debug("dropped STT final while suppressed: %s", text[:80])
            return
        if is_likely_agent_echo(text, self._recent_agent_text):
            logger.info("dropped echo STT final: %s", text[:120])
            self._recent_agent_text = None
            return
        self._recent_agent_text = None
        turn_id = self.turn_id
        final_at = time.monotonic()
        logger.info(
            "stt_final turn_id=%s t=%.3f text=%r",
            turn_id,
            final_at,
            text[:120],
        )
        self._latest_partial_text = ""
        await self.send_json(build_final_event(text, turn_id))

        speculative_active = (
            self.active_task is not None
            and not self.active_task.done()
            and self._speculative_user_text is not None
        )
        if speculative_active and utterances_match(self._speculative_user_text, text):
            self._turn_confirmed = True
            self._speculative_user_text = None
            if self.history and self.history[-1].get("role") == "user":
                self.history[-1]["content"] = text
            logger.info(
                "speculative_turn_confirmed turn_id=%s dt_since_start=%.3fs",
                turn_id,
                final_at - self._turn_started_at if self._turn_started_at else 0.0,
            )
            return

        if speculative_active:
            logger.info("final mismatched speculative; restarting turn")
            await self._cancel_speculative_turn()

        self.launch_turn(text, turn_id, speculative=False)

    async def _on_speech_error(self, message: str) -> None:
        if self._closed:
            return
        await self.send_json({"type": "error", "message": message})

    def launch_turn(
        self,
        user_text: str,
        turn_id: int,
        *,
        speculative: bool = False,
    ) -> None:
        if self.active_task and not self.active_task.done():
            self.active_task.cancel()
        self._turn_confirmed = not speculative
        self._speculative_user_text = user_text if speculative else None
        self._turn_started_at = time.monotonic()
        self._logged_first_audio_for_turn = None
        if speculative:
            logger.info(
                "speculative_turn_start turn_id=%s t=%.3f text=%r",
                turn_id,
                self._turn_started_at,
                user_text[:120],
            )
        self.active_task = asyncio.create_task(
            self.process_turn(user_text, turn_id, speculative=speculative)
        )

    def _maybe_start_speculative_turn(self) -> None:
        if self._stt_results_suppressed() or self._closed:
            return
        partial = self._latest_partial_text.strip()
        if not partial:
            return
        if self._speculative_user_text is not None:
            return
        if self.active_task and not self.active_task.done():
            return
        now = time.monotonic()
        silence_ms = (now - self._last_voice_at) * 1000.0
        if silence_ms < SPECULATIVE_SILENCE_MS:
            return
        self.launch_turn(partial, self.turn_id, speculative=True)

    async def _await_turn_confirmed(self, turn_id: int) -> bool:
        while not self._turn_confirmed:
            if turn_id != self.turn_id:
                return False
            await asyncio.sleep(0.02)
        return turn_id == self.turn_id

    async def _send_agent_audio(self, wav: bytes, turn_id: int, caption: str = "") -> bool:
        if not wav or turn_id != self.turn_id:
            return False
        if not self._agent_audio_open:
            self._agent_audio_open = True
        self.mute_stt_for_playback()
        self._playback_epoch += 1
        if not self._barge_in.armed:
            self._barge_in.arm()
            logger.info("turn_id=%s barge-in armed after first audio chunk", turn_id)
        if self._logged_first_audio_for_turn != turn_id:
            self._logged_first_audio_for_turn = turn_id
            logger.info(
                "first_audio_sent turn_id=%s t=%.3f dt_since_turn=%.3fs",
                turn_id,
                time.monotonic(),
                time.monotonic() - self._turn_started_at if self._turn_started_at else 0.0,
            )
        caption = caption.strip()
        if caption:
            await self.send_json(
                {
                    "type": "agent_caption",
                    "role": "agent",
                    "text": caption,
                    "turn_id": turn_id,
                }
            )
        await self.send_wav(wav)
        return True

    async def process_turn(
        self,
        user_text: str,
        turn_id: int,
        *,
        speculative: bool = False,
    ) -> None:
        self.ai_busy = True
        self._barge_in.reset_for_turn()
        self._barge_in.arm()
        logger.info(
            "turn_id=%s agent turn started speculative=%s tts=%s voice=%s speed=%s",
            turn_id,
            speculative,
            self.tts_backend,
            self.tts_voice,
            self.tts_speed,
        )
        first_token_logged = False
        sent_audio = False
        try:
            if turn_id != self.turn_id:
                return

            self.history.append({"role": "user", "content": user_text})

            agent_text_parts: list[str] = []
            clause_buffer = ""
            pending_clauses: list[str] = []
            first_chunk_pending = True

            async def flush_clause(clause: str) -> None:
                nonlocal sent_audio
                if not clause:
                    return
                clause = strip_spoken_markup(clause)
                if not clause:
                    return
                if not await self._await_turn_confirmed(turn_id):
                    return
                agent_text_parts.append(clause)
                wav = await tts.synth_chunk(
                    clause,
                    self.tts_backend,
                    self.tts_voice,
                    self.tts_speed,
                )
                if await self._send_agent_audio(wav, turn_id, clause):
                    sent_audio = True

            async def drain_pending_clauses() -> None:
                while pending_clauses:
                    if turn_id != self.turn_id:
                        return
                    clause = pending_clauses.pop(0)
                    await flush_clause(clause)

            async for token in llm.stream_llm_response(self.history):
                if turn_id != self.turn_id:
                    return
                if not first_token_logged:
                    first_token_logged = True
                    logger.info(
                        "first_llm_token turn_id=%s t=%.3f dt_since_turn=%.3fs",
                        turn_id,
                        time.monotonic(),
                        time.monotonic() - self._turn_started_at
                        if self._turn_started_at
                        else 0.0,
                    )
                clause_buffer += token
                while True:
                    clause, clause_buffer, first_chunk_pending = pop_next_speech_chunk(
                        clause_buffer,
                        first_chunk_pending=first_chunk_pending,
                    )
                    if not clause:
                        break
                    if self._turn_confirmed:
                        await flush_clause(clause)
                    else:
                        pending_clauses.append(clause)
                await drain_pending_clauses()

            if not self._turn_confirmed and speculative:
                if not await self._await_turn_confirmed(turn_id):
                    return
            await drain_pending_clauses()

            remainder = clause_buffer.strip()
            if remainder:
                await flush_clause(remainder)

            agent_text = " ".join(agent_text_parts).strip()
            if agent_text and turn_id == self.turn_id:
                await self.send_json(
                    {
                        "type": "transcript",
                        "role": "agent",
                        "text": agent_text,
                        "final": True,
                    }
                )
                self.history.append({"role": "assistant", "content": agent_text})
                self._recent_agent_text = agent_text
                if sent_audio:
                    self._agent_audio_open = False
                    await self.send_json(
                        {
                            "type": "agent_audio_done",
                            "turn_id": turn_id,
                            "epoch": self._playback_epoch,
                        }
                    )

        except asyncio.CancelledError:
            if speculative and self._speculative_user_text:
                self._rollback_speculative_user_message()
            raise
        except Exception as exc:
            logger.exception("process_turn failed")
            await self.send_json({"type": "error", "message": str(exc)})
        finally:
            if not sent_audio:
                self._barge_in.disarm()
            self._agent_audio_open = False
            self.ai_busy = False
            self.active_task = None
            if speculative and self._speculative_user_text is None:
                self._turn_confirmed = True

    async def handle_audio_frame(self, pcm_bytes: bytes) -> None:
        if not self._speech_ready:
            return

        frame = pcm16_to_float32(pcm_bytes)
        if frame.size == 0:
            return

        self._frames_received += 1
        if self._frames_received == 1:
            logger.info("first audio frame received bytes=%d", len(pcm_bytes))

        now = time.monotonic()
        if now - self._last_audio_log_at >= 5.0:
            boosted = amplify_pcm16(pcm_bytes)
            boosted_frame = pcm16_to_float32(boosted)
            rms_in = compute_rms(frame)
            rms_out = compute_rms(boosted_frame)
            peak = float(np.max(np.abs(boosted_frame))) if boosted_frame.size else 0.0
            logger.info(
                "audio ingress frames=%d bytes=%d rms_in=%.4f rms_out=%.4f peak=%.4f",
                self._frames_received,
                len(pcm_bytes),
                rms_in,
                rms_out,
                peak,
            )
            self._last_audio_log_at = now

        if self._discard_interrupt_audio:
            if self._barge_in.register_discard_frame(pcm_bytes):
                self._discard_interrupt_audio = False
                self._flush_stt_silence()
                self._stt_muted = False
                self._stt_ignore_finals_until = time.monotonic() + 0.35
                logger.info("interrupt utterance discarded; STT listening again")
            return

        if self._barge_in.armed and self._barge_in.register_frame(pcm_bytes):
            logger.info("barge-in fired turn_id=%s", self.turn_id)
            await self.handle_barge_in()
            return

        if self._stt_muted:
            return

        boosted = amplify_pcm16(pcm_bytes)
        boosted_frame = pcm16_to_float32(boosted)
        rms = compute_rms(boosted_frame)
        if rms > RMS_SPEECH_START_THRESHOLD:
            self._last_voice_at = now
        elif self._latest_partial_text.strip():
            self._maybe_start_speculative_turn()

        self.speech_session.push_audio(boosted)

    async def close(self) -> None:
        self._closed = True
        self._barge_in.disarm()
        await self.cancel_active_task()
        await self.speech_session.close()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_running_loop()
    session = VoiceSession(websocket, loop)

    try:
        try:
            await session.start()
        except Exception as exc:
            logger.exception("Azure Speech failed to start")
            message = f"Azure Speech failed to start: {exc}"
            try:
                await websocket.send_json(build_startup_error_event(message))
            except Exception:
                logger.exception("failed sending startup error to client")
            return

        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            if "bytes" in message and message["bytes"] is not None:
                await session.handle_audio_frame(message["bytes"])
            elif "text" in message and message["text"] is not None:
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue
                msg_type = payload.get("type")
                if msg_type == "tts_config":
                    try:
                        session.apply_tts_config(
                            payload.get("backend", ""),
                            payload.get("voice", ""),
                            payload.get("speed"),
                        )
                    except ValueError as exc:
                        await session.send_json(
                            {"type": "error", "message": str(exc)}
                        )
                elif msg_type == "playback_idle":
                    epoch = payload.get("epoch")
                    if isinstance(epoch, bool):
                        epoch = None
                    elif epoch is not None:
                        try:
                            epoch = int(epoch)
                        except (TypeError, ValueError):
                            epoch = None
                    session.handle_playback_idle(epoch)
                elif msg_type == "barge_in":
                    logger.info("client barge-in")
                    await session.handle_barge_in()

    except WebSocketDisconnect:
        pass
    finally:
        await session.close()
