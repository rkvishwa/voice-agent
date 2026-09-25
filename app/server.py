"""FastAPI WebSocket orchestration for the voice agent."""

import asyncio
import logging
import re
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from app import asr, llm, tts
from app.config import ASR_PREVIEW_INTERVAL_SEC, ASR_PREVIEW_MIN_SECONDS
from app.partial_scheduler import PartialScheduler
from app.vad import TurnDetector, is_speech_start, pcm16_to_float32

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

CLAUSE_DELIMITERS = re.compile(r"([.,!?\n])")

app = FastAPI(title="Voice Agent")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


class VoiceSession:
    """Per-connection state for turn detection, live captions, and barge-in."""

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.history: list[dict] = []
        self.turn_detector = TurnDetector()
        self.partial_scheduler = PartialScheduler(
            ASR_PREVIEW_INTERVAL_SEC,
            ASR_PREVIEW_MIN_SECONDS,
        )
        self.turn_id = 0
        self.active_task: asyncio.Task | None = None
        self.partial_task: asyncio.Task | None = None
        self.ai_busy = False

    async def send_json(self, payload: dict) -> None:
        await self.websocket.send_json(payload)

    async def send_wav(self, wav_bytes: bytes) -> None:
        if wav_bytes:
            await self.websocket.send_bytes(wav_bytes)

    def _cancel_partial_task(self) -> None:
        if self.partial_task and not self.partial_task.done():
            self.partial_task.cancel()
        self.partial_task = None

    async def cancel_active_task(self) -> None:
        self._cancel_partial_task()
        if self.active_task and not self.active_task.done():
            self.active_task.cancel()
            try:
                await self.active_task
            except asyncio.CancelledError:
                pass
        self.active_task = None
        self.ai_busy = False

    async def handle_barge_in(self) -> None:
        self.turn_id += 1
        self.partial_scheduler.reset()
        await self.cancel_active_task()
        await self.send_json({"type": "barge_in"})

    def _on_speech_start(self) -> None:
        self.turn_id += 1
        self.partial_scheduler.reset()
        self._cancel_partial_task()

    def _maybe_schedule_partial(self) -> None:
        if not self.turn_detector.in_speech:
            return
        duration = self.turn_detector.active_duration()
        now = time.monotonic()
        if not self.partial_scheduler.should_schedule(duration, now):
            return
        self.partial_scheduler.mark_scheduled(now)
        audio = self.turn_detector.active_snapshot()
        if audio.size == 0:
            return
        turn_id = self.turn_id
        self._cancel_partial_task()
        self.partial_task = asyncio.create_task(self._run_partial(turn_id, audio))

    async def _run_partial(self, turn_id: int, audio: np.ndarray) -> None:
        try:
            text = await asr.transcribe_preview(audio)
            if turn_id != self.turn_id:
                return
            if not self.turn_detector.in_speech:
                return
            if text:
                await self.send_json(
                    {
                        "type": "transcript_partial",
                        "role": "user",
                        "text": text,
                        "turn_id": turn_id,
                    }
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("partial transcription failed")

    def launch_turn(self, audio: np.ndarray, turn_id: int) -> None:
        self._cancel_partial_task()
        self.turn_detector.reset()
        self.active_task = asyncio.create_task(self.process_turn(audio, turn_id))

    async def process_turn(self, audio: np.ndarray, turn_id: int) -> None:
        self.ai_busy = True
        try:
            user_text = await asr.transcribe_final(audio)
            if turn_id != self.turn_id:
                return
            if not user_text:
                return

            await self.send_json(
                {
                    "type": "transcript",
                    "role": "user",
                    "text": user_text,
                    "turn_id": turn_id,
                    "final": True,
                }
            )
            self.history.append({"role": "user", "content": user_text})

            agent_text_parts: list[str] = []
            clause_buffer = ""

            async for token in llm.stream_llm_response(self.history):
                clause_buffer += token
                while True:
                    match = CLAUSE_DELIMITERS.search(clause_buffer)
                    if not match:
                        break
                    end = match.end()
                    clause = clause_buffer[:end].strip()
                    clause_buffer = clause_buffer[end:]
                    if clause:
                        agent_text_parts.append(clause)
                        wav = await tts.synth_chunk(clause)
                        await self.send_wav(wav)

            remainder = clause_buffer.strip()
            if remainder:
                agent_text_parts.append(remainder)
                wav = await tts.synth_chunk(remainder)
                await self.send_wav(wav)

            agent_text = " ".join(agent_text_parts).strip()
            if agent_text:
                await self.send_json(
                    {"type": "transcript", "role": "agent", "text": agent_text}
                )
                self.history.append({"role": "assistant", "content": agent_text})

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("process_turn failed")
            await self.send_json({"type": "error", "message": str(exc)})
        finally:
            self.ai_busy = False
            self.active_task = None

    async def handle_audio_frame(self, pcm_bytes: bytes) -> None:
        frame = pcm16_to_float32(pcm_bytes)
        if frame.size == 0:
            return

        if is_speech_start(frame) and self.ai_busy:
            await self.handle_barge_in()

        was_in_speech = self.turn_detector.in_speech
        turn_audio = self.turn_detector.process_frame(frame)

        if not was_in_speech and self.turn_detector.in_speech:
            self._on_speech_start()

        if self.turn_detector.in_speech:
            self._maybe_schedule_partial()

        if turn_audio is not None and turn_audio.size > 0:
            self.launch_turn(turn_audio, self.turn_id)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = VoiceSession(websocket)

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            if "bytes" in message and message["bytes"] is not None:
                await session.handle_audio_frame(message["bytes"])
            elif "text" in message and message["text"] is not None:
                pass

    except WebSocketDisconnect:
        pass
    finally:
        await session.cancel_active_task()
