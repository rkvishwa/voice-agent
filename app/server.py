"""FastAPI WebSocket orchestration for the voice agent."""

import asyncio
import logging
import re
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from app import asr, llm, tts
from app.vad import TurnDetector, pcm16_to_float32

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

CLAUSE_DELIMITERS = re.compile(r"([.,!?\n])")

app = FastAPI(title="Voice Agent")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


class VoiceSession:
    """Per-connection state for turn detection and barge-in."""

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.history: list[dict] = []
        self.turn_detector = TurnDetector()
        self.active_task: asyncio.Task | None = None
        self.ai_busy = False

    async def send_json(self, payload: dict) -> None:
        await self.websocket.send_json(payload)

    async def send_wav(self, wav_bytes: bytes) -> None:
        if wav_bytes:
            await self.websocket.send_bytes(wav_bytes)

    async def cancel_active_task(self) -> None:
        if self.active_task and not self.active_task.done():
            self.active_task.cancel()
            try:
                await self.active_task
            except asyncio.CancelledError:
                pass
        self.active_task = None
        self.ai_busy = False

    async def handle_barge_in(self) -> None:
        await self.cancel_active_task()
        await self.send_json({"type": "barge_in"})

    def launch_turn(self, audio: np.ndarray) -> None:
        self.turn_detector.reset()
        self.active_task = asyncio.create_task(self.process_turn(audio))

    async def process_turn(self, audio: np.ndarray) -> None:
        self.ai_busy = True
        try:
            user_text = await asr.transcribe(audio)
            if not user_text:
                return

            await self.send_json(
                {"type": "transcript", "role": "user", "text": user_text}
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

        from app.vad import is_speech_start

        if is_speech_start(frame) and self.ai_busy:
            await self.handle_barge_in()

        turn_audio = self.turn_detector.process_frame(frame)
        if turn_audio is not None and turn_audio.size > 0:
            self.launch_turn(turn_audio)


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
