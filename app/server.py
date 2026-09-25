"""FastAPI WebSocket orchestration for the voice agent."""

import asyncio
import logging
import re
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from app import asr, llm, tts
from app.asr import AzureSpeechSession, build_final_event, build_partial_event
from app.vad import amplify_pcm16, compute_rms, is_speech_start, pcm16_to_float32

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

CLAUSE_DELIMITERS = re.compile(r"([.,!?\n])")

app = FastAPI(title="Voice Agent")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


class VoiceSession:
    """Per-connection state for Azure Speech streaming and barge-in."""

    def __init__(self, websocket: WebSocket, loop: asyncio.AbstractEventLoop):
        self.websocket = websocket
        self.history: list[dict] = []
        self.turn_id = 0
        self.active_task: asyncio.Task | None = None
        self.ai_busy = False
        self._closed = False
        self._frames_received = 0
        self._last_audio_log_at = 0.0
        self.speech_session = AzureSpeechSession(
            loop=loop,
            on_partial=self._on_speech_partial,
            on_final=self._on_speech_final,
            on_error=self._on_speech_error,
        )

    async def start(self) -> None:
        await self.speech_session.start()

    async def send_json(self, payload: dict) -> None:
        if not self._closed:
            await self.websocket.send_json(payload)

    async def send_wav(self, wav_bytes: bytes) -> None:
        if wav_bytes and not self._closed:
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
        self.turn_id += 1
        await self.cancel_active_task()
        await self.send_json({"type": "barge_in"})

    async def _on_speech_partial(self, text: str) -> None:
        if self._closed:
            return
        await self.send_json(build_partial_event(text, self.turn_id))

    async def _on_speech_final(self, text: str) -> None:
        if self._closed or not text.strip():
            return
        turn_id = self.turn_id
        await self.send_json(build_final_event(text, turn_id))
        self.launch_turn(text, turn_id)

    async def _on_speech_error(self, message: str) -> None:
        if self._closed:
            return
        await self.send_json({"type": "error", "message": message})

    def launch_turn(self, user_text: str, turn_id: int) -> None:
        if self.active_task and not self.active_task.done():
            self.active_task.cancel()
        self.active_task = asyncio.create_task(self.process_turn(user_text, turn_id))

    async def process_turn(self, user_text: str, turn_id: int) -> None:
        self.ai_busy = True
        try:
            if turn_id != self.turn_id:
                return

            self.history.append({"role": "user", "content": user_text})

            agent_text_parts: list[str] = []
            clause_buffer = ""

            async for token in llm.stream_llm_response(self.history):
                if turn_id != self.turn_id:
                    return
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
            if agent_text and turn_id == self.turn_id:
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

        self._frames_received += 1
        if self._frames_received == 1:
            logger.info("first audio frame received bytes=%d", len(pcm_bytes))

        boosted = amplify_pcm16(pcm_bytes)

        now = time.monotonic()
        if now - self._last_audio_log_at >= 5.0:
            rms_in = compute_rms(frame)
            rms_out = compute_rms(pcm16_to_float32(boosted))
            logger.info(
                "audio ingress frames=%d bytes=%d rms_in=%.4f rms_out=%.4f",
                self._frames_received,
                len(pcm_bytes),
                rms_in,
                rms_out,
            )
            self._last_audio_log_at = now

        if is_speech_start(pcm16_to_float32(boosted)) and self.ai_busy:
            await self.handle_barge_in()

        self.speech_session.push_audio(boosted)

    async def close(self) -> None:
        self._closed = True
        await self.cancel_active_task()
        await self.speech_session.close()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_running_loop()
    session = VoiceSession(websocket, loop)
    await session.start()

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
        await session.close()
