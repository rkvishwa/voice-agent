"""Tests for STT mute during agent playback (echo loop prevention)."""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from app.config import BARGE_IN_SUSTAINED_FRAMES, RMS_BARGE_IN_PLAYBACK_THRESHOLD
from app.server import VoiceSession


class TestSttMute(unittest.IsolatedAsyncioTestCase):
    def _session(self) -> VoiceSession:
        loop = asyncio.get_running_loop()
        return VoiceSession(MagicMock(), loop)

    async def test_final_ignored_while_muted(self) -> None:
        session = self._session()
        session._stt_muted = True
        with patch.object(session, "launch_turn") as launch_turn:
            with patch.object(session, "send_json", new_callable=AsyncMock) as send_json:
                await session._on_speech_final("hello agent echo")
        launch_turn.assert_not_called()
        send_json.assert_not_called()

    async def test_final_accepted_after_playback_idle(self) -> None:
        session = self._session()
        session._stt_muted = True
        session._playback_epoch = 3
        with patch("app.server.time.monotonic", return_value=100.0):
            session.handle_playback_idle(epoch=3)
        self.assertFalse(session._stt_muted)
        with patch("app.server.time.monotonic", return_value=100.5):
            with patch.object(session, "launch_turn") as launch_turn:
                with patch.object(session, "send_json", new_callable=AsyncMock):
                    await session._on_speech_final("real user speech")
        launch_turn.assert_called_once()

    async def test_audio_not_pushed_while_muted(self) -> None:
        session = self._session()
        session._speech_ready = True
        session._stt_muted = True
        session.speech_session.push_audio = MagicMock()
        frame = b"\x00\x01" * 160
        await session.handle_audio_frame(frame)
        session.speech_session.push_audio.assert_not_called()

    async def test_playback_idle_sets_grace_window(self) -> None:
        session = self._session()
        session._stt_muted = True
        session._playback_epoch = 1
        session.handle_playback_idle(epoch=1)
        self.assertFalse(session._stt_muted)
        self.assertTrue(session._stt_results_suppressed())

    async def test_playback_idle_ignored_while_agent_audio_open(self) -> None:
        session = self._session()
        session._stt_muted = True
        session._agent_audio_open = True
        session._playback_epoch = 2
        session.handle_playback_idle(epoch=2)
        self.assertTrue(session._stt_muted)

    async def test_playback_idle_ignored_with_stale_epoch(self) -> None:
        session = self._session()
        session._stt_muted = True
        session._playback_epoch = 5
        session.handle_playback_idle(epoch=3)
        self.assertTrue(session._stt_muted)

    async def test_echo_final_does_not_launch_turn(self) -> None:
        session = self._session()
        session._recent_agent_text = "Hello! How can I help you today?"
        with patch.object(session, "launch_turn") as launch_turn:
            with patch.object(session, "send_json", new_callable=AsyncMock):
                await session._on_speech_final("How can I help you today?")
        launch_turn.assert_not_called()
        self.assertIsNone(session._recent_agent_text)

    async def test_mute_flushes_silence_to_stt(self) -> None:
        session = self._session()
        session.speech_session.push_audio = MagicMock()
        session.mute_stt_for_playback()
        self.assertTrue(session._stt_muted)
        session.speech_session.push_audio.assert_called_once()

    def _loud_pcm_bytes(self) -> bytes:
        level = RMS_BARGE_IN_PLAYBACK_THRESHOLD + 0.08
        samples = np.full(320, int(level * 30000), dtype=np.int16)
        return samples.tobytes()

    async def test_muted_playback_frames_not_sent_to_stt(self) -> None:
        session = self._session()
        session._speech_ready = True
        session._stt_muted = True
        session._barge_in.arm()
        session.speech_session.push_audio = MagicMock()
        quiet = np.zeros(320, dtype=np.int16).tobytes()
        for _ in range(BARGE_IN_SUSTAINED_FRAMES + 3):
            await session.handle_audio_frame(quiet)
        session.speech_session.push_audio.assert_not_called()

    async def test_muted_sustained_speech_triggers_barge_in(self) -> None:
        session = self._session()
        session._speech_ready = True
        session._stt_muted = True
        session._barge_in.arm()
        session.speech_session.push_audio = MagicMock()
        frame = self._loud_pcm_bytes()
        with patch.object(session, "handle_barge_in", new_callable=AsyncMock) as barge:
            for _ in range(BARGE_IN_SUSTAINED_FRAMES):
                await session.handle_audio_frame(frame)
        barge.assert_awaited_once()
        session.speech_session.push_audio.assert_not_called()

    async def test_playback_idle_disarms_barge_in(self) -> None:
        session = self._session()
        session._stt_muted = True
        session._playback_epoch = 2
        session._barge_in.arm()
        session.handle_playback_idle(epoch=2)
        self.assertFalse(session._barge_in.armed)


class TestAgentCaption(unittest.IsolatedAsyncioTestCase):
    def _session(self) -> VoiceSession:
        loop = asyncio.get_running_loop()
        return VoiceSession(MagicMock(), loop)

    async def test_send_agent_audio_emits_caption_before_wav(self) -> None:
        session = self._session()
        session.turn_id = 1
        send_json = AsyncMock()
        send_wav = AsyncMock()
        session.send_json = send_json
        session.send_wav = send_wav

        ok = await session._send_agent_audio(b"wavbytes", 1, "Hello there.")
        self.assertTrue(ok)
        send_json.assert_awaited_once()
        payload = send_json.await_args.args[0]
        self.assertEqual(payload["type"], "agent_caption")
        self.assertEqual(payload["text"], "Hello there.")
        send_wav.assert_awaited_once_with(b"wavbytes")

    async def test_send_agent_audio_skips_empty_caption(self) -> None:
        session = self._session()
        session.turn_id = 1
        send_json = AsyncMock()
        send_wav = AsyncMock()
        session.send_json = send_json
        session.send_wav = send_wav

        ok = await session._send_agent_audio(b"wavbytes", 1, "   ")
        self.assertTrue(ok)
        send_json.assert_not_awaited()
        send_wav.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
