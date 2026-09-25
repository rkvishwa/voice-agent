"""Tests for STT mute during agent playback (echo loop prevention)."""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

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


if __name__ == "__main__":
    unittest.main()
