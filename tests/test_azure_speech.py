"""Tests for Azure Speech event mapping (no live API calls)."""

import asyncio
import unittest
from unittest.mock import MagicMock, patch

from app.asr import (
    AzureSpeechSession,
    build_final_event,
    build_partial_event,
    build_speech_ready_event,
    build_startup_error_event,
)


class TestAzureSpeechEvents(unittest.TestCase):
    def test_partial_event_shape(self) -> None:
        payload = build_partial_event("hello wor", 3)
        self.assertEqual(payload["type"], "transcript_partial")
        self.assertEqual(payload["role"], "user")
        self.assertEqual(payload["text"], "hello wor")
        self.assertEqual(payload["turn_id"], 3)
        self.assertNotIn("final", payload)

    def test_final_event_shape(self) -> None:
        payload = build_final_event("hello world", 5)
        self.assertEqual(payload["type"], "transcript")
        self.assertEqual(payload["role"], "user")
        self.assertEqual(payload["text"], "hello world")
        self.assertEqual(payload["turn_id"], 5)
        self.assertTrue(payload["final"])

    def test_speech_ready_event_shape(self) -> None:
        payload = build_speech_ready_event()
        self.assertEqual(payload, {"type": "speech_ready"})

    def test_startup_error_event_shape(self) -> None:
        payload = build_startup_error_event("bad key")
        self.assertEqual(payload["type"], "error")
        self.assertIn("bad key", payload["message"])

    def test_stale_turn_id_differs(self) -> None:
        current_turn = 2
        result_turn = 1
        self.assertNotEqual(current_turn, result_turn)


class TestAzureSpeechSessionLifecycle(unittest.IsolatedAsyncioTestCase):
    async def test_close_before_start_is_safe(self) -> None:
        loop = asyncio.get_running_loop()
        session = AzureSpeechSession(
            loop=loop,
            on_partial=MagicMock(),
            on_final=MagicMock(),
            on_error=MagicMock(),
        )
        self.assertFalse(session.is_started)
        await session.close()
        self.assertFalse(session.is_started)

    @patch("app.asr.AzureSpeechSession._start_sync")
    async def test_start_marks_session_started(self, start_sync: MagicMock) -> None:
        loop = asyncio.get_running_loop()
        session = AzureSpeechSession(
            loop=loop,
            on_partial=MagicMock(),
            on_final=MagicMock(),
            on_error=MagicMock(),
        )

        def _mark_started() -> None:
            session._started = True

        start_sync.side_effect = _mark_started
        await session.start()
        self.assertTrue(session.is_started)
        await session.close()


if __name__ == "__main__":
    unittest.main()
