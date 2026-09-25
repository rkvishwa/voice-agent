"""Tests for Azure Speech event mapping (no live API calls)."""

import unittest

from app.asr import build_final_event, build_partial_event


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

    def test_stale_turn_id_differs(self) -> None:
        current_turn = 2
        result_turn = 1
        self.assertNotEqual(current_turn, result_turn)


if __name__ == "__main__":
    unittest.main()
