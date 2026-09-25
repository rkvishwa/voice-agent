"""Tests for post-barge-in interrupt phrase detection."""

import unittest

from app.interrupt_guard import is_interruption_utterance


class TestInterruptGuard(unittest.TestCase):
    def test_stop_phrases(self) -> None:
        self.assertTrue(is_interruption_utterance("Stop"))
        self.assertTrue(is_interruption_utterance("Please stop talking"))
        self.assertTrue(is_interruption_utterance("Wait, hold on"))

    def test_real_question_not_interrupt(self) -> None:
        self.assertFalse(is_interruption_utterance("What's the weather in Boston?"))
        self.assertFalse(is_interruption_utterance("Tell me about quantum physics"))


if __name__ == "__main__":
    unittest.main()
