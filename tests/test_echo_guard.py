"""Tests for agent echo text detection."""

import unittest

from app.echo_guard import is_likely_agent_echo, normalize_speech_text


class TestEchoGuard(unittest.TestCase):
    def test_normalize_strips_punctuation(self) -> None:
        self.assertEqual(
            normalize_speech_text("Hello! How are you?"),
            "hello how are you",
        )

    def test_short_hello_not_echo(self) -> None:
        agent = "Hello! How can I help you today?"
        self.assertFalse(is_likely_agent_echo("hello", agent))

    def test_agent_clause_is_echo(self) -> None:
        agent = "Hello! How can I help you today?"
        self.assertTrue(
            is_likely_agent_echo("How can I help you today?", agent),
        )

    def test_different_user_utterance_not_echo(self) -> None:
        agent = "Hello! How can I help you today?"
        self.assertFalse(is_likely_agent_echo("Wait stop please", agent))

    def test_word_overlap_echo(self) -> None:
        agent = "The weather today is sunny and warm in Boston"
        self.assertTrue(
            is_likely_agent_echo(
                "weather today sunny warm Boston",
                agent,
            ),
        )


if __name__ == "__main__":
    unittest.main()
