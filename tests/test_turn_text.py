"""Tests for utterance matching and speech chunk splitting."""

import unittest

from app.turn_text import (
    normalize_utterance,
    pop_next_speech_chunk,
    utterances_match,
)


class TestUtteranceMatch(unittest.TestCase):
    def test_ignores_case_and_trailing_punctuation(self) -> None:
        self.assertTrue(utterances_match("Hello world.", "hello world"))
        self.assertTrue(utterances_match("What's up?", "whats up"))

    def test_differs_when_words_change(self) -> None:
        self.assertFalse(utterances_match("hello world", "hello there"))


class TestPopNextSpeechChunk(unittest.TestCase):
    def test_punctuation_splits_first(self) -> None:
        chunk, rest, pending = pop_next_speech_chunk(
            "Hi there, friend",
            first_chunk_pending=True,
        )
        self.assertEqual(chunk, "Hi there,")
        self.assertEqual(rest, " friend")
        self.assertFalse(pending)

    def test_early_first_chunk_on_word_boundary(self) -> None:
        words = " ".join(f"w{i}" for i in range(8))
        chunk, rest, pending = pop_next_speech_chunk(
            words,
            first_chunk_pending=True,
        )
        self.assertEqual(chunk, " ".join(f"w{i}" for i in range(6)))
        self.assertIn("w6", rest)
        self.assertFalse(pending)

    def test_no_chunk_until_ready(self) -> None:
        chunk, rest, pending = pop_next_speech_chunk(
            "short",
            first_chunk_pending=True,
        )
        self.assertIsNone(chunk)
        self.assertEqual(rest, "short")
        self.assertTrue(pending)


class TestNormalize(unittest.TestCase):
    def test_strips_punctuation(self) -> None:
        self.assertEqual(normalize_utterance("Hello, world!"), "hello world")
