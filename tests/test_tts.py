"""Tests for Azure TTS result handling (mocked, no live API)."""

import unittest
from unittest.mock import MagicMock

from azure.cognitiveservices.speech import ResultReason

from app.tts import audio_from_synthesis_result


class TestAudioFromSynthesisResult(unittest.TestCase):
    def test_success_returns_audio_bytes(self) -> None:
        result = MagicMock()
        result.reason = ResultReason.SynthesizingAudioCompleted
        result.audio_data = b"RIFF....wav"
        self.assertEqual(audio_from_synthesis_result(result), b"RIFF....wav")

    def test_empty_audio_raises(self) -> None:
        result = MagicMock()
        result.reason = ResultReason.SynthesizingAudioCompleted
        result.audio_data = b""
        with self.assertRaises(RuntimeError) as ctx:
            audio_from_synthesis_result(result)
        self.assertIn("empty", str(ctx.exception).lower())

    def test_canceled_raises_with_details(self) -> None:
        details = MagicMock()
        details.error_details = "Invalid voice name"
        details.reason = "Error"

        result = MagicMock()
        result.reason = ResultReason.Canceled
        result.cancellation_details = details

        with self.assertRaises(RuntimeError) as ctx:
            audio_from_synthesis_result(result)
        self.assertIn("Invalid voice name", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
