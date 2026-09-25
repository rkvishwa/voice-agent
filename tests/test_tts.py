"""Tests for TTS helpers and validation."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from azure.cognitiveservices.speech import ResultReason

from app.tts import (
    TTS_SPEED_MAX,
    TTS_SPEED_MIN,
    audio_from_synthesis_result,
    azure_prosody_rate,
    build_azure_ssml,
    escape_ssml_text,
    pick_kokoro_voice,
    synth_chunk,
    validate_speed,
    validate_tts_config,
)


class TestPickKokoroVoice(unittest.TestCase):
    def test_prefers_first_available(self) -> None:
        available = {"am_adam", "af_heart", "af_bella"}
        self.assertEqual(
            pick_kokoro_voice(("af_heart", "am_adam"), available),
            "af_heart",
        )

    def test_falls_back_when_preferred_missing(self) -> None:
        available = {"am_adam", "bf_emma"}
        self.assertEqual(
            pick_kokoro_voice(("af_heart", "am_adam"), available),
            "am_adam",
        )

    def test_raises_when_no_voices(self) -> None:
        with self.assertRaises(RuntimeError):
            pick_kokoro_voice(("af_heart",), set())


class TestValidateSpeed(unittest.TestCase):
    def test_rejects_below_min(self) -> None:
        with self.assertRaises(ValueError):
            validate_speed(TTS_SPEED_MIN - 0.01)

    def test_rejects_above_max(self) -> None:
        with self.assertRaises(ValueError):
            validate_speed(TTS_SPEED_MAX + 0.01)

    def test_accepts_in_range(self) -> None:
        self.assertEqual(validate_speed(1.05), 1.05)


class TestAzureProsodyAndSsml(unittest.TestCase):
    def test_prosody_rate_mapping(self) -> None:
        self.assertEqual(azure_prosody_rate(1.0), "0%")
        self.assertEqual(azure_prosody_rate(1.2), "+20%")
        self.assertEqual(azure_prosody_rate(0.8), "-20%")

    def test_escape_ssml_text(self) -> None:
        self.assertEqual(
            escape_ssml_text("a & b < c > d"),
            "a &amp; b &lt; c &gt; d",
        )

    def test_build_azure_ssml_escapes_text(self) -> None:
        ssml = build_azure_ssml(
            "Hi & bye <test>",
            "en-US-AvaMultilingualNeural",
            1.2,
        )
        self.assertIn("Hi &amp; bye &lt;test&gt;", ssml)
        self.assertIn('rate="+20%"', ssml)


class TestValidateTtsConfig(unittest.TestCase):
    def test_rejects_unknown_backend(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            validate_tts_config("piper", "af_heart", 1.0)
        self.assertIn("backend", str(ctx.exception).lower())

    def test_rejects_unknown_azure_voice(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            validate_tts_config("azure", "not-a-real-voice", 1.0)
        self.assertIn("azure", str(ctx.exception).lower())

    def test_accepts_curated_azure_voice(self) -> None:
        backend, voice, speed = validate_tts_config(
            "azure", "en-US-AvaMultilingualNeural", 1.1
        )
        self.assertEqual(backend, "azure")
        self.assertEqual(voice, "en-US-AvaMultilingualNeural")
        self.assertEqual(speed, 1.1)


class TestSynthChunkHotPath(unittest.IsolatedAsyncioTestCase):
    async def test_synth_chunk_does_not_call_validate_tts_config(self) -> None:
        with patch("app.tts.validate_tts_config") as mock_validate:
            with patch("app.tts.list_kokoro_voices") as mock_list:
                with patch(
                    "app.tts.asyncio.to_thread",
                    new=AsyncMock(return_value=b"RIFF"),
                ):
                    result = await synth_chunk(
                        "hello",
                        "azure",
                        "en-US-AvaMultilingualNeural",
                        1.05,
                    )
                self.assertEqual(result, b"RIFF")
            mock_list.assert_not_called()
            mock_validate.assert_not_called()


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
