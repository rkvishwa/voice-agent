"""Tests for Kokoro TTS helpers."""

import unittest

from app.tts import pick_kokoro_voice


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


if __name__ == "__main__":
    unittest.main()
