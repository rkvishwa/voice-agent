"""Tests for barge-in voice activity detection."""

import unittest

import numpy as np

from app.config import FRAME_SAMPLES, RMS_SPEECH_START_THRESHOLD
from app.vad import amplify_pcm16, compute_rms, is_speech_start


def _tone(amplitude: float, samples: int = FRAME_SAMPLES) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    return (amplitude * np.sin(t)).astype(np.float32)


class TestVadThresholds(unittest.TestCase):
    def test_loud_frame_is_speech(self) -> None:
        frame = _tone(0.5)
        self.assertTrue(is_speech_start(frame))

    def test_quiet_frame_is_not_speech(self) -> None:
        frame = _tone(0.001)
        self.assertFalse(is_speech_start(frame))

    def test_rms_computation(self) -> None:
        frame = _tone(0.5)
        rms = compute_rms(frame)
        self.assertGreater(rms, RMS_SPEECH_START_THRESHOLD)

    def test_amplify_quiet_pcm(self) -> None:
        quiet = (np.ones(320, dtype=np.int16) * 100).tobytes()
        boosted = amplify_pcm16(quiet, min_peak=4000, max_gain=8.0)
        samples = np.frombuffer(boosted, dtype=np.int16)
        self.assertGreater(int(np.max(np.abs(samples))), 100)


if __name__ == "__main__":
    unittest.main()
