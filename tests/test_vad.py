"""Tests for VAD hysteresis, pre-roll, and turn-end detection."""

import unittest

import numpy as np

from app.config import (
    FRAME_SAMPLES,
    MIN_TURN_SECONDS,
    PRE_ROLL_SAMPLES,
    RMS_SPEECH_CONTINUE_THRESHOLD,
    RMS_SPEECH_START_THRESHOLD,
    SAMPLE_RATE,
    SILENCE_FRAMES_FOR_END,
)
from app.vad import (
    SampleRingBuffer,
    TurnDetector,
    compute_rms,
    is_speech_continue,
    is_speech_start,
)


def _tone(amplitude: float, samples: int = FRAME_SAMPLES) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    return (amplitude * np.sin(t)).astype(np.float32)


def _silence(samples: int = FRAME_SAMPLES) -> np.ndarray:
    return np.zeros(samples, dtype=np.float32)


class TestVadThresholds(unittest.TestCase):
    def test_hysteresis_gap(self) -> None:
        # amplitude 0.04 -> RMS ~0.028: above continue, below start
        frame = _tone(0.04)
        self.assertFalse(is_speech_start(frame))
        self.assertTrue(is_speech_continue(frame))

    def test_loud_frame_is_speech(self) -> None:
        frame = _tone(0.5)
        self.assertTrue(is_speech_start(frame))
        self.assertTrue(is_speech_continue(frame))


class TestSampleRingBuffer(unittest.TestCase):
    def test_keeps_only_recent_samples(self) -> None:
        buf = SampleRingBuffer(100)
        buf.append(np.ones(60, dtype=np.float32))
        buf.append(np.ones(60, dtype=np.float32) * 2)
        snap = buf.snapshot()
        self.assertEqual(snap.size, 100)
        self.assertEqual(snap[-1], 2.0)
        self.assertEqual(snap[0], 1.0)


class TestTurnDetector(unittest.TestCase):
    def test_pre_roll_prepended_on_speech_start(self) -> None:
        detector = TurnDetector()
        pre_frames = PRE_ROLL_SAMPLES // FRAME_SAMPLES
        for _ in range(pre_frames):
            detector.process_frame(_silence())

        speech = _tone(0.5)
        result = detector.process_frame(speech)
        self.assertIsNone(result)
        self.assertTrue(detector.in_speech)
        total = sum(chunk.size for chunk in detector.audio_buffer)
        self.assertGreaterEqual(total, PRE_ROLL_SAMPLES + FRAME_SAMPLES)

    def test_turn_commits_after_end_silence(self) -> None:
        detector = TurnDetector()
        min_frames = int(MIN_TURN_SECONDS * SAMPLE_RATE / FRAME_SAMPLES) + 1
        silence_tail = SILENCE_FRAMES_FOR_END + 2

        for _ in range(min_frames):
            detector.process_frame(_tone(0.5))

        turn = None
        for _ in range(silence_tail):
            turn = detector.process_frame(_silence())
            if turn is not None:
                break

        self.assertIsNotNone(turn)
        self.assertGreater(turn.size, 0)
        self.assertFalse(detector.in_speech)

    def test_quiet_syllable_inside_sentence_does_not_end_turn(self) -> None:
        detector = TurnDetector()
        detector.process_frame(_tone(0.5))
        quiet = _tone(
            max(RMS_SPEECH_CONTINUE_THRESHOLD * 0.5, 0.001)
        )
        self.assertFalse(is_speech_start(quiet))
        self.assertFalse(is_speech_continue(quiet))
        result = detector.process_frame(quiet)
        self.assertIsNone(result)
        self.assertTrue(detector.in_speech)


if __name__ == "__main__":
    unittest.main()
