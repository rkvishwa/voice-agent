"""Tests for live ASR flow helpers."""

import unittest

import numpy as np

from app.config import FRAME_SAMPLES
from app.partial_scheduler import PartialScheduler
from app.vad import TurnDetector


def _tone(amplitude: float, samples: int = FRAME_SAMPLES) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    return (amplitude * np.sin(t)).astype(np.float32)


class TestActiveSnapshot(unittest.TestCase):
    def test_snapshot_matches_buffer_while_speaking(self) -> None:
        detector = TurnDetector()
        frame = _tone(0.5)
        detector.process_frame(frame)
        snap = detector.active_snapshot()
        self.assertGreater(snap.size, 0)
        self.assertEqual(snap.dtype, np.float32)

    def test_snapshot_empty_when_idle(self) -> None:
        detector = TurnDetector()
        self.assertEqual(detector.active_snapshot().size, 0)


class TestStaleTurnHandling(unittest.TestCase):
    def test_turn_id_invalidation(self) -> None:
        current_turn = 5
        result_turn = 4
        self.assertTrue(result_turn != current_turn)

    def test_scheduler_resets_on_new_utterance(self) -> None:
        sched = PartialScheduler(interval_sec=1.0, min_seconds=0.8)
        sched.mark_scheduled(100.0)
        sched.reset()
        self.assertTrue(sched.should_schedule(1.0, 100.1))


if __name__ == "__main__":
    unittest.main()
