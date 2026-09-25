"""Tests for barge-in gating."""

import unittest

import numpy as np

from app.barge_in import BargeInGate
from app.config import BARGE_IN_SUSTAINED_FRAMES, RMS_BARGE_IN_PLAYBACK_THRESHOLD


class TestBargeInGate(unittest.TestCase):
    def _loud_frame(self) -> np.ndarray:
        level = RMS_BARGE_IN_PLAYBACK_THRESHOLD + 0.1
        return np.full(320, level, dtype=np.float32)

    def test_quiet_bleed_below_playback_threshold(self) -> None:
        gate = BargeInGate()
        gate.arm()
        bleed = RMS_BARGE_IN_PLAYBACK_THRESHOLD * 0.5
        quiet = np.full(320, bleed, dtype=np.float32)
        for _ in range(BARGE_IN_SUSTAINED_FRAMES + 5):
            self.assertFalse(gate.register_frame(quiet))

    def _quiet_frame(self) -> np.ndarray:
        return np.zeros(320, dtype=np.float32)

    def test_ignored_when_not_armed(self) -> None:
        gate = BargeInGate()
        for _ in range(BARGE_IN_SUSTAINED_FRAMES + 5):
            self.assertFalse(gate.register_frame(self._loud_frame()))

    def test_ignored_while_turn_in_progress_before_audio(self) -> None:
        gate = BargeInGate()
        gate.reset_for_turn()
        self.assertFalse(gate.armed)
        for _ in range(BARGE_IN_SUSTAINED_FRAMES):
            self.assertFalse(gate.register_frame(self._loud_frame()))

    def test_fires_after_armed_and_sustained_speech(self) -> None:
        gate = BargeInGate()
        gate.arm()
        fired = False
        for i in range(BARGE_IN_SUSTAINED_FRAMES):
            if gate.register_frame(self._loud_frame()):
                fired = True
                break
        self.assertTrue(fired)

    def test_single_loud_frame_does_not_fire(self) -> None:
        gate = BargeInGate()
        gate.arm()
        self.assertFalse(gate.register_frame(self._loud_frame()))
        self.assertFalse(gate.register_frame(self._quiet_frame()))

    def test_sustained_counter_resets_on_quiet(self) -> None:
        gate = BargeInGate()
        gate.arm()
        for _ in range(BARGE_IN_SUSTAINED_FRAMES - 1):
            gate.register_frame(self._loud_frame())
        gate.register_frame(self._quiet_frame())
        self.assertFalse(gate.register_frame(self._loud_frame()))


if __name__ == "__main__":
    unittest.main()
