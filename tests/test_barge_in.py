"""Tests for barge-in gating."""

import unittest
from unittest.mock import patch

import numpy as np

from app.barge_in import BargeInGate
from app.config import (
    BARGE_IN_POST_INTERRUPT_SILENCE_FRAMES,
    BARGE_IN_VOICED_MIN,
    BARGE_IN_VOICED_WINDOW,
    FRAME_SAMPLES,
    RMS_BARGE_IN_PLAYBACK_THRESHOLD,
)


def _pcm_frame(level: float = 0.05) -> bytes:
    samples = np.full(FRAME_SAMPLES, int(level * 30000), dtype=np.int16)
    return samples.tobytes()


class TestBargeInGate(unittest.TestCase):
    def test_ignored_when_not_armed(self) -> None:
        gate = BargeInGate()
        with patch.object(gate, "_barge_speech_candidate", return_value=True):
            for _ in range(BARGE_IN_VOICED_WINDOW + 2):
                self.assertFalse(gate.register_frame(_pcm_frame()))

    def test_ignored_while_turn_in_progress_before_audio(self) -> None:
        gate = BargeInGate()
        gate.reset_for_turn()
        self.assertFalse(gate.armed)

    @patch.object(BargeInGate, "_barge_speech_candidate", return_value=True)
    def test_fires_after_voiced_window(self, _mock: object) -> None:
        gate = BargeInGate()
        gate.arm()
        fired = False
        for _ in range(BARGE_IN_VOICED_WINDOW):
            if gate.register_frame(_pcm_frame()):
                fired = True
                break
        self.assertTrue(fired)

    @patch.object(BargeInGate, "_barge_speech_candidate", return_value=True)
    def test_tolerates_brief_unvoiced_in_window(self, mock_candidate: object) -> None:
        gate = BargeInGate()
        gate.arm()
        values = [True] * (BARGE_IN_VOICED_MIN - 1) + [False] + [True] * (
            BARGE_IN_VOICED_WINDOW - BARGE_IN_VOICED_MIN
        )
        mock_candidate.side_effect = values
        fired = False
        for _ in range(BARGE_IN_VOICED_WINDOW):
            if gate.register_frame(_pcm_frame()):
                fired = True
                break
        self.assertTrue(fired)

    @patch.object(BargeInGate, "_barge_speech_candidate", return_value=False)
    def test_non_speech_loud_audio_does_not_fire(self, _mock: object) -> None:
        gate = BargeInGate()
        gate.arm()
        for _ in range(BARGE_IN_VOICED_WINDOW + 5):
            self.assertFalse(gate.register_frame(_pcm_frame(0.2)))

    def test_quiet_bleed_below_rms_floor(self) -> None:
        gate = BargeInGate()
        gate.arm()
        bleed = RMS_BARGE_IN_PLAYBACK_THRESHOLD * 0.2
        quiet = np.full(FRAME_SAMPLES, int(bleed * 30000), dtype=np.int16).tobytes()
        with patch.object(gate, "_webrtc_speech", return_value=True):
            for _ in range(BARGE_IN_VOICED_WINDOW + 5):
                self.assertFalse(gate.register_frame(quiet))

    @patch.object(BargeInGate, "_webrtc_speech", return_value=False)
    def test_discard_exits_after_sustained_silence(self, _mock: object) -> None:
        gate = BargeInGate()
        gate.enter_discard_mode()
        pcm = _pcm_frame()
        for _ in range(BARGE_IN_POST_INTERRUPT_SILENCE_FRAMES - 1):
            self.assertFalse(gate.register_discard_frame(pcm))
        self.assertTrue(gate.register_discard_frame(pcm))
        self.assertFalse(gate.in_discard_mode)

    @patch.object(BargeInGate, "_webrtc_speech")
    def test_discard_resets_silence_counter_on_speech(self, mock_vad: object) -> None:
        gate = BargeInGate()
        gate.enter_discard_mode()
        pcm = _pcm_frame()
        mock_vad.side_effect = [False] * 5 + [True, False] * 20
        for _ in range(5):
            gate.register_discard_frame(pcm)
        self.assertEqual(gate._discard_silence_frames, 5)
        gate.register_discard_frame(pcm)
        self.assertEqual(gate._discard_silence_frames, 0)
        self.assertTrue(gate.in_discard_mode)


if __name__ == "__main__":
    unittest.main()
