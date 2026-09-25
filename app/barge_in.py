"""Barge-in gating: WebRTC speech detection during agent playback."""

from collections import deque

import webrtcvad

from app.config import (
    BARGE_IN_POST_INTERRUPT_SILENCE_FRAMES,
    BARGE_IN_VAD_MODE,
    BARGE_IN_VOICED_MIN,
    BARGE_IN_VOICED_WINDOW,
    FRAME_SAMPLES,
    RMS_BARGE_IN_PLAYBACK_THRESHOLD,
    SAMPLE_RATE,
)
from app.vad import compute_rms, pcm16_to_float32

_PCM_FRAME_BYTES = FRAME_SAMPLES * 2


class BargeInGate:
    """Arm during agent playback; confirm speech via WebRTC VAD + rolling window."""

    def __init__(self) -> None:
        self.armed = False
        self.in_discard_mode = False
        self._vad = webrtcvad.Vad(BARGE_IN_VAD_MODE)
        self._voice_window: deque[bool] = deque(maxlen=BARGE_IN_VOICED_WINDOW)
        self._discard_silence_frames = 0

    def reset_for_turn(self) -> None:
        self.armed = False
        self._voice_window.clear()
        self._discard_silence_frames = 0

    def arm(self) -> None:
        self.armed = True
        self._voice_window.clear()

    def disarm(self) -> None:
        self.armed = False
        self._voice_window.clear()

    def enter_discard_mode(self) -> None:
        self.in_discard_mode = True
        self._discard_silence_frames = 0
        self.disarm()

    def exit_discard_mode(self) -> None:
        self.in_discard_mode = False
        self._discard_silence_frames = 0

    def _valid_pcm(self, pcm_bytes: bytes) -> bool:
        return len(pcm_bytes) == _PCM_FRAME_BYTES

    def _rms_ok_for_barge(self, pcm_bytes: bytes) -> bool:
        frame = pcm16_to_float32(pcm_bytes)
        return compute_rms(frame) >= RMS_BARGE_IN_PLAYBACK_THRESHOLD

    def _webrtc_speech(self, pcm_bytes: bytes) -> bool:
        if not self._valid_pcm(pcm_bytes):
            return False
        try:
            return self._vad.is_speech(pcm_bytes, SAMPLE_RATE)
        except Exception:
            return False

    def _barge_speech_candidate(self, pcm_bytes: bytes) -> bool:
        if not self._rms_ok_for_barge(pcm_bytes):
            return False
        return self._webrtc_speech(pcm_bytes)

    def register_frame(self, pcm_bytes: bytes) -> bool:
        """
        Update barge-in state from one 20 ms PCM16 frame.
        Return True when confirmed user speech should interrupt the agent.
        """
        if not self.armed or not self._valid_pcm(pcm_bytes):
            return False

        self._voice_window.append(self._barge_speech_candidate(pcm_bytes))
        if len(self._voice_window) < BARGE_IN_VOICED_WINDOW:
            return False

        if sum(self._voice_window) >= BARGE_IN_VOICED_MIN:
            self._voice_window.clear()
            return True
        return False

    def register_discard_frame(self, pcm_bytes: bytes) -> bool:
        """
        While discarding the interrupt utterance, count non-speech frames.
        Return True when sustained silence allows STT to resume.
        """
        if not self.in_discard_mode or not self._valid_pcm(pcm_bytes):
            return False

        if self._webrtc_speech(pcm_bytes):
            self._discard_silence_frames = 0
            return False

        self._discard_silence_frames += 1
        if self._discard_silence_frames >= BARGE_IN_POST_INTERRUPT_SILENCE_FRAMES:
            self.exit_discard_mode()
            return True
        return False
