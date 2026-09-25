"""Barge-in gating: only interrupt the agent after playback has started."""

from app.config import BARGE_IN_SUSTAINED_FRAMES, RMS_SPEECH_START_THRESHOLD
from app.vad import compute_rms


class BargeInGate:
    """Arm after the first TTS chunk; require sustained speech before barge-in."""

    def __init__(self) -> None:
        self.armed = False
        self._speech_frames = 0

    def reset_for_turn(self) -> None:
        self.armed = False
        self._speech_frames = 0

    def arm(self) -> None:
        self.armed = True

    def disarm(self) -> None:
        self.armed = False
        self._speech_frames = 0

    def register_frame(self, audio_frame) -> bool:
        """
        Update state from one mic frame. Return True when barge-in should fire.
        """
        if not self.armed:
            return False

        if compute_rms(audio_frame) > RMS_SPEECH_START_THRESHOLD:
            self._speech_frames += 1
        else:
            self._speech_frames = 0

        if self._speech_frames >= BARGE_IN_SUSTAINED_FRAMES:
            self._speech_frames = 0
            return True
        return False
