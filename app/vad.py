"""Energy-based voice activity detection with hysteresis and pre-roll."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from app.config import (
    MIN_TURN_SECONDS,
    PRE_ROLL_SAMPLES,
    RMS_SPEECH_CONTINUE_THRESHOLD,
    RMS_SPEECH_START_THRESHOLD,
    SAMPLE_RATE,
    SILENCE_FRAMES_FOR_END,
)

logger = logging.getLogger(__name__)


def pcm16_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """Convert raw 16-bit PCM bytes to float32 samples in [-1, 1]."""
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    return samples / 32768.0


def compute_rms(audio: np.ndarray) -> float:
    """Root-mean-square energy of an audio frame."""
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio * audio)))


def is_speech_start(audio: np.ndarray) -> bool:
    """Return True when frame energy exceeds the speech-start threshold."""
    return compute_rms(audio) > RMS_SPEECH_START_THRESHOLD


def is_speech_continue(audio: np.ndarray) -> bool:
    """Return True when frame energy exceeds the lower continuation threshold."""
    return compute_rms(audio) > RMS_SPEECH_CONTINUE_THRESHOLD


class SampleRingBuffer:
    """Fixed-size ring buffer retaining the most recent audio samples."""

    def __init__(self, max_samples: int) -> None:
        self._max_samples = max_samples
        self._chunks: list[np.ndarray] = []
        self._length = 0

    def append(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        self._chunks.append(samples)
        self._length += samples.size
        while self._length > self._max_samples and self._chunks:
            overflow = self._length - self._max_samples
            first = self._chunks[0]
            if first.size <= overflow:
                self._chunks.pop(0)
                self._length -= first.size
            else:
                self._chunks[0] = first[overflow:]
                self._length -= overflow

    def snapshot(self) -> np.ndarray:
        if not self._chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(self._chunks)


@dataclass
class TurnDetector:
    """Detect end-of-turn with hysteresis VAD and leading pre-roll."""

    in_speech: bool = False
    silence_frames: int = 0
    audio_buffer: list[np.ndarray] = field(default_factory=list)
    pre_roll: SampleRingBuffer = field(
        default_factory=lambda: SampleRingBuffer(PRE_ROLL_SAMPLES)
    )
    frame_rms_values: list[float] = field(default_factory=list)

    def _classify_frame(self, frame: np.ndarray) -> bool:
        rms = compute_rms(frame)
        if self.in_speech:
            return is_speech_continue(frame)
        return is_speech_start(frame)

    def process_frame(self, frame: np.ndarray) -> np.ndarray | None:
        """
        Ingest one frame. Returns concatenated turn audio when end-of-turn
        is detected, otherwise None.
        """
        if frame.size == 0:
            return None

        speaking = self._classify_frame(frame)
        rms = compute_rms(frame)

        if not self.in_speech:
            self.pre_roll.append(frame)
            if speaking:
                self.in_speech = True
                self.silence_frames = 0
                self.frame_rms_values = [rms]
                pre = self.pre_roll.snapshot()
                self.audio_buffer = [pre, frame] if pre.size else [frame]
            return None

        self.frame_rms_values.append(rms)
        self.audio_buffer.append(frame)

        if speaking:
            self.silence_frames = 0
            return None

        self.silence_frames += 1
        duration = self._buffer_duration()
        if (
            self.silence_frames > SILENCE_FRAMES_FOR_END
            and duration > MIN_TURN_SECONDS
        ):
            audio = self._get_turn_audio()
            self._log_turn(audio, duration)
            self.reset()
            return audio

        return None

    def _buffer_duration(self) -> float:
        total = sum(chunk.size for chunk in self.audio_buffer)
        return total / SAMPLE_RATE

    def _get_turn_audio(self) -> np.ndarray:
        if not self.audio_buffer:
            return np.array([], dtype=np.float32)
        return np.concatenate(self.audio_buffer)

    def active_duration(self) -> float:
        """Duration of the in-progress utterance in seconds."""
        if not self.in_speech:
            return 0.0
        return self._buffer_duration()

    def active_snapshot(self) -> np.ndarray:
        """Thread-safe copy of audio accumulated during the active utterance."""
        if not self.in_speech or not self.audio_buffer:
            return np.array([], dtype=np.float32)
        return np.concatenate(self.audio_buffer).copy()

    def reset(self) -> None:
        """Clear active turn state while keeping the idle pre-roll buffer."""
        self.in_speech = False
        self.silence_frames = 0
        self.audio_buffer.clear()
        self.frame_rms_values.clear()

    def _log_turn(self, audio: np.ndarray, duration: float) -> None:
        if self.frame_rms_values:
            rms_min = min(self.frame_rms_values)
            rms_max = max(self.frame_rms_values)
        else:
            rms_min = rms_max = 0.0
        logger.info(
            "utterance committed duration=%.2fs rms=[%.4f, %.4f] samples=%d",
            duration,
            rms_min,
            rms_max,
            audio.size,
        )
