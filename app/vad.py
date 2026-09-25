"""Energy-based voice activity detection for barge-in."""

import numpy as np

from app.config import RMS_SPEECH_START_THRESHOLD


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
