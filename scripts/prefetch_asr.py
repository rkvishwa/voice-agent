#!/usr/bin/env python3
"""Prefetch the configured Faster-Whisper ASR model."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from faster_whisper import WhisperModel

from app.config import ASR_COMPUTE_TYPE, ASR_CPU_THREADS, ASR_MODEL


def main() -> None:
    print(f"Prefetching ASR model: {ASR_MODEL} ({ASR_COMPUTE_TYPE}, {ASR_CPU_THREADS} threads)")
    WhisperModel(
        ASR_MODEL,
        device="cpu",
        compute_type=ASR_COMPUTE_TYPE,
        cpu_threads=ASR_CPU_THREADS,
    )
    print("Whisper model cached.")


if __name__ == "__main__":
    main()
