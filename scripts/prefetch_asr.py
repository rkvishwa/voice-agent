#!/usr/bin/env python3
"""Prefetch configured Faster-Whisper preview and final models."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from faster_whisper import WhisperModel

from app.config import (
    ASR_COMPUTE_TYPE,
    ASR_CPU_THREADS,
    ASR_FINAL_MODEL,
    ASR_PREVIEW_MODEL,
)


def prefetch(model_name: str) -> None:
    print(
        f"Prefetching ASR model: {model_name} "
        f"({ASR_COMPUTE_TYPE}, {ASR_CPU_THREADS} threads)"
    )
    WhisperModel(
        model_name,
        device="cpu",
        compute_type=ASR_COMPUTE_TYPE,
        cpu_threads=ASR_CPU_THREADS,
    )
    print(f"  cached: {model_name}")


def main() -> None:
    models = []
    for model in (ASR_PREVIEW_MODEL, ASR_FINAL_MODEL):
        if model not in models:
            models.append(model)

    for model in models:
        prefetch(model)

    print("All ASR models cached.")


if __name__ == "__main__":
    main()
