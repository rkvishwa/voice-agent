"""Text-to-speech: local Kokoro or Azure Speech neural voices."""

import asyncio
import inspect
import io
import logging
from typing import Any, Literal, Optional

import azure.cognitiveservices.speech as speechsdk
import soundfile as sf
from azure.cognitiveservices.speech import ResultReason
from kokoro_onnx import Kokoro

from app.config import (
    AZURE_SPEECH_KEY,
    AZURE_SPEECH_REGION,
    AZURE_SPEECH_VOICE,
    KOKORO_MODEL_PATH,
    KOKORO_VOICE,
    KOKORO_VOICES_PATH,
    TTS_BACKEND,
    TTS_SPEED,
)

logger = logging.getLogger(__name__)

TtsBackend = Literal["kokoro", "azure"]

TTS_SPEED_MIN = 0.75
TTS_SPEED_MAX = 1.5
TTS_SPEED_STEP = 0.05

AZURE_TTS_VOICES: list[dict[str, str]] = [
    {"id": "en-US-AvaMultilingualNeural", "label": "Ava (US, multilingual)"},
    {"id": "en-US-AndrewMultilingualNeural", "label": "Andrew (US, multilingual)"},
    {"id": "en-US-JennyNeural", "label": "Jenny (US)"},
    {"id": "en-US-GuyNeural", "label": "Guy (US)"},
    {"id": "en-GB-SoniaNeural", "label": "Sonia (UK)"},
    {"id": "en-GB-RyanNeural", "label": "Ryan (UK)"},
]

_kokoro: Optional[Kokoro] = None
_kokoro_create_kwargs: dict = {}
_azure_synthesizers: dict[str, speechsdk.SpeechSynthesizer] = {}


def pick_kokoro_voice(preferred: tuple[str, ...], available: set[str]) -> str:
    """Choose the first preferred voice present in the voices bundle."""
    for name in preferred:
        if name in available:
            return name
    if not available:
        raise RuntimeError("No Kokoro voices found in voices.bin")
    return sorted(available)[0]


def azure_voice_ids() -> set[str]:
    return {v["id"] for v in AZURE_TTS_VOICES}


def default_voice_for_backend(backend: TtsBackend) -> str:
    if backend == "azure":
        if AZURE_SPEECH_VOICE in azure_voice_ids():
            return AZURE_SPEECH_VOICE
        return AZURE_TTS_VOICES[0]["id"]
    if KOKORO_VOICE:
        return KOKORO_VOICE
    return "af_heart"


def list_kokoro_voices() -> tuple[list[dict[str, str]], Optional[str]]:
    """Return Kokoro voice options and an error if models are missing."""
    if not KOKORO_MODEL_PATH.is_file() or not KOKORO_VOICES_PATH.is_file():
        return [], (
            f"Kokoro models not found under {KOKORO_MODEL_PATH.parent}. "
            "Run scripts/download_models.sh."
        )
    try:
        kokoro = Kokoro(str(KOKORO_MODEL_PATH), str(KOKORO_VOICES_PATH))
        voices = [
            {"id": name, "label": name}
            for name in sorted(kokoro.get_voices())
        ]
        return voices, None
    except Exception as exc:
        return [], str(exc)


def get_tts_catalog() -> dict:
    """Build payload for GET /api/tts."""
    backend = TTS_BACKEND if TTS_BACKEND in ("kokoro", "azure") else "kokoro"
    kokoro_voices, kokoro_error = list_kokoro_voices()
    default_kokoro = default_voice_for_backend("kokoro")
    if kokoro_voices and default_kokoro not in {v["id"] for v in kokoro_voices}:
        default_kokoro = kokoro_voices[0]["id"]

    default_speed = validate_speed(TTS_SPEED)

    return {
        "default_backend": backend,
        "default_speed": default_speed,
        "speed_min": TTS_SPEED_MIN,
        "speed_max": TTS_SPEED_MAX,
        "speed_step": TTS_SPEED_STEP,
        "default_voices": {
            "kokoro": default_kokoro,
            "azure": default_voice_for_backend("azure"),
        },
        "engines": {
            "kokoro": {
                "label": "Local Kokoro",
                "voices": kokoro_voices,
                "error": kokoro_error,
                "available": bool(kokoro_voices),
            },
            "azure": {
                "label": "Azure Speech",
                "voices": AZURE_TTS_VOICES,
                "error": None,
                "available": True,
            },
        },
    }


def validate_speed(speed: float | str | int) -> float:
    """Validate speaking rate multiplier (1.0 = normal)."""
    try:
        value = float(speed)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid TTS speed: {speed}") from None
    if value < TTS_SPEED_MIN or value > TTS_SPEED_MAX:
        raise ValueError(
            f"TTS speed must be between {TTS_SPEED_MIN} and {TTS_SPEED_MAX} "
            f"(got {value})"
        )
    return round(value, 2)


def escape_ssml_text(text: str) -> str:
    """Escape text for safe inclusion in SSML."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def azure_prosody_rate(speed: float) -> str:
    """Map speed multiplier to Azure prosody rate (1.0 -> 0%)."""
    percent = int(round((speed - 1.0) * 100))
    if percent > 0:
        return f"+{percent}%"
    if percent < 0:
        return f"{percent}%"
    return "0%"


def build_azure_ssml(text: str, voice: str, speed: float) -> str:
    rate = azure_prosody_rate(speed)
    escaped = escape_ssml_text(text)
    return (
        '<speak version="1.0" xml:lang="en-US" '
        'xmlns="http://www.w3.org/2001/10/synthesis">'
        f'<voice name="{voice}"><prosody rate="{rate}">{escaped}</prosody></voice>'
        "</speak>"
    )


def validate_tts_config(
    backend: str,
    voice: str,
    speed: float | str | int | None = None,
) -> tuple[TtsBackend, str, float]:
    """Validate backend, voice, and speed; return normalized triple."""
    if backend not in ("kokoro", "azure"):
        raise ValueError(f"Unknown TTS backend: {backend}")

    normalized_speed = validate_speed(speed if speed is not None else TTS_SPEED)

    if backend == "azure":
        if voice not in azure_voice_ids():
            raise ValueError(f"Unknown Azure TTS voice: {voice}")
        return "azure", voice, normalized_speed

    kokoro_voices, error = list_kokoro_voices()
    if error or not kokoro_voices:
        raise ValueError(error or "Kokoro TTS is not available")
    allowed = {v["id"] for v in kokoro_voices}
    if voice not in allowed:
        raise ValueError(f"Unknown Kokoro voice: {voice}")
    return "kokoro", voice, normalized_speed


def audio_from_synthesis_result(result: Any) -> bytes:
    """Extract WAV bytes from a Speech SDK synthesis result or raise with details."""
    if result.reason == ResultReason.SynthesizingAudioCompleted:
        audio = result.audio_data
        if not audio:
            raise RuntimeError("Azure Speech TTS returned empty audio")
        return bytes(audio)

    if result.reason == ResultReason.Canceled:
        details = result.cancellation_details
        if details is not None:
            message = details.error_details or str(details.reason)
        else:
            message = "unknown cancellation"
        raise RuntimeError(f"Azure Speech TTS canceled: {message}")

    raise RuntimeError(f"Azure Speech TTS failed: {result.reason}")


def _init_kokoro() -> Kokoro:
    global _kokoro, _kokoro_create_kwargs

    if _kokoro is not None:
        return _kokoro

    if not KOKORO_MODEL_PATH.is_file():
        raise RuntimeError(
            f"Kokoro model not found at {KOKORO_MODEL_PATH}. "
            "Run scripts/download_models.sh or deploy.sh."
        )
    if not KOKORO_VOICES_PATH.is_file():
        raise RuntimeError(
            f"Kokoro voices not found at {KOKORO_VOICES_PATH}. "
            "Run scripts/download_models.sh or deploy.sh."
        )

    _kokoro = Kokoro(str(KOKORO_MODEL_PATH), str(KOKORO_VOICES_PATH))
    sig = inspect.signature(_kokoro.create)
    if "lang" in sig.parameters:
        _kokoro_create_kwargs["lang"] = "en-us"

    logger.info("Kokoro TTS model loaded from %s", KOKORO_MODEL_PATH)
    return _kokoro


def _synth_kokoro_sync(text: str, voice: str, speed: float) -> bytes:
    kokoro = _init_kokoro()
    available = set(kokoro.get_voices())
    if voice not in available:
        raise RuntimeError(f"Kokoro voice {voice!r} not in voices.bin")
    samples, sample_rate = kokoro.create(
        text,
        voice=voice,
        speed=speed,
        **_kokoro_create_kwargs,
    )
    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format="WAV")
    return buffer.getvalue()


def _get_azure_synthesizer(voice: str) -> speechsdk.SpeechSynthesizer:
    if voice not in azure_voice_ids():
        raise RuntimeError(f"Azure TTS voice not allowed: {voice}")
    cached = _azure_synthesizers.get(voice)
    if cached is not None:
        return cached

    speech_config = speechsdk.SpeechConfig(
        subscription=AZURE_SPEECH_KEY,
        region=AZURE_SPEECH_REGION,
    )
    speech_config.speech_synthesis_voice_name = voice
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm,
    )
    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config,
        audio_config=None,
    )
    _azure_synthesizers[voice] = synthesizer
    return synthesizer


def _synth_azure_sync(text: str, voice: str, speed: float) -> bytes:
    synthesizer = _get_azure_synthesizer(voice)
    ssml = build_azure_ssml(text, voice, speed)
    result = synthesizer.speak_ssml_async(ssml).get()
    return audio_from_synthesis_result(result)


def _synth_sync(text: str, backend: TtsBackend, voice: str, speed: float) -> bytes:
    if backend == "kokoro":
        return _synth_kokoro_sync(text, voice, speed)
    return _synth_azure_sync(text, voice, speed)


async def synth_chunk(
    text: str,
    backend: TtsBackend,
    voice: str,
    speed: float,
) -> bytes:
    """Synthesize text into in-memory WAV bytes."""
    cleaned = text.strip()
    if not cleaned:
        return b""
    normalized_speed = validate_speed(speed)
    return await asyncio.to_thread(
        _synth_sync, cleaned, backend, voice, normalized_speed
    )
