"""Detect utterances that only mean \"stop talking\" after a barge-in."""

import re

from app.echo_guard import normalize_speech_text

# Whole-utterance matches (normalized).
_INTERRUPT_PHRASES = frozenset(
    {
        "stop",
        "stop it",
        "stop talking",
        "stop please",
        "please stop",
        "wait",
        "wait wait",
        "hold on",
        "hold up",
        "hang on",
        "one moment",
        "one second",
        "be quiet",
        "quiet",
        "shh",
        "shush",
        "enough",
        "that s enough",
        "ok stop",
        "okay stop",
        "never mind",
        "nevermind",
        "cancel",
        "listen",
        "excuse me",
        "sorry stop",
    }
)

_LEADING_FILLER = re.compile(
    r"^(?:uh+|um+|oh+|hey+|okay|ok|well|so|please|just)\s+",
    re.IGNORECASE,
)


def is_interruption_utterance(text: str) -> bool:
    """True when the transcript is only an attempt to interrupt the agent."""
    norm = normalize_speech_text(text)
    if not norm:
        return False
    if norm in _INTERRUPT_PHRASES:
        return True
    stripped = norm
    while True:
        next_text = _LEADING_FILLER.sub("", stripped).strip()
        if next_text == stripped:
            break
        stripped = next_text
    if stripped in _INTERRUPT_PHRASES:
        return True
    if len(stripped.split()) <= 4 and any(
        p in stripped for p in ("stop", "wait", "hold on", "hang on", "quiet", "enough")
    ):
        return True
    return False
