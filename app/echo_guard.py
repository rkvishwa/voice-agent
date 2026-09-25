"""Detect when STT finals are likely agent playback echo."""

import re

_MIN_WORDS_FOR_ECHO = 3
_MIN_CHARS_FOR_ECHO = 12
_WORD_OVERLAP_RATIO = 0.75

_NON_ALNUM = re.compile(r"[^a-z0-9\s]+")


def normalize_speech_text(text: str) -> str:
    """Lowercase and strip punctuation for comparison."""
    lowered = text.lower().strip()
    cleaned = _NON_ALNUM.sub(" ", lowered)
    return " ".join(cleaned.split())


def _words(text: str) -> list[str]:
    norm = normalize_speech_text(text)
    return norm.split() if norm else []


def is_likely_agent_echo(user_text: str, recent_agent_text: str | None) -> bool:
    """
    Return True when user_text is probably the agent's own speech transcribed.

    Short utterances are never treated as echo so real replies like \"hello\" work.
    """
    if not recent_agent_text or not user_text.strip():
        return False

    user_norm = normalize_speech_text(user_text)
    agent_norm = normalize_speech_text(recent_agent_text)
    if not user_norm or not agent_norm:
        return False

    user_words = _words(user_text)
    if len(user_norm) < _MIN_CHARS_FOR_ECHO and len(user_words) < _MIN_WORDS_FOR_ECHO:
        return False

    if user_norm in agent_norm:
        return True

    agent_word_set = set(_words(recent_agent_text))
    if not agent_word_set or len(user_words) < _MIN_WORDS_FOR_ECHO:
        return False

    overlap = sum(1 for w in user_words if w in agent_word_set)
    if overlap / len(user_words) >= _WORD_OVERLAP_RATIO:
        return True

    return False
