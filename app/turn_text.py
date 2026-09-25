"""Utterance matching and streaming TTS chunk boundaries."""

import re

CLAUSE_DELIMITERS = re.compile(r"([.,!?\n])")

FIRST_CHUNK_MAX_WORDS = 6
FIRST_CHUNK_MAX_CHARS = 40

_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MD_BOLD = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
_MD_ITALIC = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)|(?<!_)_(?!_)(.+?)(?<!_)_(?!_)")


def strip_spoken_markup(text: str) -> str:
    """Remove markdown formatting so TTS and captions read natural speech."""
    if not text:
        return ""

    cleaned = text
    cleaned = _MD_LINK.sub(r"\1", cleaned)
    cleaned = _MD_BOLD.sub(lambda m: m.group(1) or m.group(2) or "", cleaned)
    cleaned = _MD_ITALIC.sub(lambda m: m.group(1) or m.group(2) or "", cleaned)
    cleaned = cleaned.replace("`", "")
    cleaned = re.sub(r"^\s*#{1,6}\s+", "", cleaned)
    cleaned = re.sub(r"^\s*[-*]\s+", "", cleaned)
    cleaned = re.sub(r"^\s*\d+\.\s+", "", cleaned)
    cleaned = cleaned.replace("*", "").replace("_", "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def normalize_utterance(text: str) -> str:
    """Compare STT partial vs final: ignore case and trailing punctuation."""
    cleaned = text.strip().lower()
    cleaned = re.sub(r"[^\w\s]", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def utterances_match(a: str, b: str) -> bool:
    return normalize_utterance(a) == normalize_utterance(b)


def first_chunk_ready(buffer: str) -> bool:
    stripped = buffer.strip()
    if not stripped:
        return False
    return len(stripped) >= FIRST_CHUNK_MAX_CHARS or len(stripped.split()) >= FIRST_CHUNK_MAX_WORDS


def pop_first_chunk_at_word_boundary(buffer: str) -> tuple[str | None, str]:
    """
    If the buffer is long enough for an early first chunk, return (chunk, remainder).
    """
    if not first_chunk_ready(buffer):
        return None, buffer

    words = buffer.split()
    if not words:
        return None, buffer

    take = 0
    length = 0
    for word in words:
        add = len(word) + (1 if take else 0)
        if take >= FIRST_CHUNK_MAX_WORDS:
            break
        if take > 0 and length + add > FIRST_CHUNK_MAX_CHARS:
            break
        take += 1
        length += add

    if take == 0:
        take = 1

    chunk = " ".join(words[:take]).strip()
    if not chunk:
        return None, buffer

    idx = buffer.find(chunk)
    if idx < 0:
        return None, buffer
    remainder = buffer[idx + len(chunk) :].lstrip()
    return chunk, remainder


def pop_next_speech_chunk(
    clause_buffer: str,
    *,
    first_chunk_pending: bool,
) -> tuple[str | None, str, bool]:
    """
    Extract the next speakable chunk from the LLM token buffer.

    Returns (chunk or None, updated buffer, whether first-chunk early flush still pending).
    """
    match = CLAUSE_DELIMITERS.search(clause_buffer)
    if match:
        end = match.end()
        clause = clause_buffer[:end].strip()
        rest = clause_buffer[end:]
        if clause:
            return clause, rest, False
        return None, rest, first_chunk_pending

    if first_chunk_pending:
        chunk, rest = pop_first_chunk_at_word_boundary(clause_buffer)
        if chunk:
            return chunk, rest, False

    return None, clause_buffer, first_chunk_pending
