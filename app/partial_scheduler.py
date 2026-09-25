"""Debouncing logic for live partial transcription updates."""


class PartialScheduler:
    """Decide when to schedule the next preview decode."""

    def __init__(self, interval_sec: float, min_seconds: float) -> None:
        self.interval_sec = interval_sec
        self.min_seconds = min_seconds
        self.last_partial_at: float = 0.0

    def should_schedule(self, duration_sec: float, now: float) -> bool:
        if duration_sec < self.min_seconds:
            return False
        if self.last_partial_at == 0.0:
            return True
        return (now - self.last_partial_at) >= self.interval_sec

    def mark_scheduled(self, now: float) -> None:
        self.last_partial_at = now

    def reset(self) -> None:
        self.last_partial_at = 0.0
