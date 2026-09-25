"""Tests for live partial transcription scheduling."""

import unittest

from app.partial_scheduler import PartialScheduler


class TestPartialScheduler(unittest.TestCase):
    def test_waits_for_minimum_duration(self) -> None:
        sched = PartialScheduler(interval_sec=1.0, min_seconds=0.8)
        self.assertFalse(sched.should_schedule(0.5, 10.0))
        self.assertTrue(sched.should_schedule(0.8, 10.0))

    def test_debounces_by_interval(self) -> None:
        sched = PartialScheduler(interval_sec=1.0, min_seconds=0.8)
        self.assertTrue(sched.should_schedule(1.0, 10.0))
        sched.mark_scheduled(10.0)
        self.assertFalse(sched.should_schedule(1.5, 10.5))
        self.assertTrue(sched.should_schedule(2.0, 11.1))

    def test_reset_allows_immediate_schedule(self) -> None:
        sched = PartialScheduler(interval_sec=1.0, min_seconds=0.8)
        sched.mark_scheduled(10.0)
        sched.reset()
        self.assertTrue(sched.should_schedule(1.0, 10.2))


if __name__ == "__main__":
    unittest.main()
