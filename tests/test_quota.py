"""Günlük üretim sınırı."""

from datetime import date, timedelta

import pytest

from sketch2code import quota


@pytest.mark.parametrize(("raw", "expected"), [("", 0), ("5", 5), ("-3", 0), ("çok", 0), (" 2 ", 2)])
def test_limit_from_env(raw: str, expected: int) -> None:
    assert quota.limit_from_env({quota.LIMIT_ENV: raw}) == expected


def test_quota_counts_per_visitor_and_resets_next_day() -> None:
    today = [date(2026, 11, 20)]
    counter = quota.DailyQuota(2, today=lambda: today[0])

    assert counter.consume("a") and counter.consume("a")
    assert not counter.consume("a")
    assert counter.remaining("a") == 0
    assert counter.remaining("b") == 2  # başka ziyaretçi etkilenmez

    today[0] += timedelta(days=1)
    assert counter.remaining("a") == 2
    assert counter.consume("a")
