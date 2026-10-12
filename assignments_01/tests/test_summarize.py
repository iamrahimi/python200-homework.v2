
import pytest

from weatherkit.records import HourlyReading
from weatherkit.summarize import DailyAggregator


@pytest.fixture
def sample_readings():
    return [
        HourlyReading("2026-04-08T00:00", 10.0, 0.1),
        HourlyReading("2026-04-08T01:00", 15.0, 0.2),
        HourlyReading("2026-04-08T02:00", 8.0, 0.3),
        HourlyReading("2026-04-09T00:00", 20.0, 0.4),
        HourlyReading("2026-04-09T01:00", 18.0, 0.6),
    ]


def test_grouping_produces_two_daily_summaries(sample_readings):
    aggregator = DailyAggregator(min_hours=1)

    summaries = aggregator.summarize(sample_readings)

    assert len(summaries) == 2
    assert [summary.date for summary in summaries] == [
        "2026-04-08",
        "2026-04-09",
    ]


def test_daily_maximum_and_minimum_temperatures(sample_readings):
    aggregator = DailyAggregator(min_hours=1)

    summaries = aggregator.summarize(sample_readings)
    first_day = summaries[0]

    assert first_day.temp_max == 15.0
    assert first_day.temp_min == 8.0


def test_precipitation_sum_is_correct(sample_readings):
    aggregator = DailyAggregator(min_hours=1)

    summaries = aggregator.summarize(sample_readings)

    assert summaries[0].precipitation_sum == pytest.approx(0.6)
    assert summaries[1].precipitation_sum == pytest.approx(1.0)


def test_incomplete_day_is_dropped(sample_readings):
    aggregator = DailyAggregator(min_hours=3)

    summaries = aggregator.summarize(sample_readings)

    assert [summary.date for summary in summaries] == ["2026-04-08"]
    assert aggregator.incomplete_days() == ["2026-04-09"]


def test_lowering_min_hours_keeps_the_same_day(sample_readings):
    aggregator = DailyAggregator(min_hours=2)

    summaries = aggregator.summarize(sample_readings)

    assert [summary.date for summary in summaries] == [
        "2026-04-08",
        "2026-04-09",
    ]
    assert aggregator.incomplete_days() == []


@pytest.mark.parametrize(
    "expected_range",
    [7.0],
)
def test_temperature_range(sample_readings, expected_range):
    aggregator = DailyAggregator(min_hours=1)

    summaries = aggregator.summarize(sample_readings)

    assert summaries[0].temp_range() == expected_range
