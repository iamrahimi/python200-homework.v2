import pytest

from weatherkit import HourlyReading, DailyAggregator


@pytest.fixture
def sample_readings():
    """Provide hourly readings spanning two dates."""

    return [
        HourlyReading(
            timestamp="2026-04-08T00:00",
            temperature_c=10.0,
            precipitation_mm=0.1,
        ),
        HourlyReading(
            timestamp="2026-04-08T01:00",
            temperature_c=15.0,
            precipitation_mm=0.2,
        ),
        HourlyReading(
            timestamp="2026-04-08T02:00",
            temperature_c=8.0,
            precipitation_mm=0.3,
        ),
        HourlyReading(
            timestamp="2026-04-09T00:00",
            temperature_c=12.0,
            precipitation_mm=0.4,
        ),
        HourlyReading(
            timestamp="2026-04-09T01:00",
            temperature_c=18.0,
            precipitation_mm=0.5,
        ),
    ]


def test_grouping_creates_two_daily_summaries(sample_readings):
    """Readings from two dates should produce two summaries."""

    aggregator = DailyAggregator(min_hours=1)

    summaries = aggregator.summarize(sample_readings)

    assert len(summaries) == 2
    assert summaries[0].date == "2026-04-08"
    assert summaries[1].date == "2026-04-09"


def test_maximum_and_minimum_temperature(sample_readings):
    """Daily maximum and minimum temperatures should be correct."""

    aggregator = DailyAggregator(min_hours=1)

    summaries = aggregator.summarize(sample_readings)

    assert summaries[0].temp_max == 15.0
    assert summaries[0].temp_min == 8.0
    assert summaries[1].temp_max == 18.0
    assert summaries[1].temp_min == 12.0


def test_precipitation_sum(sample_readings):
    """Daily precipitation should equal the sum of hourly values."""

    aggregator = DailyAggregator(min_hours=1)

    summaries = aggregator.summarize(sample_readings)

    assert summaries[0].precipitation_sum == pytest.approx(0.6)
    assert summaries[1].precipitation_sum == pytest.approx(0.9)


def test_incomplete_day_is_dropped(sample_readings):
    """Days below the minimum should be excluded and reported."""

    aggregator = DailyAggregator(min_hours=3)

    summaries = aggregator.summarize(sample_readings)
    incomplete = aggregator.incomplete_days(sample_readings)

    assert len(summaries) == 1
    assert summaries[0].date == "2026-04-08"
    assert incomplete == ["2026-04-09"]


def test_lower_min_hours_keeps_the_day(sample_readings):
    """Lowering min_hours should allow a previously excluded day."""

    aggregator = DailyAggregator(min_hours=2)

    summaries = aggregator.summarize(sample_readings)

    assert len(summaries) == 2
    assert aggregator.incomplete_days(sample_readings) == []


@pytest.mark.parametrize(
    "temperatures, expected_range",
    [
        ([10.0, 15.0, 8.0], 7.0),
        ([5.0, 5.0], 0.0),
        ([-5.0, 5.0], 10.0),
    ],
)
def test_temperature_range(temperatures, expected_range):
    """Temperature range should equal maximum minus minimum."""

    readings = [
        HourlyReading(
            timestamp=f"2026-04-08T{i:02d}:00",
            temperature_c=temperature,
            precipitation_mm=0.0,
        )
        for i, temperature in enumerate(temperatures)
    ]

    aggregator = DailyAggregator(min_hours=1)
    summary = aggregator.summarize(readings)[0]

    assert summary.temp_range() == pytest.approx(expected_range)