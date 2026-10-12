
def test_incomplete_day_is_dropped(sample_readings):
    aggregator = DailyAggregator(min_hours=3)

    summaries = aggregator.summarize(sample_readings)

    assert [summary.date for summary in summaries] == ["2026-04-08"]
    assert aggregator.incomplete_days(sample_readings) == ["2026-04-09"]


def test_lowering_min_hours_keeps_the_same_day(sample_readings):
    aggregator = DailyAggregator(min_hours=2)

    summaries = aggregator.summarize(sample_readings)

    assert [summary.date for summary in summaries] == [
        "2026-04-08",
        "2026-04-09",
    ]
    assert aggregator.incomplete_days(sample_readings) == []
