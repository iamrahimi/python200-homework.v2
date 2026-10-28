from collections import defaultdict
from dataclasses import dataclass
from weatherkit.records import HourlyReading

@dataclass
class DailySummary:
    """Represent a summary of weather measurements for one day."""

    date: str
    temp_max: float
    temp_min: float
    precipitation_sum: float
    hours_observed: int
    def temp_range(self) -> float:
        """Return the difference between maximum and minimum temperature."""
        return self.temp_max - self.temp_min

class DailyAggregator:

    """Group hourly readings into daily weather summaries."""
    def __init__(self, min_hours: int = 24):
        """Initialize the aggregator.
        Args:
            min_hours: Minimum observations required to report a day.
        """
        self.min_hours = min_hours

    def summarize(
        self,
        readings: list[HourlyReading],
    ) -> list[DailySummary]:

        """Group hourly readings and calculate daily summaries.
        Args:
            readings: Hourly weather readings.
        Returns:
            Daily summaries sorted by date, excluding incomplete days.

        """

        grouped = defaultdict(list)

        for reading in readings:
            date = reading.timestamp[:10]
            grouped[date].append(reading)

        summaries = []

        for date in sorted(grouped):
            day_readings = grouped[date]
            if len(day_readings) < self.min_hours:
                continue

            temperatures = [
                reading.temperature_c
                for reading in day_readings

            ]

            precipitation = sum(
                reading.precipitation_mm
                for reading in day_readings

            )
            summary = DailySummary(
                date=date,
                temp_max=max(temperatures),
                temp_min=min(temperatures),
                precipitation_sum=precipitation,
                hours_observed=len(day_readings),

            )

            summaries.append(summary)
        return summaries

    def incomplete_days(
        self,
        readings: list[HourlyReading],

    ) -> list[str]:
        """Return dates excluded because they have too few observations.
        Args:
            readings: Hourly weather readings.
        Returns:
            Sorted dates with fewer than the required observations.

        """
        grouped = defaultdict(int)
        for reading in readings:
            date = reading.timestamp[:10]
            grouped[date] += 1

        return sorted(
            date
            for date, count in grouped.items()
            if count < self.min_hours

        )