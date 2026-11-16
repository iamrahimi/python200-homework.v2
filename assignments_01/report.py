import json
from pathlib import Path

from weatherkit import WeatherResponse, to_readings, DailyAggregator

def main() -> None:
    """Load, validate, process, and report daily weather summaries."""

    file_path = Path(__file__).parent / "weather_raw.json"

    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    # Validate the external API response.
    response = WeatherResponse.model_validate(data)

    print("Latitude:", response.latitude)
    print("Timezone:", response.timezone)

    # Convert the validated response into hourly readings.
    readings = to_readings(response)

    # Generate daily summaries.
    aggregator = DailyAggregator(min_hours=24)
    summaries = aggregator.summarize(readings)

    # Print the table.
    print()
    print(
        f"{'Date':<12}"
        f"{'High (°C)':>12}"
        f"{'Low (°C)':>12}"
        f"{'Precip (mm)':>15}"
        f"{'Range (°C)':>13}"
    )

    print("-" * 64)

    for summary in summaries:
        print(
            f"{summary.date:<12}"
            f"{summary.temp_max:>12.1f}"
            f"{summary.temp_min:>12.1f}"
            f"{summary.precipitation_sum:>15.1f}"
            f"{summary.temp_range():>13.1f}"
        )

    # Warn about days that were excluded.
    incomplete = aggregator.incomplete_days(readings)

    if incomplete:
        print()
        print("WARNING: The following days were excluded:")
        print(", ".join(incomplete))
    else:
        print()
        print("All days met the minimum observation requirement.")


if __name__ == "__main__":
    # Without this guard, importing report.py would execute main()
    # and unexpectedly load the file and print the report.
    main()


# Reflection
#
# 1. Should the schema reject a file if one temperature is null?
# Strict validation is useful when every measurement is required,
# such as when a downstream calculation cannot handle missing data.
# However, a weather pipeline may need to tolerate a missing measurement
# so it can process the remaining valid observations.
# To tolerate gaps, temperature_2m could be typed as list[float | None].
# The conversion and aggregation functions would then need to handle
# missing temperatures safely.
#
# 2. What happens if the pipeline runs at noon?
# The current day may have only 12 observations, so it will be excluded
# by the default minimum of 24. incomplete_days() reports the excluded
# date rather than letting it disappear silently.
#
# 3. Why use a package?
# The Week 10 pipeline can import and reuse the validation, conversion,
# and aggregation functions without copying the code into another file.