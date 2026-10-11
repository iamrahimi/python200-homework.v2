import json
from pathlib import Path
from weatherkit import WeatherResponse, HourlyReading, to_readings

def load_weather_response():
    """Load and validate the real weather JSON file."""

    file_path = Path(__file__).parent.parent / "weather_raw.json"
    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return WeatherResponse.model_validate(data)


def test_to_readings_preserves_order():
    """Conversion should preserve the first and last timestamps."""

    response = load_weather_response()
    readings = to_readings(response)

    assert len(readings) == 168
    assert readings[0].timestamp == response.hourly.time[0]
    assert readings[-1].timestamp == response.hourly.time[-1]


def test_reading_values_match_original_indexes():
    """Each reading should use measurements from the matching index."""

    response = load_weather_response()
    readings = to_readings(response)

    for i, reading in enumerate(readings):
        assert reading.timestamp == response.hourly.time[i]
        assert reading.temperature_c == response.hourly.temperature_2m[i]
        assert reading.precipitation_mm == response.hourly.precipitation[i]


def test_identical_hourly_readings_are_equal():
    """Dataclass instances with identical fields should compare equal."""

    first = HourlyReading(
        timestamp="2026-04-08T00:00",
        temperature_c=10.0,
        precipitation_mm=0.0,
    )

    second = HourlyReading(
        timestamp="2026-04-08T00:00",
        temperature_c=10.0,
        precipitation_mm=0.0,
    )

    assert first == second