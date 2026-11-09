
import json
from pathlib import Path

import pytest

from weatherkit.schemas import WeatherResponse
from weatherkit.records import HourlyReading, to_readings


@pytest.fixture
def response():
    file_path = Path(__file__).parent.parent / "weather_raw.json"

    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return WeatherResponse.model_validate(data)


@pytest.fixture
def readings(response):
    return to_readings(response)


def test_to_readings_preserves_order_and_count(response, readings):
    assert len(readings) == len(response.hourly.time)

    assert readings[0].timestamp == response.hourly.time[0]
    assert readings[-1].timestamp == response.hourly.time[-1]


@pytest.mark.parametrize("index", [0, 50, 167])
def test_reading_values_match_input_index(response, readings, index):
    reading = readings[index]

    assert reading.timestamp == response.hourly.time[index]
    assert reading.temperature_c == response.hourly.temperature_2m[index]
    assert reading.precipitation_mm == response.hourly.precipitation[index]


def test_identical_hourly_readings_compare_equal():
    reading1 = HourlyReading(
        timestamp="2026-04-08T00:00",
        temperature_c=10.0,
        precipitation_mm=0.0,
    )

    reading2 = HourlyReading(
        timestamp="2026-04-08T00:00",
        temperature_c=10.0,
        precipitation_mm=0.0,
    )

    assert reading1 == reading2
