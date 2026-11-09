
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from weatherkit.schemas import WeatherResponse


@pytest.fixture
def weather_data():
    """Load the real weather JSON file."""
    # Build the path from this test file so it works regardless
    # of the directory from which pytest is run.
    file_path = Path(__file__).parent.parent / "weather_raw.json"

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def test_valid_weather_response_has_168_hours(weather_data):
    response = WeatherResponse.model_validate(weather_data)

    assert len(response.hourly.time) == 168


def test_invalid_latitude_raises_validation_error(weather_data):
    weather_data["latitude"] = 200.0

    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(weather_data)


def test_mismatched_list_lengths_raise_validation_error():
    data = {
        "latitude": 40.0,
        "longitude": -74.0,
        "timezone": "UTC",
        "elevation": 10.0,
        "hourly": {
            "time": ["2026-04-08T00:00", "2026-04-08T01:00", "2026-04-08T02:00"],
            "temperature_2m": [10.0, 12.0],
            "precipitation": [0.0, 0.1, 0.0],
        },
    }

    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(data)


def test_null_temperature_raises_validation_error(weather_data):
    weather_data["hourly"]["temperature_2m"][0] = None

    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(weather_data)
