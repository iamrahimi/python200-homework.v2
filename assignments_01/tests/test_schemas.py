import json
from pathlib import Path
import pytest
from pydantic import ValidationError
from weatherkit import WeatherResponse

def test_valid_real_weather_response():
    """The real JSON file should contain 168 valid hourly observations."""

    # Use the test file's location rather than the current working
    # directory, because pytest can be run from different directories.
    file_path = Path(__file__).parent.parent / "weather_raw.json"

    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    response = WeatherResponse.model_validate(data)

    assert len(response.hourly.time) == 168


def test_invalid_latitude():
    """Latitude outside the allowed range should fail validation."""

    data = {
        "latitude": 200.0,
        "longitude": -80.0,
        "timezone": "America/New_York",
        "elevation": 100.0,
        "hourly": {
            "time": ["2026-04-08T00:00"],
            "temperature_2m": [10.0],
            "precipitation": [0.0],
        },
    }

    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(data)


def test_mismatched_list_lengths():
    """Hourly measurement lists must have equal lengths."""

    data = {
        "latitude": 35.2,
        "longitude": -80.8,
        "timezone": "America/New_York",
        "elevation": 200.0,
        "hourly": {
            "time": [
                "2026-04-08T00:00",
                "2026-04-08T01:00",
            ],
            "temperature_2m": [10.0],
            "precipitation": [0.0, 0.1],
        },
    }

    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(data)


def test_null_temperature_fails_validation():
    """A null temperature should fail validation."""

    data = {
        "latitude": 35.2,
        "longitude": -80.8,
        "timezone": "America/New_York",
        "elevation": 200.0,
        "hourly": {
            "time": ["2026-04-08T00:00"],
            "temperature_2m": [None],
            "precipitation": [0.0],
        },
    }

    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(data)