import json
from dataclasses import dataclass
from pathlib import Path

import pytest


# Warmup Question 1: Load JSON once and count features.
class ModelInfo:
    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as file:
            self.data = json.load(file)

    def feature_count(self) -> int:
        return len(self.data["features"])


# Loading in __init__ avoids repeated file reads and JSON parsing when
# feature_count() is called many times; it reuses the stored data instead.


# Warmup Question 2: Describe a daily forecast with a dataclass.
@dataclass
class DayForecast:
    date: str
    high_c: float
    will_rain: bool

    def summary(self) -> str:
        rain_message = "rain expected" if self.will_rain else "no rain expected"
        return f"{self.date}: high {self.high_c}C, {rain_message}"


# @dataclass generates __init__ for us, so we do not have to write
# the assignments for date, high_c, and will_rain by hand.


# Warmup Question 3: Parse a temperature and test valid and invalid inputs.
def parse_temperature(text: str) -> float:
    try:
        return float(text)
    except ValueError as error:
        raise ValueError(f"Invalid temperature: {text!r}") from error


def test_parse_temperature_valid():
    assert parse_temperature("24.1") == pytest.approx(24.1)


def test_parse_temperature_bad_input_raises():
    with pytest.raises(ValueError, match="temperature"):
        parse_temperature("hot")


# Without match, pytest.raises(ValueError) would accept a ValueError with
# the wrong message, even if the message did not mention temperature.


if __name__ == "__main__":
    # Question 1 example: create a small JSON file beside this Python file.
    json_path = Path(__file__).with_name("model_info_sample.json")
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump({"features": ["temperature", "humidity", "wind_speed"]}, file)

    model = ModelInfo(json_path)
    print("Feature count:", model.feature_count())

    # Question 2 examples.
    forecast_one = DayForecast("2023-06-15", 24.1, True)
    forecast_two = DayForecast("2023-06-16", 26.5, False)
    print(forecast_one.summary())
    print(forecast_two.summary())

    # Question 3 example.
    print("Parsed temperature:", parse_temperature("24.1"))

