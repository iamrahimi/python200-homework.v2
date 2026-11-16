from pathlib import Path

from weather_model import WeatherClassifier


def main() -> None:
    """Load the real pipeline and print four readable predictions."""
    path = Path(__file__).resolve().parent / "models" / "weather_classifier.pkl"
    classifier = WeatherClassifier(path)
    examples = [
        ("Clearly good", 22.0, 12.0, 0.0, 5.0),
        ("Clearly bad", -2.0, -8.0, 20.0, 55.0),
        ("Borderline cool", 11.0, 4.0, 4.0, 23.0),
        ("Borderline warm", 29.0, 18.0, 1.0, 18.0),
    ]
    days = [
        {
            "temperature_2m_max": high,
            "temperature_2m_min": low,
            "precipitation_sum": rain,
            "wind_speed_10m_max": wind,
        }
        for _, high, low, rain, wind in examples
    ]
    for example, day, prediction in zip(examples, days, classifier.predict(days)):
        print(
            f"{example[0]} | high {day['temperature_2m_max']:.1f} C, "
            f"low {day['temperature_2m_min']:.1f} C, "
            f"rain {day['precipitation_sum']:.1f} mm, "
            f"wind {day['wind_speed_10m_max']:.1f} km/h | "
            f"{prediction.label} | P(good)={prediction.probability:.1%}"
        )


# Without the guard, calling main() here would also run it during imports.
# Importing this module to reuse a helper would load the model and print results,
# or fail if the saved model were missing.
if __name__ == "__main__":
    main()


# Reflection 1:
# A caller might insert wind_speed_10m_max first and temperature_2m_max last.
# Using list(day.values()) could treat wind speed 55 as a temperature of 55 C.
# Selecting named features in the training order prevents this swapped-input bug.
#
# Reflection 2:
# The tiny fixture model makes tests independent of a large or missing real
# file and changes caused by retraining the Week 3 model. tmp_path isolates
# each test. These tests check the component interface; real-model quality
# should be evaluated separately using held-out weather data.
#
# Reflection 3:
# The component accepts batches of dictionaries and returns structured
# Prediction objects in order, so a database pipeline can call it directly.
# Before production, I would validate numeric values and reject NaN, infinite,
# or physically impossible measurements with useful error messages.
