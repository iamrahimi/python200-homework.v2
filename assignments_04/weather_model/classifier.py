"""A reusable interface to the saved running-weather pipeline."""

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

FEATURES = (
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
)


@dataclass
class Prediction:
    """A good/skip label and the probability that running weather is good."""

    label: str
    probability: float


class WeatherClassifier:
    """Load a saved binary pipeline with classes 0=skip and 1=good.

    Raises:
        FileNotFoundError: If the saved model file does not exist.
    """

    def __init__(self, model_path: str | Path) -> None:
        """Load the pipeline once.

        Raises:
            FileNotFoundError: If model_path does not point to a file.
        """
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Weather model file not found: {path}. "
                "Place your saved Week 3 pipeline at this path."
            )
        self.model = joblib.load(path)

    def predict(self, days: list[dict[str, float]]) -> list[Prediction]:
        """Predict each day in input order using the four required features.

        Extra keys are ignored. The pipeline must use classes [0, 1], so
        probability column 1 represents good running weather.

        Raises:
            ValueError: If required features are missing or the model rejects
                the supplied feature values.
        """
        if not days:
            return []

        problems = []
        for index, day in enumerate(days):
            missing = [feature for feature in FEATURES if feature not in day]
            if missing:
                problems.append(f"day {index + 1}: {', '.join(missing)}")
        if problems:
            raise ValueError("Missing required features: " + "; ".join(problems))

        frame = pd.DataFrame(days).loc[:, list(FEATURES)]
        labels = self.model.predict(frame)
        probabilities = self.model.predict_proba(frame)[:, 1]
        return [
            Prediction(
                label="good" if label == 1 else "skip",
                probability=float(probability),
            )
            for label, probability in zip(labels, probabilities)
        ]
