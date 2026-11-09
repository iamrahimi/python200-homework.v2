"""Component tests independent of the real Week 3 saved model."""

import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from weather_model import Prediction, WeatherClassifier
from weather_model.classifier import FEATURES

GOOD_DAY = dict(zip(FEATURES, [22.0, 12.0, 0.0, 5.0]))
BAD_DAY = dict(zip(FEATURES, [-2.0, -8.0, 20.0, 55.0]))
BORDERLINE_DAY = dict(zip(FEATURES, [11.0, 4.0, 4.0, 23.0]))


@pytest.fixture
def classifier(tmp_path):
    """Train, save, and load a tiny deterministic pipeline per test."""
    frame = pd.DataFrame(
        [
            [22, 12, 0, 5], [20, 10, 0, 8], [24, 14, 0.2, 6],
            [18, 9, 0, 10], [-2, -8, 20, 55], [0, -5, 15, 45],
            [3, -2, 12, 40], [5, 0, 10, 35],
        ], columns=FEATURES,
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(random_state=0))
    model.fit(frame, [1, 1, 1, 1, 0, 0, 0, 0])
    path = tmp_path / "test_weather.pkl"
    joblib.dump(model, path)
    return WeatherClassifier(path)


@pytest.mark.parametrize(
    "day, expected_label",
    [(GOOD_DAY, "good"), (BAD_DAY, "skip")],
    ids=["good-day", "bad-day"],
)
def test_predict_labels(classifier, day, expected_label):
    assert classifier.predict([day])[0].label == expected_label


# Mutation check: always returning "skip" caused
# test_predict_labels[good-day] to fail. The deliberate change was undone.


def test_probability_is_between_zero_and_one(classifier):
    for prediction in classifier.predict([GOOD_DAY, BAD_DAY, BORDERLINE_DAY]):
        assert 0.0 <= prediction.probability <= 1.0


def test_three_predictions_preserve_order(classifier):
    days = [GOOD_DAY, BAD_DAY, BORDERLINE_DAY]
    results = classifier.predict(days)
    assert len(results) == 3
    assert all(isinstance(result, Prediction) for result in results)
    expected = [classifier.predict([day])[0] for day in days]
    assert [result.label for result in results] == [item.label for item in expected]
    assert [result.probability for result in results] == pytest.approx(
        [item.probability for item in expected]
    )
    assert results[0].label == "good"
    assert results[1].label == "skip"


def test_empty_input_returns_empty_list(classifier):
    assert classifier.predict([]) == []


def test_missing_features_raise_value_error(classifier):
    incomplete = GOOD_DAY.copy()
    del incomplete["precipitation_sum"]
    del incomplete["wind_speed_10m_max"]
    with pytest.raises(
        ValueError, match="day 2: precipitation_sum, wind_speed_10m_max"
    ):
        classifier.predict([GOOD_DAY, incomplete])


def test_missing_model_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="Weather model file not found:.*missing.pkl"):
        WeatherClassifier(tmp_path / "missing.pkl")


def test_dictionary_order_does_not_change_prediction(classifier):
    reordered = dict(reversed(list(GOOD_DAY.items())))
    reordered["unneeded_feature"] = 999
    assert classifier.predict([reordered]) == classifier.predict([GOOD_DAY])


def test_probability_is_for_good_class(classifier):
    days = [GOOD_DAY, BAD_DAY, BORDERLINE_DAY]
    frame = pd.DataFrame(days).loc[:, list(FEATURES)]
    expected = classifier.model.predict_proba(frame)[:, 1]
    assert [p.probability for p in classifier.predict(days)] == pytest.approx(expected)
