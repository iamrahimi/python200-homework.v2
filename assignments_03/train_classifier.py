
import os
import json
import sys
import requests
import numpy as np
import pandas as pd
import sklearn
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


# --- Task 1: Fetch the Data ---

print("\n--- Task 1: Fetch the Data ---")

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": 47.6062,
    "longitude": -122.3321,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/Los_Angeles",
}

response = requests.get(url, params=params, timeout=30)
response.raise_for_status()

df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)

# I chose Seattle, Washington, and historical weather data from 2023.

print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())


# --- Task 2: Engineer the Label ---

print("\n--- Task 2: Engineer the Label ---")

feature_cols = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]

# Remove rows with missing values in required columns.
df = df.dropna(subset=feature_cols).copy()

# A good day has a high between 7°C and 26°C,
# a low of at least 0°C, precipitation below 3 mm,
# and maximum wind below 30 km/h.
df["good_for_running"] = (
    df["temperature_2m_max"].between(7, 26)
    & (df["temperature_2m_min"] >= 0)
    & (df["precipitation_sum"] < 3)
    & (df["wind_speed_10m_max"] < 30)
).astype(int)

good_days = int((df["good_for_running"] == 1).sum())
skip_days = int((df["good_for_running"] == 0).sum())
good_fraction = good_days / len(df)

print("Good days:", good_days)
print("Skip days:", skip_days)
print(f"Fraction of good days: {good_fraction:.2%}")

# This fraction helps show how often Seattle meets the chosen rules.
# If classes are imbalanced, examine precision, recall, and F1,
# not just accuracy.


# --- Task 3: Build and Evaluate a KNN Classifier ---

print("\n--- Task 3: KNN Classifier ---")

X = df[feature_cols]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

k_values = [3, 5, 7, 9, 11, 15]
knn_results = {}

for k in k_values:
    knn_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k)),
        ]
    )

    scores = cross_val_score(
        knn_pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy",
    )

    knn_results[k] = scores.mean()
    print(f"k={k:2d}, Mean CV accuracy: {scores.mean():.3f}")

best_k = max(knn_results, key=knn_results.get)

print("Selected k:", best_k)

knn_model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=best_k)),
    ]
)

knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)

knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_f1 = f1_score(y_test, knn_predictions, zero_division=0)

print("KNN test accuracy:", knn_accuracy)
print("KNN test F1:", knn_f1)
print("\nKNN classification report:")
print(
    classification_report(
        y_test,
        knn_predictions,
        labels=[0, 1],
        target_names=["skip", "good"],
        zero_division=0,
    )
)


# --- Task 4: Build and Evaluate Logistic Regression ---

print("\n--- Task 4: Logistic Regression ---")

logreg_model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000)),
    ]
)

logreg_model.fit(X_train, y_train)
logreg_predictions = logreg_model.predict(X_test)

logreg_accuracy = accuracy_score(y_test, logreg_predictions)
logreg_precision = precision_score(
    y_test, logreg_predictions, zero_division=0
)
logreg_recall = recall_score(
    y_test, logreg_predictions, zero_division=0
)
logreg_f1 = f1_score(
    y_test, logreg_predictions, zero_division=0
)

print("Logistic regression test accuracy:", logreg_accuracy)
print("Precision for good days:", logreg_precision)
print("Recall for good days:", logreg_recall)
print("F1 for good days:", logreg_f1)

print("\nLogistic regression classification report:")
print(
    classification_report(
        y_test,
        logreg_predictions,
        labels=[0, 1],
        target_names=["skip", "good"],
        zero_division=0,
    )
)

print("Feature coefficients:")

coefficients = logreg_model.named_steps["logreg"].coef_[0]

for name, coef in zip(feature_cols, coefficients):
    print(f"{name:25s}: {coef:+.3f}")

# A positive coefficient pushes the prediction toward good (1).
# A negative coefficient pushes it toward skip (0).
# Larger absolute coefficients indicate stronger effects per
# standardized feature unit, rather than per original measurement unit.


# --- Task 5: Compare and Choose ---

print("\n--- Task 5: Compare and Choose ---")

print(f"KNN accuracy: {knn_accuracy:.3f}")
print(f"Logistic regression accuracy: {logreg_accuracy:.3f}")
print(f"KNN F1: {knn_f1:.3f}")
print(f"Logistic regression F1: {logreg_f1:.3f}")

# Logistic regression is selected for deployment because it is
# interpretable, fast, and provides probability confidence scores.
# Compare the scores before deciding whether it is suitable.
#
# For a running app, false positives may be especially concerning:
# they recommend running when the weather fails the chosen rules.
# False negatives miss a good running day. Precision helps measure
# false positives among recommended days; recall measures how many
# good days the model successfully identifies.
#
# Accuracy alone is not enough to make the deployment decision.


# --- Task 6: Save the Model and Metadata ---

print("\n--- Task 6: Save the Model ---")

os.makedirs("models", exist_ok=True)

model_path = "models/weather_classifier.pkl"
metadata_path = "models/weather_classifier_metadata.json"

joblib.dump(logreg_model, model_path)

metadata = {
    "city": "Seattle, Washington",
    "year": 2023,
    "python_version": sys.version,
    "scikit_learn_version": sklearn.__version__,
    "feature_names": feature_cols,
    "label_rule": (
        "good=1 when temperature_2m_max is between 7 and 26°C "
        "inclusive, temperature_2m_min >= 0°C, "
        "precipitation_sum < 3 mm, and wind_speed_10m_max "
        "< 30 km/h; otherwise skip=0."
    ),
    "good_label": 1,
    "skip_label": 0,
    "test_accuracy": float(logreg_accuracy),
    "test_f1": float(logreg_f1),
}

with open(metadata_path, "w") as file:
    json.dump(metadata, file, indent=4)

print("Model saved to:", model_path)
print("Metadata saved to:", metadata_path)
print("Both files were written successfully.")
