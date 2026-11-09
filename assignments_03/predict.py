import json
import os
import joblib
import pandas as pd


# --- Task 7: Load the Model and Predict ---
print("\n--- Task 7: Load and Predict ---")

model_path = "models/weather_classifier.pkl"
metadata_path = "models/weather_classifier_metadata.json"

# Give a helpful message if training has not been run yet.
if not os.path.exists(model_path) or not os.path.exists(metadata_path):
    raise SystemExit(
        "Model files not found. Run train_classifier.py first "
        "to train and save the model."
    )

loaded_model = joblib.load(model_path)

with open(metadata_path, "r") as file:
    metadata = json.load(file)

print("City:", metadata["city"])
print("Features:", metadata["feature_names"])
print(f"Test accuracy: {metadata['test_accuracy']:.3f}")

feature_cols = metadata["feature_names"]

# Four hypothetical weather scenarios.
# The values use the same units as the training dataset.
weather_days = [
    {
        "day": "Clearly good day",
        "temperature_2m_max": 20.0,
        "temperature_2m_min": 12.0,
        "precipitation_sum": 0.0,
        "wind_speed_10m_max": 8.0,
    },
    {
        "day": "Clearly bad day",
        "temperature_2m_max": 5.0,
        "temperature_2m_min": -3.0,
        "precipitation_sum": 12.0,
        "wind_speed_10m_max": 40.0,
    },
    {
        "day": "Borderline day",
        "temperature_2m_max": 26.5,
        "temperature_2m_min": 1.0,
        "precipitation_sum": 2.8,
        "wind_speed_10m_max": 29.5,
    },
    {
        "day": "Rainy day",
        "temperature_2m_max": 15.0,
        "temperature_2m_min": 7.0,
        "precipitation_sum": 5.0,
        "wind_speed_10m_max": 12.0,
    },
]

# Use the feature names from metadata to keep columns in the
# same order as the features used during training.
weather_df = pd.DataFrame(weather_days)
X_new = weather_df[feature_cols]

predictions = loaded_model.predict(X_new)
probabilities = loaded_model.predict_proba(X_new)

# Find which probability column corresponds to the good class (1).
classes = list(loaded_model.named_steps["logreg"].classes_)
good_index = classes.index(1)

good_probabilities = probabilities[:, good_index]

for i, row in weather_df.iterrows():
    predicted_label = "good" if predictions[i] == 1 else "skip"
    confidence = good_probabilities[i]

    print(f"\n{row['day']}")
    for feature in feature_cols:
        print(f"  {feature}: {row[feature]}")

    print("  Predicted label:", predicted_label)
    print(f"  Probability of good: {confidence:.2f}")

    if row["day"] == "Borderline day":
        print(f"  Borderline probability: {confidence:.2f}")


# --- Task 8: Reflect ---

# 1. Borderline day:
# Check the printed probability for the borderline day.
# A probability near 0.50 means the model is uncertain.
# If the model predicts 0.52, I would treat it as uncertain
# and avoid making a strong recommendation based on that score alone.
#
# 2. Running before training:
# The model and metadata files would not exist, so prediction
# could not run. The file-existence check provides a helpful
# message telling the user to run train_classifier.py first.
#
# 3. Week 10 cloud pipeline:
# This script currently uses four hypothetical days written
# directly in the code. A cloud pipeline would need to receive
# real daily weather data automatically, validate the input,
# and process new observations without manually editing this file.
