
# --- Imports ---

import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# Create the outputs folder if it does not exist
os.makedirs("outputs", exist_ok=True)


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

# I chose Seattle, Washington, for this weather prediction project.

print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())


# --- Task 2: Look at the Features ---

print("\n--- Task 2: Look at the Features ---")

print("\nDescriptive statistics:")
print(df.describe())

print("\nMissing values in each column:")
print(df.isnull().sum())

# If any required values are missing, remove those rows.
# This keeps the models from receiving incomplete observations.
required_cols = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]

missing_before = df[required_cols].isnull().sum().sum()

if missing_before > 0:
    df = df.dropna(subset=required_cols).copy()
    print("\nRows with missing required values were removed.")
else:
    print("\nNo missing values were found in the required features.")

print("Dataset shape after checking missing values:", df.shape)

print("\nCorrelation with daily high, sorted:")
correlations = (
    df[required_cols]
    .corr()["temperature_2m_max"]
    .drop("temperature_2m_max")
    .sort_values(ascending=False)
)
print(correlations)

# The feature with the largest positive correlation is most strongly
# positively associated with the daily high. Correlation does not
# necessarily mean that a feature causes the temperature to change.

# Plot 1: Histogram of daily high temperatures
plt.figure(figsize=(8, 5))
plt.hist(df["temperature_2m_max"], bins=20, edgecolor="black")
plt.title("Distribution of Daily High Temperatures in Seattle")
plt.xlabel("Daily High Temperature (°C)")
plt.ylabel("Number of Days")
plt.tight_layout()
plt.savefig("outputs/daily_high_histogram.png")
plt.close()

# The histogram shows how frequently different daily high temperatures
# occurred during the year.

# Plot 2: Daily low versus daily high
plt.figure(figsize=(8, 5))
plt.scatter(
    df["temperature_2m_min"],
    df["temperature_2m_max"],
    alpha=0.7
)
plt.title("Daily High vs Low Temperature in Seattle")
plt.xlabel("Daily Low Temperature (°C)")
plt.ylabel("Daily High Temperature (°C)")
plt.tight_layout()
plt.savefig("outputs/high_vs_low_project.png")
plt.close()

# The scatter plot should show a positive relationship: days with
# higher minimum temperatures generally have higher maximum temperatures.


# --- Task 3: Engineer a Feature ---

print("\n--- Task 3: Engineer a Feature ---")

df["is_summer"] = df["date"].dt.month.isin([6, 7, 8]).astype(int)

summer_days = (df["is_summer"] == 1).sum()
non_summer_days = (df["is_summer"] == 0).sum()

print("Summer days:", summer_days)
print("Non-summer days:", non_summer_days)

# I expect is_summer to help because Seattle is generally warmer in
# summer than in other seasons. This feature identifies the season
# even when two days have similar minimum temperatures.


# --- Task 4: Baseline Model ---

print("\n--- Task 4: Baseline Model ---")

X = df[["temperature_2m_min"]].values
y = df["temperature_2m_max"].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

baseline_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
baseline_r2 = r2_score(y_test, y_pred)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
print("Slope:", model.coef_[0])
print("Baseline RMSE (°C):", baseline_rmse)
print("Baseline test R²:", baseline_r2)

# The slope estimates how much the predicted daily high changes when
# the daily low increases by 1°C.
#
# RMSE is measured in degrees Celsius. For example, an RMSE of 2.5°C
# means the typical prediction error is around 2.5°C in magnitude,
# although RMSE is not exactly the average absolute error.


# --- Task 5: Full Model ---

print("\n--- Task 5: Full Model ---")

feature_cols = [
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "is_summer",
]

X = df[feature_cols].values
y = df["temperature_2m_max"].values

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model_full = LinearRegression()
model_full.fit(X_train_full, y_train_full)

y_train_pred_full = model_full.predict(X_train_full)
y_pred_full = model_full.predict(X_test_full)

train_r2 = r2_score(y_train_full, y_train_pred_full)
test_r2 = r2_score(y_test_full, y_pred_full)
full_rmse = np.sqrt(mean_squared_error(y_test_full, y_pred_full))

print("Training R²:", train_r2)
print("Full model test R²:", test_r2)
print("Full model test RMSE (°C):", full_rmse)

print("\nFeature coefficients:")
for name, coef in zip(feature_cols, model_full.coef_):
    print(f"{name:22s}: {coef:+.3f}")

# Each coefficient estimates how the predicted daily high changes
# when that feature increases by one unit while the other features
# remain constant.
#
# A positive coefficient raises the predicted high, while a negative
# coefficient lowers it. For is_summer, the coefficient estimates
# the difference between summer and non-summer predictions, holding
# the other features constant.
#
# Compare train R² and test R². Similar scores suggest that the model
# performs similarly on training and unseen data. A much higher train
# score can indicate overfitting.
#
# Compare the full model test R² with baseline_r2 to see whether the
# additional features improved predictions.


# --- Task 6: Evaluate and Summarize ---

print("\n--- Task 6: Evaluate and Summarize ---")

# Plot predicted versus actual daily highs
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_full, y_test_full, alpha=0.7)

min_value = min(y_pred_full.min(), y_test_full.min())
max_value = max(y_pred_full.max(), y_test_full.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted Daily High (°C)")
plt.ylabel("Actual Daily High (°C)")
plt.tight_layout()
plt.savefig("outputs/predicted_vs_actual_high.png")
plt.close()

# Points above the diagonal represent underpredictions: actual values
# are higher than predicted. Points below it represent overpredictions.
# Look for larger deviations at the high or low end to see whether
# the model struggles more in one part of the temperature range.

# --- Summary ---

print("\n--- Project Summary ---")

print("City: Seattle, Washington")
print("Dataset size:", len(df), "days")
print("Test set size:", len(y_test_full), "days")
print(f"Baseline test R²: {baseline_r2:.3f}")
print(f"Full model test R²: {test_r2:.3f}")
print(f"Full model test RMSE: {full_rmse:.2f}°C")
print(f"Change in test R²: {test_r2 - baseline_r2:+.3f}")

largest_effect_index = np.argmax(np.abs(model_full.coef_))
largest_effect_name = feature_cols[largest_effect_index]
largest_effect_coef = model_full.coef_[largest_effect_index]

print(
    f"Largest absolute coefficient: {largest_effect_name} "
    f"({largest_effect_coef:+.3f})"
)

print(f"Training R²: {train_r2:.3f}")
print(f"Test R²: {test_r2:.3f}")

# Plain-language summary:
# - The dataset contains one year of Seattle weather observations.
# - The test set contains 20% of the usable observations.
# - RMSE estimates the typical prediction error on the Celsius scale.
# - R² describes how much variation in daily highs the model explains.
# - The largest absolute coefficient identifies the largest numerical
#   coefficient, but coefficients use different units and should not
#   automatically be treated as directly comparable effect sizes.
# - Use the printed scores and scatter plot to describe model accuracy,
#   whether the added features help, and one surprising result.
