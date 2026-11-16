
# --- scikit-learn API ---

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# Q1: Create, fit, and predict

temp_min = np.array([1, 4, 8, 12, 16, 20]).reshape(-1, 1)
temp_max = np.array([8, 10, 15, 18, 23, 27])

model = LinearRegression()
model.fit(temp_min, temp_max)

predictions = model.predict(np.array([6, 18]).reshape(-1, 1))

print("--- scikit-learn Q1 ---")
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Predicted high for low of 6 degrees:", predictions[0])
print("Predicted high for low of 18 degrees:", predictions[1])


# Q2: Reshape a 1D array into a 2D array

x = np.array([10, 20, 30, 40, 50])

print("\n--- scikit-learn Q2 ---")
print("Original shape:", x.shape)

x_2d = x.reshape(-1, 1)

print("New shape:", x_2d.shape)

# scikit-learn needs X to be 2D because rows represent observations
# and columns represent features, even when there is only one feature.


# --- Linear Regression ---

rng = np.random.default_rng(42)
num_days = 120
temp_min = rng.uniform(-5, 22, num_days)
is_rainy = rng.integers(0, 2, num_days).astype(float)
temp_max = 1.0 * temp_min + 6 - 4 * is_rainy + rng.normal(0, 2, num_days)


# Q1: Visualize daily high vs low

print("\n--- Linear Regression Q1 ---")

plt.figure(figsize=(8, 6))
plt.scatter(temp_min, temp_max, c=is_rainy, cmap="coolwarm")
plt.title("Daily High vs Low")
plt.xlabel("Daily Low Temperature")
plt.ylabel("Daily High Temperature")
plt.tight_layout()
plt.savefig("assignments_02/outputs/high_vs_low.png")
plt.close()

# Two visible groups appear because rainy days generally have lower
# high temperatures than dry days with similar low temperatures.
# This suggests that is_rainy affects the daily high.


# Q2: Split into training and test sets

print("\n--- Linear Regression Q2 ---")

X = temp_min.reshape(-1, 1)
y = temp_max

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# Q3: Fit a model using temperature low only

print("\n--- Linear Regression Q3 ---")

model = LinearRegression()
model.fit(X_train, y_train)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = model.score(X_test, y_test)

print("RMSE:", rmse)
print("Test R²:", r2)

# The slope estimates how many degrees the daily high increases
# on average when the daily low increases by one degree,
# without accounting for whether it rains.


# Q4: Add rain status as a second feature

print("\n--- Linear Regression Q4 ---")

X_full = np.column_stack([temp_min, is_rainy])

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)

model_full = LinearRegression()
model_full.fit(X_train_full, y_train_full)

y_pred_full = model_full.predict(X_test_full)

r2_full = model_full.score(X_test_full, y_test_full)

print("One-feature model test R²:", r2)
print("Two-feature model test R²:", r2_full)
print("temp_min coefficient:", model_full.coef_[0])
print("is_rainy coefficient:", model_full.coef_[1])

# The is_rainy coefficient estimates the difference in predicted
# daily high between rainy and dry days with the same daily low.
# A negative coefficient means rainy days tend to have lower highs.


# Q5: Plot predicted vs actual values

print("\n--- Linear Regression Q5 ---")

plt.figure(figsize=(8, 6))
plt.scatter(y_pred_full, y_test_full)

# The diagonal represents perfect predictions.
min_value = min(y_pred_full.min(), y_test_full.min())
max_value = max(y_pred_full.max(), y_test_full.max())
plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.title("Predicted vs Actual")
plt.xlabel("Predicted High Temperature")
plt.ylabel("Actual High Temperature")
plt.tight_layout()
plt.savefig("assignments_02/outputs/predicted_vs_actual_warmup.png")
plt.close()

# A point above the diagonal means the actual value is higher
# than predicted, so the model underestimated the high.
# A point below the diagonal means the actual value is lower
# than predicted, so the model overestimated the high.
