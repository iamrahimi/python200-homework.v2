
# --- Setup ---

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)
import joblib

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

df = pd.read_csv("assignments_03/weather_classification.csv")
numeric = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
X = df[numeric]
y = df["good_for_running"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# --- Preprocessing ---

# Q1: StandardScaler

print("\n--- Preprocessing Q1 ---")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaled training column means:")
print(pd.Series(X_train_scaled.mean(axis=0), index=numeric))

# Fit the scaler only on training data to avoid leaking information
# from the test set into the training process.


# Q2: One-hot encode season

print("\n--- Preprocessing Q2 ---")

encoder = OneHotEncoder(sparse_output=False)
season_encoded = encoder.fit_transform(df[["season"]])

print("Encoded array shape:", season_encoded.shape)
print("Encoded category names:")
print(encoder.get_feature_names_out(["season"]))

# The season column becomes one column per unique season category.
# The number of columns equals the number of unique seasons in the data.


# Q3: ColumnTransformer and Pipeline

print("\n--- Preprocessing Q3 ---")

numeric_features = numeric
categorical_features = ["season"]

X_all = df[numeric + categorical_features]

X_train_pipe, X_test_pipe, y_train_pipe, y_test_pipe = train_test_split(
    X_all, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

preprocessing_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)

preprocessing_pipeline.fit(X_train_pipe, y_train_pipe)
preprocessing_accuracy = preprocessing_pipeline.score(
    X_test_pipe, y_test_pipe
)

print("Pipeline test accuracy:", preprocessing_accuracy)

# The pipeline automatically scales numeric columns, encodes the season,
# and passes the transformed data to logistic regression.


# --- KNN ---

# Q1: KNN with k=5

print("\n--- KNN Q1 ---")

knn_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
    ]
)

knn_pipeline.fit(X_train, y_train)
knn_predictions = knn_pipeline.predict(X_test)

print("KNN test accuracy:", accuracy_score(y_test, knn_predictions))
print("\nClassification report:")
print(classification_report(y_test, knn_predictions, zero_division=0))


# Q2: Compare different k values using cross-validation

print("\n--- KNN Q2 ---")

k_values = [1, 3, 5, 7, 9, 11, 15, 21]
cv_results = {}

for k in k_values:
    knn_cv_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k)),
        ]
    )

    scores = cross_val_score(
        knn_cv_pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy",
    )

    cv_results[k] = scores.mean()
    print(f"k={k:2d}, Mean CV accuracy: {scores.mean():.3f}")

best_k = max(cv_results, key=cv_results.get)
print("Best k by mean CV accuracy:", best_k)

# Choose the k with the highest mean cross-validation accuracy.
# If multiple values tie, consider choosing the larger, simpler option.


# Q3: Compare KNN with and without scaling

print("\n--- KNN Q3 ---")

knn_raw = KNeighborsClassifier(n_neighbors=5)
knn_raw.fit(X_train, y_train)
raw_predictions = knn_raw.predict(X_test)

knn_scaled = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
    ]
)

knn_scaled.fit(X_train, y_train)
scaled_predictions = knn_scaled.predict(X_test)

raw_accuracy = accuracy_score(y_test, raw_predictions)
scaled_accuracy = accuracy_score(y_test, scaled_predictions)

print("Accuracy without scaling:", raw_accuracy)
print("Accuracy with scaling:", scaled_accuracy)

# Scaling helps prevent features with larger numerical ranges from
# dominating distance calculations. Compare the two accuracies to see
# whether it helped on this dataset.


# --- Logistic Regression ---

# Q1: Evaluate logistic regression

print("\n--- Logistic Regression Q1 ---")

pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000)),
    ]
)

pipeline.fit(X_train, y_train)
logreg_predictions = pipeline.predict(X_test)

print("Test accuracy:", accuracy_score(y_test, logreg_predictions))
print(
    "Precision:",
    precision_score(y_test, logreg_predictions, pos_label=1, zero_division=0),
)
print(
    "Recall:",
    recall_score(y_test, logreg_predictions, pos_label=1, zero_division=0),
)
print(
    "F1:",
   f1_score(y_test, logreg_predictions, pos_label=1, zero_division=0),
)


# Q2: Examine logistic regression coefficients

print("\n--- Logistic Regression Q2 ---")

coefficients = pipeline.named_steps["logreg"].coef_[0]

for name, coef in zip(numeric, coefficients):
    print(f"{name:25s}: {coef:+.3f}")

# Positive coefficients push predictions toward the "good" class,
# while negative coefficients push predictions toward the other class.
# Because the inputs are scaled, the coefficients reflect changes per
# standard deviation of each feature, not per original unit.


# Q3: Predict probabilities for the first five test days

print("\n--- Logistic Regression Q3 ---")

probabilities = pipeline.predict_proba(X_test)
class_names = list(pipeline.named_steps["logreg"].classes_)
good_index = class_names.index(1)

good_probabilities = probabilities[:, good_index]

print("First five probabilities of good weather:")
print(np.round(good_probabilities[:5], 2))

# A probability near 0.5 means the model is relatively uncertain
# between the two classes.


# --- Evaluation ---

# Q1: Confusion matrix

print("\n--- Evaluation Q1 ---")

cm = confusion_matrix(
    y_test,
    logreg_predictions,
    labels=[0, 1],
)

print("Confusion matrix:")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["skip", "good"],
)

display.plot(cmap="Blues")
plt.title("Logistic Regression Confusion Matrix")
plt.tight_layout()
plt.savefig("assignments_03/outputs/logreg_confusion_matrix.png")
plt.close()

# Rows represent actual labels and columns represent predicted labels.
# False positives are actual "skip" days predicted as "good".
# False negatives are actual "good" days predicted as "skip".
#
# With labels ordered as ["skip", "good"]:
false_positives = cm[0, 1]
false_negatives = cm[1, 0]

print("False positives:", false_positives)
print("False negatives:", false_negatives)

# For a running app, false positives could be more concerning because
# the app might recommend running on an unsuitable day. False negatives
# could also be frustrating because a good running day is missed.


# --- joblib ---

# Q1: Save, load, and compare predictions

print("\n--- joblib Q1 ---")

joblib.dump(pipeline, "assignments_03/models/warmup_model.pkl")
loaded = joblib.load("assignments_03/models/warmup_model.pkl")

assert (loaded.predict(X_test) == pipeline.predict(X_test)).all()

print("Predictions match.")

# Saving only logistic regression would omit the fitted scaler.
# Predictions on raw data would then use a different feature scale
# than the model saw during training, potentially causing wrong results.
