
import json
import joblib
import pandas as pd

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score
)

from xgboost import XGBClassifier


# ---------------------------------------
# 1. Project paths
# ---------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "fraud_model_data.csv"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "fraud_detection_model.pkl"
METRICS_PATH = PROJECT_ROOT / "artifacts" / "model_metrics.json"


# ---------------------------------------
# 2. Load prepared dataset
# ---------------------------------------

print("Loading prepared dataset...")

df = pd.read_csv(DATA_PATH, low_memory=False)

# Sort transactions chronologically
df = df.sort_values("TransactionDT").reset_index(drop=True)

# Convert all categorical columns to consistent strings
categorical_features = [
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "P_emaildomain"
]

for column in categorical_features:
    df[column] = df[column].astype("string").fillna("Unknown")

X = df.drop(columns=["isFraud"])
y = df["isFraud"]

print("Dataset shape:", df.shape)


# ---------------------------------------
# 3. Identify feature types
# ---------------------------------------

categorical_features = [
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "P_emaildomain"
]

numerical_features = [
    column for column in X.columns
    if column not in categorical_features
]

print("\nCategorical features:", len(categorical_features))
print("Numerical features:", len(numerical_features))


# ---------------------------------------
# 4. Split data chronologically
# ---------------------------------------

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_valid = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_valid = y.iloc[split_index:]

print("\nTraining records:", len(X_train))
print("Validation records:", len(X_valid))

print("\nTraining fraud distribution:")
print(y_train.value_counts())

print("\nValidation fraud distribution:")
print(y_valid.value_counts())


# ---------------------------------------
# 5. Build preprocessing pipeline
# ---------------------------------------

numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)


# ---------------------------------------
# 6. Calculate class imbalance
# ---------------------------------------

negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())

scale_pos_weight = negative_count / positive_count

print("\nClass imbalance ratio:", round(scale_pos_weight, 2))


# ---------------------------------------
# 7. Create XGBoost model
# ---------------------------------------

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)


# ---------------------------------------
# 8. Combine preprocessing and model
# ---------------------------------------

model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", xgb_model)
])


# ---------------------------------------
# 9. Train the model
# ---------------------------------------

print("\nTraining XGBoost model...")
print("This may take a few minutes.")

model_pipeline.fit(X_train, y_train)

print("\nModel training completed!")


# ---------------------------------------
# 10. Make predictions
# ---------------------------------------

print("\nEvaluating model...")

y_prob = model_pipeline.predict_proba(X_valid)[:, 1]

# Initial classification threshold
threshold = 0.5

y_pred = (y_prob >= threshold).astype(int)


# ---------------------------------------
# 11. Evaluate model
# ---------------------------------------

print("\n========== CLASSIFICATION REPORT ==========")

report = classification_report(
    y_valid,
    y_pred,
    zero_division=0
)

print(report)

print("\n========== CONFUSION MATRIX ==========")

cm = confusion_matrix(y_valid, y_pred)

print(cm)

print("\n========== MODEL METRICS ==========")

pr_auc = average_precision_score(y_valid, y_prob)
roc_auc = roc_auc_score(y_valid, y_prob)

print(f"PR-AUC: {pr_auc:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

metrics = {
    "threshold": threshold,
    "pr_auc": float(pr_auc),
    "roc_auc": float(roc_auc),
    "confusion_matrix": cm.tolist()
}


# ---------------------------------------
# 12. Save model and metrics
# ---------------------------------------

joblib.dump(model_pipeline, MODEL_PATH)

with open(METRICS_PATH, "w") as file:
    json.dump(metrics, file, indent=4)

print("\nModel saved at:", MODEL_PATH)
print("Metrics saved at:", METRICS_PATH)

print("\nXGBoost fraud detection training completed successfully!")
