
import joblib
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "fraud_model_data.csv"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "fraud_detection_model.pkl"

print("Loading dataset and trained model...")
df = pd.read_csv(DATA_PATH, low_memory=False)

df = df.sort_values("TransactionDT").reset_index(drop=True)

# Apply the same categorical preprocessing used during training
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

split_index = int(len(df) * 0.80)

X_valid = df.drop(columns=["isFraud"]).iloc[split_index:]
y_valid = df["isFraud"].iloc[split_index:]

model = joblib.load(MODEL_PATH)

print("Generating fraud probabilities...")

y_prob = model.predict_proba(X_valid)[:, 1]

print("\nPR-AUC:", round(
    average_precision_score(y_valid, y_prob), 4
))

thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]

results = []

print("\n========== THRESHOLD COMPARISON ==========")

for threshold in thresholds:

    y_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(
        y_valid, y_pred, zero_division=0
    )

    recall = recall_score(
        y_valid, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_valid, y_pred, zero_division=0
    )

    results.append({
        "Threshold": threshold,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1_Score": round(f1, 4),
        "Fraud_Alerts": int(y_pred.sum())
    })

results_df = pd.DataFrame(results)

print(results_df.to_string(index=False))

print("\nThreshold comparison completed!")
