import pandas as pd
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "train_transaction.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "fraud_model_data.csv"

# Features selected for our first model
selected_columns = [
    "isFraud",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "P_emaildomain",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "D1",
    "D2",
    "D3"
]

# Categorical features
categorical_columns = [
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

print("Loading selected transaction features...")

df = pd.read_csv(
    INPUT_PATH,
    usecols=selected_columns
)

print("Dataset loaded:", df.shape)

# Convert categorical features to strings.
# Missing values are represented as "Unknown".
for column in categorical_columns:
    df[column] = df[column].astype("string").fillna("Unknown")

# Keep the target separate from the input features.
# Missing numerical values will be handled later in the ML pipeline.
print("\nChecking missing values...")

print(df.isnull().sum().sort_values(ascending=False).head(10))

# Save the prepared dataset
df.to_csv(OUTPUT_PATH, index=False)

print("\nPrepared dataset saved successfully!")
print("Output location:", OUTPUT_PATH)
print("Final dataset shape:", df.shape)

print("\nFraud distribution:")
print(df["isFraud"].value_counts())