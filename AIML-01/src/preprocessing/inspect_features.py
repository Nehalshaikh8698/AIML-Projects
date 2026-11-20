import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train_transaction.csv"

# Features we want to investigate
selected_columns = [
    "TransactionID",
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
    "D3",
    "M1",
    "M2",
    "M3"
]

df = pd.read_csv(DATA_PATH, usecols=selected_columns)

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
missing = df.isnull().sum()
missing_percentage = (missing / len(df)) * 100

missing_report = pd.DataFrame({
    "Missing_Count": missing,
    "Missing_Percentage": missing_percentage
})

print(missing_report.sort_values(
    by="Missing_Percentage",
    ascending=False
))

print("\n========== CATEGORICAL FEATURE VALUES ==========")

for column in ["ProductCD", "card4", "card6", "P_emaildomain"]:
    print(f"\n{column}:")
    print(df[column].value_counts().head(10))

print("\n========== FRAUD VS TRANSACTION AMOUNT ==========")

print(
    df.groupby("isFraud")["TransactionAmt"]
    .agg(["count", "mean", "median", "min", "max"])
)

print("\nFeature inspection completed!")