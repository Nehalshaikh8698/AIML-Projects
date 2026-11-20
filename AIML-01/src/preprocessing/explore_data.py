import pandas as pd
from pathlib import Path

# Find the project directory automatically
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset location
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train_transaction.csv"

# Process the large CSV in smaller chunks to save memory
chunk_size = 100_000

total_rows = 0
fraud_counts = {0: 0, 1: 0}
missing_values = None
sample_data = None
columns = None

print("Loading and exploring the training dataset...")

for chunk in pd.read_csv(DATA_PATH, chunksize=chunk_size):

    total_rows += len(chunk)

    # Count fraudulent and non-fraudulent transactions
    counts = chunk["isFraud"].value_counts()

    fraud_counts[0] += int(counts.get(0, 0))
    fraud_counts[1] += int(counts.get(1, 0))

    # Count missing values in every column
    chunk_missing = chunk.isnull().sum()

    if missing_values is None:
        missing_values = chunk_missing
        columns = chunk.columns.tolist()
        sample_data = chunk.head(5)
    else:
        missing_values += chunk_missing

print("\n========== DATASET OVERVIEW ==========")

print(f"Total rows: {total_rows}")
print(f"Total columns: {len(columns)}")

print("\nFirst 5 rows:")
print(sample_data.head())

print("\n========== FRAUD DISTRIBUTION ==========")

total_fraud = fraud_counts[1]
total_non_fraud = fraud_counts[0]

print(f"Non-fraud transactions: {total_non_fraud}")
print(f"Fraud transactions: {total_fraud}")

fraud_percentage = (total_fraud / total_rows) * 100

print(f"Fraud percentage: {fraud_percentage:.2f}%")

print("\n========== TOP 20 COLUMNS WITH MISSING VALUES ==========")

missing_summary = missing_values.sort_values(ascending=False).head(20)

for column, count in missing_summary.items():
    percentage = (count / total_rows) * 100
    print(f"{column}: {count} missing ({percentage:.2f}%)")

print("\nDataset exploration completed successfully!")