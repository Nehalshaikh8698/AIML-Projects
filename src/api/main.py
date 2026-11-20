
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# Initialize FastAPI
app = FastAPI(
    title="Real-Time Financial Fraud Detection API",
    description="API for predicting fraudulent financial transactions",
    version="1.0"
)

# Load trained model
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

MODEL_PATH = os.path.join(
    BASE_DIR, "artifacts", "fraud_detection_model.pkl"
)

model = joblib.load(MODEL_PATH)

# Define transaction input structure
class Transaction(BaseModel):
    TransactionDT: float
    TransactionAmt: float
    ProductCD: str
    card1: str
    card2: str
    card3: str
    card4: str
    card5: str
    card6: str
    addr1: str
    P_emaildomain: str
    C1: float
    C2: float
    C3: float
    C4: float
    C5: float
    C6: float
    D1: float
    D2: float
    D3: float

# Root endpoint
@app.get("/")
def home():
    return {
        "message": "Financial Fraud Detection API is running"
    }

# Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

# Fraud prediction endpoint
@app.post("/predict")
def predict_fraud(transaction: Transaction):

    # Convert input into DataFrame
    transaction_data = pd.DataFrame(
        [transaction.model_dump()]
    )

    # Generate fraud probability
    probability = float(
        model.predict_proba(transaction_data)[0][1]
    )

    # Apply selected threshold
    threshold = 0.70

    prediction = (
        "Fraudulent"
        if probability >= threshold
        else "Legitimate"
    )

    return {
        "fraud_probability": round(probability, 4),
        "prediction": prediction,
        "threshold": threshold
    }

from typing import List

class BatchTransactions(BaseModel):
    transactions: List[Transaction]


@app.post("/predict/batch")
def predict_batch(batch: BatchTransactions):

    transaction_data = pd.DataFrame(
        [transaction.model_dump() for transaction in batch.transactions]
    )

    probabilities = model.predict_proba(transaction_data)[:, 1]

    threshold = 0.70

    predictions = []

    for index, probability in enumerate(probabilities):

        prediction = (
            "Fraudulent"
            if probability >= threshold
            else "Legitimate"
        )

        predictions.append({
            "transaction_number": index + 1,
            "fraud_probability": round(float(probability), 4),
            "prediction": prediction
        })

    return {
        "total_transactions": len(predictions),
        "fraudulent_transactions": sum(
            item["prediction"] == "Fraudulent"
            for item in predictions
        ),
        "legitimate_transactions": sum(
            item["prediction"] == "Legitimate"
            for item in predictions
        ),
        "results": predictions
    }
