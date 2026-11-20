
import json
import os
import mlflow
import joblib

# Project paths
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

METRICS_PATH = os.path.join(
    BASE_DIR, "artifacts", "model_metrics.json"
)

MODEL_PATH = os.path.join(
    BASE_DIR, "artifacts", "fraud_detection_model.pkl"
)

# Load saved evaluation metrics
with open(METRICS_PATH, "r") as file:
    metrics = json.load(file)

# MLflow tracking configuration
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Real-Time-Financial-Fraud-Detection")

# Start an MLflow run
with mlflow.start_run(run_name="XGBoost_Fraud_Detection"):

    # Log model parameters
    mlflow.log_param("model", "XGBoost")
    mlflow.log_param("threshold", metrics["threshold"])
    mlflow.log_param("dataset", "IEEE-CIS Fraud Detection")
    mlflow.log_param("training_split", "80% chronological")
    mlflow.log_param("validation_split", "20% chronological")

    # Log evaluation metrics
    mlflow.log_metric("PR_AUC", metrics["pr_auc"])
    mlflow.log_metric("ROC_AUC", metrics["roc_auc"])

    # Log confusion matrix values
    tn = metrics["confusion_matrix"][0][0]
    fp = metrics["confusion_matrix"][0][1]
    fn = metrics["confusion_matrix"][1][0]
    tp = metrics["confusion_matrix"][1][1]

    mlflow.log_metric("True_Negatives", tn)
    mlflow.log_metric("False_Positives", fp)
    mlflow.log_metric("False_Negatives", fn)
    mlflow.log_metric("True_Positives", tp)

    # Calculate and log precision, recall, and F1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    mlflow.log_metric("Precision", precision)
    mlflow.log_metric("Recall", recall)
    mlflow.log_metric("F1_Score", f1)

    # Log trained model as an artifact
    model = joblib.load(MODEL_PATH)
    mlflow.log_artifact(
    MODEL_PATH,
    artifact_path="fraud_detection_model"
)

    print("MLflow experiment tracking completed!")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
