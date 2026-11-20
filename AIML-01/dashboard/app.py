
import streamlit as st
import requests
import pandas as pd
from datetime import datetime

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="💳",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000/predict"
BATCH_API_URL = "http://127.0.0.1:8000/predict/batch"

# Store transaction history during this dashboard session
if "transaction_history" not in st.session_state:
    st.session_state.transaction_history = []

# ---------------- DASHBOARD HEADER ----------------

st.title("💳 Financial Fraud Detection Dashboard")

st.write(
    "Monitor financial transactions and identify "
    "potentially fraudulent activities using machine learning."
)

st.divider()

# ---------------- TRANSACTION INPUT ----------------

st.header("🔍 Analyze a Transaction")

with st.form("transaction_form"):

    col1, col2 = st.columns(2)

    with col1:
        TransactionDT = st.number_input(
            "Transaction Time",
            min_value=0.0,
            value=86400.0
        )

        TransactionAmt = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=150.0
        )

        ProductCD = st.selectbox(
            "Product Code",
            ["W", "C", "R", "H", "S"]
        )

        card1 = st.text_input("Card 1", "12345")
        card2 = st.text_input("Card 2", "111")
        card3 = st.text_input("Card 3", "150")

        card4 = st.selectbox(
            "Card Type",
            ["visa", "mastercard", "american express", "discover"]
        )

        card5 = st.text_input("Card 5", "226")

        card6 = st.selectbox(
            "Card Category",
            ["debit", "credit"]
        )

        addr1 = st.text_input("Address Code", "315")

        P_emaildomain = st.text_input(
            "Email Domain", "gmail.com"
        )

    with col2:
        C1 = st.number_input("C1", value=1.0)
        C2 = st.number_input("C2", value=1.0)
        C3 = st.number_input("C3", value=0.0)
        C4 = st.number_input("C4", value=0.0)
        C5 = st.number_input("C5", value=1.0)
        C6 = st.number_input("C6", value=1.0)

        D1 = st.number_input("D1", value=10.0)
        D2 = st.number_input("D2", value=10.0)
        D3 = st.number_input("D3", value=5.0)

    submitted = st.form_submit_button(
        "🔍 Predict Fraud",
        use_container_width=True
    )

# ---------------- SINGLE TRANSACTION PREDICTION ----------------

if submitted:

    transaction_data = {
        "TransactionDT": TransactionDT,
        "TransactionAmt": TransactionAmt,
        "ProductCD": ProductCD,
        "card1": card1,
        "card2": card2,
        "card3": card3,
        "card4": card4,
        "card5": card5,
        "card6": card6,
        "addr1": addr1,
        "P_emaildomain": P_emaildomain,
        "C1": C1,
        "C2": C2,
        "C3": C3,
        "C4": C4,
        "C5": C5,
        "C6": C6,
        "D1": D1,
        "D2": D2,
        "D3": D3
    }

    try:
        response = requests.post(
            API_URL,
            json=transaction_data,
            timeout=30
        )

        if response.status_code == 200:

            result = response.json()

            probability = result["fraud_probability"]
            prediction = result["prediction"]

            record = {
                "Time": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "Amount": TransactionAmt,
                "Fraud Probability (%)": round(
                    probability * 100, 2
                ),
                "Prediction": prediction
            }

            st.session_state.transaction_history.append(record)

            st.success("Transaction analyzed successfully!")

            st.subheader("Prediction Result")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Fraud Probability",
                    f"{probability * 100:.2f}%"
                )

            with col2:
                st.metric(
                    "Classification Threshold",
                    f"{result['threshold'] * 100:.0f}%"
                )

            st.progress(probability)

            if prediction == "Fraudulent":
                st.error(
                    "⚠️ Potential Fraudulent Transaction Detected!"
                )
            else:
                st.success(
                    "✅ Transaction Classified as Legitimate"
                )

        else:
            st.error(f"API Error: {response.status_code}")
            st.write(response.text)

    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to FastAPI. "
            "Please make sure the API server is running."
        )

    except requests.exceptions.RequestException as error:
        st.error(f"Request failed: {error}")

# ---------------- SINGLE TRANSACTION ANALYTICS ----------------

st.divider()

st.header("📊 Fraud Analytics")

history = st.session_state.transaction_history

if history:

    history_df = pd.DataFrame(history)

    total_transactions = len(history_df)

    fraud_count = (
        history_df["Prediction"] == "Fraudulent"
    ).sum()

    legitimate_count = (
        history_df["Prediction"] == "Legitimate"
    ).sum()

    fraud_rate = (
        fraud_count / total_transactions
    ) * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Transactions",
        total_transactions
    )

    col2.metric(
        "Potential Fraud",
        int(fraud_count)
    )

    col3.metric(
        "Legitimate",
        int(legitimate_count)
    )

    col4.metric(
        "Potential Fraud Rate",
        f"{fraud_rate:.2f}%"
    )

    st.subheader("Transaction Distribution")

    chart_data = pd.DataFrame({
        "Transaction Type": [
            "Fraudulent",
            "Legitimate"
        ],
        "Count": [
            int(fraud_count),
            int(legitimate_count)
        ]
    })

    st.bar_chart(
        chart_data.set_index("Transaction Type")
    )

    st.subheader("Transaction History")

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    if st.button("Clear Transaction History"):
        st.session_state.transaction_history = []
        st.rerun()

else:
    st.info(
        "No transactions analyzed yet. "
        "Submit a transaction above to see analytics."
    )

# ---------------- CSV BATCH PREDICTION ----------------

st.divider()

st.header("📁 Bulk Fraud Detection")

st.write(
    "Upload a CSV file containing multiple transactions "
    "to predict fraud for all transactions at once."
)

uploaded_file = st.file_uploader(
    "Upload Transaction CSV",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        csv_data = pd.read_csv(uploaded_file)

        st.subheader("CSV Preview")

        st.dataframe(
            csv_data.head(),
            use_container_width=True
        )

        st.write(
            f"Total rows uploaded: {len(csv_data)}"
        )

        required_columns = [
            "TransactionDT", "TransactionAmt", "ProductCD",
            "card1", "card2", "card3", "card4", "card5",
            "card6", "addr1", "P_emaildomain",
            "C1", "C2", "C3", "C4", "C5", "C6",
            "D1", "D2", "D3"
        ]

        missing_columns = [
            column for column in required_columns
            if column not in csv_data.columns
        ]

        if missing_columns:

            st.error(
                "Missing required columns: "
                + ", ".join(missing_columns)
            )

        elif st.button("🚀 Predict All Transactions"):

            batch_data = csv_data[required_columns].copy()

            categorical_columns = [
                "ProductCD", "card1", "card2", "card3",
                "card4", "card5", "card6", "addr1",
                "P_emaildomain"
            ]

            for column in categorical_columns:
                batch_data[column] = (
                    batch_data[column]
                    .fillna("Unknown")
                    .astype(str)
                )

            batch_data = batch_data.astype(object)

            batch_data = batch_data.where(
                pd.notna(batch_data), None
            )

            transactions = batch_data.to_dict(
                orient="records"
            )

            try:

                with st.spinner(
                    "Analyzing transactions..."
                ):

                    response = requests.post(
                        BATCH_API_URL,
                        json={"transactions": transactions},
                        timeout=180
                    )

                if response.status_code == 200:

                    result = response.json()

                    st.success(
                        "Batch prediction completed successfully!"
                    )

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Total Transactions",
                        result["total_transactions"]
                    )

                    col2.metric(
                        "Potential Fraud",
                        result["fraudulent_transactions"]
                    )

                    col3.metric(
                        "Legitimate",
                        result["legitimate_transactions"]
                    )

                    # ---------------- RESULTS TABLE ----------------

                    results_df = pd.DataFrame(
                        result["results"]
                    )

                    results_df.insert(
                        1,
                        "Transaction Amount",
                        csv_data["TransactionAmt"].values
                    )

                    results_df["Fraud Probability (%)"] = (
                        results_df["fraud_probability"] * 100
                    ).round(2)

                    st.subheader("Batch Prediction Results")

                    st.dataframe(
                        results_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    # ---------------- FRAUD PROBABILITY CHART ----------------

                    st.divider()

                    st.subheader(
                        "📈 Fraud Probability Analysis"
                    )

                    results_df["Transaction Label"] = (
                        "Transaction "
                        + results_df["transaction_number"].astype(str)
                    )

                    probability_chart = results_df[
                        [
                            "Transaction Label",
                            "Fraud Probability (%)"
                        ]
                    ].set_index("Transaction Label")

                    st.write(
                        "Fraud probability for each transaction"
                    )

                    st.bar_chart(
                        probability_chart
                    )

                    # ---------------- RISK LEVEL ANALYSIS ----------------

                    st.subheader(
                        "🚨 Transaction Risk Levels"
                    )

                    def classify_risk(probability):

                        if probability >= 0.70:
                            return "High Risk"

                        elif probability >= 0.40:
                            return "Medium Risk"

                        else:
                            return "Low Risk"

                    results_df["Risk Level"] = (
                        results_df["fraud_probability"]
                        .apply(classify_risk)
                    )

                    risk_counts = (
                        results_df["Risk Level"].value_counts()
                    )

                    risk_summary = pd.DataFrame({
                        "Risk Level": [
                            "High Risk",
                            "Medium Risk",
                            "Low Risk"
                        ],
                        "Number of Transactions": [
                            risk_counts.get("High Risk", 0),
                            risk_counts.get("Medium Risk", 0),
                            risk_counts.get("Low Risk", 0)
                        ]
                    })

                    st.bar_chart(
                        risk_summary.set_index("Risk Level")
                    )

                    # ---------------- HIGH RISK ALERTS ----------------

                    high_risk_transactions = results_df[
                        results_df["Risk Level"] == "High Risk"
                    ]

                    if not high_risk_transactions.empty:

                        st.error(
                            f"⚠️ {len(high_risk_transactions)} "
                            "high-risk transaction(s) detected!"
                        )

                        st.subheader(
                            "High-Risk Transaction Details"
                        )

                        st.dataframe(
                            high_risk_transactions,
                            use_container_width=True,
                            hide_index=True
                        )

                    else:

                        st.success(
                            "No high-risk transactions detected."
                        )

                    # ---------------- DOWNLOAD RESULTS ----------------

                    st.divider()

                    st.subheader(
                        "⬇️ Download Prediction Report"
                    )

                    csv_output = results_df.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        "Download Prediction Results",
                        data=csv_output,
                        file_name="fraud_predictions.csv",
                        mime="text/csv"
                    )

                else:

                    st.error(
                        f"API Error: {response.status_code}"
                    )

                    st.write(response.text)

            except requests.exceptions.RequestException as error:

                st.error(
                    f"Batch prediction failed: {error}"
                )

    except Exception as error:

        st.error(
            f"Unable to read CSV file: {error}"
        )
