"""
app_streamlit.py
-----------------
Streamlit web app for the Credit Score classification model.

Run locally:
    streamlit run app_streamlit.py

Expects the following artifacts (produced by `src/pipeline.py`) in ./artifacts:
    - best_model.pkl       (the winning tree-based classifier)
    - preprocessor.pkl     (fitted FeaturePreprocessor: imputers/scalers/encoders)
    - target_encoder.pkl   (fitted TargetEncoder for the 3 credit-score classes)
    - best_model_name.txt  (name of the winning model, for display)
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import preprocessing

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August"]
OCCUPATIONS = [
    "Accountant", "Architect", "Developer", "Doctor", "Engineer", "Entrepreneur",
    "Journalist", "Lawyer", "Manager", "Mechanic", "Media_Manager", "Musician",
    "Scientist", "Teacher", "Writer",
]
CREDIT_MIX = ["Good", "Standard", "Bad"]
PAYMENT_MIN_AMOUNT = ["Yes", "No", "NM"]
PAYMENT_BEHAVIOUR = [
    "High_spent_Large_value_payments", "High_spent_Medium_value_payments",
    "High_spent_Small_value_payments", "Low_spent_Large_value_payments",
    "Low_spent_Medium_value_payments", "Low_spent_Small_value_payments",
]

FEATURE_ORDER = [
    "Month", "Age", "Occupation", "Annual_Income", "Monthly_Inhand_Salary",
    "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan",
    "Delay_from_due_date", "Num_of_Delayed_Payment", "Changed_Credit_Limit",
    "Num_Credit_Inquiries", "Credit_Mix", "Outstanding_Debt",
    "Credit_Utilization_Ratio", "Payment_of_Min_Amount", "Total_EMI_per_month",
    "Amount_invested_monthly", "Payment_Behaviour", "Monthly_Balance",
    "Credit_History_Age_Months",
]


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(ARTIFACT_DIR, "best_model.pkl"))
    preprocessor = joblib.load(os.path.join(ARTIFACT_DIR, "preprocessor.pkl"))
    target_encoder = joblib.load(os.path.join(ARTIFACT_DIR, "target_encoder.pkl"))
    best_name_path = os.path.join(ARTIFACT_DIR, "best_model_name.txt")
    best_name = open(best_name_path).read().strip() if os.path.exists(best_name_path) else "Best Model"
    return model, preprocessor, target_encoder, best_name


def build_input_form():
    st.subheader("Input Data Nasabah")
    col1, col2, col3 = st.columns(3)

    with col1:
        month = st.selectbox("Month", MONTHS)
        age = st.number_input("Age", min_value=14, max_value=100, value=30)
        occupation = st.selectbox("Occupation", OCCUPATIONS)
        annual_income = st.number_input("Annual Income", min_value=0.0, value=45000.0, step=1000.0)
        monthly_inhand_salary = st.number_input("Monthly Inhand Salary", min_value=0.0, value=3500.0, step=100.0)
        num_bank_accounts = st.number_input("Num Bank Accounts", min_value=0, max_value=20, value=4)
        num_credit_card = st.number_input("Num Credit Card", min_value=0, max_value=20, value=5)

    with col2:
        interest_rate = st.number_input("Interest Rate (%)", min_value=0, max_value=40, value=13)
        num_of_loan = st.number_input("Num of Loan", min_value=0, max_value=15, value=3)
        delay_from_due_date = st.number_input("Delay from Due Date (days)", min_value=0, max_value=120, value=18)
        num_of_delayed_payment = st.number_input("Num of Delayed Payment", min_value=0, max_value=30, value=12)
        changed_credit_limit = st.number_input("Changed Credit Limit", value=9.0, step=0.5)
        num_credit_inquiries = st.number_input("Num Credit Inquiries", min_value=0, max_value=30, value=5)
        credit_mix = st.selectbox("Credit Mix", CREDIT_MIX)

    with col3:
        outstanding_debt = st.number_input("Outstanding Debt", min_value=0.0, value=1200.0, step=50.0)
        credit_utilization_ratio = st.number_input("Credit Utilization Ratio", min_value=0.0, max_value=100.0, value=32.0)
        payment_of_min_amount = st.selectbox("Payment of Min Amount", PAYMENT_MIN_AMOUNT)
        total_emi_per_month = st.number_input("Total EMI per Month", min_value=0.0, value=100.0, step=10.0)
        amount_invested_monthly = st.number_input("Amount Invested Monthly", min_value=0.0, value=150.0, step=10.0)
        payment_behaviour = st.selectbox("Payment Behaviour", PAYMENT_BEHAVIOUR)
        monthly_balance = st.number_input("Monthly Balance", value=350.0, step=10.0)

    credit_history_age_months = st.slider(
        "Credit History Age (months)", min_value=0, max_value=420, value=220
    )

    data = {
        "Month": month, "Age": age, "Occupation": occupation,
        "Annual_Income": annual_income, "Monthly_Inhand_Salary": monthly_inhand_salary,
        "Num_Bank_Accounts": num_bank_accounts, "Num_Credit_Card": num_credit_card,
        "Interest_Rate": interest_rate, "Num_of_Loan": num_of_loan,
        "Delay_from_due_date": delay_from_due_date, "Num_of_Delayed_Payment": num_of_delayed_payment,
        "Changed_Credit_Limit": changed_credit_limit, "Num_Credit_Inquiries": num_credit_inquiries,
        "Credit_Mix": credit_mix, "Outstanding_Debt": outstanding_debt,
        "Credit_Utilization_Ratio": credit_utilization_ratio,
        "Payment_of_Min_Amount": payment_of_min_amount, "Total_EMI_per_month": total_emi_per_month,
        "Amount_invested_monthly": amount_invested_monthly, "Payment_Behaviour": payment_behaviour,
        "Monthly_Balance": monthly_balance, "Credit_History_Age_Months": credit_history_age_months,
    }
    return pd.DataFrame([data])[FEATURE_ORDER]


def plot_probabilities(classes, probabilities):
    colors = {"Good": "#2ecc71", "Standard": "#f1c40f", "Poor": "#e74c3c"}
    bar_colors = [colors.get(c, "#3498db") for c in classes]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(classes, probabilities, color=bar_colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Predicted Class Probability Distribution")
    for bar, p in zip(bars, probabilities):
        ax.text(bar.get_x() + bar.get_width() / 2, p + 0.02, f"{p:.1%}", ha="center", fontweight="bold")
    fig.tight_layout()
    st.pyplot(fig)


def main():
    st.set_page_config(page_title="Credit Score Classifier", page_icon="💳", layout="wide")
    st.title("💳 Credit Score Classification")
    st.caption("Prediksi kategori credit score nasabah: Good / Standard / Poor")

    if not os.path.exists(os.path.join(ARTIFACT_DIR, "best_model.pkl")):
        st.error(
            "Artifacts belum ditemukan. Jalankan `python src/pipeline.py` terlebih dahulu "
            "untuk menghasilkan artifacts/best_model.pkl, preprocessor.pkl, dan target_encoder.pkl."
        )
        st.stop()

    model, preprocessor, target_encoder, best_name = load_artifacts()
    st.sidebar.header("Model Info")
    st.sidebar.write(f"**Model digunakan:** {best_name}")
    st.sidebar.write(f"**Kelas target:** {', '.join(target_encoder.classes_)}")

    input_df = build_input_form()

    if st.button("Predict Credit Score", type="primary"):
        X_transformed = preprocessor.transform(input_df)
        pred_encoded = model.predict(X_transformed)[0]
        pred_label = target_encoder.inverse_transform([pred_encoded])[0]
        probabilities = model.predict_proba(X_transformed)[0]

        st.divider()
        st.subheader("Hasil Prediksi")

        badge_color = {"Good": "green", "Standard": "orange", "Poor": "red"}.get(pred_label, "blue")
        st.markdown(f"### Predicted Credit Score: :{badge_color}[{pred_label}]")

        col_a, col_b = st.columns([1, 1])
        with col_a:
            plot_probabilities(list(target_encoder.classes_), probabilities)
        with col_b:
            prob_df = pd.DataFrame({
                "Class": target_encoder.classes_,
                "Probability": probabilities,
            }).sort_values("Probability", ascending=False)
            st.dataframe(prob_df, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
