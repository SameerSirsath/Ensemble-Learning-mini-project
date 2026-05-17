import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- CUSTOM CSS FOR BRIGHT/AESTHETIC LOOK ----------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e9eef5 100%);
    }
    .css-1r6slb0, .css-1v3fvcr, .stButton button {
        border-radius: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        background-color: rgba(255,255,255,0.9);
        backdrop-filter: blur(2px);
    }
    h1, h2, h3 {
        color: #1e3c72;
        font-weight: 600;
    }
    .stButton button {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: transform 0.2s;
    }
    .stButton button:hover {
        transform: scale(1.02);
        background: linear-gradient(90deg, #00c6fb 0%, #005bea 100%);
    }
    .css-1d391kg {
        background-color: #ffffffcc;
        backdrop-filter: blur(10px);
        border-radius: 20px;
        margin: 10px;
        padding: 10px;
    }
    .stMetric {
        background: white;
        border-radius: 20px;
        padding: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .stAlert {
        border-radius: 15px;
        border-left: 5px solid;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ---------- LOAD MODELS & RESOURCES ----------
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        "Logistic Regression": "../models/logistic_regression.pkl",
        "Decision Tree": "../models/decision_tree.pkl",
        "Naïve Bayes": "../models/naive_bayes.pkl",
        "XGBoost": "../models/xgboost.pkl",
        "AdaBoost": "../models/adaboost.pkl"
    }
    for name, path in model_files.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            st.warning(f"⚠️ Model {name} not found. Please train it first.")
    return models


@st.cache_data
def load_feature_names():
    with open("../data/processed/feature_names.txt", "r") as f:
        return [line.strip() for line in f]


@st.cache_resource
def load_scaler():
    return joblib.load("../models/scaler.pkl")


def preprocess(raw_dict, scaler, expected_features):
    """Exact preprocessing as in training notebook."""
    df = pd.DataFrame([raw_dict])
    df['balance_diff_orig'] = df['newbalanceOrig'] - df['oldbalanceOrg']
    df['balance_diff_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
    df['error_orig'] = abs(df['oldbalanceOrg'] - df['newbalanceOrig'] - df['amount'])
    df['error_dest'] = abs(df['oldbalanceDest'] - df['newbalanceDest'] - df['amount'])
    df['amount_to_orig_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
    df.drop(['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'], axis=1, inplace=True)
    type_dummies = pd.get_dummies(df['type'], prefix='type')
    df = pd.concat([df, type_dummies], axis=1)
    df.drop('type', axis=1, inplace=True)
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_features]
    return scaler.transform(df)


# Load everything
models = load_models()
scaler = load_scaler()
feature_names = load_feature_names()

if not models:
    st.error("❌ No models loaded. Please train and save models first.")
    st.stop()

# ---------- PERFORMANCE COMPARISON TABLE (mock values – replace with actual) ----------
perf_data = {
    "Model": list(models.keys()),
    "Accuracy": [0.95, 0.96, 0.92, 0.99, 0.98][:len(models)],
    "Precision": [0.92, 0.94, 0.88, 0.99, 0.97][:len(models)],
    "Recall": [0.88, 0.90, 0.85, 0.98, 0.96][:len(models)],
    "F1": [0.90, 0.92, 0.86, 0.98, 0.96][:len(models)],
    "ROC-AUC": [0.96, 0.97, 0.91, 0.99, 0.98][:len(models)]
}
perf_df = pd.DataFrame(perf_data)

# ---------- SIDEBAR: USER INPUTS ----------
st.sidebar.image("https://img.icons8.com/color/96/000000/fraud-detection.png", width=80)
st.sidebar.title("🔍 Transaction Details")

# Auto-calculate toggle
auto_calc = st.sidebar.checkbox("✨ Auto‑calculate new balances (recommended)", value=True)
st.sidebar.caption(
    "When enabled, Sender's New Balance = Old Balance - Amount, and Receiver's New Balance = Old Balance + Amount. You can still edit manually.")

with st.sidebar.form("input_form"):
    step = st.number_input("Step (time unit)", min_value=0, value=1, help="Hour or sequential step")
    type_val = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"])
    amount = st.number_input("Amount", min_value=0.0, value=1000.0)
    oldbalanceOrg = st.number_input("Sender's Old Balance", min_value=0.0, value=5000.0)
    oldbalanceDest = st.number_input("Receiver's Old Balance", min_value=0.0, value=2000.0)

    # Auto-calculate new balances if enabled
    if auto_calc:
        # Standard assumption: sender loses amount, receiver gains amount
        default_new_orig = oldbalanceOrg - amount
        default_new_dest = oldbalanceDest + amount
        # Ensure non-negative (balances cannot be negative in real banking)
        default_new_orig = max(0.0, default_new_orig)
        default_new_dest = max(0.0, default_new_dest)
    else:
        default_new_orig = 0.0
        default_new_dest = 0.0

    newbalanceOrig = st.number_input("Sender's New Balance", min_value=0.0, value=default_new_orig, step=100.0)
    newbalanceDest = st.number_input("Receiver's New Balance", min_value=0.0, value=default_new_dest, step=100.0)

    submitted = st.form_submit_button("🔮 Predict Fraud")

# ---------- MAIN AREA: TITLE & METRICS ----------
st.title("💳 Online Payment Fraud Detection")
st.markdown("### *Intelligent, real‑time fraud prevention*")

col1, col2 = st.columns(2)
with col1:
    st.subheader("📊 Model Performance")
    st.dataframe(perf_df, use_container_width=True)
with col2:
    st.subheader("✨ Smart Selector")
    # st.markdown("""
    # **Auto‑select best model** – for each transaction, the system runs **all models** and picks the one with the **highest fraud probability** (most confident).  
    # You can also choose a model manually below.
    # """)

# Manual override option
manual_override = st.checkbox("🔧 Choose model manually (disable auto-select)")
if manual_override:
    selected_model_name = st.selectbox("Select Model", list(models.keys()))
else:
    selected_model_name = None  # will be auto

# ---------- PREDICTION LOGIC ----------
if submitted:
    raw_input = {
        'step': step,
        'type': type_val,
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest
    }

    try:
        processed = preprocess(raw_input, scaler, feature_names)

        if not manual_override:
            # Auto select: run all models, pick highest fraud probability
            best_model = None
            best_prob = -1.0
            best_proba_dict = {}
            for name, model in models.items():
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(processed)[0][1]
                    best_proba_dict[name] = proba
                    if proba > best_prob:
                        best_prob = proba
                        best_model = name
                else:
                    pred = model.predict(processed)[0]
                    proba = 1.0 if pred == 1 else 0.0
                    best_proba_dict[name] = proba
                    if proba > best_prob:
                        best_prob = proba
                        best_model = name

            selected_model = models[best_model]
            selected_model_name = best_model
            fraud_prob = best_prob
            prediction = 1 if fraud_prob >= 0.5 else 0

            st.info(f"🤖 **Auto-selected model:** `{best_model}` (highest fraud probability = {fraud_prob:.2%})")

            # Bar chart of all model probabilities
            prob_df = pd.DataFrame(list(best_proba_dict.items()), columns=["Model", "Fraud Probability"])
            prob_df = prob_df.sort_values("Fraud Probability", ascending=False)
            st.caption("🔍 All model probabilities:")
            st.bar_chart(prob_df.set_index("Model"))
        else:
            # Manual selection
            model = models[selected_model_name]
            if hasattr(model, "predict_proba"):
                fraud_prob = model.predict_proba(processed)[0][1]
                prediction = model.predict(processed)[0]
            else:
                prediction = model.predict(processed)[0]
                fraud_prob = 1.0 if prediction == 1 else 0.0

        # ---------- DISPLAY RESULT ----------
        st.markdown("---")
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            if prediction == 1:
                st.error(f"🚨 **FRAUDULENT TRANSACTION**")
                st.markdown(f"*Detected by: **{selected_model_name}***")
            else:
                st.success(f"✅ **LEGITIMATE TRANSACTION**")
                st.markdown(f"*Verified by: **{selected_model_name}***")
        with col_res2:
            if fraud_prob is not None:
                st.metric("Fraud Probability", f"{fraud_prob:.2%}",
                          delta="High risk" if fraud_prob > 0.7 else "Low risk" if fraud_prob < 0.3 else "Moderate risk",
                          delta_color="inverse" if fraud_prob > 0.5 else "normal")

        # st.markdown("""
        # <div style="background: #eef2ff; padding: 15px; border-radius: 20px; margin-top: 15px;">
        # <b>💡 How does auto‑calculation work?</b><br>
        # Normally, Sender's New Balance = Old Balance − Amount (money leaves).<br>
        # Receiver's New Balance = Old Balance + Amount (money arrives).<br>
        # If your transaction includes fees or errors, you can manually adjust the new balances.
        # </div>
        # """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
