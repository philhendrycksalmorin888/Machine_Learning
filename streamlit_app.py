import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from src.cybersecurity_pipeline import run_pipeline, F2_THRESHOLD

st.set_page_config(page_title="ML Cybersecurity Detection", page_icon="🛡️", layout="wide")
st.title("🛡️ Machine Learning–Based Detection of Malicious Network Traffic")
st.caption("Random Forest prototype for binary malicious network-event classification")

st.markdown("""
Upload the course-supplied `cybersecurity.csv` file to reproduce the project pipeline: data preprocessing, model training, evaluation, threshold comparison, and predictions.
""")

uploaded = st.file_uploader("Upload cybersecurity.csv", type=["csv"])

if uploaded is None:
    st.info("Upload the CSV file to start the demonstration.")
    st.stop()

with open("uploaded_cybersecurity.csv", "wb") as f:
    f.write(uploaded.getbuffer())

with st.spinner("Training Random Forest and evaluating held-out test set..."):
    model, results, split, prepared_df = run_pipeline("uploaded_cybersecurity.csv")

X_train, X_test, y_train, y_test, proba = split

st.subheader("Dataset Profile")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows after validation", f"{len(prepared_df):,}")
c2.metric("Attack prevalence", f"{prepared_df['label'].mean()*100:.2f}%")
c3.metric("Training events", f"{len(X_train):,}")
c4.metric("Test events", f"{len(X_test):,}")

st.subheader("Evaluation Results")
summary = pd.DataFrame([
    {"Operating threshold":"0.50 default", **results["threshold_0_50"]},
    {"Operating threshold":f"F2-selected {F2_THRESHOLD:.3f}", **results["threshold_f2"]},
])
show_cols = ["Operating threshold", "accuracy", "precision", "recall", "f1", "false_positive_rate", "tn", "fp", "fn", "tp"]
st.dataframe(summary[show_cols], use_container_width=True)

col_a, col_b = st.columns(2)
for col, key, title in [(col_a, "threshold_0_50", "Confusion Matrix @ 0.50"), (col_b, "threshold_f2", "Confusion Matrix @ F2 Threshold")]:
    with col:
        r = results[key]
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        fig, ax = plt.subplots(figsize=(4, 3.2))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(title)
        ax.set_xticks([0, 1], ["Benign", "Attack"])
        ax.set_yticks([0, 1], ["Benign", "Attack"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontweight="bold")
        st.pyplot(fig)

st.subheader("Try a Manual Prediction")
st.write("Enter a sample network event. The model returns the estimated probability that the event is malicious.")
with st.form("predict_form"):
    col1, col2, col3 = st.columns(3)
    src_port = col1.number_input("Source port", value=443, min_value=0, max_value=65535)
    dst_port = col2.number_input("Destination port", value=80, min_value=0, max_value=65535)
    protocol = col3.selectbox("Protocol", ["TCP", "UDP", "ICMP", "HTTP", "HTTPS", "Other"])
    col4, col5, col6 = st.columns(3)
    bytes_sent = col4.number_input("Bytes sent", value=1000, min_value=0)
    bytes_received = col5.number_input("Bytes received", value=5000, min_value=0)
    is_internal = col6.selectbox("Internal traffic", [True, False])
    user_agent = st.text_input("User agent", value="Mozilla/5.0")
    url = st.text_input("URL", value="https://example.com/login")
    timestamp = st.text_input("Timestamp", value="2025-10-01 12:00:00")
    submitted = st.form_submit_button("Predict")

if submitted:
    sample = pd.DataFrame([{
        "timestamp": timestamp,
        "src_ip": "0.0.0.0",
        "dst_ip": "0.0.0.0",
        "src_port": src_port,
        "dst_port": dst_port,
        "protocol": protocol,
        "bytes_sent": bytes_sent,
        "bytes_received": bytes_received,
        "user_agent": user_agent,
        "url": url,
        "is_internal_traffic": is_internal,
        "label": 0,
        "attack_type": "unknown"
    }])
    from src.cybersecurity_pipeline import prepare_features, FEATURES
    x_sample = prepare_features(sample)[FEATURES]
    p = float(model.predict_proba(x_sample)[:, 1][0])
    st.metric("Predicted attack probability", f"{p:.2%}")
    st.write("Classification @ 0.50:", "Attack" if p >= 0.50 else "Benign")
    st.write(f"Classification @ F2 threshold ({F2_THRESHOLD:.3f}):", "Attack" if p >= F2_THRESHOLD else "Benign")

st.caption("This prototype is for coursework demonstration only and is not a production intrusion-detection system.")
