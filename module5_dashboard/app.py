"""
MODULE 5 — STREAMLIT DASHBOARD
================================
Run from project root:
    streamlit run module5_dashboard/app.py
"""

import os
import json
import time
import smtplib
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Suppress warnings in terminal
warnings.filterwarnings('ignore')

# ── 1. Page Configuration ──
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Predictive Maintenance — Real-Time Turbofan Monitoring")
st.markdown("Real-time telemetry monitoring, remaining useful life (RUL) estimation, and failure probability tracking.")

# ── 2. Email Configuration (SMTP) ──
SENDER_EMAIL = "your_email@gmail.com"        # Replace with valid Gmail
SENDER_PASSWORD = "your_app_password"         # Replace with 16-digit App Password
RECEIVER_EMAIL = "receiver_email@gmail.com"   # Replace with recipient email

def send_alert_email(engine_id, cycle, risk_percentage, rul_estimate):
    """Sends an automated email notification when risk crosses the threshold."""
    if SENDER_EMAIL == "your_email@gmail.com":
        st.sidebar.warning("⚠️ Email credentials not set. Skipping live email dispatch.")
        return False
    try:
        subject = f"🚨 CRITICAL ALERT: Engine {engine_id} Failure Risk at {risk_percentage:.1f}%"
        body = f"""
        HIGH-RISK DEGRADATION DETECTED
        ========================================
        Engine ID            : {engine_id}
        Current Cycle        : {cycle}
        Failure Probability  : {risk_percentage:.1f}%
        Estimated RUL        : {rul_estimate:.0f} cycles remaining
        ========================================
        
        Action Required:
        Schedule immediate maintenance and inspection for Engine {engine_id} to prevent unscheduled operational downtime.
        
        -- Automated Predictive Maintenance Alert System
        """
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        st.sidebar.error(f"Email failure: {e}")
        return False

# ── 3. Load Artifacts ──
@st.cache_resource
def load_models_and_scaler():
    rf_model = joblib.load('models/random_forest.pkl')
    xgb_model = joblib.load('models/xgboost_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    lstm_model = None
    lstm_path = 'models/lstm_model.h5'
    if os.path.exists(lstm_path):
        try:
            import tensorflow as tf
            lstm_model = tf.keras.models.load_model(lstm_path, compile=False)
        except Exception as e:
            st.sidebar.warning(f"⚠️ LSTM Model found but failed to load: {e}")

    with open('outputs/feature_cols.json', 'r') as f:
        feature_cols = json.load(f)

    return rf_model, xgb_model, lstm_model, scaler, feature_cols

try:
    rf_model, xgb_model, lstm_model, scaler, feature_cols = load_models_and_scaler()
    st.sidebar.success("✅ Models & Scaler Loaded Successfully")
except Exception as e:
    st.error(f"❌ Error loading model artifacts: {e}")
    st.stop()

# ── 4. Sidebar Controls ──
st.sidebar.header("🕹️ Simulation Controls")
uploaded_file = st.sidebar.file_uploader("Upload Test Data (test_FD001.txt)", type=["txt", "csv"])

selected_engine = st.sidebar.number_input("Select Engine ID", min_value=1, max_value=100, value=50, step=1)
speed = st.sidebar.slider("Simulation Speed (Delay in Sec)", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
alert_threshold = st.sidebar.slider("Alert Failure Threshold (%)", min_value=50, max_value=95, value=80, step=5)
enable_email = st.sidebar.checkbox("Enable Live Email Alerts", value=False)

run_simulation = st.sidebar.button("🚀 Start Real-Time Simulation")

# Status Helper Function
def get_status(prob):
    if prob >= alert_threshold:
        return "CRITICAL FAULT RISK", "status-critical"
    elif prob >= 40:
        return "WARNING — DEGRADATION DETECTED", "status-warning"
    else:
        return "NORMAL OPERATION", "status-normal"

st.markdown("""
<style>
    .status-normal { background-color: #27ae60; color: white; padding: 10px; border-radius: 5px; font-weight: bold; text-align: center; }
    .status-warning { background-color: #f39c12; color: white; padding: 10px; border-radius: 5px; font-weight: bold; text-align: center; }
    .status-critical { background-color: #c0392b; color: white; padding: 10px; border-radius: 5px; font-weight: bold; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ── 5. Main Simulation Logic ──
if uploaded_file is not None:
    col_names = ['unit_number', 'time_cycles', 'setting_1', 'setting_2', 'setting_3'] + [f's{i}' for i in range(1, 22)]
    df = pd.read_csv(uploaded_file, sep=r'\s+', header=None, names=col_names)

    if run_simulation:
        engine_data = df[df['unit_number'] == selected_engine].copy().reset_index(drop=True)

        if engine_data.empty:
            st.error(f"Engine ID {selected_engine} not found in uploaded file.")
            st.stop()

        raw_features = engine_data[feature_cols].values
        features_scaled = scaler.transform(raw_features)

        status_placeholder = st.empty()
        metrics_placeholder = st.empty()
        graph_placeholder = st.empty()
        alert_placeholder = st.empty()

        prob_history, cycle_nums, sequence_buffer = [], [], []
        SEQUENCE_LENGTH = 30
        email_sent = False

        for i in range(len(features_scaled)):
            cycle = i + 1
            row_scaled = features_scaled[i]
            row_values = row_scaled.reshape(1, -1)

            # 1. Classification Predictions (RF & XGB)
            rf_prob = float(rf_model.predict_proba(row_values)[0][1] if len(rf_model.predict_proba(row_values)[0]) > 1 else rf_model.predict_proba(row_values)[0][0]) * 100
            xgb_prob = float(xgb_model.predict_proba(row_values)[0][1] if len(xgb_model.predict_proba(row_values)[0]) > 1 else xgb_model.predict_proba(row_values)[0][0]) * 100
            avg_prob = (rf_prob + xgb_prob) / 2

            # 2. LSTM RUL Estimation with Dynamic Sequence Padding
            sequence_buffer.append(row_scaled)
            rul_pred = None

            if lstm_model is not None:
                try:
                    if len(sequence_buffer) < SEQUENCE_LENGTH:
                        pad_needed = SEQUENCE_LENGTH - len(sequence_buffer)
                        seq_data = [sequence_buffer[0]] * pad_needed + list(sequence_buffer)
                    else:
                        seq_data = sequence_buffer[-SEQUENCE_LENGTH:]

                    seq_input = np.array(seq_data).reshape(1, SEQUENCE_LENGTH, len(feature_cols))
                    raw_rul = float(lstm_model.predict(seq_input, verbose=0)[0][0])
                    rul_pred = max(0.0, raw_rul * 125 if raw_rul <= 1.0 else raw_rul)
                except Exception:
                    rul_pred = None

            # Fallback estimation if LSTM is unavailable
            if rul_pred is None:
                estimated_max_life = max(125, len(features_scaled))
                rul_pred = max(0.0, (1.0 - (avg_prob / 100.0)) * (estimated_max_life - cycle))

            prob_history.append(avg_prob)
            cycle_nums.append(cycle)

            # UI Status Header
            status_text, status_class = get_status(avg_prob)
            status_placeholder.markdown(
                f'<div class="{status_class}">Engine {selected_engine} — Cycle {cycle} — {status_text}</div>',
                unsafe_allow_html=True
            )

            # Metrics Bar
            with metrics_placeholder.container():
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Cycle", f"{cycle} / {len(features_scaled)}")
                prev_prob = prob_history[-2] if len(prob_history) > 1 else avg_prob
                m2.metric("Failure Prob", f"{avg_prob:.1f}%", delta=f"{avg_prob - prev_prob:.1f}%")
                m3.metric("RUL Estimate", f"{rul_pred:.0f} cycles")
                m4.metric("RF / XGB", f"{rf_prob:.0f}% / {xgb_prob:.0f}%")

            # Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=cycle_nums, y=prob_history, mode='lines+markers', name='Failure Probability',
                line=dict(color='#E74C3C', width=2), marker=dict(size=4)
            ))
            fig.add_hline(y=alert_threshold, line_dash="dash", line_color="orange", annotation_text=f"Alert ({alert_threshold}%)")
            fig.update_layout(
                title=f'Engine {selected_engine} — Real-Time Failure Risk',
                xaxis_title='Cycle', yaxis_title='Failure Probability (%)', yaxis=dict(range=[0, 105]), height=380
            )
            graph_placeholder.plotly_chart(fig, use_container_width=True)

            # Alert Trigger & Email Dispatch
            if avg_prob >= alert_threshold:
                alert_placeholder.error(f"⚠️ HIGH RISK ALERT at Cycle {cycle}: Failure Probability reached {avg_prob:.1f}%!")
                
                if enable_email and not email_sent:
                    email_sent = True
                    sent_success = send_alert_email(selected_engine, cycle, avg_prob, rul_pred)
                    if sent_success:
                        st.sidebar.success(f"📩 Alert email dispatched to {RECEIVER_EMAIL}!")

            time.sleep(speed)

        st.success(f"✅ Simulation completed for Engine {selected_engine}")

    # ── 6. SHAP Explainability (Only Renders If File Exists) ──
    if os.path.exists('outputs/shap_summary.png'):
        st.divider()
        with st.expander("🔍 Model Explainability & Feature Importance (SHAP Analysis)"):
            st.markdown("""
            **Understanding Sensor Contributions:**  
            SHAP (SHapley Additive exPlanations) highlights how individual engine sensor readings influence model predictions. 
            Higher values in core sensors (e.g., temperatures and pressures) directly increase the predicted failure probability.
            """)
            st.image('outputs/shap_summary.png', caption="SHAP Summary Plot — Feature Impact on Failure Risk", use_container_width=True)
else:
    st.info("👈 Please upload `test_FD001.txt` in the sidebar to begin testing.")