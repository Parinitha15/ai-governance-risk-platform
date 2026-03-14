import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq

import os
from dotenv import load_dotenv

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="AI Governance Platform", layout="wide")

# -------------------------------------------------
# CUSTOM DARK STYLE
# -------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #0E1117;
    color: white;
}
[data-testid="stMetric"] {
    background-color: #161B22;
    padding: 20px;
    border-radius: 12px;
}
.section-card {
    background: linear-gradient(135deg, #161B22, #1F2937);
    padding: 25px;
    border-radius: 15px;
    margin-bottom: 20px;
}
.critical-glow {
    text-shadow: 0 0 12px #D50000;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
model = joblib.load("models/risk_model.pkl")
feature_importance = pd.read_csv("models/feature_importance.csv")

class_mapping = {
    "Low": 20,
    "Moderate": 50,
    "High": 75,
    "Critical": 95
}

risk_colors = {
    "Low": "#00C853",
    "Moderate": "#FFD600",
    "High": "#FF6D00",
    "Critical": "#D50000"
}

# -------------------------------------------------
# SIDEBAR INPUTS
# -------------------------------------------------
st.sidebar.title("⚙️ System Configuration")

with st.sidebar.expander("📊 Data Risk Factors", expanded=True):
    st.selectbox("Sensitive Attributes", ["No", "Yes"], key="sensitive_attributes")
    st.selectbox("Demographic Imbalance", ["Low","Medium","High"], key="demographic_imbalance")
    st.selectbox("Proxy Risk", ["Low","Medium","High"], key="proxy_risk")
    st.selectbox("PII Usage", ["None","Partial","Extensive"], key="pii_usage")
    st.selectbox("Anonymization Applied", ["No","Yes"], key="anonymization")

with st.sidebar.expander("🤖 Model Risk Factors"):
    st.selectbox("Model Type", ["Interpretable","Semi-black-box","Black-box"], key="model_type")
    st.selectbox("Fairness Testing", ["No","Yes"], key="fairness_testing")
    st.selectbox("Explanation Available", ["No","Yes"], key="explanation_available")
    st.selectbox("Documentation Quality", ["Low","Medium","High"], key="documentation_quality")

with st.sidebar.expander("⚙️ Operational Risk"):
    st.selectbox("Automation Level", ["Advisory","Semi-automated","Fully automated"], key="automation_level")
    st.selectbox("Impact Severity", ["Low","Medium","High"], key="impact_severity")
    st.selectbox("Human Review", ["No","Yes"], key="human_review")

with st.sidebar.expander("🛡 Governance Controls"):
    st.selectbox("Encryption", ["No","Yes"], key="encryption")
    st.selectbox("Audit Conducted", ["No","Yes"], key="audit_conducted")
    st.selectbox("Retention Policy", ["No","Yes"], key="retention_policy")
    st.selectbox("Third-party Data", ["No","Yes"], key="third_party_data")
    st.selectbox("Appeals Mechanism", ["No","Yes"], key="appeals_mechanism")
    st.selectbox("Historical Bias Present", ["No","Yes"], key="historical_bias")

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.title("🛡 AI Governance Risk Dashboard")
st.markdown("Enterprise AI Compliance & Ethical Risk Monitoring Platform")
st.divider()

# -------------------------------------------------
# ENCODE FUNCTION
# -------------------------------------------------
def encode_inputs():
    return pd.DataFrame([{
        "sensitive_attributes": 1 if st.session_state.sensitive_attributes == "Yes" else 0,
        "demographic_imbalance": ["Low","Medium","High"].index(st.session_state.demographic_imbalance),
        "historical_bias": 1 if st.session_state.historical_bias == "Yes" else 0,
        "fairness_testing": 1 if st.session_state.fairness_testing == "Yes" else 0,
        "proxy_risk": ["Low","Medium","High"].index(st.session_state.proxy_risk),
        "pii_usage": ["None","Partial","Extensive"].index(st.session_state.pii_usage),
        "anonymization": 1 if st.session_state.anonymization == "Yes" else 0,
        "retention_policy": 1 if st.session_state.retention_policy == "Yes" else 0,
        "third_party_data": 1 if st.session_state.third_party_data == "Yes" else 0,
        "encryption": 1 if st.session_state.encryption == "Yes" else 0,
        "model_type": ["Interpretable","Semi-black-box","Black-box"].index(st.session_state.model_type),
        "explanation_available": 1 if st.session_state.explanation_available == "Yes" else 0,
        "documentation_quality": ["Low","Medium","High"].index(st.session_state.documentation_quality),
        "audit_conducted": 1 if st.session_state.audit_conducted == "Yes" else 0,
        "automation_level": ["Advisory","Semi-automated","Fully automated"].index(st.session_state.automation_level),
        "human_review": 1 if st.session_state.human_review == "Yes" else 0,
        "impact_severity": ["Low","Medium","High"].index(st.session_state.impact_severity),
        "appeals_mechanism": 1 if st.session_state.appeals_mechanism == "Yes" else 0
    }])

# -------------------------------------------------
# GENERATE REPORT FUNCTION (GROQ)
# -------------------------------------------------
def generate_governance_report(score, risk_level, confidence):
    prompt = f"""
You are an AI governance and compliance expert.

An AI system has been evaluated using a governance risk assessment platform.

Risk Level: {risk_level}
Exposure Score: {score}/100
Model Confidence: {confidence:.2f}%

Write a short professional governance assessment report that includes:

1. Explanation of the risk level
2. Key governance concerns
3. Recommended mitigation actions

Keep it clear, professional, and under 150 words.
"""
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        if "429" in str(e) or "rate_limit" in str(e).lower():
            return "⚠️ API quota exceeded. Please wait a moment and try again."
        return f"⚠️ Report generation failed: {str(e)}"

# -------------------------------------------------
# RUN ASSESSMENT
# -------------------------------------------------
if st.sidebar.button("🚀 Run Assessment"):
    input_data = encode_inputs()
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    confidence = np.max(probabilities) * 100

    risk_score = sum([
        probabilities[i] * class_mapping[model.classes_[i]]
        for i in range(len(model.classes_))
    ])
    risk_score = int(risk_score)

    st.session_state["evaluated"] = True
    st.session_state["input_data"] = input_data
    st.session_state["prediction"] = prediction
    st.session_state["confidence"] = confidence
    st.session_state["risk_score"] = risk_score

# -------------------------------------------------
# DASHBOARD DISPLAY
# -------------------------------------------------
if "evaluated" in st.session_state:

    risk = st.session_state["prediction"]
    score = st.session_state["risk_score"]
    risk_color = risk_colors.get(risk, "white")

    st.markdown(f"""
    <div class="section-card">
        <h3>Governance Exposure Index</h3>
        <h1 style="color:{risk_color};">{score}/100</h1>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    if risk == "Critical":
        risk_display = f'<h2 class="critical-glow" style="color:{risk_color};">{risk}</h2>'
    else:
        risk_display = f'<h2 style="color:{risk_color};">{risk}</h2>'

    col1.markdown(f"""
    <div class="section-card">
        <h4>Risk Classification</h4>
        {risk_display}
    </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
    <div class="section-card">
        <h4>Model Confidence</h4>
        <h2>{st.session_state["confidence"]:.1f}%</h2>
    </div>
    """, unsafe_allow_html=True)

    # Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": risk_color}}
    ))
    fig_gauge.update_layout(paper_bgcolor="#0E1117", font_color="white")
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.divider()

    # Feature Chart
    st.subheader("📊 Top Governance Risk Drivers")
    top_features = feature_importance.head(8)

    fig = px.bar(
        top_features,
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale="reds"
    )
    fig.update_layout(plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", font_color="white")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # -------------------------------------------------
    # MITIGATION SIMULATION
    # -------------------------------------------------
    st.markdown("""
    <div class="section-card">
        <h3>🔁 Mitigation Strategy Simulation</h3>
        <p>Test structural governance improvements and measure exposure reduction.</p>
    </div>
    """, unsafe_allow_html=True)

    colA, colB = st.columns(2)

    with colA:
        sim_fairness = st.checkbox("Enable Fairness Testing")
        sim_human_review = st.checkbox("Add Human Review")

    with colB:
        sim_encryption = st.checkbox("Enable Encryption")
        sim_audit = st.checkbox("Conduct Audit")

    if st.button("Run Simulation"):

        simulated_input = st.session_state["input_data"].copy()

        if sim_fairness:
            simulated_input["fairness_testing"] = 1
        if sim_human_review:
            simulated_input["human_review"] = 1
        if sim_encryption:
            simulated_input["encryption"] = 1
        if sim_audit:
            simulated_input["audit_conducted"] = 1

        sim_prediction = model.predict(simulated_input)[0]
        sim_prob = model.predict_proba(simulated_input)[0]
        sim_confidence = np.max(sim_prob) * 100

        sim_score = sum([
            sim_prob[i] * class_mapping[model.classes_[i]]
            for i in range(len(model.classes_))
        ])
        sim_score = int(sim_score)

        original_score = score
        score_difference = original_score - sim_score

        col1, col2 = st.columns(2)

        col1.markdown(f"""
        <div class="section-card">
            <h4>Before Mitigation</h4>
            <h2 style="color:#FF6D00;">{original_score}/100</h2>
            <p>Risk Level: {risk}</p>
        </div>
        """, unsafe_allow_html=True)

        col2.markdown(f"""
        <div class="section-card">
            <h4>After Mitigation</h4>
            <h2 style="color:#00C853;">{sim_score}/100</h2>
            <p>Risk Level: {sim_prediction}</p>
        </div>
        """, unsafe_allow_html=True)

        if score_difference > 0:
            st.markdown(f"""
            <div style="background-color:#0F2A1D;padding:15px;border-radius:12px;margin-top:15px;">
                <h4 style="color:#00C853;">⬇ Risk Reduced by {score_difference} Points</h4>
            </div>
            """, unsafe_allow_html=True)
        elif score_difference < 0:
            st.markdown(f"""
            <div style="background-color:#2A0F0F;padding:15px;border-radius:12px;margin-top:15px;">
                <h4 style="color:#D50000;">⬆ Risk Increased by {abs(score_difference)} Points</h4>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color:#1A1F2B;padding:15px;border-radius:12px;margin-top:15px;">
               <h4 style="color:#FFD600;">No Exposure Change</h4>
            </div>
            """, unsafe_allow_html=True)

    # -------------------------------------------------
    # GENERATIVE AI GOVERNANCE REPORT
    # -------------------------------------------------
    st.divider()
    st.subheader("🧠 AI Governance Report")

    if st.button("Generate AI Risk Report"):
        st.session_state["report"] = generate_governance_report(
            st.session_state["risk_score"],
            st.session_state["prediction"],
            st.session_state["confidence"]
        )

    if "report" in st.session_state:
        st.markdown(st.session_state["report"])