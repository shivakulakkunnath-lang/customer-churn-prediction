import streamlit as st
import pandas as pd
import joblib

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f;
    color: #e8e8f0;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 0%, #1a0a2e 0%, #0a0a0f 50%),
                radial-gradient(ellipse at 80% 100%, #0d1a2e 0%, transparent 50%);
    min-height: 100vh;
}

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* Hero header */
.hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
    position: relative;
}
.hero-tag {
    display: inline-block;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #7c5cfc;
    border: 1px solid #7c5cfc44;
    padding: 0.3rem 1rem;
    border-radius: 20px;
    margin-bottom: 1rem;
    background: #7c5cfc11;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 5vw, 3.5rem);
    font-weight: 800;
    color: #fff;
    line-height: 1.1;
    margin-bottom: 0.5rem;
}
.hero h1 span { color: #7c5cfc; }
.hero p {
    color: #888;
    font-size: 0.95rem;
    font-weight: 300;
}

/* Section labels */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #7c5cfc;
    margin-bottom: 1rem;
    margin-top: 0.5rem;
    padding-left: 2px;
}

/* Cards */
.card {
    background: #12121a;
    border: 1px solid #ffffff0d;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.card:hover { border-color: #7c5cfc33; }

/* Streamlit widget overrides */
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] > div > div > input {
    background: #1c1c28 !important;
    border: 1px solid #ffffff15 !important;
    border-radius: 10px !important;
    color: #e8e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSelectbox"] > div > div:focus-within,
[data-testid="stNumberInput"] > div > div > input:focus {
    border-color: #7c5cfc !important;
    box-shadow: 0 0 0 2px #7c5cfc22 !important;
}

label, [data-testid="stWidgetLabel"] p {
    color: #aaa !important;
    font-size: 0.82rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 400 !important;
    letter-spacing: 0.01em !important;
}

/* Slider */
[data-testid="stSlider"] > div { padding: 0.2rem 0; }
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #7c5cfc !important;
    border: 2px solid #fff !important;
}

/* Button */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #7c5cfc, #5c8afc) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.85rem 2rem !important;
    transition: all 0.2s !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px #7c5cfc44 !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* Result cards */
.result-churn {
    background: linear-gradient(135deg, #2a0a0a, #1a0505);
    border: 1px solid #ff4d4d44;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-safe {
    background: linear-gradient(135deg, #0a2a1a, #051a10);
    border: 1px solid #00e57644;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-icon { font-size: 3rem; margin-bottom: 0.5rem; }
.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    margin-bottom: 0.3rem;
}
.result-churn .result-title { color: #ff6b6b; }
.result-safe .result-title { color: #00e576; }
.result-sub { color: #888; font-size: 0.85rem; }

/* Probability meter */
.prob-bar-wrap {
    background: #1c1c28;
    border-radius: 999px;
    height: 10px;
    margin: 1.2rem 0 0.4rem;
    overflow: hidden;
}
.prob-bar {
    height: 100%;
    border-radius: 999px;
    transition: width 0.8s ease;
}
.prob-label {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    margin-top: 0.8rem;
}

/* Stat pills */
.stat-row {
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
    margin-top: 1rem;
    justify-content: center;
}
.stat-pill {
    background: #1c1c28;
    border: 1px solid #ffffff0d;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-size: 0.78rem;
    color: #aaa;
}
.stat-pill strong { color: #e8e8f0; display: block; font-size: 1rem; }

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #ffffff08;
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("customer_churn_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, feature_columns

try:
    model, feature_columns = load_model()
    model_loaded = True
except:
    model_loaded = False

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-tag">🔬 ML-Powered Analytics</div>
    <h1>Customer <span>Churn</span> Predictor</h1>
    <p>Enter customer details to predict churn probability using Random Forest</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ Model files not found. Place `customer_churn_model.pkl` and `feature_columns.pkl` in the same directory.")
    st.stop()

# ── Layout ────────────────────────────────────────────────────────────────────
left, right = st.columns([3, 2], gap="large")

with left:
    # ── Account Info ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Account Information</div>', unsafe_allow_html=True)
    with st.container():
        c1, c2, c3 = st.columns(3)
        with c1:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
        with c2:
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        with c3:
            payment = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])

        c4, c5, c6 = st.columns(3)
        with c4:
            monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.5)
        with c5:
            total = st.number_input("Total Charges ($)", 0.0, 10000.0, float(tenure * monthly), step=1.0)
        with c6:
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Demographics ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Demographics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with c2:
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    with c3:
        partner = st.selectbox("Partner", ["No", "Yes"])
    with c4:
        dependents = st.selectbox("Dependents", ["No", "Yes"])

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Services ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Services Subscribed</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    with c2:
        security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    with c3:
        support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("⚡ Predict Churn Risk", use_container_width=True)

# ── Right panel ───────────────────────────────────────────────────────────────
with right:
    st.markdown('<div class="section-label">Prediction Result</div>', unsafe_allow_html=True)

    if not predict_btn:
        st.markdown("""
        <div style="background:#12121a;border:1px dashed #ffffff15;border-radius:16px;
                    padding:3rem 2rem;text-align:center;color:#444;">
            <div style="font-size:2.5rem;margin-bottom:1rem;">📡</div>
            <div style="font-family:'Syne',sans-serif;font-size:1rem;color:#555;">
                Fill in customer details<br>and click Predict
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Build input
        input_df = pd.DataFrame(0, index=[0], columns=feature_columns)

        input_df["SeniorCitizen"] = 1 if senior == "Yes" else 0
        input_df["tenure"] = tenure
        input_df["MonthlyCharges"] = monthly
        input_df["TotalCharges"] = total

        if gender == "Male":       input_df["gender_Male"] = 1
        if partner == "Yes":       input_df["Partner_Yes"] = 1
        if dependents == "Yes":    input_df["Dependents_Yes"] = 1
        if phone == "Yes":         input_df["PhoneService_Yes"] = 1
        if paperless == "Yes":     input_df["PaperlessBilling_Yes"] = 1

        if multiple == "No phone service": input_df["MultipleLines_No phone service"] = 1
        elif multiple == "Yes":            input_df["MultipleLines_Yes"] = 1

        if internet == "Fiber optic": input_df["InternetService_Fiber optic"] = 1
        elif internet == "No":        input_df["InternetService_No"] = 1

        if security == "No internet service": input_df["OnlineSecurity_No internet service"] = 1
        elif security == "Yes":               input_df["OnlineSecurity_Yes"] = 1

        if backup == "No internet service": input_df["OnlineBackup_No internet service"] = 1
        elif backup == "Yes":               input_df["OnlineBackup_Yes"] = 1

        if device == "No internet service": input_df["DeviceProtection_No internet service"] = 1
        elif device == "Yes":               input_df["DeviceProtection_Yes"] = 1

        if support == "No internet service": input_df["TechSupport_No internet service"] = 1
        elif support == "Yes":               input_df["TechSupport_Yes"] = 1

        if tv == "No internet service": input_df["StreamingTV_No internet service"] = 1
        elif tv == "Yes":               input_df["StreamingTV_Yes"] = 1

        if movies == "No internet service": input_df["StreamingMovies_No internet service"] = 1
        elif movies == "Yes":              input_df["StreamingMovies_Yes"] = 1

        if contract == "One year":  input_df["Contract_One year"] = 1
        elif contract == "Two year": input_df["Contract_Two year"] = 1

        if payment == "Credit card (automatic)":   input_df["PaymentMethod_Credit card (automatic)"] = 1
        elif payment == "Electronic check":         input_df["PaymentMethod_Electronic check"] = 1
        elif payment == "Mailed check":             input_df["PaymentMethod_Mailed check"] = 1

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        pct = int(probability * 100)

        # Risk level
        if pct < 30:
            risk_label, risk_color = "Low Risk", "#00e576"
        elif pct < 60:
            risk_label, risk_color = "Medium Risk", "#ffc107"
        else:
            risk_label, risk_color = "High Risk", "#ff6b6b"

        if prediction == 1:
            st.markdown(f"""
            <div class="result-churn">
                <div class="result-icon">⚠️</div>
                <div class="result-title">Likely to Churn</div>
                <div class="result-sub">This customer is at risk of leaving</div>
                <div class="prob-bar-wrap">
                    <div class="prob-bar" style="width:{pct}%;background:linear-gradient(90deg,#ff6b6b,#ff4d4d);"></div>
                </div>
                <div class="prob-label" style="color:#ff6b6b;">{pct}%</div>
                <div style="color:#888;font-size:0.78rem;">Churn Probability</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-safe">
                <div class="result-icon">✅</div>
                <div class="result-title">Likely to Stay</div>
                <div class="result-sub">This customer shows low churn signals</div>
                <div class="prob-bar-wrap">
                    <div class="prob-bar" style="width:{pct}%;background:linear-gradient(90deg,#00e576,#00c853);"></div>
                </div>
                <div class="prob-label" style="color:#00e576;">{pct}%</div>
                <div style="color:#888;font-size:0.78rem;">Churn Probability</div>
            </div>
            """, unsafe_allow_html=True)

        # Stats row
        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-pill"><strong style="color:{risk_color}">{risk_label}</strong>Risk Level</div>
            <div class="stat-pill"><strong>{tenure} mo</strong>Tenure</div>
            <div class="stat-pill"><strong>${monthly:.0f}</strong>Monthly</div>
            <div class="stat-pill"><strong>{contract.split()[0]}</strong>Contract</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Retention tips
        st.markdown('<div class="section-label">Retention Signals</div>', unsafe_allow_html=True)

        tips = []
        if contract == "Month-to-month":
            tips.append(("📋", "Month-to-month contract", "Highest churn risk contract type"))
        if internet == "Fiber optic":
            tips.append(("🌐", "Fiber optic user", "Fiber users churn at higher rates"))
        if tenure < 12:
            tips.append(("⏱️", f"Only {tenure} months tenure", "New customers churn most"))
        if support == "No":
            tips.append(("🛠️", "No tech support", "Correlated with higher churn"))
        if monthly > 70:
            tips.append(("💸", f"High monthly charge (${monthly:.0f})", "Above average billing"))

        if tips:
            for icon, title, desc in tips[:4]:
                st.markdown(f"""
                <div style="background:#12121a;border:1px solid #ffffff0d;border-radius:10px;
                            padding:0.8rem 1rem;margin-bottom:0.5rem;display:flex;gap:0.8rem;align-items:center;">
                    <span style="font-size:1.2rem">{icon}</span>
                    <div>
                        <div style="font-size:0.82rem;color:#e8e8f0;font-weight:500;">{title}</div>
                        <div style="font-size:0.72rem;color:#666;">{desc}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#12121a;border:1px solid #ffffff0d;border-radius:10px;
                        padding:1rem;color:#555;font-size:0.82rem;text-align:center;">
                No major churn risk signals detected
            </div>
            """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2rem 0 1rem;color:#333;font-size:0.75rem;">
    Random Forest · ROC-AUC 0.787 · Telco Customer Churn Dataset
</div>
""", unsafe_allow_html=True)