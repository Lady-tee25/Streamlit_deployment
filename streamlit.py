import pickle
import numpy as np
import pandas as pd
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="AI Loan Intelligence System",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Modern Glassmorphism & Dark Mode Custom Styling
st.markdown(
    """
    <style>
    /* Global Page Styling */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.7) !important;
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Hero Banner Component */
    .hero-container {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(168, 85, 247, 0.2) 100%);
        border: 1px solid rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(16px);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 400;
    }

    /* Input Card Container */
    .input-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
        padding: 1.5rem;
        border-radius: 16px;
        margin-bottom: 1rem;
    }

    /* Styled Action Button */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
        color: #ffffff;
        border: none;
        padding: 0.85rem 1.5rem;
        font-size: 1.1rem;
        font-weight: 700;
        border-radius: 12px;
        box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.4);
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 25px -5px rgba(168, 85, 247, 0.5);
    }

    /* Result Notification Cards */
    .result-approved {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10b981;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.2);
    }
    .result-rejected {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 0 25px rgba(239, 68, 68, 0.2);
    }
    </style>
""",
    unsafe_allow_html=True,
)


# 3. Model Loader
@st.cache_resource
def load_model():
    with open("logistic_regression_model.pkl", "rb") as file:
        return pickle.load(file)


try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "❌ Model file `logistic_regression_model.pkl` not found. Please verify the file path."
    )
    st.stop()

# 4. Sidebar Controls & Metadata
with st.sidebar:
    st.title("⚡ Settings & Info")
    st.info(
        "This evaluation engine uses a trained Logistic Regression model to assess financial default risk based on real-time metrics."
    )
    st.markdown("---")
    st.caption("Model Version: v1.0.4")
    st.caption("Engine: Scikit-Learn / Logistic Regression")

# 5. Header Banner
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">Loan Qualification Predictor</div>
        <div class="hero-subtitle">Evaluate loan applications instantly using machine learning risk modeling</div>
    </div>
""",
    unsafe_allow_html=True,
)

# 6. Main Inputs Form Design
st.markdown("### 📝 Enter Applicant Information")

col1, col2 = st.columns(2, gap="medium")

with col1:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown("#### 💵 Financial Overview")

    income = st.number_input(
        "Annual Income ($)",
        min_value=0,
        max_value=2000000,
        value=350000,
        step=5000,
        help="Gross annual income of the primary applicant",
    )

    credit_score = st.slider(
        "Credit Score (FICO)",
        min_value=300,
        max_value=850,
        value=720,
        help="Standard credit score range between 300 and 850",
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown("#### 💼 Employment & Liabilities")

    employment_years = st.number_input(
        "Employment Duration (Years)",
        min_value=0,
        max_value=50,
        value=6,
        step=1,
        help="Consecutive years in active employment",
    )

    debt_ratio = st.slider(
        "Debt-to-Income (DTI) Ratio",
        min_value=0.00,
        max_value=1.00,
        value=0.35,
        step=0.01,
        help="Monthly debt obligations divided by gross monthly income",
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 7. Action Button & Prediction Execution
if st.button("🚀 Analyze Credit Risk"):
    features = np.array(
        [[income, credit_score, employment_years, debt_ratio]]
    )

    prediction = model.predict(features)[0]
    probabilities = (
        model.predict_proba(features)[0]
        if hasattr(model, "predict_proba")
        else None
    )

    st.markdown("<br>", unsafe_allow_html=True)

    res_col1, res_col2 = st.columns([1, 1], gap="medium")

    with res_col1:
        if prediction == 1:
            st.markdown(
                """
                <div class="result-approved">
                    <h2 style="color: #10b981; margin:0;">✅ APPROVED</h2>
                    <p style="color: #e2e8f0; margin-top: 10px;">The applicant meets the eligibility threshold for loan approval.</p>
                </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="result-rejected">
                    <h2 style="color: #ef4444; margin:0;">❌ REJECTED</h2>
                    <p style="color: #e2e8f0; margin-top: 10px;">The applicant poses a high risk profile for default.</p>
                </div>
            """,
                unsafe_allow_html=True,
            )

    with res_col2:
        if probabilities is not None:
            approval_prob = probabilities[1] * 100
            st.markdown("#### Confidence Breakdown")
            st.metric(
                label="Probability of Approval", value=f"{approval_prob:.1f}%"
            )
            st.progress(float(probabilities[1]))