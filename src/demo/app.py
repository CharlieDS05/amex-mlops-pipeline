"""
Streamlit demo for the AmEx Default Risk API.
Demonstrates the model's binary decision-making behavior.
"""
import streamlit as st
import requests
import json
import os
from pathlib import Path

API_URL = os.getenv("API_URL", "http://localhost:8000")
PRESETS_PATH = Path(__file__).parent / "preset_customers.json"


@st.cache_data
def load_presets():
    if PRESETS_PATH.exists():
        with open(PRESETS_PATH) as f:
            return json.load(f)
    return None


# Page config
st.set_page_config(
    page_title="AmEx Default Risk Predictor",
    page_icon="💳",
    layout="centered",
)

st.title("AmEx Default Risk Predictor")
st.caption("Interactive demo · Powered by XGBoost + FastAPI + MLflow")
st.markdown("---")


# Sidebar
with st.sidebar:
    st.header("⚙️ System Status")
    
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=3)
        if health_response.status_code == 200 and health_response.json().get("model_loaded"):
            st.success("✅ API online")
            st.success("✅ Model loaded")
        else:
            st.warning("⚠️ Service degraded")
    except Exception:
        st.error("❌ Cannot reach API")
    
    st.markdown("---")
    st.caption("API endpoint:")
    st.code(API_URL, language="text")
    
    st.markdown("---")
    st.caption(
        "**About this model**\n\n"
        "XGBoost classifier trained on 460K AmEx customer profiles "
        "achieving M Score 0.7929 (OOF) and ROC-AUC 0.9616. "
        "The model exhibits high discriminative power, producing confident "
        "binary classifications with a sharp decision boundary."
    )


# Load presets
presets = load_presets()
if presets is None:
    st.error("Preset profiles not found.")
    st.stop()


# Header explanation
st.markdown(
    "This demo shows how the model classifies real customer profiles "
    "from the AmEx dataset. Select a profile to see the model's prediction."
)
st.markdown("---")


# Profile selection 
st.subheader("Customer profile")

profile_choice = st.radio(
    "Select a customer profile",
    options=["low_risk", "borderline", "high_risk"],
    format_func=lambda x: {
        "low_risk": "🟢 Low-risk profile (good payment history)",
        "borderline": "🟡 Borderline case (synthetic interpolation)",
        "high_risk": "🔴 High-risk profile (concerning patterns)",
    }[x],
    horizontal=False,
)

st.info(
    "💡 **Note**: The borderline case is a synthetic profile created by "
    "interpolating between the low-risk and high-risk customers. "
    "This reveals the model's decision boundary."
)

customer_id = st.text_input(
    "Customer ID",
    value=f"CUST_DEMO_{profile_choice.upper()}",
)

st.markdown("---")


# Prediction button
if st.button("Predict Default Risk", type="primary", use_container_width=True):
    
    payload = {
        "customer_id": customer_id,
        "features": presets[profile_choice],
    }
    
    with st.spinner("Querying model..."):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=30,
            )
            
            if response.status_code == 200:
                result = response.json()
                
                st.markdown("### 📊 Prediction Result")
                
                col_a, col_b, col_c = st.columns(3)
                
                col_a.metric(
                    "Default probability",
                    f"{result['default_probability']:.4f}",
                )
                
                col_b.metric(
                    "Risk tier",
                    result["risk_tier"].upper(),
                )
                
                col_c.metric(
                    "Decision",
                    "DECLINE" if result['default_probability'] >= 0.5 else "APPROVE",
                )
                
                # Decision message
                prob = result["default_probability"]
                if prob >= 0.5:
                    st.error(
                        "🚨 **DECLINE** — Model predicts this customer is "
                        "likely to default. Recommend rejection or manual "
                        "review by credit risk analyst."
                    )
                else:
                    st.success(
                        "✅ **APPROVE** — Model predicts low default risk. "
                        "Standard approval workflow applies."
                    )
                
                with st.expander("🔍 View raw API response"):
                    st.json(result)
                    
            else:
                st.error(f"API returned status {response.status_code}")
                st.code(response.text)
                
        except requests.exceptions.Timeout:
            st.error("Request timed out.")
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot connect to API at {API_URL}")
        except Exception as e:
            st.error(f"Error: {e}")


# Educational section
st.markdown("---")
with st.expander("ℹ️ Why does this model produce binary predictions?"):
    st.markdown("""
    The XGBoost model trained on the AmEx dataset shows **high discriminative power**, 
    consistently classifying customers as either very likely (>90%) or very unlikely (<10%) 
    to default. Few customers fall in the middle.
    
    **This is desirable behavior for credit risk**:
    - In the banking sector it is required to make confident decisions, not vague probabilities
    - High-confidence predictions allow automated approvals
    - The sharp decision boundary correlates with high model performance (M Score: 0.7929)
    
    **Trade-off**: The model commits firmly, which is great for production deployment 
    but means cases truly "in the middle" require manual review.
    """)


# Footer
st.markdown("---")
st.caption(
    "End-to-end MLOps pipeline · "
    "[GitHub source](https://github.com/CharlieDS05/amex-mlops-pipeline)"
)