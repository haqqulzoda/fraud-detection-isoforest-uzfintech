import streamlit as st
import pandas as pd
import numpy as np
import joblib 
import os


# 1. Setup & Styling

st.set_page_config(page_title="UzFintech FireGuard AI", page_icon='🛡️', layout='centered')

st.markdown("""
    <style>
        .big-font { font-size: 30px !important; font-weight: bold; }
        .risk-high {color: #ff4b4b;}
        .risk-low { color: #21c354; }
        .stButton>button { width: 100%; border-radius: 10px; height: 3em; }
     </style>
 """, unsafe_allow_html=True)

 # 2. Load Artifacts (Cached)
@st.cache_resource
def load_artifacts():
     # Helper to find paths relative to this file
     base_dir = os.path.dirname(os.path.abspath(__file__))
     models_dir = os.path.join(base_dir, "..", 'models')

     # Load all 4 models
     iso_forest = joblib.load(os.path.join(models_dir, 'iso_forest.joblib'))
     robust_scaler = joblib.load(os.path.join(models_dir, 'robust_scaler.joblib'))
     score_scaler = joblib.load(os.path.join(models_dir, 'score_scaler.joblib'))
     model_cols = joblib.load(os.path.join(models_dir, 'model_columns.joblib'))

     return iso_forest, robust_scaler, score_scaler, model_cols

# 3. Load Data (For Simulation only)
@st.cache_data
def load_data():
    # We load the processed data just to pick random rows from it
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', 'data', 'creditcard_sample.csv') 
    df = pd.read_csv(data_path)
    return df

try: 
    model, r_scaler, s_scaler, cols = load_artifacts()
    df = load_data()
except Exception as e:
    st.error(f"Error loading files: {str(e)}")
    st.stop()

# 4. Header
st.title("🛡️ FraudGuard: Anomaly Detection UzFinTech")
st.markdown("Real-time Unsupervised Learning Engine (Isolation Forest)")
st.divider()

# 5. Session State (To keep track of the current transaction)
if 'curr_row' not in st.session_state:
    st.session_state.curr_row = None

# 6. Button: "Simulate New Transaction"
col1, col2 = st.columns([1, 2])
with col1:
    if st.button("Simulate New Transaction"):
        # Randomly pick 1 row
        random_idx = np.random.randint(0,len(df))
        st.session_state.curr_row = df.iloc[[random_idx]]
      
# 7. Prediction Logic
if st.session_state.curr_row is not None:
    row = st.session_state.curr_row
    
    # True Label (Hidden from model, but we show it for demo)
    true_label = row['Class'].values[0]
    true_text = "FRAUD" if true_label == 1 else "Legit"

    # A. Preprocessing (Must match training!)
    # We need to drop 'Class' and ensure columns match
    input_data = row.drop('Class', axis=1)
    # Ensure correct column order
    input_data = input_data[cols]

    # Note: 'Amount' is already scaled in the CSV we loaded, so we don't scale it again here.
    # In a real app with raw input, we would apply robust_scaler here.
    
    # B. Inference
    raw_score = model.decision_function(input_data)    

    # C. Post-Processing (Convert to 0-100)
    inverted_score = raw_score * -1
    risk_score = s_scaler.transform(inverted_score.reshape(-1,1))[0][0]

    # 8. Display Dashboard
    with col2:
        st.metric('Risk Score [0-100]', f"{risk_score:.1f}")

    st.divider()

    # Visualization Columns
    c1, c2, c3 = st.columns(3)
    # DECISION ENGINE
    if risk_score > 40: # Threshold we chose based on your 13 vs 60 gap
        st.error("🚨 HIGH RISK BLOCKED")
        st.markdown(f"**Reason:** Anomaly Score is abnormally high ({risk_score:.1f})")
    else:
        st.success("✅ APPROVED")
        st.markdown(f"**Reason:** Transaction pattern looks normal ({risk_score:.1f})")
        
    # "Truth" Reveal (For Demo purposes)
    with st.expander("👮 View Ground Truth (For Analysts)"):
        st.write(f"Actual Database Label: **{true_text}**")
        st.write("Transaction Features (Anonymized V1-V28):")
        st.dataframe(input_data)