# 💎 Diamond Dynamics - Price Prediction & Market Segmentation Streamlit App
# Paste the whole file into /content/app.py or run as a single cell (if using Colab, write to file)
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import json
from sklearn.pipeline import Pipeline

# ------------------- Streamlit Page Setup -------------------
st.set_page_config(
    page_title="Diamond Dynamics 💎",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("💎 Diamond Dynamics — Price Prediction & Market Segmentation")

# ------------------- Helper Functions -------------------
@st.cache_data
def load_csv(uploaded_file):
    '''Read and clean CSV data'''
    df = pd.read_csv(uploaded_file)
    df.columns = [c.strip() for c in df.columns]  # strip unwanted spaces
    return df

def try_load_models_from_folder(folder='models'):
    '''Auto-load models from 'models' folder if exists'''
    models = {'regressor': None, 'clusterer': None}
    if os.path.exists(folder):
        for fname in os.listdir(folder):
            if fname.endswith('.pkl') or fname.endswith('.joblib'):
                path = os.path.join(folder, fname)
                try:
                    with open(path, 'rb') as f:
                        obj = pickle.load(f)
                    if hasattr(obj, 'predict'):
                        # heuristic: if it has feature_importances_ assume regressor, else if cluster_centers_ assume clusterer
                        if hasattr(obj, 'feature_importances_'):
                            models['regressor'] = obj
                        elif hasattr(obj, 'cluster_centers_'):
                            models['clusterer'] = obj
                        else:
                            if models['regressor'] is None:
                                models['regressor'] = obj
                except Exception as e:
                    st.sidebar.warning(f"Could not load {fname}: {e}")
    return models

def load_uploaded_model(uploaded_file):
    try:
        return pickle.load(uploaded_file)
    except Exception as e:
        st.sidebar.error(f"Model load error: {e}")
        return None

def predict_from_model(model, X_df):
    try:
        return model.predict(X_df)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

# ------------------- Sidebar: Data & Models -------------------
st.sidebar.header("📂 Data & Models")

uploaded = st.sidebar.file_uploader("Upload CSV dataset (optional)", type=['csv'])
models = try_load_models_from_folder()

reg_model_file = st.sidebar.file_uploader("Upload regression model (.pkl)", type=['pkl','joblib'], key='reg')
clust_model_file = st.sidebar.file_uploader("Upload clustering model (.pkl)", type=['pkl','joblib'], key='clust')

if reg_model_file is not None:
    models['regressor'] = load_uploaded_model(reg_model_file)
if clust_model_file is not None:
    models['clusterer'] = load_uploaded_model(clust_model_file)

st.sidebar.write(f"✅ Regressor loaded: {'Yes' if models['regressor'] else 'No'}")
st.sidebar.write(f"✅ Clusterer loaded: {'Yes' if models['clusterer'] else 'No'}")

if uploaded is not None:
    try:
        df_uploaded = load_csv(uploaded)
        st.sidebar.write("Uploaded dataset preview:")
        st.sidebar.dataframe(df_uploaded.head())
    except Exception as e:
        st.sidebar.error(f"Could not read uploaded CSV: {e}")

# ------------------- Main Input Form -------------------
st.header("📥 Diamond Attributes (Same Form for Both Modules)")
with st.form("diamond_input_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        carat = st.number_input("Carat", min_value=0.0, value=0.7, step=0.01)
        x = st.number_input("x (Length mm)", min_value=0.0, value=5.8, step=0.01)
        y = st.number_input("y (Width mm)", min_value=0.0, value=5.7, step=0.01)
    with col2:
        z = st.number_input("z (Depth mm)", min_value=0.0, value=3.6, step=0.01)
        depth = st.number_input("Depth (%)", min_value=0.0, value=61.5, step=0.1)
        table = st.number_input("Table (%)", min_value=0.0, value=57.0, step=0.1)
    with col3:
        default_cut = ["Ideal", "Premium", "Very Good", "Good", "Fair"]
        default_color = ["D", "E", "F", "G", "H", "I", "J"]
        default_clarity = ["IF","VVS1","VVS2","VS1","VS2","SI1","SI2","I1"]
        cut = st.selectbox("Cut", default_cut)
        color = st.selectbox("Color", default_color)
        clarity = st.selectbox("Clarity", default_clarity)

    submitted = st.form_submit_button("Save Inputs")

input_data = pd.DataFrame({
    'carat': [carat],
    'x': [x],
    'y': [y],
    'z': [z],
    'depth': [depth],
    'table': [table],
    'cut': [cut],
    'color': [color],
    'clarity': [clarity]
})
st.write("### 💾 Current Input Data")
st.dataframe(input_data)

# ------------------- Cluster Name Mapping -------------------
st.sidebar.header("🗂 Cluster Name Mapping")
cluster_map_text = st.sidebar.text_area(
    "Map cluster numbers to names (JSON format)",
    value='{"0":"Affordable Small Diamonds","1":"Premium Heavy Diamonds","2":"Mid-range Balanced Diamonds","3":"Luxury Unique Diamonds"}'
)
try:
    cluster_map = json.loads(cluster_map_text)
    if not isinstance(cluster_map, dict):
        cluster_map = {}
except Exception:
    cluster_map = {}

# ------------------- Prediction Buttons -------------------
colA, colB = st.columns(2)

with colA:
    if st.button("💰 Predict Price"):
        if models['regressor'] is None:
            st.error("Please upload or place a regression model first.")
        else:
            processed_input = input_data.copy()
            pred = predict_from_model(models['regressor'], processed_input)
            if pred is not None:
                price_inr = float(pred[0])
                st.success(f"Predicted Price: ₹{price_inr:,.2f} INR")

with colB:
    if st.button("📈 Predict Market Segment"):
        if models['clusterer'] is None:
            st.error("Please upload or place a clustering model first.")
        else:
            processed_input_cluster = input_data.copy()
            X_num = processed_input_cluster.select_dtypes(include=[np.number])
            try:
                if hasattr(models['clusterer'], 'predict'):
                    cluster_label = int(models['clusterer'].predict(X_num)[0])
                elif hasattr(models['clusterer'], 'cluster_centers_'):
                    centers = models['clusterer'].cluster_centers_
                    dists = ((centers - X_num.values) ** 2).sum(axis=1)
                    cluster_label = int(dists.argmin())
                else:
                    cluster_label = None
                if cluster_label is not None:
                    cluster_name = cluster_map.get(str(cluster_label), "Unknown Segment")
                    st.success(f"Cluster #{cluster_label}: {cluster_name}")
                    st.write(f"📝 **Cluster Insight:** {cluster_name} represents diamonds with similar market characteristics.")
            except Exception as e:
                st.error(f"Cluster prediction error: {e}")

# ------------------- Optional Insights -------------------
st.write("---")
st.subheader("ℹ️ Model Information & Usage Notes")
st.markdown(
    "- Upload trained `.pkl` models or keep them in a `models/` folder.\n"
    "- Regressor predicts **price**; clusterer predicts **market segment**.\n"
    "- Make sure both models were trained with the same features (columns) as this app input.\n"
    "- Cluster names can be customized in sidebar (JSON format)."
)
st.caption("App generated for project: Diamond Dynamics — Developed for Colab deployment.")
