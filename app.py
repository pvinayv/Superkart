import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download
import os

st.title("Sales Forecast App")

# Download the model from Hugging Face Model Hub
REPO_ID = os.getenv("HF_MODEL_REPO", "pvinayv/superkart-model") # Replace with your repo ID
MODEL_FILENAME = "xgboost_model.pkl"

@st.cache_resource
def load_model():
    try:
        model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

if model:
    st.write("Model loaded successfully. Please input the features to predict sales.")
    
    col1, col2 = st.columns(2)
    with col1:
        product_weight = st.number_input("Product Weight", min_value=0.0, value=10.0)
        product_sugar_content = st.selectbox("Product Sugar Content", ["Low Fat", "Regular"])
        product_allocated_area = st.number_input("Product Allocated Area", min_value=0.0, value=0.05)
        product_mrp = st.number_input("Product MRP", min_value=0.0, value=150.0)
    with col2:
        store_establishment_year = st.number_input("Store Establishment Year", min_value=1980, max_value=2024, value=1999)
        store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
        store_location_city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
        store_type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Grocery Store"])

    if st.button("Predict Sales"):
        # Map inputs to model expected format (must match data preparation steps)
        # Assuming basic label encoding matching training step
        input_data = pd.DataFrame({
            "Product_Weight": [product_weight],
            "Product_Allocated_Area": [product_allocated_area],
            "Product_MRP": [product_mrp],
            "Store_Establishment_Year": [store_establishment_year],
            # To be properly encoded in real app (simplified here)
            "Product_Sugar_Content_Regular": [1 if product_sugar_content == "Regular" else 0],
            "Store_Size_Medium": [1 if store_size == "Medium" else 0],
            "Store_Size_High": [1 if store_size == "High" else 0],
            "Store_Location_City_Type_Tier 2": [1 if store_location_city_type == "Tier 2" else 0],
            "Store_Location_City_Type_Tier 3": [1 if store_location_city_type == "Tier 3" else 0],
            "Store_Type_Supermarket Type1": [1 if store_type == "Supermarket Type1" else 0],
            "Store_Type_Supermarket Type2": [1 if store_type == "Supermarket Type2" else 0],
            "Store_Type_Supermarket Type3": [1 if store_type == "Supermarket Type3" else 0],
        })
        
        # Adjust input_data columns to match the model's exact features...
        try:
            prediction = model.predict(input_data)
            st.success(f"Predicted Sales: ${prediction[0]:.2f}")
        except Exception as e:
            st.error(f"Prediction Error: {e}. Ensure input features match the trained model's features.")
