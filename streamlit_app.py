import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load("rf_model.pkl")

# Load label encoders for categorical variables (if you used them)
# These should be saved before in your model code if you used them
# For example, for "Gender", "Smoking", etc.
gender_encoder = joblib.load("gender_encoder.pkl")  # Assuming this is saved earlier

# Function to make a prediction
def predict(input_data):
    # Standardize the input data (assuming model expects scaled data)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform([input_data])

    # Predict using the model
    prediction = model.predict(scaled_data)
    return prediction[0]

# Streamlit user interface
st.title("Thyroid Recurrence Prediction")

# Collect user inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.selectbox("Gender", ("Female", "Male"))
smoking = st.selectbox("Smoking", ("Yes", "No"))
hx_smoking = st.selectbox("History of Smoking", ("Yes", "No"))
hx_radiotherapy = st.selectbox("History of Radiotherapy", ("Yes", "No"))
thyroid_function = st.selectbox("Thyroid Function", ("Euthyroid", "Clinical Hyperthyroidism"))

# Physical examination (left or right)
physical_examination = st.selectbox("Physical Examination", ("Single nodular goiter-left", "Single nodular goiter-right", "Multinodular goiter"))

# Pathology type
pathology = st.selectbox("Pathology", ("Micropapillary", "Other"))

# Focality (uni or multi-focal)
focality = st.selectbox("Focality", ("Uni-Focal", "Multi-Focal"))

# Risk level
risk = st.selectbox("Risk", ("Low", "High"))

# Tumor, Nodes, Metastasis and Stage
t = st.selectbox("T", ("T1a", "T1b", "T2", "T3", "T4"))
n = st.selectbox("N", ("N0", "N1", "N2", "N3"))
m = st.selectbox("M", ("M0", "M1"))
stage = st.selectbox("Stage", ("I", "II", "III", "IV"))

# Collect all inputs into a list (or pandas DataFrame)
input_data = [
    age,
    gender_encoder.transform([gender])[0],  # Encoding gender
    smoking == "Yes",
    hx_smoking == "Yes",
    hx_radiotherapy == "Yes",
    thyroid_function == "Euthyroid",
    1 if "left" in physical_examination else 0,  # Left=1, Right=0
    pathology == "Micropapillary",  # assuming binary classification for pathology
    1 if focality == "Multi-Focal" else 0,  # Uni-Focal=0, Multi-Focal=1
    1 if risk == "High" else 0,  # High=1, Low=0
    1 if t == "T1a" else 0,  # Simplified encoding for T (more detailed encoding could be done)
    1 if n == "N0" else 0,  # Similarly for N
    1 if m == "M0" else 0,  # Similarly for M
    1 if stage == "I" else 0,  # Stage I=1, others=0
]

# Button to make prediction
if st.button("Predict Recurrence"):
    prediction = predict(input_data)
    if prediction == 0:
        st.write("Prediction: No recurrence.")
    else:
        st.write("Prediction: Recurrence likely.")
