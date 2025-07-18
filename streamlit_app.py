import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load("rf_model.pkl")

# Load original dataset to extract structure
data = pd.read_csv("classificationDataset.csv")

# Drop target column to prepare encoding/scaling like during training
X_full = data.drop(columns=["Recurred"])
categorical_cols = X_full.drop(columns=["Age"]).columns.tolist()

# Encode categorical columns as category codes
X_encoded = X_full.copy()
for col in categorical_cols:
    X_encoded[col] = X_encoded[col].astype("category").cat.codes

# Fit the scaler on the encoded full dataset
scaler = StandardScaler()
scaler.fit(X_encoded)

# Build label encoders from original data
category_mappings = {
    col: dict(enumerate(data[col].astype("category").cat.categories))
    for col in categorical_cols
}
inverse_mappings = {
    col: {v: k for k, v in mapping.items()} for col, mapping in category_mappings.items()
}

# Streamlit App
st.title("Recurrent Cancer Prediction")
st.write("Enter patient data to predict if cancer **recurred**.")

# Collect user input
user_input = {}
for col in categorical_cols:
    options = list(inverse_mappings[col].keys())
    user_input[col] = st.selectbox(f"{col}", options)

age = st.number_input("Age", min_value=0, max_value=120, value=50)

# When user clicks predict
if st.button("Predict"):
    # Encode input using same encoding as training
    input_df = pd.DataFrame([user_input])
    for col in categorical_cols:
        input_df[col] = input_df[col].map(inverse_mappings[col])
    
    input_df["Age"] = age
    input_df = input_df[categorical_cols + ["Age"]]  # Maintain column order

    # Scale the input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)

    # Show result only if recurrence is predicted
    if prediction[0] == 1:
        st.error("⚠️ Prediction: Cancer Recurred")
    else:
        st.success("✅ Prediction: No Recurrence")
