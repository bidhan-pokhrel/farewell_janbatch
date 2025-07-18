import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load("rf_model.pkl")

# Load the original dataset for encoding info
data = pd.read_csv("classificationDataset.csv")

# Prepare categorical encoders
datanonnumeric = data.drop(columns=["Age"])
categorical_columns = datanonnumeric.columns.tolist()

# Build encoders
category_encoders = {
    col: dict(enumerate(data[col].astype("category").cat.categories))
    for col in categorical_columns
}
inverse_encoders = {
    col: {v: k for k, v in enc.items()} for col, enc in category_encoders.items()
}

# Streamlit UI
st.title("Recurrent Cancer Prediction")
st.write("Enter patient information to predict if cancer **recurred**.")

# Input form
user_input = {}
for col in categorical_columns:
    options = list(category_encoders[col].values())
    user_input[col] = st.selectbox(f"{col}", options)

age = st.number_input("Age", min_value=0, max_value=120, value=50)

if st.button("Predict"):
    # Build DataFrame from input
    input_df = pd.DataFrame([user_input])
    input_df["Age"] = age

    # Encode categorical data
    for col in categorical_columns:
        input_df[col] = input_df[col].map(inverse_encoders[col])

    # Scale the input
    full_data = data.drop(columns="Recurred")
    scaler = StandardScaler().fit(full_data)
    input_scaled = scaler.transform(input_df[categorical_columns + ["Age"]])

    # Predict
    prediction = model.predict(input_scaled)

    # Show prediction only if it's "Recurred"
    if prediction[0] == 1:
        st.subheader("⚠️ Prediction: Recurred")
    else:
        st.subheader("✅ No recurrence detected")
