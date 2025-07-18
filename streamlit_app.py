import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load("rf_model.pkl")

# Load the dataset to get categories and column structure
data = pd.read_csv("classificationDataset.csv")

# Prepare data like in training
datanonnumeric = data.drop(columns=["Age"])
categorical_columns = datanonnumeric.columns.tolist()

# Get category encodings
category_encoders = {
    col: dict(enumerate(data[col].astype("category").cat.categories))
    for col in categorical_columns
}
inverse_encoders = {
    col: {v: k for k, v in enc.items()} for col, enc in category_encoders.items()
}

st.title("Classification Model Prediction")
st.write("Enter feature values to predict recurrence")

# Collect user input
user_input = {}
for col in categorical_columns:
    options = list(category_encoders[col].values())
    user_input[col] = st.selectbox(f"{col}", options)

age = st.number_input("Age", min_value=0, max_value=120, value=50)

# Create a DataFrame for the input
input_df = pd.DataFrame([user_input])
input_df["Age"] = age

# Encode categorical inputs
for col in categorical_columns:
    input_df[col] = input_df[col].map(inverse_encoders[col])

# Ensure column order matches training
input_encoded = input_df[categorical_columns + ["Age"]]

# Scale the input (standard scaling like in training)
scaler = StandardScaler()
X_scaled_all = scaler.fit_transform(data.drop(columns="Recurred"))  # Fit scaler on full data
input_scaled = scaler.transform(input_encoded)

# Make prediction
prediction = model.predict(input_scaled)

# Show result
st.subheader("Prediction:")
st.write("Recurred" if prediction[0] == 1 else "Not Recurred")
