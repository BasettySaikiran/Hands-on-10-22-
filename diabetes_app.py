import streamlit as st
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load your custom deep learning model
model = tf.keras.models.load_model(r'C:\Users\saiki\Handson(10-22)\DLOptimized_model.h5')


# Preprocessing function for user input
def preprocess_input(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return np.array(data_scaled)

# Streamlit UI
st.title("Diabetes Prediction App")

# Add input fields for features
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=100, step=1)
bmi = st.number_input("BMI", min_value=0.0, step=0.1)
blood_pressure = st.number_input("Blood Pressure", min_value=0.0, step=0.1)
glucose = st.number_input("Glucose Level", min_value=0.0, step=0.1)

# Prediction button
if st.button("Predict"):
    # Encode the 'Gender' feature
    gender_encoded = 1 if gender == "Male" else 0

    # Prepare input data
    input_data = [[gender_encoded, age, bmi, blood_pressure, glucose]]
    input_data_preprocessed = preprocess_input(input_data)

    # Make prediction
    prediction = model.predict(input_data_preprocessed)[0]

    # Display result
    if prediction > 0.5:
        st.write("The model predicts that the patient is likely to have diabetes.")
    else:
        st.write("The model predicts that the patient is unlikely to have diabetes.")
