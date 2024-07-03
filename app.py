import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('gradient_boosting_best_model.pkl')

st.title("Gradient Boosting Model Predictor")

# Get user input for passenger count and trip distance
passenger_count = st.number_input('Passenger Count', min_value=1, max_value=10, value=1)
trip_distance = st.number_input('Trip Distance', min_value=0.0, value=1.0)

if st.button('Predict'):
    # Create a feature array with the relevant inputs
    features = np.array([[passenger_count, trip_distance]])
    # Predict the value using the model
    prediction = model.predict(features)
    # Display the prediction
    st.write(f"Predicted Fare Amount: ${prediction[0]:.2f}")

st.write("Note: Please input valid values for prediction.")
