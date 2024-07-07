import streamlit as st
import joblib
import numpy as np
import openai
# Load the trained model
model = joblib.load('gradient_boosting_best_model.pkl')

st.title("Taxi Fare Predictor")

# Get user input
passenger_count = st.number_input('Passenger Count', min_value=1, max_value=10, value=1)
trip_distance = st.number_input('Trip Distance(in miles)', min_value=0.0, value=1.0)


with st.expander("Optional Inputs"):
    PULocationID = st.selectbox('PULocationID', options=range(1, 301), index=0)
    transaction_day = st.selectbox('Transaction Day', options=range(1, 32), index=0)
    transaction_hour = st.selectbox('Transaction Hour', options=range(0, 24), index=0)

if st.button('Predict'):
    features = np.array([[passenger_count, trip_distance, PULocationID, transaction_day, transaction_hour]])
    prediction = model.predict(features)
    st.write(f"Predicted Fare: ${prediction[0]:.2f}")

st.write("Note: Please input valid values for prediction.")

# Chatbot function using OpenAI API
def generate_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.9,
    )
    return response.choices[0].text.strip()

# Chatbot interaction
st.header("Chat with Our Transit Fare Assistant")
chat_input = st.text_input("You: ", "Hello, how can I get assistance with transit fares?")
if st.button("Send"):
    if chat_input:
        response = generate_response(chat_input)
        st.write(f"Chatbot: {response}")
