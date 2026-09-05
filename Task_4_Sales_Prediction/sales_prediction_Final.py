import streamlit as st
import pickle
import numpy as np

with open("sales_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("📈 Sales Prediction")

st.write("Enter advertising expenditure:")

tv = st.number_input(
    "TV Advertising",
    min_value=0.0,
    value=100.0
)

radio = st.number_input(
    "Radio Advertising",
    min_value=0.0,
    value=20.0
)

newspaper = st.number_input(
    "Newspaper Advertising",
    min_value=0.0,
    value=20.0
)

if st.button("Predict Sales"):

    input_data = np.array([
        [tv, radio, newspaper]
    ])

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Sales: {prediction[0]:.2f}"
    )