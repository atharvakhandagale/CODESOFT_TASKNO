import streamlit as st
import joblib
import numpy as np
model = joblib.load("iris_model.pkl")
flower_names = [
    "Iris Setosa",
    "Iris Versicolor",
    "Iris Virginica"
]
st.title("🌸 Iris Flower Classification")
st.write("Enter the measurements of the Iris flower:")
sepal_length = st.number_input(
    "Sepal Length (cm)",
    min_value=0.0,
    value=5.1
)
sepal_width = st.number_input(
    "Sepal Width (cm)",
    min_value=0.0,
    value=3.5
)
petal_length = st.number_input(
    "Petal Length (cm)",
    min_value=0.0,
    value=1.4
)
petal_width = st.number_input(
    "Petal Width (cm)",
    min_value=0.0,
    value=0.2
)
if st.button("Predict Flower"):
    input_data = np.array([
        [sepal_length, sepal_width, petal_length, petal_width]
    ])

    prediction = model.predict(input_data)
    predicted_flower = flower_names[prediction[0]]
    st.success(f"Predicted Flower: {predicted_flower}")
    st.info("Model Accuracy: 96.67%")