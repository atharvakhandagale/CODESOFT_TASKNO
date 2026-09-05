import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("titanic_model.pkl", "rb") as file:
    model = pickle.load(file)

# Title
st.title("🚢 Titanic Survival Prediction")

st.write("Enter passenger details:")

# Inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])

sex = st.selectbox("Sex", ["Male", "Female"])

age = st.number_input("Age", min_value=0.0, max_value=100.0, value=25.0)

sibsp = st.number_input(
    "Siblings/Spouses Aboard",
    min_value=0,
    max_value=10,
    value=0
)

parch = st.number_input(
    "Parents/Children Aboard",
    min_value=0,
    max_value=10,
    value=0
)

fare = st.number_input(
    "Fare",
    min_value=0.0,
    value=30.0
)

embarked = st.selectbox("Embarked", ["S", "C", "Q"])

# Convert categorical values
sex_value = 0 if sex == "Female" else 1

embarked_value = {
    "S": 0,
    "C": 1,
    "Q": 2
}[embarked]

# Prediction
if st.button("Predict Survival"):

    input_data = np.array([[
        pclass,
        sex_value,
        age,
        sibsp,
        parch,
        fare,
        embarked_value
    ]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("🎉 Passenger Survived")
    else:
        st.error("❌ Passenger Did Not Survive") 