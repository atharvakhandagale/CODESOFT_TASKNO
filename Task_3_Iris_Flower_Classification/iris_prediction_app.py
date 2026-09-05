import joblib
import numpy as np
model = joblib.load("iris_model.pkl")
flower_names = [
    "Iris Setosa",
    "Iris Versicolor",
    "Iris Virginica"
]

print("Iris Flower Classification")
print("---------------------------")

sepal_length = float(input("Enter sepal length (cm): "))
sepal_width = float(input("Enter sepal width (cm): "))
petal_length = float(input("Enter petal length (cm): "))
petal_width = float(input("Enter petal width (cm): "))
input_data = np.array([
    [sepal_length, sepal_width, petal_length, petal_width]
])

prediction = model.predict(input_data)
print("\nPredicted Flower:", flower_names[prediction[0]])
