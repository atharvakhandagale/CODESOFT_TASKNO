import pandas as pd

data = pd.read_csv("advertising.csv")

print(data.head())
print("\nDataset Information:")
print(data.info())

print("\nBasic Statistics:")
print(data.describe())
print("\nMissing Values:")
print(data.isnull().sum())

X = data[["TV", "Radio", "Newspaper"]]
y = data["Sales"]

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data size:", X_train.shape)
print("Testing data size:", X_test.shape)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel training completed successfully!")

y_pred = model.predict(X_test)

print("\nPredicted Sales:")
print(y_pred[:10])

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\nModel Evaluation:")
print("R2 Score:", r2)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
# Save the trained model

import pickle

with open("sales_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved successfully!")