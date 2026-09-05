import pandas as pd

data = pd.read_csv("train.csv")

print(data.head())

print(data.info())

print(data.describe())
print("\nDataset Shape:")
print(data.shape)

print("\nColumn Names:")
print(data.columns)

print("\nMissing Values:")
print(data.isnull().sum())

data = data.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

data["Age"] = data["Age"].fillna(data["Age"].median())

data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

print("\nMissing values after cleaning:")
print(data.isnull().sum())

data["Sex"] = data["Sex"].map({"male": 0, "female": 1})

data["Embarked"] = data["Embarked"].map({
    "S": 0,
    "C": 1,
    "Q": 2
})

print("\nData after encoding:")
print(data.head())

X = data.drop("Survived", axis=1)
y = data["Survived"]

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

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("\nModel training completed successfully!")

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy * 100, "%")

from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

import pickle

with open("titanic_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved successfully!")