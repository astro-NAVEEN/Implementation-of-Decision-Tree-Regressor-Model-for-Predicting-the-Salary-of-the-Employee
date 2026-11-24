# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and create/load the employee dataset containing years of experience and corresponding salary.
2. Split the dataset into training and testing sets.
3. Train the Decision Tree Regressor model using the training data and predict the salary for the test data.
4. Evaluate the model and predict the salary for a new employee based on years of experience.

## Program:

~~~py
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error, r2_score

# Sample Dataset
# Feature: Years of Experience
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])

# Target: Salary (in Rupees)
y = np.array([25000, 30000, 38000, 45000, 52000,
              60000, 70000, 82000, 95000, 110000])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Decision Tree Regressor
model = DecisionTreeRegressor(random_state=42)

model.fit(X_train, y_train)

# Predict salaries
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print("Actual Salaries:")

print(y_test)

print("\nPredicted Salaries:")

print(y_pred.astype(int))

print("\nMean Squared Error:", round(mse, 2))

print("R2 Score:", round(r2, 2))

# Predict salary for a new employee
experience = np.array([[6.5]])

predicted_salary = model.predict(experience)

print("\nPredicted Salary for 6.5 years of experience: ₹{:.2f}".format(predicted_salary[0]))

~~~

## Output:

<img width="801" height="611" alt="image" src="https://github.com/user-attachments/assets/5d7abd15-7cb2-48cf-8010-0dd463a4b3e0" />



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
