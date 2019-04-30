# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Splitting data sets into Training sets and Test sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

# Feature Scaling - Sets values to the same scale, such that there is no "overlooking" of data
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

# Fitting SLR to Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting Test set results
Y_pred = regressor.predict(X_test)

# Visualizing the Training set results
# Training Set Graph
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs. Experience (Training Sets)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Test Set Graph
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs. Experience (Test Sets)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()