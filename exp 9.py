import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import operator

# Generating synthetic data
np.random.seed(0)
X = 2 - 3 * np.random.normal(0, 1, 100)
y = X - 2 * (X ** 2) + np.random.normal(-3, 3, 100)

X = X[:, np.newaxis]

# Linear Regression
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)
y_pred_linear = linear_regressor.predict(X)

# Polynomial Regression
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)

poly_regressor = LinearRegression()
poly_regressor.fit(X_poly, y)
y_pred_poly = poly_regressor.predict(X_poly)

# Plotting the results
plt.scatter(X, y, color='blue', s=10, label='Data Points')

# Linear Regression Line
plt.plot(X, y_pred_linear, color='green', label='Linear Regression')

# Polynomial Regression Line
# Sort the values of X before line plot (to get a smooth curve)
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(X, y_pred_poly), key=sort_axis)
X_sorted, y_poly_sorted = zip(*sorted_zip)
plt.plot(X_sorted, y_poly_sorted, color='red', label='Polynomial Regression')

plt.legend()
plt.show()
