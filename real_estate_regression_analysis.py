import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

# Load the data
data = pd.read_csv('real_estate_data.csv')  # Make sure to replace this with the actual file path.

# Multiple Linear Regression
X = data[['size', 'year']]
y = data['price']
X = sm.add_constant(X)  # Adding a constant for the intercept

# Fit the model
model = sm.OLS(y, X).fit()
print(model.summary())

# R-squared and Adjusted R-squared
r_squared = model.rsquared
adjusted_r_squared = model.rsquared_adj
print(f'R-squared: {r_squared}, Adjusted R-squared: {adjusted_r_squared}')

# P-values
p_values = model.pvalues
print(f'P-values: {p_values}')

# Make a prediction for 750 sq.ft in 2009
new_data = pd.DataFrame({'const': 1, 'size': [750], 'year': [2009]})
predicted_price = model.predict(new_data)
print(f'Predicted price for 750 sq.ft in 2009: ${predicted_price[0]:.2f}')

# Simple Linear Regression for comparison
X_simple = data[['size']]
y_simple = data['price']
model_simple = LinearRegression().fit(X_simple, y_simple)

# Prediction for simple linear regression
predicted_price_simple = model_simple.predict(np.array([[750]]))
print(f'Predicted price using simple linear regression for 750 sq.ft: ${predicted_price_simple[0]:.2f}')
