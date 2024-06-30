import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Data Collection
# Assuming you have a CSV file with historical sales data
data = pd.read_csv('sales_data.csv')

# 2. Data Preprocessing
data['date'] = pd.to_datetime(data['date'])
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['day_of_week'] = data['date'].dt.dayofweek

# Drop columns that won't be used for prediction
data = data.drop(['date'], axis=1)

# Handle missing values, if any
data = data.fillna(0)

# 3. Feature Selection
features = ['year', 'month', 'day', 'day_of_week']
X = data[features]
y = data['sales']

# 4. Splitting the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 5. Choosing and Training the Model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Evaluating the Model
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse}')

# 7. Forecasting
future_dates = pd.date_range(start='2024-07-01', end='2024-12-31', freq='D')
future_data = pd.DataFrame({
    'year': future_dates.year,
    'month': future_dates.month,
    'day': future_dates.day,
    'day_of_week': future_dates.dayofweek
})
future_predictions = model.predict(future_data)
future_data['predicted_sales'] = future_predictions

print(future_data)
