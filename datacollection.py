import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load the dataset
data = pd.read_csv("sales_data.csv")

# Convert 'date' column to datetime
data['date'] = pd.to_datetime(data['date'])

# Extract features from date
data['month'] = data['date'].dt.month
data['year'] = data['date'].dt.year


data = pd.get_dummies(data, columns=['product_name', 'day_of_week'], drop_first=True)


X = data.drop(columns=['date', 'quantity_sold'])
y = data['quantity_sold']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
