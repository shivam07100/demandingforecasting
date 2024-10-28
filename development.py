import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load the dataset
data = pd.read_csv("sales_data.csv")  


data['date'] = pd.to_datetime(data['date'])


data['month'] = data['date'].dt.month
data['year'] = data['date'].dt.year

# One-hot encode the product_name and day_of_week columns
data = pd.get_dummies(data, columns=['product_name', 'day_of_week'], drop_first=True)

X = data.drop(columns=['date', 'quantity_sold'])
y = data['quantity_sold']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")
