import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv("sales_data.csv")

# Preprocess the dataset
data['date'] = pd.to_datetime(data['date'])
data['month'] = data['date'].dt.month
data['year'] = data['date'].dt.year
data['day_of_week'] = data['date'].dt.dayofweek  # 0 = Monday, ..., 6 = Sunday
data = pd.get_dummies(data, columns=['product_name'], drop_first=True)


X = data.drop(columns=['date', 'quantity_sold'])
y = data['quantity_sold']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)

def forecast_demand(input_data):
   
    input_df = pd.DataFrame([input_data])
    
    # One-hot encode the product_name and day_of_week columns
    input_df = pd.get_dummies(input_df, columns=['product_name'], drop_first=True)
    
   
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
   
    prediction = model.predict(input_df[X.columns])
    return prediction[0]

# Example input for the application
input_data = {
    'product_id': 1,
    'is_holiday': 0,
    'month': 10,
    'year': 2024,
    'product_name': 'Fruits',  
    'day_of_week': 5  # 5 representing Friday 
}


if model is not None and X is not None:
    forecasted_demand = forecast_demand(input_data)
    print(f"Forecasted demand: {forecasted_demand:.2f}")
else:
    print("Model or feature set is not defined.")
