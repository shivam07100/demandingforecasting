# Hyperpure Demand Forecasting Application

This project is a demand forecasting application for Hyperpure’s farm-to-fork model, which provides next-day delivery of perishable products (fruits, vegetables, frozen food, etc.) to clients in the hospitality sector. The objective is to provide clients with accurate demand predictions for optimal inventory levels, minimizing wastage and shortages.

## Project Overview

The application uses historical sales data to predict optimal product order quantities based on factors such as seasonality, customer demand trends, and special events. This is achieved using a machine learning model (Linear Regression) to forecast demand over a given period.

### Directory Structure

- **data/sales_data.csv**: The simulated historical sales data used for model training.
- **scripts/datacollection.py**: Loads and preprocesses data.
- **scripts/model.py**: Trains the demand forecasting model and saves it.
- **scripts/application.py**: Forecasts demand for user-provided inputs.
- **requirements.txt**: Lists the required Python libraries.
- **forecast_report.pdf**: Brief report outlining the approach and evaluation.

### Installation

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/finaldemandforecasting.git
   cd finaldemandforecasting
2. Install the required dependencies:
pip install -r requirements.txt

### Usage
Step 1: Data Collection 
python scripts/datacollection.py

Step 2: Development
python scripts/development.py

Step 3: Run the Application
python scripts/application.py

### Evaluation Metrics
The model's performance is evaluated using Mean Absolute Error (MAE) to measure accuracy. This metric is suitable for assessing demand forecasting applications due to its interpretability and effectiveness.

### Dependencies
Python 3.x
Pandas
Scikit-learn
Joblib
