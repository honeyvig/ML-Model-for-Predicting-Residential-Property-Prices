# ML-Model-for-Predicting-Residential-Property-Prices
This project used machine learning algorithms to predict residential property prices and price changes in Hong Kong.

The objective was to utilize artificial intelligence (AI) and existing property data to analyze the factors driving property prices and price changes to predict future property price trends in Hong Kong.

The project used two databases. One was a large residential sales dataset prepared by one of the largest property agencies in Hong Kong. This comprised 159,676 entries containing property sales transactions from 2020 to 2023 from 18 geographical districts. A second dataset incorporated macroeconomic indicators based on the census and statistics department of the government of Hong Kong. The two datasets were then merged.

The merged dataset was then reviewed and cleaned up before various machine learning algorithms were applied. This involved several steps: first, examining and understanding the data, performing data analysis, cleaning, managing, and normalizing the dataset. Then, the merged dataset was analyzed using various machine-learning techniques, including regression, decision trees, XGBoost, MLP and LSTM.

In the end the best results were with LSTM and ensemble techniques.

Additionally, hyperparameter optimization was carried out to the machine learning algorithms to enhance the accuracy of the results and predictions.

The machine learning algorithms identified a high correlation between XXX, XXX and property prices and XXX [placeholder]

[However, the study was limited by datasets that were incomplete, included the COVID-19 period and were over a short period time (3 years), and improving the datasets would have been time-consuming and beyond the scope of the study.]
=================
To implement a machine learning model for predicting residential property prices and price changes in Hong Kong, we need to follow these steps:

    Data Collection: Load the datasets for property sales transactions and macroeconomic indicators.
    Data Preprocessing: Clean the data, handle missing values, and normalize the datasets.
    Feature Engineering: Merge the two datasets (property sales and macroeconomic indicators) and create relevant features.
    Modeling: Apply various machine learning algorithms, including regression, decision trees, XGBoost, MLP, and LSTM. Optimize hyperparameters for better predictions.
    Evaluation: Evaluate the model performance using metrics like RMSE, MAE, etc.

Python Code

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error

# Load datasets
property_sales_df = pd.read_csv('property_sales_data.csv')
macroeconomic_df = pd.read_csv('macroeconomic_data.csv')

# Step 1: Data Exploration and Cleaning

# Check for missing values
print(property_sales_df.isnull().sum())
print(macroeconomic_df.isnull().sum())

# Handle missing values (drop rows with missing values as a simple strategy)
property_sales_df = property_sales_df.dropna()
macroeconomic_df = macroeconomic_df.dropna()

# Merge datasets on a common key (e.g., 'district' or 'date')
merged_df = pd.merge(property_sales_df, macroeconomic_df, on='date', how='inner')

# Step 2: Feature Engineering
# Select relevant features for the model
X = merged_df[['sqft', 'num_bedrooms', 'district', 'macro_indicator1', 'macro_indicator2']]  # example features
y = merged_df['price']

# Convert categorical variables (e.g., 'district') to numerical values
X = pd.get_dummies(X, columns=['district'], drop_first=True)

# Normalize data using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 3: Apply Various Machine Learning Models

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# Decision Tree Regressor
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# Random Forest Regressor
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# Gradient Boosting Regressor
gb = GradientBoostingRegressor(random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)

# XGBoost Regressor
xgb_model = xgb.XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

# Evaluate models
def evaluate_model(predictions, true_values):
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predictions)
    r2 = r2_score(true_values, predictions)
    return mse, rmse, mae, r2

# Evaluation of all models
models = [lr_pred, dt_pred, rf_pred, gb_pred, xgb_pred]
model_names = ['Linear Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'XGBoost']

for i, model_pred in enumerate(models):
    mse, rmse, mae, r2 = evaluate_model(model_pred, y_test)
    print(f"Model: {model_names[i]}")
    print(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, R^2: {r2:.2f}\n")

# Step 4: Apply LSTM Model for Time-Series Predictions
# Prepare data for LSTM model (considering 'price' as the target variable and using past 30 days data)
sequence_length = 30
X_lstm, y_lstm = [], []
for i in range(sequence_length, len(merged_df)):
    X_lstm.append(merged_df['price'][i-sequence_length:i].values)
    y_lstm.append(merged_df['price'][i])

X_lstm = np.array(X_lstm)
y_lstm = np.array(y_lstm)

# Reshaping for LSTM
X_lstm = X_lstm.reshape(X_lstm.shape[0], X_lstm.shape[1], 1)

# Define LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(units=64, return_sequences=True, input_shape=(X_lstm.shape[1], 1)))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=32, return_sequences=False))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(units=1))

lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Train LSTM model
lstm_model.fit(X_lstm, y_lstm, epochs=10, batch_size=32)

# Predict using LSTM
lstm_pred = lstm_model.predict(X_lstm[-len(y_test):])

# Evaluate LSTM model
mse, rmse, mae, r2 = evaluate_model(lstm_pred, y_test)
print(f"LSTM Model Evaluation:\nMSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, R^2: {r2:.2f}\n")

# Step 5: Hyperparameter Optimization for XGBoost (as an example)
xgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6],
    'learning_rate': [0.01, 0.1]
}

xgb_grid = GridSearchCV(estimator=xgb.XGBRegressor(random_state=42), param_grid=xgb_params, cv=3, scoring='neg_mean_squared_error')
xgb_grid.fit(X_train, y_train)

print(f"Best Parameters for XGBoost: {xgb_grid.best_params_}")
best_xgb_model = xgb_grid.best_estimator_
xgb_grid_pred = best_xgb_model.predict(X_test)
mse, rmse, mae, r2 = evaluate_model(xgb_grid_pred, y_test)
print(f"Optimized XGBoost Model Evaluation:\nMSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, R^2: {r2:.2f}")

# Step 6: Visualization of Results
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual Prices', color='blue')
plt.plot(xgb_pred, label='Predicted Prices (XGBoost)', color='red')
plt.title('Actual vs Predicted Property Prices')
plt.xlabel('Sample')
plt.ylabel('Price')
plt.legend()
plt.show()

Steps Explained:

    Data Preprocessing:
        We load and clean the property sales and macroeconomic datasets.
        Missing values are dropped for simplicity, though more sophisticated techniques like imputation could be used.
        The datasets are merged based on a common key (e.g., date or district).

    Feature Engineering:
        We select features that might influence property prices, such as square footage, number of bedrooms, district, and macroeconomic indicators.
        Categorical variables like 'district' are one-hot encoded to make them suitable for machine learning models.

    Modeling:
        We apply multiple machine learning models: Linear Regression, Decision Trees, Random Forest, Gradient Boosting, and XGBoost.
        We also implement an LSTM (Long Short-Term Memory) model for time-series prediction, which might be useful for capturing trends and seasonality in property prices.

    Evaluation:
        We evaluate each model using MSE, RMSE, MAE, and R² metrics.
        The LSTM model is specifically evaluated on time-series predictions.

    Hyperparameter Optimization:
        We perform hyperparameter tuning for the XGBoost model to improve its performance using GridSearchCV.

    Visualization:
        A comparison between the actual and predicted property prices is visualized.

Next Steps:

    The LSTM model and other machine learning algorithms can be further fine-tuned.
    Additional preprocessing steps like normalization or imputation could improve results.
    Feature engineering can be extended by adding more relevant features from the data.
