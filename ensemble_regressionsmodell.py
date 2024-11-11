"""Train, evaluate, test and validate ensemble regression models for forecasting education costs."""

# Import necessary libraries
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set project paths
DB_PATH = r"C:\Users\girli\OneDrive\Desktop\Utbildningskostnader_Födelsetal_Prognos\ds_database.db"
sys.path.append(r"C:\Users\girli\OneDrive\Desktop\Utbildningskostnader_Födelsetal_Prognos")
MODEL_DIR = "pretrained_ensemble_regression_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Define model paths
MODEL_PATHS = {
    f"{level}_{type}": os.path.join(MODEL_DIR, f"{level}_{type}_model.joblib")
    for level in ["primary", "secondary"] for type in ["fixed_costs", "current_costs"]
}

# Import data loading function
from EDA.data_loading import load_data

# Load and prepare data
def load_cost_data(db_path):
    """Load and prepare grundskola and gymnasieskola costs per region data."""
    tables = load_data(db_path)
    grundskola_data = tables('grundskola_costs_per_region')
    gymnasieskola_data = tables('gymnasieskola_costs_per_region')
    data = pd.merge(
        grundskola_data.rename(columns={'Fixed_Cost_Per_Child_SEK': 'Primary_Fixed_Costs', 'Current_Cost_Per_Child_SEK': 'Primary_Current_Costs'}),
        gymnasieskola_data.rename(columns={'Fixed_Cost_Per_Child_SEK': 'Secondary_Fixed_Costs', 'Current_Cost_Per_Child_SEK': 'Secondary_Current_Costs'}),
        on='Year', how='outer', validate='one_to_one'
    )
    return data[['Year']], data[['Primary_Fixed_Costs', 'Secondary_Fixed_Costs']], data[['Primary_Current_Costs', 'Secondary_Current_Costs']]

# Ensemble model creation
def create_ensemble():
    """Create an ensemble model with Linear Regression, Random Forest, and Gradient Boosting Regressors."""
    return VotingRegressor(estimators=[
        ('lr', LinearRegression()),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42, max_features='sqrt', min_samples_leaf=1)),
        ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5, learning_rate=0.1))
    ])

# Model evaluation
def evaluate_model(model, X_test, y_test, label):
    """Evaluate the model using RMSE, MAE, and R2 metrics."""
    preds = model.predict(X_test)
    metrics = {
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "MAE": mean_absolute_error(y_test, preds),
        "R2": r2_score(y_test, preds)
    }
    print(f"{label}: RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}, R2={metrics['R2']:.4f}")
    return model, metrics

# Training, evaluation, and model saving
# Train and save model function
def train_and_save_models(X, y, label_prefix):
    model = create_ensemble()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    evaluate_model(model, X_test, y_test, label_prefix)
    joblib.dump(model, MODEL_PATHS[label_prefix])

# Forecasting and visualization
def forecast_and_plot(X, y_actual, label):
    """Forecast future years and plot the actual vs forecasted values."""
    future_years = np.arange(2024, 2036).reshape(-1, 1)
    X_extended = StandardScaler().fit_transform(np.vstack([X['Year'].values.reshape(-1, 1), future_years]))

    plt.figure(figsize=(10, 6))
    plt.plot(X, y_actual, label='Actual', marker='o')
    model = joblib.load(MODEL_PATHS[label])
    predictions = model.predict(X_extended)
    plt.plot(np.concatenate([X['Year'], future_years.flatten()]), predictions, '--', label='Forecast')
    plt.title(f"{label.replace('_', ' ').title()} Education Costs (2007-2035)")
    plt.xlabel("Year")
    plt.ylabel("Cost (SEK)")
    plt.legend()
    plt.grid()
    plt.show()

# Main execution
X, target_fixed, target_current = load_cost_data(DB_PATH)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[['Year']])
for target, label in [(target_fixed, "fixed_costs"), (target_current, "current_costs")]:
    for column, level in zip(target.columns, ["primary", "secondary"]):
        train_and_save_models(X_scaled, target[column], f"{level}_{label}")

# Plotting forecasts for each model
for label, y_data in zip(["primary_fixed_costs", "primary_current_costs", "secondary_fixed_costs", "secondary_current_costs"],
                         [target_fixed['Primary_Fixed_Costs'], target_current['Primary_Current_Costs'], target_fixed['Secondary_Fixed_Costs'], target_current['Secondary_Current_Costs']]):
    forecast_and_plot(X, y_data, label)
