"""Model training, evaluation, testing, validation, and prediction using LSTM."""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add project path for module imports
sys.path.append(r"C:\Users\girli\OneDrive\Desktop\Utbildningskostnader_Födelsetal_Prognos")

# Custom imports
from EDA.coefficients import calculate_birth_mortality_migration_birth_rates
from EDA.data_loading import load_data

# Database and model paths
DB_PATH = r"C:\Users\girli\OneDrive\Desktop\Utbildningskostnader_Födelsetal_Prognos\ds_database.db"
MODEL_PATH = r"C:\Users\girli\OneDrive\Desktop\Utbildningskostnader_Födelsetal_Prognos\pretrained_models_forecasting_birth\best_lstm_model.keras"
SCALER_PATH = r"C:\Users\girli\OneDrive\Desktop\Utbildningskostnader_Födelsetal_Prognos\pretrained_models_forecasting_birth\scaler.joblib"

def load_and_prepare_data(db_path):
    """Load and prepare population and coefficient data."""
    tables = load_data(db_path)
    population_df = pd.concat([
        tables('population_0_16_per_region')[['Year', 'Total_Population']],
        tables('population_17_19_per_region')[['Year', 'Total_Population']]
    ], ignore_index=True)

    avg_mortality, avg_migration, avg_birth, _ = calculate_birth_mortality_migration_birth_rates()
    for df, name in [(avg_birth, "Birth"), (avg_migration, "Migration"), (avg_mortality, "Mortality")]:
        if 'Year' not in df.columns:
            df['Year'] = population_df['Year']
            print(f"Added 'Year' column to {name} DataFrame")

    combined_df = (
        avg_birth.merge(avg_migration, on=['Region_Code', 'Year'], how='inner')
                 .merge(avg_mortality, on=['Region_Code', 'Year'], how='inner')
    )

    if 'Year' in combined_df.columns:
        combined_df.set_index('Year', inplace=True)
    else:
        raise ValueError("Error: 'Year' column not found in combined_df after merging.")

    return combined_df

def scale_data(df, scaler_path):
    """Scale features and save the scaler."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Region_Code', 'Birth_Coefficient', 'Migration_Coefficient', 'Mortality_Coefficient']])
    joblib.dump(scaler, scaler_path)
    return scaled_data

def create_lstm_sequences(data, time_step=1):
    """Create sequences for LSTM model training."""
    X, y = [], []

    # Ensure there is enough data for sequences
    if len(data) <= time_step:
        print("Not enough data for the given time_step")
        return np.array(X), np.array(y)

    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])

    X, y = np.array(X), np.array(y)

    # Debug: Print shapes of X and y
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    return X, y

def build_lstm_model(time_step):
    """Build and compile an LSTM model."""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def train_and_evaluate_lstm(data, time_step, model_path):
    """Train and evaluate LSTM model using TimeSeriesSplit."""
    tscv = TimeSeriesSplit(n_splits=5)
    best_model, best_rmse = None, float('inf')

    for fold, (train_idx, val_idx) in enumerate(tscv.split(data), 1):
        train, val = data[train_idx], data[val_idx]

        # Check if train and val are non-empty
        if len(train) <= time_step or len(val) <= time_step:
            print(f"Skipping fold {fold} due to insufficient data for time_step {time_step}")
            continue

        X_train, y_train = create_lstm_sequences(train, time_step)
        X_val, y_val = create_lstm_sequences(val, time_step)

        # Skip fold if sequences are empty
        if X_train.shape[0] == 0 or X_val.shape[0] == 0:
            print(f"Skipping fold {fold} due to empty sequences.")
            continue

        # Debug: Print shapes of X_train and X_val before reshaping
        print(f"Fold {fold} - X_train shape before reshaping: {X_train.shape}")
        print(f"Fold {fold} - X_val shape before reshaping: {X_val.shape}")

        X_train, X_val = X_train.reshape((X_train.shape[0], X_train.shape[1], 1)), X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

        # Debug: Print shapes of X_train and X_val after reshaping
        print(f"Fold {fold} - X_train shape after reshaping: {X_train.shape}")
        print(f"Fold {fold} - X_val shape after reshaping: {X_val.shape}")
        print(f"Fold {fold} - Train length: {len(train)}, Val length: {len(val)}")


        model = build_lstm_model(time_step)
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, verbose=0)

        rmse = np.sqrt(mean_squared_error(y_val, model.predict(X_val).flatten()))
        print(f'Fold {fold} - RMSE: {rmse:.4f}')

        if rmse < best_rmse:
            best_rmse, best_model = rmse, model

    # Check if a best model was found
    if best_model is not None:
        best_model.save(model_path)
        print(f'Best Model RMSE: {best_rmse:.4f}')
    else:
        print("No model was trained due to insufficient data.")
    return best_model

def main():
    """Main function to load data, scale, and train LSTM model."""
    combined_df = load_and_prepare_data(DB_PATH)
    scaled_data = scale_data(combined_df, SCALER_PATH)
    train_and_evaluate_lstm(scaled_data, time_step=1, model_path=MODEL_PATH)

if __name__ == "__main__":
    main()
