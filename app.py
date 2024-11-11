"""Streamlit application for forecasting education costs and birth rates in Sweden."""
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Paths to pretrained models
MODEL_PATHS = {
    "primary_fixed_costs": r"C:\Users\girli\OneDrive\Desktop\Utbildningskostnader_Födelsetal_Prognos\pretrained_ensemble_regression_models\primary_fixed_costs_model.joblib",
    "primary_current_costs": r"C:\Users\girli\OneDrive\Desktop\Utbildningskostnader_Födelsetal_Prognos\pretrained_ensemble_regression_models\primary_current_costs_model.joblib",
    "secondary_fixed_costs": r"C:\Users\girli\OneDrive\Desktop\Utbildningskostnader_Födelsetal_Prognos\pretrained_ensemble_regression_models\secondary_fixed_costs_model.joblib",
    "secondary_current_costs": r"C:\Users\girli\OneDrive\Desktop\Utbildningskostnader_Födelsetal_Prognos\pretrained_ensemble_regression_models\secondary_current_costs_model.joblib",
    "birth_forecast": r"C:\Users\girli\OneDrive\Desktop\Utbildningskostnader_Födelsetal_Prognos\pretrained_models_forecasting_birth\best_lstm_model.keras",
    "scaler": r"C:\Users\girli\OneDrive\Desktop\Utbildningskostnader_Födelsetal_Prognos\pretrained_models_forecasting_birth\scaler.joblib"
}

# Load models
education_models = {key: joblib.load(path) for key, path in MODEL_PATHS.items() if 'costs' in key}
birth_model = load_model(MODEL_PATHS['birth_forecast'])
scaler = joblib.load(MODEL_PATHS['scaler'])

# Region mapping
region_mapping = {
    "01": "Stockholms län", "03": "Uppsala län", "04": "Södermanlands län", "05": "Östergötlands län",
    "06": "Jönköpings län", "07": "Kronobergs län", "08": "Kalmar län", "09": "Gotlands län",
    "10": "Blekinge län", "12": "Skåne län", "13": "Hallands län", "14": "Västra Götalands län",
    "17": "Värmlands län", "18": "Örebro län", "19": "Västmanlands län", "20": "Dalarnas län",
    "21": "Gävleborgs län", "22": "Västernorrlands län", "23": "Jämtlands län", "24": "Västerbottens län",
    "25": "Norrbottens län"
}

# Sidebar selections
st.sidebar.title("Prognos Inställningar")
forecast_type = st.sidebar.selectbox("Välja Prognostyp", ["Utbildningskostnader", "Födelsetal"])

# Assuming we need only the year feature
scaler = MinMaxScaler()

# Fit scaler on the year data only (assuming it's just one column as input)
# Example: fitting on a range of years for demonstration
scaler.fit(np.array(range(1968, 2024)).reshape(-1, 1))

# Your main code
if forecast_type == "Utbildningskostnader":
    # UI input as before
    region_code = st.sidebar.selectbox("Län", options=region_mapping.keys(), format_func=lambda x: f"{x} - {region_mapping[x]}")
    school_level = st.sidebar.selectbox("Skolnivå", options=["Grundskola", "Gymnasieskola"])
    cost_type = st.sidebar.selectbox("Kostnadstyp", options=["Fasta Kostnader", "Löpande Kostnader"])
    years = st.sidebar.multiselect("Select Year(s)", options=range(2025, 2036), default=2025)
    fetch_button = st.sidebar.button("Beräkna")

    # Prepare model key
    level_map = {"Grundskola": "primary", "Gymnasieskola": "secondary"}
    cost_map = {"Fasta Kostnader": "fixed_costs", "Löpande Kostnader": "current_costs"}
    model_key = f"{level_map[school_level]}_{cost_map[cost_type]}"

    if fetch_button:
        # Ensure single-feature array (n_samples, 1)
        input_years = np.array(years).reshape(-1, 1)

        # Scaling single-feature data
        input_scaled = scaler.transform(input_years)

        # Prediction
        predictions = education_models[model_key].predict(input_scaled)
        forecast_df = pd.DataFrame({"Year": years, "Forecasted Cost (SEK)": predictions.flatten()})

        # Display results
        st.subheader(f"Forecast for {school_level} - {cost_type} in {region_mapping[region_code]}")
        st.write(forecast_df)

        # Interactive plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast_df["Year"], y=forecast_df["Forecasted Cost (SEK)"],
                                 mode='lines+markers', name="Forecasted Cost"))
        fig.update_layout(title=f"{school_level} - {cost_type} Forecast", xaxis_title="Year", yaxis_title="Cost (SEK)")
        st.plotly_chart(fig)
