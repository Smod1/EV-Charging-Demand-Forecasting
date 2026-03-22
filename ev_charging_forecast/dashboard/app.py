import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st #we will use this as our front end dashboard.

#Configuration of the page
st.set_page_config(page_title="EV Charge Demand Forecast by Sarun Modi", page_icon="!!",layout="wide", #We use the full browser width
                    initial_sidebar_state="expanded") #sidebar is opened by default 


# ── Custom CSS ────────────────────────────────────────────────────────────────
#st.markdown("""
#<style>
#@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');

#html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

#.metric-card {
#    background: linear-gradient(135deg, #1a1d2e, #0f1117);
#    border: 1px solid #2a2d3e;
#    border-radius: 12px;
#    padding: 20px;
#    text-align: center;
#}
#.metric-value { font-size: 2rem; font-weight: 700; color: #4C9BE8; }
#.metric-label { font-size: 0.85rem; color: #888; margin-top: 4px; }
#.badge-best   { background: #2A9D8F; color: white; border-radius: 6px;
#                padding: 2px 8px; font-size: 0.75rem; }
#</style>
#""", unsafe_allow_html=True)

from data.data_pipeline import generate_synthetic_ev_data, engineer_features
from models.forecasters import (LinearForecaster, RandomForestForecaster, HAS_XGB)
from utils.visualisation import (plotly_forecast_24h, plotly_heatmap, plotly_model_comparison)

if HAS_XGB:
    from models.forecasters import XGBoostForecaster



@st.cache_data(show_spinner="Generating dataset") #the spinner is a loading indicator in streamlit that spins
def load_data():
    raw = generate_synthetic_ev_data(start_date="2022-01-01", end_date="2024-01-01")
    feat = engineer_features(raw)
    return raw, feat

TARGET = "demand_kWh"
DROP_COLS = {"demand_kWh", "sessions", "location", "tod_factor", "seasonal_factor", "temp_effect", "rain_effect", "trend_factor", "holiday_effect"}

@st.cache_resource(show_spinner="Training models")
def train_models(feature_cols):
    raw, feat = load_data()
    split_pt  = feat.index[-1] - pd.DateOffset(months=3)
    train = feat[feat.index <= split_pt]
    test = feat[feat.index > split_pt] #ensures no data overlapping
    X_tr, y_tr = train[list(feature_cols)], train[TARGET]
    X_te, y_te = test[list(feature_cols)], test[TARGET]
    models_to_train = [LinearForecaster(), RandomForestForecaster(n_estimators=200, max_depth=10),]
    if HAS_XGB:
        models_to_train.append(XGBoostForecaster(n_estimators=300))
    results, predictions = [], {}
    for model in models_to_train:
        from models.forecasters import evaluate
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        result = evaluate(y_te.values, preds, model.name)
        results.append(result)
        predictions[model.name] = preds
    return models_to_train, results, predictions, y_te, X_te


def make_24h_forecast(model, feature_dataframe, feature_cols):
    last = feature_dataframe.iloc[-1].copy() #Uses last known row as a template
    future_rows = []
    base_timestamp = feature_dataframe.index[-1] #last timestamp in data
    for hour in range(1, 25): #Forecast hours 1 through to 24 inclusive 
        timestamp = base_timestamp + timedelta(hours=hour) #This is the future timestamp 
        row = last.copy()
        #We now update time-based columns 
        row["hour"] = timestamp.hour
        row["day_of_week"] = timestamp.dayofweek
        row["month"] = timestamp.month
        row["hour_sin"] = np.sin(2 * np.pi * timestamp.hour / 24)
        row["hour_cos"] = np.cos(2 * np.pi * timestamp.hour / 24)
        row["dow_sin"] = np.sin(2 * np.pi * timestamp.dayofweek / 7)
        row["dow_cos"] = np.cos(2 * np.pi * timestamp.dayofweek / 7)
        row["month_sin"] = np.sin(2 * np.pi * timestamp.month / 12)
        row["month_cos"] = np.cos(2 * np.pi * timestamp.month / 12)
        future_rows.append(row)
    X_fut = pd.DataFrame(future_rows)[feature_cols] #Creates a dataframe from 24 rows and keeps only features the moel expects
    predictions = model.predict(X_fut)
    index = pd.date_range(start=base_timestamp + timedelta(hours=1), periods=24, freq="h") #Builds a datetime index for the forecast period
    return pd.Series(predictions, index=index, name=TARGET)

def main():
    #We first markdown the header of the dashboard
    st.markdown("  EV Charging Demand Forecasting  ")
    st.markdown("Real-time ML predictions for UK EV charging infrastructure")
    #Now we load the data
    raw_dataframe, feature_dataframe = load_data()
    feature_cols = [col for col in feature_dataframe.columns if col not in DROP_COLS]
    #And implement a sidebar
    with st.sidebar: #Below will only contain things inside the left sidebar
        st.markdown("Controls:")
        location = st.selectbox("Location:", ["London", "Manchester", "Birmingham", "Bristol"])
        st.markdown("\n")
        if HAS_XGB:
             model_choice = st.selectbox("Forecast Model:",["RandomForest", "LinearRegression", "XGBoost"])
        else:
             model_choice = st.selectbox("Forecast Model:",["RandomForest", "LinearRegression", ""])
        st.markdown("\n")
        show_heatmap = st.toggle("Show Demand Heatmap", value=True) #These are toggle switches, a bit like checkboxes 
        show_compare = st.toggle("Show Model Comparison", value=True)
        show_raw = st.toggle("Show Raw Demand Chart", value=False) #default off 
        st.markdown("\n")
        st.markdown("Dataset Info:")
        st.caption(f"Records: {len(raw_dataframe):,}") #adds captions 
        st.caption(f"Period: {raw_dataframe.index[0].date()} to {raw_dataframe.index[-1].date()}")
        st.caption(f"Features: {len(feature_cols)}")
    #We now train the models
    with st.spinner("Training models!"): 
        trained, results, predictions, y_test, X_test = train_models(tuple(feature_cols)) #tuple ensures column list is hashable for caching 
    #And now map model choice to trained object
    model_map = {model.name: model for model in trained}
    selected_model = model_map.get(model_choice, trained[0])
    best_result = min(results, key=lambda result: result["RMSE"])#finds the results dictionary with smallest RMSE (best model overall)
    result_selected = next((result for result in results if result["model"] == model_choice), best_result) #Finds the result for the user selected model
    #If not found then uses best result.
    #We now create columns to display metrics 
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Test Demand:", f"{y_test.sum():,.0f} kWh")
    with col2:
        st.metric(f"{model_choice} Root mean square error: ", f"{result_selected['RMSE']:.2f}")
    with col3:
        st.metric(f"Mean absolute error: ", f"{result_selected['MAE']:.2f}")
    with col4:
        st.metric("Best Model!! \n", best_result["model"])
    st.markdown("\n")
    #We now add the section that displays a 24 hour demand forecast
    st.subheader("24-Hour Demand Forecast!")
    forecast_24h = make_24h_forecast(selected_model, feature_dataframe, feature_cols)
    history = raw_dataframe["demand_kWh"]
    fig_forecast = plotly_forecast_24h(history, forecast_24h, model_name=model_choice) #Gets the forecast over 24 hours as a go.Figure object
    st.plotly_chart(fig_forecast, use_container_width=True) #ensures fills container width and plots the chart
    #And now we make a table for the forecast data
    with st.expander("View Forecast Data"): #User can click to expand.
        forecast_dataframe = forecast_24h.reset_index()
        forecast_dataframe.columns = ["Timestamp", "Predicted Demand (kWh)"]
        forecast_dataframe["Predicted Demand (kWh)"] = forecast_dataframe["Predicted Demand (kWh)"].round(2)
        st.dataframe(forecast_dataframe, use_container_width=True)
        st.download_button("Download Forecast CSV here!", forecast_dataframe.to_csv(index=False), file_name="ev_forecast_24h.csv", mime="text/csv")
        #Adds a download_button to the streamlit dashboard for users that want the CSV file of the forecast.
    #We now want to compare actual vs predictied demand
    st.subheader("Test Set: Predictions vs Actual")
    n_plot = st.slider("Hours to display", 48, 168, 72, step=24) #User can slide to change how many hours they wish to display
    predictions = predictions.get(model_choice, list(predictions.values())[0])
    index = y_test.index[:n_plot]
    plot_dataframe = pd.DataFrame({"Actual": y_test.values[:n_plot], "Predicted": predictions[:n_plot]}, index=index) #gets the dataframe for the plot.
    import plotly.graph_objects as go
    fig_test = go.Figure()
    fig_test.add_trace(go.Scatter(x=index, y=plot_dataframe["Actual"], name="Actual", line=dict(color="#264653", width=2))) #Added a line for the scatter plot
    fig_test.add_trace(go.Scatter(x = index, y = plot_dataframe["Predicted"], name = "Predicted", line = dict(color = "#4C9BE8", width = 2, dash = "dash")))
    fig_test.update_layout(template = "plotly_dark", height = 380, xaxis_title = "Time", yaxis_title = "Demand (kWh)")
    st.plotly_chart(fig_test, use_container_width=True)
    #And now if selected, display the heatmap:
    if show_heatmap:
        st.subheader("Demand Heatmap (Day × Hour)")
        fig_heatmap = plotly_heatmap(raw_dataframe)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    #And now display a summary of the comparison in performance between models.
    if show_compare:
        st.subheader("Model Performance Comparison:")
        fig_compare = plotly_model_comparison(results)
        st.plotly_chart(fig_compare, use_container_width=True)
        st.dataframe(pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True), use_container_width=True)
    #Finally, we display the raw data that was used.
    if show_raw:
        st.subheader("Raw Dataset Sample:")
        st.dataframe(raw_dataframe.head(100), use_container_width=True)
    #We now display the footer of the dashboard.
    st.markdown("\n")
    st.caption("EV Charging Demand Forecasting System - Built with Python, scikit-learn, XGBoost & Streamlit by Sarun Modi")


if __name__ == "__main__":
    main()
