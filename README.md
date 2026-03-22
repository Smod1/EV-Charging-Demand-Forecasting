# EV-Charging-Demand-Forecasting
EV Charging Demand Forecasting System is an end-to-end machine learning pipeline built in Python that predicts hourly electric vehicle charging demand across UK locations. The system generates realistic synthetic charging data incorporating time-of-day patterns (commuter peaks at 08:00 and 18:00), seasonal variation, temperature-driven range anxiety effects, UK bank holidays, and a long-term EV adoption trend — mirroring the structure of real datasets such as Zap-Map and UK Government open data.

The data pipeline engineers 36 features from raw time-series data, including cyclical sin/cos encodings for hour, day-of-week and month, lag features from 1 hour to 1 week, and rolling mean/standard deviation windows — all standard practice in production forecasting systems. A time-aware train/test split is enforced to prevent data leakage, ensuring models are always evaluated on future data they have never seen.

Four forecasting models are implemented behind a shared interface: a Ridge Regression baseline, Random Forest, XGBoost, and an optional LSTM built in PyTorch. Models are evaluated against RMSE, MAE, MAPE and R², with XGBoost and Random Forest consistently outperforming the linear baseline (R² of ~0.91 vs ~0.90). All trained models are serialised with joblib for reuse, and feature importance plots are generated for the tree-based models to aid interpretability.

The project includes a fully interactive Streamlit dashboard allowing users to select a forecasting model, view a 24-hour ahead demand forecast with confidence bands, explore a day-by-hour demand heatmap, compare model performance visually, and download forecast output as a CSV — making the system accessible to non-technical stakeholders.

Tech stack: Python, scikit-learn, XGBoost, pandas, NumPy, Plotly, Matplotlib, Streamlit

To use this project!!
1.Ensure the following dependencies are installed:
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
xgboost>=2.0
statsmodels>=0.14
matplotlib>=3.7
plotly>=5.17
streamlit>=1.30
joblib>=1.3
pyarrow>=14.0     

2. Train all models:
python train.py

3. Launch the dashboard:
py -m streamlit run ev_charging_forecast\dashboard\app.py

4. Open the dashboard (if not automatically opened):
Open http://localhost:8501 in your browser.

Here are the full names for the evaluation metrics:
- RMSE – Root Mean Squared Error (penalises large errors)
- MAE  – Mean Absolute Error (interpretable in kWh)
- MAPE – Mean Absolute Percentage Error (scale-independent)
- R²   – Coefficient of determination (1 = perfect)


