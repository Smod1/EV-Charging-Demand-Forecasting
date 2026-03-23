import argparse
import numpy as np
import pandas as pd
import sys
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from data.data_pipeline import generate_synthetic_ev_data, engineer_features, temporal_train_test_split
from models.forecasters import LinearForecaster, RandomForestForecaster, get_all_models, evaluate, HAS_XGB
from utils.visualisation import plot_raw_demand, plot_predictions, plot_model_comparison, plot_feature_importance
import json

#These will be my global variables
TARGET = "demand_kWh"
DROP_COLS = {"demand_kWh", "sessions", "location","tod_factor", "seasonal_factor", "temp_effect","rain_effect", "trend_factor", "holiday_effect"}


def main(quick = False, no_plots = False):
    banner = """
╔══════════════════════════════════════════════════════╗
║   Sarun Modi EV Charging Demand Forecasting System   ║ 
╚══════════════════════════════════════════════════════╝
    """ #This banner was AI generated (ChatGPT) purely for presentational purposes.
    print(banner)

#We first generate and load the data   
    print("STEP 1: Data Pipeline!")
    print("━" * 60)
    start = "2022-01-01"
    if quick:
        end = "2023-06-01"
    else:
        end = "2024-01-01"
    print("Generating synthetic UK EV charging data from " + start + " to " + end) 
    raw_data = generate_synthetic_ev_data(start_date=start, end_date=end)
    print(f"Raw data shape: {raw_data.shape}")
    print(f"Columns: {list(raw_data.columns)}")
    if not no_plots:
        print("Plotting data …")
        plot_raw_demand(raw_data, save=True)

#We must now transform raw data into features which improve accuracy and the model's performance
#The engineer_features function does this for us
    print("\nSTEP 2: Feature Engineering!")
    print("━" * 60)
    features = engineer_features(raw_data, target_col=TARGET)
    feature_cols = [col for col in features.columns if col not in DROP_COLS]
    print(f"Feature matrix: {features.shape}")
    print(f"Feature count: {len(feature_cols)}")
    print(f"Sample features: {feature_cols[:8]}")

#Now we train the model, and split the test with it.
    print("\nSTEP 3: Temporal Train / Test Split!")
    print("━" * 60)
    if quick:
        test_months = 2
    else:
        test_months = 3
    train,test = temporal_train_test_split(features, test_months)
    X_train = train[feature_cols]
    y_train = train[TARGET]
    X_test  = test[feature_cols]
    y_test  = test[TARGET]

#And now we test the model.
    print("\nSTEP 4: Model Training And Evaluation!")
    print("━" * 60)
    models = get_all_models()
    if quick:
        models = [LinearForecaster(),RandomForestForecaster(n_estimators=100, max_depth=8),] #these were imported.
    results = []
    predictions = {}
    trained_models = []
    for model in models:
        print(model.name)
        result = model.fit_predict(X_train, y_train, X_test, y_test)
        prediction = model.predict(X_test)
        results.append(result)
        predictions[model.name] = prediction
        trained_models.append(model)
        model.save()
    print("\nSTEP 5: Summary")
    print("━" * 60)
    summary = pd.DataFrame(results).sort_values("RMSE") #organises results into rows and columns (pandas dataframe), the values are sorted by
    #RMSE, a column given by the imported library, standing for Root Mean Squared errors  ascending.
    print(summary.to_string(index=False))
    best = summary.iloc[0]["model"] #iloc gets the first row i.e the lowest RMSE value
    print("Best model: " + best)

    #We now produce plots of predictions and compare models.
    if not no_plots:
        print("\nSTEP 6: Generating Figures!!")
        print("━" * 60)
        plot_predictions(y_test, predictions, n_hours=24 * 7, save=True)
        plot_model_comparison(results, save=True)
        for m in trained_models:
            if hasattr(m, "feature_importances_"): #Check if it has the attribute, i.e. if it is tree based (Random Forest)
                plot_feature_importance(m, feature_cols, top_n=20, save=True) #we plot the top 20 most important features to interpret which features
                #drive the model's performance

    #Saves the data and models to csv files 
    features.to_csv(ROOT / "data" / "featured_data.csv")
    summary.to_csv(ROOT / "models" / "results_summary.csv", index=False)
    meta = {"feature_cols": feature_cols,"target_col": TARGET, "drop_cols": list(DROP_COLS), "best_model": best, "train_end": str(train.index[-1].date()),
        "test_end": str(test.index[-1].date()),}
    with open(ROOT / "models" / "meta.json", "w") as f: json.dump(meta, f, indent=2) #Saves meta dictionary to a JSON file 

    print("\n All data and models were successfully saved!")
    print("Run streamlit dashboard python file to launch the dashboard.")
    return results, predictions, y_test, feature_cols


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick",    action="store_true", help="Fast run (less data, fewer estimators)")
    parser.add_argument("--no-plots", action="store_true", help="Skip matplotlib figure generation")
    args = parser.parse_args()
    main(quick=args.quick, no_plots=args.no_plots)
