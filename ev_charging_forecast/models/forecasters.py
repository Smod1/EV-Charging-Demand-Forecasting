import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
try:
    import xgboost as xgb 
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("xgboost not installed – skipping XGBoostForecaster") #Ensures no importing error is thrown by client.
try:
    from statsmodels.tsa.arima.model import ARIMA as ARIMA
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("statsmodels not installed – skipping ARIMAForecaster")
SAVE_DIR = Path(__file__).parent.parent / "models" / "saved"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def evaluate(y_true, y_pred, name = ""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) #Calculates the root mean squared error 
    mae = mean_absolute_error(y_true, y_pred) #Calculates the mean absolute error 
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1.0))) * 100 #Calculates the mean absolute percentage error 
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2) #Calculates the r-squared value (proportion of variance) useful in stats
    result = {"model": name, "RMSE": round(rmse, 4),"MAE": round(mae, 4), "MAPE": round(mape, 2), "R2": round(r2, 4)}
    print(f"{name:30s}  RMSE={rmse:.3f}  MAE={mae:.3f}  MAPE={mape:.1f}%  R^2={r2:.3f}") #Rounds to required precision number of decimal places.
    return result


#This class creates an interface for the base forecaster
class BaseForecaster:
    name = "BaseForecaster"
    
    def fit(self, X_train, y_train):
        raise NotImplementedError
    
    def predict(self, X_test):
        raise NotImplementedError
    
    def fit_predict(self, X_train, y_train, X_test, y_test):
        self.fit(X_train, y_train)
        predictions = self.predict(X_test)
        if hasattr(y_test, "values"): #If there is a test and values then we return the evaluation of the model for the test
            return evaluate(y_test.values)
        else:
            return y_test, preds, self.name

    def save(self):
        joblib.dump(self, SAVE_DIR / f"{self.name}.pkl") #saves as a pickle file 
        print(f"Saved to {SAVE_DIR / self.name}.pkl")

    @classmethod
    def load(a_class, name):
        return joblib.load(SAVE_DIR / f"{name}.pkl")


class LinearForecaster(BaseForecaster):
    name = "LinearRegression"

    def __init__(self, alpha = 1.0): #constructor 
        self.model = Pipeline([("scaler", StandardScaler()), ("ridge",  Ridge(alpha=alpha))]) #standardscaler standardises features to 0 mean and 1 variance
        #This ensures all features contribute equally. Ridge is a linear regression model 

    def fit(self, X, y):
        self.model.fit(X, y) #scales the features and fits the ridge regression 
        return self

    def predict(self, X):
        return np.maximum(0, self.model.predict(X)) #predicts target values for new features stored in feature matrix X
    #scales X then computes the linear prediction. Demand cannot be negative 


class ARIMAForecaster(BaseForecaster):
    name = "ARIMA"

    def __init__(self, order=(2, 1, 2), seasonal_order=(1, 1, 1, 24)): #constructor
        #sets default ARIMA orders
        #order(2,1,2) means 2 autoregressive terms, 1 differencing term, 2 moving average terms
        #seasonal_order(1,1,1,24) means 1 seasonal autoregressive term, 1 seasonal difference, 1 seasonal moving average term, 24 period (hourly data)
        if not HAS_STATSMODELS:
            raise ImportError("statsmodels required for ARIMAForecaster")
        self.order = order
        self.seasonal_order = seasonal_order
        self._fitted = None
        self._last_train = None

    def fit(self, X, y): #ARIMA is univariate so only uses the target series
        if hasattr(y, "values"): #i.e. target is a series already 
            self._last_train = y
        else:
            self._last_train = pd.Series(y) #ensures target is a series 
        model = ARIMA(self._last_train, order=self.order, seasonal_order=self.seasonal_order) #creates an instance of the seasonal ARIMA model 
        self._fitted = model.fit(disp=False) #estimates the parameters of the model and stores the result. disp = False means no output here.
        return self

    def predict(self, X) -> np.ndarray:
        numSteps_to_forecast = len(X) #X tells us how many future steps to generate
        forecast = self._fitted.forecast(steps=numSteps_to_forecast) #gets the panda series with the forecasted values of that many steps ahead.
        return np.maximum(0, forecast.values)


class RandomForestForecaster(BaseForecaster):
    name = "RandomForest"

    def __init__(self, n_estimators = 300, max_depth = 12, n_jobs = -1, random_state = 42):
        self.model = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, min_samples_leaf = 2, n_jobs = n_jobs,
        random_state=random_state) #n_estimators acts as the 'number of trees in the forest', we set a max depth to contorl overfitting
        #We use all available CPU cores for parallel training 

    def fit(self, X, y):
        self.model.fit(X, y) #trains the random forest and gives parameters required 
        return self 

    def predict(self, X): #predicts target values using fitted model 
        return np.maximum(0, self.model.predict(X))

    @property
    def feature_importances_(self):
        return self.model.feature_importances_ #feature importance array useful for analyzing which engineered features contributed most to predictions
    #E.g. was it lags or rolling stats or cyclical nature?


# ─────────────────────────────────────────────
# 4. XGBoost
# ─────────────────────────────────────────────

class XGBoostForecaster(BaseForecaster):
    name = "XGBoost"

    def __init__(self, n_estimators = 500, learning_rate = 0.05, max_depth = 6, subsample = 0.8, colsample_bytree = 0.8, random_state = 42):
        if not HAS_XGB:
            raise ImportError("xgboost required for XGBoostForecaster")
        self.model = xgb.XGBRegressor(n_estimators = n_estimators, learning_rate = learning_rate, max_depth = max_depth, subsample = subsample,
            colsample_bytree = colsample_bytree, random_state = random_state, verbosity = 0, n_jobs=-1)
        #n_estimators is number of boosting rounds in this case
        #The learning rate is the step size shrinkage
        #subsample is the fraction of training data to sample for each tree
        #colsample by tree is the fraction of features to sample for each tree
        #verbosity = 0 means no output during training
        
    def fit(self, X, y):
        self.model.fit(X, y, eval_set=[(X, y)], verbose=False) #trains the model, eval_set is used to monitor performance
        #on the training set 
        return self

    def predict(self, X):
        return np.maximum(0, self.model.predict(X))

    @property
    def feature_importances_(self):
        return self.model.feature_importances_




def get_all_models():
    models = [LinearForecaster(),RandomForestForecaster(),]
    if HAS_XGB:
        models.append(XGBoostForecaster())
    return models #makes a list of all the models trained and tested.


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent)) #Allows importing modules such as data.data_pipeline even if script is run from subfolder
    from data.data_pipeline import (generate_synthetic_ev_data, engineer_features, temporal_train_test_split)
    TARGET = "demand_kWh"
    DROP = ["demand_kWh", "sessions", "location", "tod_factor","seasonal_factor", "temp_effect", "rain_effect", "trend_factor", "holiday_effect"]
    #These are columns to be excluded from the feature set - should not be used as input features
    print("Building dataset")
    raw = generate_synthetic_ev_data() #gets synthetic time-series dataframe
    feat = engineer_features(raw) #Applies feature engineering 
    train, test = temporal_train_test_split(feat) #splits the data chronologically. Test set contains most recent data 
    feature_cols = [col for col in feat.columns if col not in DROP]
    X_train, y_train = train[feature_cols], train[TARGET]
    X_test,  y_test  = test[feature_cols],  test[TARGET]
    print(f"\nFeatures: {len(feature_cols)}\n")
    results = []
    for model in get_all_models():
        print(f"Model: {model.name}")
        result = model.fit_predict(X_train, y_train, X_test, y_test)
        results.append(result)
        model.save()

    print("\nSummary:")
    summary = pd.DataFrame(results).sort_values("RMSE") #ascending by default
    print(summary.to_string(index=False)) #prints summary table without row indices to show performance of all models.
