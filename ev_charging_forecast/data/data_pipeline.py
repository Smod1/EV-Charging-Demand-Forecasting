import numpy as np
import pandas as pd
from datetime import datetime, timedelta
#Generates synthetic hourly data in years 2022-2024 (mimics properties and variation of real data)
#Include time of day matterns, weekday vs weekend comparison, seasonal variation
#Temperature effects, holiday effects, and some random noise 
def generate_synthetic_ev_data(start_date = "2022-01-01", end_date = "2024-01-01", location = "London", seed = 42):
    np.random.seed(seed)
    dates = pd.date_range(start=start_date, end=end_date, freq="h")
    numDates = len(dates)
    data_frame = pd.DataFrame({"timestamp": dates})
    data_frame["location"] = location
    data_frame["hour"] = data_frame["timestamp"].dt.hour
    data_frame["day_of_week"] = data_frame["timestamp"].dt.dayofweek   #Where 0 = Monday, ... , 6 = Sunday
    data_frame["month"] = data_frame["timestamp"].dt.month
    data_frame["year"] = data_frame["timestamp"].dt.year
    data_frame["day_of_year"] = data_frame["timestamp"].dt.dayofyear
    data_frame["week_of_year"] = data_frame["timestamp"].dt.isocalendar().week.astype(int) #isocalendar().week returns the week number from the iso calendar
    #It is then converted from a Series to an int.
    data_frame["is_weekend"] = (data_frame["day_of_week"] >= 5).astype(int)


    #Returns a relative demand level, which is created by using Gaussian-like peaks
    #If weekday, then there is a morning peak at 8am (commuter rush) and 18:00 (when work finishes)
    #If weekend, There is a peak at around midday (family travels), and a smaller peak at 20:00 (when returning back home)
    #Both formulae will add a standard consistent baseline of 0.05
    #We square negative terms as part of the Gaussian-like algorithm found online.
    def time_of_day_pattern(hour, is_weekend):
        if is_weekend:
            relative_demand = 0.3 * np.exp(-((hour - 12) ** 2) / 18) + 0.1 * np.exp(-((hour - 20) ** 2) / 8) + 0.05
        else:
            relative_demand = 0.4 * np.exp(-((hour - 8)  ** 2) / 4) + 0.5 * np.exp(-((hour - 18) ** 2) / 6) + 0.05
        return relative_demand
    data_frame["tod_factor"] = [time_of_day_pattern(hour, isWeekend) for hour, isWeekend in zip(data_frame["hour"], data_frame["is_weekend"])]
    #We now consider seasons - winter => cold => peak, summer => dip
    #We have modelled the seasonal factor as a harmonic seasonal component, as suggested online
    #Using the formula seasonal factor = baseline + amplitude * cos(2pi(day-dayshift)/365)
    #In this case, we use the baseline = 1, amplitude = 0.25 and the time shift is 15 because the middle of winter months e.g. January is usually colder
    #This ensures that the peak occurs on day 15 rather than day 1
    #We model the base temperature similarly
    data_frame["seasonal_factor"] = 1.0 + 0.25 * np.cos(2 * np.pi * (data_frame["day_of_year"] - 15) / 365)
    base_temp = 10 + 8 * np.sin(2 * np.pi * (data_frame["day_of_year"] - 80) / 365)
    data_frame["temperature"] = base_temp + np.random.normal(0, 3, numDates)
    #When it is cold, demand should be higher. This should also happen when it is very hot.
    data_frame["temp_effect"] = 1.0 + 0.015 * np.maximum(0, 5 - data_frame["temperature"])
    #We model precipitation as a Normal(1.5,2^2) distribution and adds 0.5 to the precipitation when it is winter (october - feb)
    #As winter months are wetter. Precipitation is in mm.
    data_frame["precipitation"] = np.maximum(0, np.random.normal(1.5, 2, numDates) + 0.5 * (data_frame["month"].isin([10,11,12,1,2])))
    data_frame["rain_effect"] = 1.0 + 0.05 * (data_frame["precipitation"] > 2).astype(float)
    days_since_start = (data_frame["timestamp"] - data_frame["timestamp"].iloc[0]).dt.days #Gets the number of days from first timestamp in dataset
    data_frame["trend_factor"] = 1.0 + 0.0003 * days_since_start   # approx 10.95% increase in trend factor per year
    #We multiplied by 0.0003 to get a linear increase 
    #Now let us say that holidays reduce the demand by 20%
    #Then the holiday effect would be 0.8
    uk_holidays = uk_bank_holidays(start_date, end_date) # gets UK bank holidays
    data_frame["is_holiday"] = data_frame["timestamp"].dt.date.astype(str).isin(uk_holidays).astype(int) #checks if date is a bank holiday
    data_frame["holiday_effect"] = 1.0 - 0.20 * data_frame["is_holiday"] 
    base_demand = 80.0 # we set a base demand before including all the factors to account for.
    data_frame["demand_kWh"] = (base_demand * data_frame["tod_factor"] * data_frame["seasonal_factor"] * data_frame["temp_effect"] * data_frame["rain_effect"]
        * data_frame["trend_factor"]* data_frame["holiday_effect"] + np.random.normal(0, 3, numDates) #Gaussian noise randomly included.
        + np.random.exponential(1.5, numDates) #Occasional spikes due to events etc.
        ).clip(lower=0) # replaces any value less than 0 with 0 
    # ── Number of sessions (loosely correlated with demand) ───────────────
    data_frame["sessions"] = (data_frame["demand_kWh"] / np.random.uniform(8, 15, numDates)).round().astype(int).clip(lower=0)
    #Gets how many charging sessions occured in a day
    #We model the energy per session as a Unif(8, 15) distribution as a typical EV charging session uses approx 8-15 kWh (online source)
    data_frame = data_frame.set_index("timestamp").sort_index()
    #Sorts the rows by the timestamp index to ensure chronological order 
    return data_frame

#Returns the UK bank holidays (hard-coded)
def uk_bank_holidays(start, end):
    years = range(int(start[:4]), int(end[:4]) + 1)
    holidays = []
    for year in years:
        holidays = holidays + [f"{year}-01-01", f"{year}-12-25", f"{year}-12-26", f"{year}-05-01", f"{year}-08-26",]
    return holidays

def engineer_features(data_frame, target_col = "demand_kWh"):
    """Add lag, rolling, and cyclical features to the DataFrame."""
    data_frame = data_frame.copy() #we don't want to modify the original data
    #We preserve cyclical nature and periodicity. Machine learning models don't inherently understand circularity (hour 23 and 0 are close)
    data_frame["hour_sin"] = np.sin(2 * np.pi * data_frame["hour"] / 24) 
    data_frame["hour_cos"] = np.cos(2 * np.pi * data_frame["hour"] / 24)
    data_frame["dow_sin"] = np.sin(2 * np.pi * data_frame["day_of_week"] / 7)
    data_frame["dow_cos"] = np.cos(2 * np.pi * data_frame["day_of_week"] / 7)
    data_frame["month_sin"] = np.sin(2 * np.pi * data_frame["month"] / 12)
    data_frame["month_cos"] = np.cos(2 * np.pi * data_frame["month"] / 12)

    #Lag features allow the model to see recent history 
    for lag in [1, 2, 3, 6, 12, 24, 48, 168]:   #hours up to one week
        data_frame[f"lag_{lag}h"] = data_frame[target_col].shift(lag)
    #We can compute summary statistics over previous hours excluding the current hour
    #This is called rolling statistics
    for window in [3, 6, 12, 24, 168]:
        data_frame[f"roll_mean_{window}h"] = (data_frame[target_col].shift(1).rolling(window).mean()) #Gets the mean over last 'window' hours
        data_frame[f"roll_std_{window}h"] = (data_frame[target_col].shift(1).rolling(window).std()) #Gets the standard deviation over last 'window' hours
    #Useful extra statistics - get the demand for same hour last week and same hour yesterday.
    data_frame["same_hour_last_week"] = data_frame[target_col].shift(168)
    data_frame["same_hour_yesterday"] = data_frame[target_col].shift(24)
    data_frame = data_frame.dropna() #deletes rows with missing data
    return data_frame

#We perform a time-based split ensuring that the test set always contains the most recent observations (3 months)
def temporal_train_test_split(data_frame, test_months = 3):
    split_point = data_frame.index[-1] - pd.DateOffset(months=test_months) #gives a split point exactly test_months before end of data
    train = data_frame[data_frame.index <= split_point]
    test = data_frame[data_frame.index > split_point] #This ensures no data leakage from future into past.
    print(f"Train: {train.index[0].date()} to {train.index[-1].date()} ({len(train):,} rows)")
    print(f"Test : {test.index[0].date()}  to {test.index[-1].date()}  ({len(test):,} rows)")
    return train, test #returns the two data frames as a tuple.

#This acts as a quick testing function.
if __name__ == "__main__":
    print("Generating synthetic EV data!")
    raw = generate_synthetic_ev_data()
    print(raw[["demand_kWh", "sessions", "temperature"]].describe().round(2))
    print("\nEngineering features!")
    features = engineer_features(raw)
    print(f"Feature matrix: {features.shape}")
    print("\nSplitting data!")
    train, test = temporal_train_test_split(features)
    print("Finished process!")
