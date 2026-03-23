import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # we will use matplotlib to plot arrays
import matplotlib.dates as mdates 
from matplotlib.gridspec import GridSpec #This allows layouts in matplotlib 
from pathlib import Path 
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
FIGURES_DIR = Path(__file__).parent.parent / "figures"#Navigates to a new directory two levels in hierarchy above this directory 
FIGURES_DIR.mkdir(parents=True, exist_ok=True) #Adds to this directory 'figures' 
HAS_PLOTLY = True # This is assumed!
#Colours
PALETTE = {"LinearRegression": "#4C9BE8", "ARIMA": "#F4A261", "RandomForest": "#2A9D8F", "XGBoost": "#E76F51", "LSTM": "#9B5DE5", "actual": "#264653",}
#I have mapped each model for predictions to a distinct colour and included the actual outcomes too.

def style_plot():
    plt.rcParams.update({"figure.facecolor": "#0F1117", "axes.facecolor": "#0F1117", "axes.edgecolor": "#333", "axes.labelcolor": "#CCC", "xtick.color": "#999",
        "ytick.color": "#999", "text.color": "#EEE", "grid.color": "#222", "grid.linestyle": "--", "legend.facecolor": "#1A1D27", "legend.edgecolor": "#333",})
    #This is a function taken from the documentation of the matplotlib.pyplot library, it gives each of the features of the graph a colour
    #I have given it a darker theme.
    


def plot_raw_demand(data_frame, save = True):
    style_plot()
    figure, axes = plt.subplots(3, 1, figsize=(16, 12), tight_layout=True)  #Sets to 3 panels structure 
    figure.suptitle("EV Charging Demand – Exploratory Analysis", fontsize=16, y=1.01)
    #We consider the panel for daily total demand
    daily = data_frame["demand_kWh"].resample("D").sum() #Groups the data by calendar day and adds up all the demands within each day 
    axes[0].fill_between(daily.index, daily.values, alpha=0.7, color="#4C9BE8") #Fills the line plot 
    axes[0].plot(daily.index, daily.values, lw=0.5, color="#4C9BE8") #Plots the line with the same color.
    axes[0].set_title("Daily Total Demand (kWh)") 
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y")) #sets a formatter for the major tick labels (labels after each month) and
    #displays date as abbreviated month (%b) and 4-digit years (%Y)
    axes[0].tick_params(axis="x", rotation=30) #Rotates the x axis tick labels by 30 degrees to avoid overlapping 
    axes[0].grid(True) #puts on grid 
    #Plots average demand by the hour
    hourly_avg = data_frame.groupby("hour")["demand_kWh"].mean() # takes the mean demand over each hour
    axes[1].bar(hourly_avg.index, hourly_avg.values, color=["#2A9D8F" if hour in range(7, 21) else "#555" for hour in hourly_avg.index]) #Bar chart but
    #Highlights hours 7-20 in a different colour for the purposes
    axes[1].set_title("Average Demand by Hour of Day")
    axes[1].set_xlabel("Hour")
    axes[1].grid(True, axis="y")

    #Plots a temperature vs demand scatter
    sample = data_frame.sample(min(5000, len(data_frame)), random_state=42) # Takes a sample of at max 5000 points
    axes[2].scatter(sample["temperature"], sample["demand_kWh"],alpha=0.3, s=6, c="#E76F51") #alpha sets transparency, s sets marker size to small
    #as there are many points
    axes[2].set_title("Temperature vs Demand")
    axes[2].set_xlabel("Temperature (°C)")
    axes[2].set_ylabel("Demand (kWh)")
    axes[2].grid(True)
    if save:
        path = FIGURES_DIR / "eda.png"
        figure.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0F1117") #Saves the plots to a .png file 
        print(f"Saved to {path}")
    return figure


#Plots actual vs predicted values over a week's worth of hours.
def plot_predictions(y_test, predictions, n_hours = 24 * 7, save = True):
    style_plot()
    figure, axes = plt.subplots(len(predictions), 1, figsize=(16, 4 * len(predictions)), tight_layout=True) #Creates n panels where n is num predictions 
    if len(predictions) == 1:
        axes = [axes]
    index = y_test.index[:n_hours] #takes the first n_hours from the series of timestamps
    for axis, (name, preds) in zip(axes, predictions.items()):
        p = preds[:n_hours]
        axis.plot(index, y_test.values[:n_hours], label="Actual", color=PALETTE["actual"], lw=1.5) # lw is the line thickness 
        axis.plot(index, p, label=name, color=PALETTE.get(name, "#888"), lw=1.2, linestyle="--")
        axis.fill_between(index, y_test.values[:n_hours], p, alpha=0.15, color=PALETTE.get(name, "#888"))
        axis.set_title(f"{name} – Predicted vs Actual (first {n_hours}hours of test set)")
        axis.legend() # adds legend automatically in best place to help readers understand what each line or marker represents 
        axis.grid(True)
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))#%d is a zero-padded day e.g. 01 
    if save:
        path = FIGURES_DIR / "predictions.png"
        figure.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0F1117")
        print(f"Saved to {path}")
    return figure


def plot_model_comparison(results, save = True):
    style_plot()
    data_frame = pd.DataFrame(results).sort_values("RMSE") #sorts values by RMSE ascending 
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), tight_layout=True) # Sets 1 panel row with 3 columns
    figure.suptitle("Model Comparison", fontsize=14)
    colours = [PALETTE.get(model, "#888") for model in data_frame["model"]]
    for axis, metric in zip(axes, ["RMSE", "MAE", "R2"]):
        bars = axis.barh(data_frame["model"], data_frame[metric], color=colours) #creates a horizontal bar chart on the subplot
        axis.set_title(metric)
        axis.grid(True, axis="x")
        for bar, value in zip(bars, data_frame[metric]):
            axis.text(value * 1.01, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=9)
    if save:
        path = FIGURES_DIR / "model_comparison.png"
        figure.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0F1117")
        print(f"Saved to {path}")
    return figure


def plot_feature_importance(model, feature_names, top_n = 20, save = True):
    style_plot()
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n] #::-1 reverses the order of the indices 
    names = [feature_names[index] for index in indices]
    values = importances[indices]
    figure, axis = plt.subplots(figsize=(10, 6), tight_layout=True)
    axis.barh(names[::-1], values[::-1], color=PALETTE.get(model.name, "#4C9BE8"))
    axis.set_title(f"{model.name} – Top {top_n} Feature Importances")
    axis.grid(True, axis="x")
    if save:
        path = FIGURES_DIR / f"feature_importance_{model.name}.png"
        figure.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0F1117")
        print(f"Saved to {path}")
    return figure


#We now need helpers for streamlit dashboard.

def plotly_forecast_24h(history, forecast, model_name = "Model", history_hours = 72):
    if not HAS_PLOTLY:
        raise ImportError("plotly required")
    figure = go.Figure() #creates a new plotly figure
    historical_window = history.iloc[-history_hours:] # the last history hours of the input series 
    figure.add_trace(go.Scatter(x=historical_window.index, y=historical_window.values,name="Historical", line=dict(color=PALETTE["actual"], width=2),mode="lines"))
    #Adds a trace for the historical data, it is a solid line
    figure.add_trace(go.Scatter(x=forecast.index, y=forecast.values, name=f"Forecast ({model_name})", line=dict(color=PALETTE.get(model_name, "#4C9BE8"), width=2.5, dash="dash"),mode="lines"))
    #Adds a trace for the forecast (dashed line)
    
    #Confidence band for heuristic, 15% 
    upper = forecast.values * 1.15
    lower = forecast.values * 0.85
    figure.add_trace(go.Scatter(x=list(forecast.index) + list(forecast.index[::-1]), y=list(upper) + list(lower[::-1]),fill="toself", fillcolor=f"rgba(76,155,232,0.15)", line=dict(color="rgba(0,0,0,0)"), name="Confidence band", showlegend=True))
    #Adds a filled area between the upper and lower bounds
    #X coordinates are forwards then reversed
    #Y coordinates are upper then reversed lower
    #We fill the area enclosed
    #We do a semi-transparent blue fill with no border line     
    figure.update_layout(title=f"24-Hour EV Charging Demand Forecast – {model_name}",xaxis_title="Time",yaxis_title="Demand (kWh)",template="plotly_dark",legend=dict(orientation="h", y=-0.2),height=420,margin=dict(t=50, b=80))
    return figure


def plotly_heatmap(data_frame):
    if not HAS_PLOTLY:
        raise ImportError("plotly required")
    pivot = data_frame.groupby(["day_of_week", "hour"])["demand_kWh"].mean().unstack() #computes the average demand for each day and hour
    #and converts the panda series into a data frame where the rows becomes the days and columns become hours
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    figure = go.Figure(go.Heatmap(z=pivot.values,x=[f"{hour:02d}:00" for hour in pivot.columns],y=day_labels,colorscale="Viridis",colorbar=dict(title="kWh")))
    #draws a heatmap where z is the numeric matrix of days and hours
    #formats each hour as a string in 24 hour format e.g. 00:00
    #The color scale is the scheme of colors Viridis
    #We add a color bar with the title KWh
    figure.update_layout(title="Average Demand Heatmap (Day × Hour)", xaxis_title="Hour of Day", yaxis_title="Day of Week", template="plotly_dark", height=320)
    #Adds a title, axis and labels with the dark theme 
    return figure


def plotly_model_comparison(results):
    if not HAS_PLOTLY:
        raise ImportError("plotly required")
    data_frame = pd.DataFrame(results).sort_values("RMSE")
    colours = [PALETTE.get(model, "#888") for model in data_frame["model"]]
    figure = make_subplots(rows=1, cols=3, subplot_titles=["RMSE", "MAE", "R²"])#Creates a 1x3 of panels 
    for col, metric in enumerate(["RMSE", "MAE", "R2"], 1):
        figure.add_trace(go.Bar(x=data_frame["model"], y=data_frame[metric], marker_color=colours, showlegend=False, name=metric), row=1, col=col) #adds a bar
        #for each subplot, which is a bar chart.
    figure.update_layout(template="plotly_dark", height=380,title="Model Performance Comparison") #uses dark template 
    return figure
