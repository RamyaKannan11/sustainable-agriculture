# src/utils.py
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# ---------------------------
# Load and clean dataset
# ---------------------------
def load_and_clean(csv_path: str):
    df = pd.read_csv(csv_path)

    # Standardize column names
    df = df.rename(columns={
        "Value": "Yield",   # rename Value -> Yield
        "Item": "Crop"      # rename Item -> Crop
    })

    # Ensure numeric types
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Yield"] = pd.to_numeric(df["Yield"], errors="coerce")

    # Drop rows with missing values
    df = df.dropna(subset=["Year", "Yield"])

    return df

# ---------------------------
# Convert to supervised format (lag features)
# ---------------------------
def make_supervised_series(df, crop_name, n_lags=3):
    crop_df = df[df["Crop"] == crop_name].sort_values("Year")
    series = crop_df["Yield"].values

    X, y = [], []
    for i in range(len(series) - n_lags):
        X.append(series[i:i+n_lags])
        y.append(series[i+n_lags])

    return np.array(X), np.array(y), crop_df

# ---------------------------
# Forecast recursively
# ---------------------------
def recursive_forecast(model, history, steps=5):
    """
    model   : trained regression model
    history : last observed values (array-like)
    steps   : number of years to predict
    """
    history = list(history)
    preds = []

    for _ in range(steps):
        x_input = np.array(history[-len(history):]).reshape(1, -1)
        yhat = model.predict(x_input)[0]
        preds.append(yhat)
        history.append(yhat)

    return preds

# ---------------------------
# Train a simple regression model
# ---------------------------
def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)

    # Evaluate
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    return model, mse, r2
