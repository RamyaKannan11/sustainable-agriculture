# train_forecast.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# ---------------------------
# Load and prepare the dataset
# ---------------------------
def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    # Keep only useful columns
    df = df[["Item", "Year", "Value"]]
    df = df.dropna()
    return df

# ---------------------------
# Train model for one crop
# ---------------------------
def train_model_for_crop(df, crop_name, model_dir="models"):
    crop_df = df[df["Item"] == crop_name]

    if crop_df.empty:
        print(f"No data found for {crop_name}")
        return None

    X = crop_df[["Year"]].values
    y = crop_df["Value"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Crop: {crop_name} | MSE: {mse:.2f} | R2: {r2:.2f}")

    # Save model
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, f"{crop_name}_model.pkl"))

    # Plot results
    plt.figure(figsize=(8, 5))
    plt.scatter(X_test, y_test, color="blue", label="Actual")
    plt.plot(X_test, y_pred, color="red", linewidth=2, label="Predicted")
    plt.title(f"{crop_name} Yield Prediction")
    plt.xlabel("Year")
    plt.ylabel("Yield (kg/ha)")
    plt.legend()
    plt.show()

    return model

# ---------------------------
# Train for multiple crops
# ---------------------------
def train_all_crops(csv_path, crops):
    df = load_dataset(csv_path)
    for crop in crops:
        train_model_for_crop(df, crop)

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    data_path = "data/Sustainable_Agriculture_dataset.csv"

    crops_to_train = ["Rice, paddy", "Wheat", "Maize", "Bananas", "Apples"]

    train_all_crops(data_path, crops_to_train)
