import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_and_clean, make_supervised_series, recursive_forecast



# -------------------------------
# Title & File Upload
# -------------------------------
st.title("🌱 Sustainable Agriculture - Crop Yield Prediction")

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    df = load_and_clean(uploaded_file)
    st.write("### Preview of Data", df.head())

    # -------------------------------
    # Basic Dataset Information
    # -------------------------------
    if "Year" not in df.columns or "Yield" not in df.columns:
        st.error("❌ Dataset must have 'Year' and 'Yield' columns")
    else:
        st.success("✅ Dataset looks good!")

        # Plot yield over years
        st.write("### Yield Trend")
        fig, ax = plt.subplots()
        ax.plot(df["Year"], df["Yield"], marker="o")
        ax.set_xlabel("Year")
        ax.set_ylabel("Yield")
        ax.set_title("Crop Yield over Years")
        st.pyplot(fig)

        # -------------------------------
        # Train-Test Split
        # -------------------------------
        X = df[["Year"]]
        y = df["Yield"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        st.write(f"📉 Model Mean Squared Error: **{mse:.2f}**")

        # -------------------------------
        # Forecast Next Year
        # -------------------------------
        last_year = df["Year"].max()
        next_year = last_year + 1
        next_pred = model.predict([[next_year]])[0]

        st.success(f"🌾 Predicted Yield for {next_year}: **{next_pred:.2f}**")

        # -------------------------------
        # Forecast multiple years
        # -------------------------------
        st.write("### Forecast Future Yields")
        n_years = st.slider("Select number of future years to forecast", 1, 10, 3)

        future_years = list(range(last_year + 1, last_year + n_years + 1))
        future_preds = model.predict(pd.DataFrame(future_years, columns=["Year"]))

        forecast_df = pd.DataFrame({"Year": future_years, "Predicted Yield": future_preds})
        st.write(forecast_df)

        # Plot forecast
        fig2, ax2 = plt.subplots()
        ax2.plot(df["Year"], df["Yield"], marker="o", label="Actual")
        ax2.plot(forecast_df["Year"], forecast_df["Predicted Yield"],
                 marker="x", linestyle="--", color="red", label="Forecast")
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Yield")
        ax2.legend()
        st.pyplot(fig2)
