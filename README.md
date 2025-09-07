# 🌾 Sustainable Agriculture — Crop Yield Prediction & Forecasting

End-to-end ML project that predicts next-year crop yields for India using historical FAOSTAT-like data.

## 📁 Project Structure
```
sustainable_agriculture_yield_forecast/
├─ data/
│  └─ Sustainable_Agriculture_dataset.csv        # ← Put your CSV here
├─ src/
│  ├─ train_forecast.py                          # Train & forecast for all crops
│  └─ utils.py                                   # Data loading & helpers
├─ models/                                       # Saved models per crop (*.joblib)
├─ outputs/
│  ├─ metrics.csv                                # Model performance per crop
│  ├─ forecasts.csv                              # Forecasted values per crop
│  └─ plots/                                     # PNG plots (actual vs forecast)
├─ app/
│  └─ streamlit_app.py                           # Interactive dashboard
├─ requirements.txt
└─ README.md
```

## ✅ What You Can Do
- Clean the dataset and extract **(Year, Item, Value)** for yield in India.
- Train a **time-aware regression model** per crop using lag features.
- Produce **next-k-year forecasts** for each crop.
- Visualize actual vs predicted yields.
- Run an interactive **Streamlit dashboard**.

---

## 🚀 Quick Start (Step-by-Step)

> **Prerequisite:** Python 3.10+ recommended.

1) **Clone / unzip this folder** and move your CSV into `data/` with name:
```
data/Sustainable_Agriculture_dataset.csv
```

2) **Create a virtual environment and install dependencies:**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

3) **Run training & forecasting** (forecasts next 3 years by default):
```bash
python src/train_forecast.py --csv data/Sustainable_Agriculture_dataset.csv --horizon 3
```
Outputs will be saved in `models/` and `outputs/`.

4) **Launch the dashboard:**
```bash
streamlit run app/streamlit_app.py
```
Then open the local URL shown in the terminal.

---

## 🧠 Method Overview

- **Preprocessing:** Filter to `Area == "India"`, `Element == "Yield"` (case-insensitive), keep `Unit` = kg/ha when present. Convert `Year` and `Value` to numeric.
- **Features:** For each crop (Item), sort by year and create:
  - Lags: `lag1`, `lag2`, `lag3`
  - Rolling mean (3): `roll3`
  - Year index: `year`
- **Models:** 
  - Baseline: `LinearRegression`
  - Main: `RandomForestRegressor`
  - TimeSeriesSplit for evaluation. Best model is chosen based on RMSE.
- **Forecasting:** Recursive one-step-ahead for `horizon` years using the trained model and recent lags.

---

## 📈 Files Produced
- `outputs/metrics.csv` — MAE/RMSE for each crop and chosen model.
- `outputs/forecasts.csv` — Historical last year + future `horizon` years for each crop.
- `outputs/plots/<Item>.png` — Actual vs forecast line plot.

---

## 🧪 Tips
- You can change `--min-years` to require a minimum history length per crop (default 6).
- You can change the lags/rolling window in `src/train_forecast.py` if you want to experiment.
- Extend the project by merging rainfall/temperature per year to improve features.

---

## ✍️ Citation / Source
This template expects a FAOSTAT-like dataset with columns:
`["Area","Element","Item","Year","Unit","Value","Flag","Flag Description"]` (extra columns are ignored).

Good luck, and happy modeling! 🌿
