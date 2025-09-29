import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import sklearn
df = pd.read_csv("bitcoin.csv")

# =========================
# STEP 1: Initial Data Analysis 
# =========================

# Parse and sort date column
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df = df.set_index('Date')

# Ensure numeric
df['btc_market_price'] = pd.to_numeric(df['btc_market_price'], errors='coerce')

# Missing values
missing = df.isna().mean().sort_values(ascending=False) * 100
print("\%Percent of Missing Values by Column:\n", missing)

# Zero values
zeros = (df == 0).mean().sort_values(ascending=False) * 100
print("\n%Percent of Zero Values by Column:\n", zeros)

# Column summary
summary = pd.DataFrame({
    "dtype": df.dtypes,
    "min": df.min(),
    "max": df.max()
})
print("\nColumn summary:\n", summary)

# Convert to numpy for plotting
x = df.index.to_numpy()
y = df['btc_market_price'].to_numpy(dtype=float)
plt.figure(figsize=(12,6))
plt.plot(x, y, label="BTC Market Price", color="blue")

# Find the first non-zero date
first_nonzero_date = df.loc[df['btc_market_price'] > 0].index.min()
first_nonzero_price = df.loc[first_nonzero_date, 'btc_market_price']

# Mark the point on the plot
plt.axvline(first_nonzero_date, color="red", linestyle="--", alpha=0.7, label=f"First non-zero price: {first_nonzero_date.date()}")
plt.scatter([first_nonzero_date], [first_nonzero_price], color="red", zorder=5)
plt.title("Bitcoin Market Price Over Time")
plt.xlabel("Date")
plt.ylabel("BTC Market Price (USD)")
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# STEP 2: Build modeling dataset (clean)
# =========================

# Choose modeling start at first non-zero BTC price
model_start = first_nonzero_date

# Remove all leading zero BTC values
df_model = df.loc[df.index >= model_start].copy()

# Enforce daily frequency
df_model = df_model.asfreq("D")
df_model['btc_market_price'] = df_model['btc_market_price'].ffill()  # price can be forward-filled between missing days

print("Modeling frame shape after cut & daily asfreq:", df_model.shape)
print("Modeling frame head:\n", df_model.head(3))
print("Any remaining NaNs in btc_market_price?", df_model['btc_market_price'].isna().any())

# =========================
# STEP 3: Define target & feature engineering
# =========================

# Horizon: predict next-day price
h = 1
df_model['target_next_price'] = df_model['btc_market_price'].shift(-h)

# Helper function to engineer features
def engineer_features(df, col, lags=[1,3,7,14,30], rolls=[3,7,14,30], add_returns=True):
    """Generate lag, return, and rolling stats for one column."""
    if add_returns:
        df[f"{col}_return_1d"] = df[col].pct_change()

    for L in lags:
        df[f"{col}_lag_{L}"] = df[col].shift(L)
        if add_returns:
            df[f"{col}_return_lag_{L}"] = df[f"{col}_return_1d"].shift(L)

    for W in rolls:
        df[f"{col}_roll_mean_{W}"] = df[col].rolling(W).mean()
        df[f"{col}_roll_std_{W}"]  = df[col].rolling(W).std()

    return df

# Choose which columns to build features for
selected_features = [
    "btc_market_price",
    "btc_market_cap",
    "btc_trade_volume",
    "btc_total_bitcoins",
    "btc_n_unique_addresses"
]

# Apply feature engineering
for col in selected_features:
    df_model = engineer_features(df_model, col)

# Drop rows with missing target
df_model = df_model.dropna(subset=['target_next_price'])

print("After feature engineering:", df_model.shape)

# =========================
# STEP 4: Time-based split (hold out last ~20% for test)
# =========================

split_idx = int(len(df_model) * 0.8)
split_date = df_model.index[split_idx]
train = df_model.loc[:split_date].copy()
test  = df_model.loc[split_date + pd.Timedelta(days=1):].copy()

feature_cols = [c for c in df_model.columns 
                if any(c.startswith(col) for col in selected_features)]
target_col = 'target_next_price'

print(f"Train range: {train.index.min().date()} → {train.index.max().date()}  (n={len(train)})")
print(f"Test  range: {test.index.min().date()} → {test.index.max().date()}  (n={len(test)})")
print("Using features:", feature_cols)

# =========================
# STEP 5: Baseline (naïve: predict yesterday's price)
# =========================

# Naïve prediction equals yesterday's price (btc_market_price_lag_1)
test_baseline_pred = test['btc_market_price_lag_1']
y_true = test[target_col]

from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import numpy as np

mae = mean_absolute_error(y_true, test_baseline_pred)
rmse = root_mean_squared_error(y_true, test_baseline_pred)
print(f"Naïve baseline — MAE: {mae:.4f}, RMSE: {rmse:.4f}")

# Plot baseline vs actual for quick visual check
x_test = test.index.to_numpy()                              
y_true_np = y_true.to_numpy()                               
test_baseline_pred_np = test_baseline_pred.to_numpy()     

plt.figure(figsize=(12,6))
plt.plot(x_test, y_true_np, label="Actual next-day price")                      
plt.plot(x_test, test_baseline_pred_np, label="Naïve (lag-1) prediction", alpha=0.8) 
plt.title("Baseline Check: Actual vs Naïve (Lag-1) on Test")
plt.xlabel("Date")
plt.ylabel("BTC Market Price (USD)")
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# ADDED: STEP 6 — Ridge Regression
# =========================

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

X_train = train[feature_cols].to_numpy()
y_train = train[target_col].to_numpy()
X_test  = test[feature_cols].to_numpy()
y_test  = y_true.to_numpy()

# time-series aware CV
tscv = TimeSeriesSplit(n_splits=5)

pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", Ridge())
])

param_grid = {
    "model__alpha": [0.1, 1.0, 3.0, 10.0, 30.0, 100.0]
}

search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",
    cv=tscv,
    n_jobs=-1,
    refit=True
)

search.fit(X_train, y_train)
best_alpha = search.best_params_["model__alpha"]
print(f"Best alpha from TimeSeries CV (MAE): {best_alpha}")

# Refit model is already on full train; make test predictions
test_pred = search.predict(X_test)

test_mae_ridge = mean_absolute_error(y_test, test_pred)
test_rmse = root_mean_squared_error(y_test, test_pred)
print(f"Ridge on test — MAE: {test_mae_ridge:.4f}, RMSE: {test_rmse:.4f}")

# Compare vs baseline
improvement_mae  = mae - test_mae_ridge
improvement_rmse = rmse - test_rmse
print(f"Improvement over naïve — ΔMAE: {improvement_mae:.4f}, ΔRMSE: {improvement_rmse:.4f}")

# Plot model vs actual
x_test = test.index.to_numpy()

plt.figure(figsize=(12,6))
plt.plot(x_test, y_test, label="Actual next-day price")
plt.plot(x_test, test_pred, label="Ridge prediction", alpha=0.9)
plt.title("Test Set: Actual vs Ridge Prediction")
plt.xlabel("Date")
plt.ylabel("BTC Market Price (USD)")
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# STEP 7: Lasso Regression
# =========================
from sklearn.linear_model import Lasso

# Lasso pipeline
pipe_lasso = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", Lasso(random_state=42))
])

# define hyperperam grid for Lasso's alpha
param_grid_lasso = {
    "model__alpha": [0.1, 1.0, 10.0, 50.0, 100.0]
}

# GridSearchCV
search_lasso = GridSearchCV(
    estimator=pipe_lasso,
    param_grid=param_grid_lasso,
    scoring="neg_mean_absolute_error",
    cv=tscv,
    n_jobs=-1
)

print("Running GridSearchCV for Lasso...")
search_lasso.fit(X_train, y_train)
print("Best Lasso params:", search_lasso.best_params_)

# preds and eval
test_pred_lasso = search_lasso.predict(X_test)
test_mae_lasso = mean_absolute_error(y_test, test_pred_lasso)
test_rmse_lasso = root_mean_squared_error(y_test, test_pred_lasso)
print(f"Lasso on test — MAE: {test_mae_lasso:.4f}, RMSE: {test_rmse_lasso:.4f}")

improvement_mae_lasso  = mae - test_mae_lasso
improvement_rmse_lasso = rmse - test_rmse_lasso
print(f"Improvement over naïve — ΔMAE: {improvement_mae_lasso:.4f}, ΔRMSE: {improvement_rmse_lasso:.4f}")

# actual vs. pred for Lasso
plt.figure(figsize=(12, 6))
plt.plot(x_test, y_test, label="Actual next-day price")
plt.plot(x_test, test_pred_lasso, label="Lasso prediction", alpha=0.9)
plt.title("Test Set: Actual vs. Lasso Prediction")
plt.xlabel("Date")
plt.ylabel("BTC Market Price (USD)")
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# STEP 8: Random Forest
# =========================
from sklearn.ensemble import RandomForestRegressor

# define random forest pipeline
pipe_rf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
])

#hyperparam grid for Random Forest
#tried preventing model from getting lazy but didnt work
param_grid_rf = {
    'model__n_estimators': [200],
    'model__max_depth': [15]
}

#gridsearch
search_rf = GridSearchCV(
    estimator=pipe_rf,
    param_grid=param_grid_rf,
    scoring="neg_mean_absolute_error",
    cv=tscv,
    n_jobs=-1 # Also use all CPU cores for the search itself
)

print("\nRunning GridSearchCV for Random Forest...")
search_rf.fit(X_train, y_train)
print("Best Random Forest params:", search_rf.best_params_)

# predict + evaluate
test_pred_rf = search_rf.predict(X_test)
test_mae_rf = mean_absolute_error(y_test, test_pred_rf)
test_rmse_rf = root_mean_squared_error(y_test, test_pred_rf)
print(f"Random Forest on test — MAE: {test_mae_rf:.4f}, RMSE: {test_rmse_rf:.4f}")
improvement_mae_rf  = mae - test_mae_rf
improvement_rmse_rf = rmse - test_rmse_rf
print(f"Improvement over naïve — ΔMAE: {improvement_mae_rf:.4f}, ΔRMSE: {improvement_rmse_rf:.4f}")

#plot actual vs predicted for Random Forest
plt.figure(figsize=(12, 6))
plt.plot(x_test, y_test, label="Actual next-day price")
plt.plot(x_test, test_pred_rf, label="Random Forest prediction", alpha=0.9)
plt.title("Test Set: Actual vs. Random Forest Prediction")
plt.xlabel("Date")
plt.ylabel("BTC Market Price (USD)")
plt.legend()
plt.tight_layout()
plt.show()


# =========================
# STEP 9: Final model comparison
# =========================

print("\n--- Final Model MAE Comparison ---")
print(f"{'Model':<20} | {'MAE':<10}")
print("-" * 33)
print(f"{'Naïve baseline':<20} | {mae:.4f}")
print(f"{'Ridge regression':<20} | {test_mae:.4f}")
print(f"{'Lasso regression':<20} | {test_mae_lasso:.4f}")
print(f"{'Random Forest':<20} | {test_mae_rf:.4f}")