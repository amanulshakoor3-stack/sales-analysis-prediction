"""Model training, evaluation and monthly revenue forecasting.

Produces:
- model/: trained RandomForest pickle + Ridge forecast model
- model/metrics.json          (R2/MAE/RMSE for the units model)
- model/predictions.csv       (held-out test predictions)
- model/forecast_output.csv   (3-month revenue forecast)
- plots/forecast_revenue.png  (actual vs forecast for the hold-out window)
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Project root (one level above src/) + the top-level plots directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOTS = PROJECT_ROOT / "plots"

sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
# Units-sold prediction (per line item)
# ---------------------------------------------------------------------------
def train_units_model(X: pd.DataFrame, y: pd.Series, out_dir: Path) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "r2": round(float(r2_score(y_test, pred)), 4),
        "mae": round(float(mean_absolute_error(y_test, pred)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, pred))), 2),
        "n_test": int(len(y_test)),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "units_model.pkl", "wb") as fh:
        pickle.dump(model, fh)
    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    pd.DataFrame({"y_true": y_test.values, "y_pred": pred}).to_csv(
        out_dir / "predictions.csv", index=False
    )

    # Feature importance (top 12)
    importance = (
        pd.Series(model.feature_importances_, index=X.columns)
        .sort_values()
        .tail(12)
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    importance.plot(kind="barh", ax=ax, color="#1f77b4")
    ax.set_title("Top Feature Importances (RandomForest: units_sold)")
    ax.set_xlabel("Importance")
    _save(fig, PLOTS / "feature_importance.png")
    plt.close(fig)

    return metrics


# ---------------------------------------------------------------------------
# Monthly revenue forecasting
# ---------------------------------------------------------------------------
def forecast_monthly_revenue(df: pd.DataFrame, out_dir: Path) -> dict:
    """Train a Ridge on month-of-year + linear trend, validate the last 3
    months on the held-out window, then forecast the 3 months after the data."""
    d = df.copy()
    out_dir.mkdir(parents=True, exist_ok=True)
    d["ym"] = d["date"].dt.to_period("M")
    monthly = (
        d.groupby("ym")["line_total"].sum().sort_index().reset_index()
    )
    monthly["t"] = np.arange(len(monthly))
    monthly["m"] = monthly["ym"].dt.month
    monthly = monthly.rename(columns={"line_total": "revenue"})

    y = monthly["revenue"].values

    # month-of-year dummy matrix, kept as a DataFrame so column labels are easy
    # to align the forecast dummies against later.
    month_dummies = pd.get_dummies(monthly["m"], prefix="m", drop_first=True)

    # feature matrix: linear trend + month-of-year seasonality
    X = np.column_stack([monthly["t"].values, month_dummies.values])
    n = len(monthly)
    train = slice(0, max(3, n - 3))  # first 21 months fit, last 3 for validation

    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X[train], y[train])
    with open(out_dir / "forecast_model.pkl", "wb") as fh:
        pickle.dump(model, fh)
    y_pred = model.predict(X)

    # Positions 0..train.stop-1 are used for fitting; the rest is the
    # hold-out validation window and the target of the following 3-month
    # revenue forecast.
    test_months = monthly["ym"].iloc[train.stop:].astype(str).tolist()
    metrics = {
        "method": "Ridge(revenue) on month-of-year + linear trend",
        "train_months": int(train.stop),
        "test_months": int(n - train.stop),
        "mae": round(float(mean_absolute_error(y[train], y_pred[train])), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y[train], y_pred[train]))), 2),
    }

    # Forecast the window immediately after the data (held out from the fit).
    last_ym = monthly["ym"].iloc[-1]
    future = pd.period_range(start=last_ym + 1, periods=3, freq="M")
    future_t = np.arange(n, n + 3)
    future_months = np.array([p.month for p in future])
    future_dummies = pd.get_dummies(future_months, prefix="m", drop_first=True)
    # Align forecast dummies to the training month labels (missing months -> 0)
    future_dummies = future_dummies.reindex(columns=month_dummies.columns, fill_value=0)
    future_X = np.column_stack([future_t, future_dummies.values])
    future_pred = model.predict(future_X)

    forecast_df = pd.DataFrame({
        "year_month": [str(p) for p in future],
        "forecast_revenue": np.round(future_pred, 2),
    })

    # Actual-vs-forecast plot for the hold-out test window (last 3 months)
    test_actual = y[train.stop:]
    test_pred = y_pred[train.stop:]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(test_months))
    ax.plot(x, test_actual, marker="o", label="Actual")
    ax.plot(x, test_pred, marker="s", linestyle="--", label="Forecast")
    ax.set_xticks(x)
    ax.set_xticklabels(test_months, rotation=30)
    ax.set_title("Monthly Revenue: Actual vs Forecast (hold-out window)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue ($)")
    ax.legend()
    _save(fig, PLOTS / "forecast_revenue.png")
    plt.close(fig)

    return {"metrics": metrics, "forecast": forecast_df}


def _save(fig: plt.Figure, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path
