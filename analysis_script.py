"""End-to-end sales analytics & prediction pipeline.

Run:
    python analysis_script.py

Artifacts under `sales-analysis-prediction/`:
    data/raw/sales_data.csv
    data/processed/cleaned_sales.csv
    plots/*.png
    model/units_model.pkl          (RandomForest predicting units_sold)
    model/forecast_model.pkl       (Ridge monthly-revenue model)
    model/metrics.json
    model/predictions.csv
    model/forecast_output.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# Make `src` importable when running this file from the project root.
import sys
from pathlib import Path as _P

sys.path.insert(0, str(_P(__file__).resolve().parent / "src"))

from data import clean_data, generate_sales_data, save_data  # noqa: E402
from features import build_model_matrix  # noqa: E402
from src import visualize  # noqa: E402
from train import forecast_monthly_revenue, train_units_model  # noqa: E402

PROJECT = _P(__file__).resolve().parent
RAW = PROJECT / "data" / "raw"
PROCESSED = PROJECT / "data" / "processed"
PLOTS = PROJECT / "plots"
MODEL = PROJECT / "model"


def main() -> None:
    # 1. Data -------------------------------------------------
    data = generate_sales_data(n=6000, seed=42)
    save_data(data, RAW / "sales_data.csv", PROCESSED / "cleaned_sales.csv")
    clean = clean_data(data)
    print(f"[1] data -> {RAW/'sales_data.csv'} ({len(data)} rows)\n"
          f"     cleaned -> {PROCESSED/'cleaned_sales.csv'} ({len(clean)} rows)")

    # 2. Visualizations ---------------------------------------
    viz = visualize
    tasks = [
        ("plot_revenue_over_time", "revenue_over_time.png"),
        ("plot_monthly_seasonality", "monthly_seasonality.png"),
        ("plot_revenue_by_region", "revenue_by_region.png"),
        ("plot_revenue_by_category", "revenue_by_category.png"),
        ("plot_channel_segment_mix", "channel_segment_mix.png"),
        ("plot_discount_distribution", "discount_distribution.png"),
        ("plot_correlation_heatmap", "correlation_heatmap.png"),
        ("plot_scatter_units_vs_price", "units_vs_price.png"),
    ]
    for fn, name in tasks:
        getattr(viz, fn)(clean, PLOTS / name)
        print(f"[2] plot -> {PLOTS/name}")

    # 3. Prediction (units_sold) ------------------------------
    X, y = build_model_matrix(clean, target="units_sold")
    units_metrics = train_units_model(X, y, MODEL / "units")
    print(f"[3] units model -> R2={units_metrics['r2']}  MAE={units_metrics['mae']}  RMSE={units_metrics['rmse']}")

    # 4. Monthly revenue forecast ------------------------------
    fc = forecast_monthly_revenue(clean, MODEL / "forecast")
    fc["forecast"].to_csv(MODEL / "forecast_output.csv", index=False)
    print(f"[4] revenue forecast -> {json.dumps(fc['metrics'])}")
    print(f"     forecast.csv -> {MODEL/'forecast_output.csv'}")

    # 5. Headline numbers --------------------------------------
    summary = pd.DataFrame({
        "metric": ["Total revenue", "Total profit", "Avg transaction",
                   "Unique regions", "Unique products"],
        "value": [
            f"${clean['line_total'].sum():,.0f}",
            f"${clean['gross_profit'].sum():,.0f}",
            f"${clean['line_total'].mean():,.0f}",
            clean["region"].nunique(),
            clean["category"].nunique(),
        ],
    })
    summary.to_csv(PROJECT / "model" / "summary.csv", index=False)
    print(f"[5] summary -> {PROJECT/'model'/'summary.csv'}")

    print("\nPipeline complete. See plots/ and model/ for artifacts.")


if __name__ == "__main__":
    main()
