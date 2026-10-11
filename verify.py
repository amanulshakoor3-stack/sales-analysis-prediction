"""Independent verification of the project results (not relying on analysis_script output)."""
import json

import pandas as pd

df = pd.read_csv("data/processed/cleaned_sales.csv")

print("ROWS:", len(df))
print("NULLS:\n", df.isna().sum().to_string())
print("DUPLICATES:", int(df.duplicated().sum()))
print("COLUMNS:", list(df.columns))

print("\nREVENUE by region:\n", df.groupby("region")["line_total"].sum().round(0).to_string())
print("\nCATEGORY revenue:\n", df.groupby("category")["line_total"].sum().round(0).to_string())
print("\nTOP5 sub-category:\n", df.groupby("sub_category")["line_total"].sum().sort_values(ascending=False).head(5).round(0).to_string())
print("\nTOTAL REVENUE (line_total):", round(df["line_total"].sum(), 2))
print("TOTAL PROFIT (gross_profit):", round(df["gross_profit"].sum(), 2))
print("AVG TRANSACTION:", round(df["line_total"].mean(), 2))
print("UNIT PROFIT (profit/units):", round(df["gross_profit"].sum() / df["units_sold"].sum(), 4))

# Units model held-out metrics stored in model/units/metrics.json
m = json.loads(open("model/units/metrics.json").read())
print("\nUNITS MODEL metrics:", m)

# Forecast output
print("\nFORECAST OUTPUT:\n", open("model/forecast_output.csv").read())

# Sanity: model predictions plausible (target scale ~1..80 units)
pred = pd.read_csv("model/units/predictions.csv")
print("\nPREDICTIONS y_true range:", pred["y_true"].min(), "-", pred["y_true"].max(),
      "| y_pred range:", pred["y_pred"].min(), "-", pred["y_pred"].max())
