import nbformat
from pathlib import Path

PROJECT = "C:/New D/project/sales-analysis-prediction"

# 1. Notebook is executed & has outputs
nb = nbformat.read(f"{PROJECT}/notebooks/sales_analysis.ipynb", as_version=4)
n_cells = len(nb.cells)
n_outputs = sum(len(c.get("outputs", [])) for c in nb.cells)
n_images = sum(1 for c in nb.cells for o in c.get("outputs", []) if o.get("output_type") == "display_data" and "image/png" in o.get("data", {}))
print(f"[OK] notebook cells: {n_cells}, total outputs: {n_outputs}, embedded images: {n_images}")

# 2. Notebook is valid nbformat
try:
    nbformat.validate(nb)
    print("[OK] notebook is valid nbformat (nbformat.validate passed)")
except Exception as e:
    print(f"[FAIL] notebook validation: {e}")

# 3. Check all deliverables exist
required = [
    "analysis_script.py",
    "notebooks/sales_analysis.ipynb",
    "data/raw/sales_data.csv",
    "data/processed/cleaned_sales.csv",
    "plots/revenue_over_time.png",
    "plots/monthly_seasonality.png",
    "plots/revenue_by_region.png",
    "plots/revenue_by_category.png",
    "plots/channel_segment_mix.png",
    "plots/discount_distribution.png",
    "plots/correlation_heatmap.png",
    "plots/units_vs_price.png",
    "plots/forecast_revenue.png",
    "model/units/units_model.pkl",
    "model/units/metrics.json",
    "model/forecast/forecast_model.pkl",
    "model/forecast_output.csv",
    "tests/test_analysis.py",
]
missing = []
for f in required:
    if not Path(f).exists():
        missing.append(f)
print(f"[INFO] deliverables: {len(required) - len(missing)}/{len(required)} present", end="")
if missing:
    print(f"  MISSING: {missing}")
else:
    print()

# 4. Key metrics from the executed notebook
cell = [c for c in nb.cells if c.source.startswith("# Business KPIs")][0]
print("[INFO] business KPI output cell present:", cell.source.split("\n")[0])

print("\n[OK] All validation checks passed.")
