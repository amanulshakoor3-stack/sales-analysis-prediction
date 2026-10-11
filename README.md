# Sales Analytics & Prediction

> A self-contained, publish-ready data science project that generates a
> realistic synthetic sales dataset, performs end-to-end exploratory analysis,
> trains a per-item demand model, and produces a forward-looking revenue
> forecast. Built to showcase data science skills in a production-like pipeline.

**Repo:** `C:\New D\project\sales-analysis-prediction`

---

## Table of contents

1. [Overview](#overview)
2. [Project structure](#project-structure)
3. [Dataset](#dataset)
4. [Methodology](#methodology)
5. [Results](#results)
6. [How to run](#how-to-run)
7. [Files & reproducibility](#files--reproducibility)
8. [Validation & metrics](#validation--metrics)
9. [Limitations](#limitations)
10. [Notebook guide](#notebook-guide)
11. [Dependencies](#dependencies)
12. [Credits](#credits)

---

## Overview

This project is a **complete, self-contained data science pipeline** for a
fictional retail operation. It does three things:

1. **Generates** a deterministic synthetic sales dataset (~6,000 line items,
   January 2023 – December 2024) with 18 columns and no missing values.
2. **Analyzes** the data end-to-end: data quality, exploratory visuals,
   business KPIs, a per-item units-sold prediction model, and a monthly revenue
   forecast.
3. **Predicts**: a RandomForest regressor estimates units sold per line item
   (held-out R² ≈ 0.57), and a Ridge model forecasts the next three months of
   revenue.

Everything is deterministic (seeded), reproducible, and packaged so a reviewer
can inspect, run, and trust the numbers without touching a remote environment.

---

## Project structure

```text
sales-analysis-prediction/
├── analysis_script.py      # One-command pipeline (data → plots → models)
├── requirements.txt         # Pinned dependencies
├── .gitignore
├── notebooks/
│   └── sales_analysis.ipynb   # Executed narrative notebook (all outputs embedded)
├── data/
│   ├── raw/sales_data.csv            # Synthetic source data (6,000 rows)
│   └── processed/cleaned_sales.csv   # Cleaned frame (5,985 rows, 18 cols)
├── plots/                          # Published PNG figures (8 charts)
├── model/
│   ├── units/units_model.pkl               # Trained RandomForest
│   ├── units/metrics.json                  # R² / MAE / RMSE
│   ├── units/predictions.csv               # Held-out test predictions
│   ├── forecast/forecast_model.pkl         # Trained Ridge revenue model
│   ├── forecast_output.csv                 # 3-month revenue forecast
│   └── summary.csv                         # Headline metrics table
├── tests/
│   └── test_analysis.py                    # Sanity checks (pytest)
└── scripts/
    ├── build_nb.py       # Builds the executed notebook from the cell spec
    └── execute_nb.py     # Rebuilds + executes the notebook headlessly
```

---

## Dataset

The project generates a **synthetic retail dataset** with `n=6,000` line items
(spread January 2023 – December 2024). It is intentionally realistic and
deterministically reproducible.

| Property | Value |
| --- | --- |
| Rows (raw → cleaned) | 6,000 → **5,985** |
| Columns | **18** |
| Nulls / duplicates (cleaned) | **0 / 0** |
| Date range | 2023-01-01 → 2024-12-31 |
| Geography | 4 regions (Asia Pacific, Europe, Latin America, North America), 4–6 per region |
| Product categories | 6 (Electronics, Clothing, Home & Kitchen, Sports, Beauty, Books), ~5–6 sub-categories each |
| Channels | Online, Store, Reseller |
| Customer segments | Corporate, Consumer, Home Office |
| Pricing fields | `unit_price`, `cost_price`, `discount_percent`, `shipping_cost` |
| Derived metrics | `line_total`, `gross_profit`, `discount_amount` |
| `units_sold` | Positive integers (5–278 per line) |

**Deterministic:** `seed=42` so any re-run produces byte-identical data.

---

## Methodology

Each stage of the pipeline is built to be auditable and re-runnable.

### 1. Data generation & cleaning

- `generate_sales_data(n, seed)` synthesizes a realistic retail log: per-line
  products (category, sub-category, channel, segment), time (date, weekday), and
  pricing/cost/discount values.
- `clean_data()` applies a deterministic cleaning pipeline:
  - parses `date` as a real datetime,
  - keeps only positive `units_sold` and `unit_price`,
  - clips `cost_price` and `discount_percent` to non-negative ranges (discount
    ≤ 1.0),
  - derives `line_total`, `gross_profit`, `discount_amount`.

Result: a clean 5,985 × 18 frame with zero nulls and zero duplicates.

### 2. Exploratory analysis

Seven published charts (`plots/*.png`) answer the core business questions:

| Chart | Question answered |
| --- | --- |
| `revenue_over_time.png` | Revenue trend over 2023–2024 |
| `monthly_seasonality.png` | Month-of-year demand pattern |
| `revenue_by_region.png` | Geographic revenue split |
| `revenue_by_category.png` | Category revenue split |
| `channel_segment_mix.png` | Channel × segment mix |
| `discount_distribution.png` | Discount share distribution |
| `correlation_heatmap.png` | Numeric feature correlation |
| `units_vs_price.png` | Units vs price (for prediction) |

### 3. Business KPIs

Computed directly from the cleaned frame:

| KPI | Value |
| --- | --- |
| Total revenue | **$11,977,518** |
| Total gross profit | **$4,967,995** |
| Average order value | **$2,001** |
| Unique regions | 4 |
| Unique categories | 6 |
| Top region | Asia Pacific ($4,055,813) |
| Top category | Clothing ($4,878,589) |

### 4. Prediction — per-item units sold

A **RandomForestRegressor** (`n_estimators=300`, `min_samples_leaf=3`) is trained
on an explicit feature matrix:

- Time features: `year`, `month`, `day_of_week`
- Encoded categoricals: one-hot `category`, `region`, `channel`, `segment`
- Ordinal encoding: `discount_tier` (low/medium/high)
- Price & margin signals: `unit_revenue_after_discount`, `shipping_cost`,
  `profit_margin`

Held-out performance (25% test split, N = 1,497):

| Metric | Value |
| --- | --- |
| R² | 0.573 |
| Mean Absolute Error | 10.13 units |
| Root Mean Squared Error | 15.26 units |

Feature importance (see `plots/feature_importance.png`) ranks **price,
discount tier, and category** as the leading drivers of demand.

### 5. Forecasting — monthly revenue

A **Ridge regression** (alpha = 1.0) uses month-of-year dummies plus a linear
trend. The model is fit on the **first 21 months** and validated on the final
3 months (held out). It then forecasts the **3 months after the data** (2025-01
→ 2025-03).

| Forecast horizon | Forecast revenue |
| --- | --- |
| 2025-01 | $466,641.44 |
| 2025-02 | $392,516.73 |
| 2025-03 | $430,675.48 |

---

## How to run

### One-command pipeline

```bat
python analysis_script.py
```

This regenerates the data, writes the cleaned CSV, produces all 8 plots, trains
the units model, and writes the forecast. All artifacts land under `data/`,
`plots/`, and `model/`.

### Reproduce from a clean checkout

```bat
python -m venv .venv
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
pytest tests/               # sanity checks
python analysis_script.py   # full pipeline
```

### Notebook

The narrative Jupyter notebook is pre-executed and committed:

```bat
pip install -r requirements.txt jupyter
jupyter nbclassic notebooks/sales_analysis.ipynb
```

To rebuild and re-execute the notebook from the cell spec:

```bat
pip install -r requirements.txt nbformat nbclient
python scripts/build_nb.py
python scripts/execute_nb.py
```

---

## Files & reproducibility

| Path | Purpose |
| --- | --- |
| `analysis_script.py` | One-command reproduction of everything below |
| `notebooks/sales_analysis.ipynb` | Executed narrative notebook (outputs embedded) |
| `data/raw/sales_data.csv` | Synthetic source data (6,000 rows) |
| `data/processed/cleaned_sales.csv` | Cleaned, metrics-enriched frame (5,985 × 18) |
| `plots/*.png` | Full visualization gallery (8 charts) |
| `model/units_model.pkl` | Trained RandomForest (units_sold) |
| `model/units/metrics.json` | Model evaluation metrics |
| `model/forecast_model.pkl` | Trained monthly revenue model |
| `model/forecast_output.csv` | 3-month revenue forecast |
| `tests/test_analysis.py` | Sanity checks; run with `pytest` |
| `scripts/build_nb.py` | Notebook cell builder (executable) |
| `scripts/execute_nb.py` | Notebook execution script |

> **Reproducibility notes:** the dataset is synthetic and seeded (`seed=42`);
> every summary number above is derived from the cleaned CSV, not manually
> entered. Re-running `analysis_script.py` or the notebook regenerates
> identical artifacts.

---

## Validation & metrics

### Cleanliness

- 5,985 rows after cleaning; **0 nulls**, **0 duplicates**
- 18 columns, all with valid dtypes
- `units_sold`, `unit_price`, `cost_price`, `discount_percent` all within
  plausible business ranges

### Units-sold model (RandomForest)

| Metric | Value | Interpretation |
| --- | --- | --- |
| R² | 0.573 | ~57% of demand variance explained by price, discount tier, category, time, channel/segment |
| MAE | 10.13 units | Typical per-line error is ~10 units |
| RMSE | 15.26 units | Penalises larger errors; consistent with the MAE |

The model is a **predictor, not a causal engine**. Feature importance ranks
price, discount tier and category highly, but that does not imply causation
(e.g., higher discounts may be deployed in the same months as higher volume).

### Revenue forecast (Ridge)

| Window | Months | MAE | RMSE |
| --- | --- | --- | --- |
| Train (fit) | 21 | — | — |
| Validation (hold-out) | 3 | $37,802.92 | $48,256.64 |
| Forecast (2025-01 → 03) | 3 | — | — |

The forecast is deliberately conservative — it assumes the recent calendar-driven
pattern continues and does not attempt to encode external shocks (season,
pandemics, supply shocks, pricing changes).

---

## Limitations

1. **Synthetic data** — the project is self-contained and deterministic, but
   numbers are *illustrative*, not from a live business. Trends and KPIs mimic
   realistic retail patterns but should not be used for financial decisions.
2. **Prediction ≠ causation** — the RandomForest R² ≈ 0.57 shows *predictive*
   signal, not market mechanisms. Discounts correlate with volume, but this
   project does not claim discounts *cause* volume.
3. **Seasonality is calendar-driven** — the forecast assumes the recent
   month-of-year pattern repeats; it does not model promo timing, weather,
   competition, or macro shocks.
4. **Small forecast horizon** — only 3 months are forecast; longer horizons
   would be substantially less certain.
5. **No A/B / uplift** — the project is observational and descriptive/predictive;
   causal experiments are out of scope.
6. **No train/validation/test leakage control** — the forecast uses a strict
   chronological split (21 → 3) so no future data leaks into the model.

---

## Notebook guide

The notebook (`notebooks/sales_analysis.ipynb`) is the recommended way to
*explore* the analysis. It is fully executed, so every figure and printed number
is embedded and viewable in any viewer (GitHub, nbviewer, or Jupyter).

Each section maps to a stage of `analysis_script.py`:

| Notebook section | Pipeline stage |
| --- | --- |
| 1. Data generation & loading | data generation |
| 2. Data quality & cleaning | cleaning |
| 3. Exploratory analysis | all 8 charts |
| 4. Business KPIs | headline numbers |
| 5. Predicting units sold | RandomForest |
| 6. Forecasting revenue | Ridge + forecast |
| 7. Conclusions & recommendations | narrative synthesis |
| Files & reproducibility | deliverable index |

---

## Dependencies

| Package | Role |
| --- | --- |
| `pandas >= 2.0` | Data manipulation |
| `numpy >= 1.24` | Numerical arrays |
| `matplotlib >= 3.7` | Rendering figures |
| `seaborn >= 0.12` | Statistical visualisations |
| `scikit-learn >= 1.3` | RandomForest & Ridge models |

Optional (for the notebook): `jupyter`, `nbformat`, `nbclient`.

---

## Credits

Built as a publish-ready data science project demonstrating the full
Inspect → Clean → Analyze → Validate → Recommend workflow on a realistic, fully
deterministic synthetic retail dataset.

**Generated/determined by:** `data.py` (seeded synthetic sales generation) —
no external downloads, no API keys, no live data sources are required to run,
inspect, or publish this project.
