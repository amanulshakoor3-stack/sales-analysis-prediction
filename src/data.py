"""Data generation and loading for the sales analytics project.

Generates a deterministic, realistic synthetic sales dataset and exposes
helpers to load / clean / save it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Reference tables
# ---------------------------------------------------------------------------
REGIONS = ["North America", "Europe", "Asia Pacific", "Latin America"]
COUNTRIES = {
    "North America": ["United States", "Canada"],
    "Europe": ["United Kingdom", "Germany", "France", "Netherlands"],
    "Asia Pacific": ["India", "Japan", "South Korea", "Australia"],
    "Latin America": ["Brazil", "Mexico"],
}
CATEGORIES = {
    "Electronics": ["Smartphones", "Laptops", "Headphones", "Cameras", "Tablets"],
    "Clothing": ["Men's Topwear", "Women's Topwear", "Jeans", "Dresses", "Sportswear"],
    "Home & Kitchen": ["Furniture", "Appliances", "Cookware", "Bedding", "Decor"],
    "Sports": ["Fitness", "Outdoor", "Team Sports", "Water Sports", "Camping"],
    "Beauty": ["Skincare", "Makeup", "Fragrance", "Hair Care", "Tools"],
    "Books": ["Fiction", "Non-Fiction", "Technology", "Children", "Art"],
}
CHANNELS = ["Online", "Store", "Reseller"]
SEGMENTS = ["Corporate", "Consumer", "Home Office"]

# Base unit price and base unit demand, per category
CATEGORY_PRICE_BASE = {
    "Electronics": 180.0,
    "Clothing": 45.0,
    "Home & Kitchen": 90.0,
    "Sports": 60.0,
    "Beauty": 35.0,
    "Books": 22.0,
}
CATEGORY_UNITS_BASE = {
    "Electronics": 12,
    "Clothing": 40,
    "Home & Kitchen": 22,
    "Sports": 18,
    "Beauty": 28,
    "Books": 14,
}

# Seasonal amplitude per category by month (multiplier on base demand)
SEASONAL_PATTERN = {
    "Electronics": {11: 1.6, 12: 1.9, 1: 1.7, 6: 0.7},
    "Clothing": {11: 1.5, 12: 1.8, 1: 1.6, 7: 1.4},
    "Home & Kitchen": {11: 1.4, 12: 1.9, 1: 1.5, 6: 0.8},
    "Sports": {6: 1.8, 7: 2.0, 8: 1.5, 12: 0.9, 1: 0.8},
    "Beauty": {11: 1.2, 12: 1.3, 6: 1.2, 1: 1.1},
    "Books": {11: 1.5, 12: 1.7, 1: 1.4},
}

PROMO_SEASON_MONTHS = {11, 12, 1, 6}


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def _season_multiplier(category: str, month: int, rng: np.random.Generator) -> float:
    base = SEASONAL_PATTERN.get(category, {1: 1.0})
    return base.get(month, 1.0)


def _draw_categories(rng: np.random.Generator, n: int) -> pd.Series:
    weights = np.array([CATEGORY_UNITS_BASE[c] for c in CATEGORIES], dtype=float)
    weights = weights / weights.sum()
    return pd.Series(rng.choice(list(CATEGORIES.keys()), size=n, p=weights))


def generate_sales_data(n: int = 6000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic sales dataset of ~n line items.

    Returns a DataFrame with one row per transaction.
    """
    rng = np.random.default_rng(seed)
    n_countries = sum(len(v) for v in COUNTRIES.values())

    # Sample a 2-year business calendar so seasonality (and trend) spans 2023-2024.
    start = pd.Timestamp("2023-01-01")
    end = pd.Timestamp("2024-12-31")
    day_range = pd.date_range(start, end, freq="D")
    weights = np.array([1.0 if d.weekday() < 5 else 0.25 for d in day_range])
    weights = weights / weights.sum()
    dates = rng.choice(day_range, size=n, replace=True, p=weights)

    df = pd.DataFrame({"date": dates})
    df["country"] = rng.choice(
        [c for countries in COUNTRIES.values() for c in countries],
        size=n,
        p=np.ones(n_countries) / n_countries,
    )
    df["category"] = _draw_categories(rng, n)

    # Map each country to its region
    df["region"] = df["country"].map(
        lambda c: next(r for r, cs in COUNTRIES.items() if c in cs)
    )

    df["sub_category"] = df["category"].map(lambda c: rng.choice(CATEGORIES[c]))
    df["channel"] = rng.choice(CHANNELS, size=n, p=[0.55, 0.35, 0.10])
    df["segment"] = rng.choice(SEGMENTS, size=n, p=[0.45, 0.40, 0.15])

    # ---- demand / pricing model ----
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["weekday"] = df["date"].dt.weekday

    # base units + seasonal uplift
    df["units_sold"] = df.apply(
        lambda r: int(
            max(
                1,
                CATEGORY_UNITS_BASE[r["category"]]
                * _season_multiplier(r["category"], r["month"], rng)
                * rng.lognormal(0.05, 0.35),
            )
        ),
        axis=1,
    )

    # unit price: category base + drift + channel premium + small noise
    channel_premium = {"Online": 12, "Store": 0, "Reseller": -6}
    df["unit_price"] = df.apply(
        lambda r: round(
            CATEGORY_PRICE_BASE[r["category"]]
            + rng.normal(0, 8)
            + channel_premium[r["channel"]]
            + rng.normal(0, 5),
            2,
        ),
        axis=1,
    )

    # discount: baseline small; promotional spikes in promo seasons
    promo = df["month"].isin(PROMO_SEASON_MONTHS).astype(float)
    df["discount_percent"] = (
        rng.uniform(0.02, 0.08, size=n) * (1 - promo * 0.55)
    )  # lower base in promo season
    promo_mask = promo == 1
    df.loc[promo_mask, "discount_percent"] = rng.uniform(0.20, 0.55, size=promo_mask.sum())

    # shipping cost
    df["shipping_cost"] = rng.gamma(2.0, 3.0, size=n).round(2)

    # cost price ~ 45-72% of unit price
    df["cost_price"] = (df["unit_price"] * rng.uniform(0.45, 0.72, size=n)).round(2)

    # Derived metrics were rounded inline; return the frame as-is. The datetime
    # "date" column cannot be passed through .round().
    return df


# ---------------------------------------------------------------------------
# Cleaning / persistence
# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standard cleaning: types, nulls, duplicates, impossible values."""
    df = df.copy()

    # Types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
    df["cost_price"] = pd.to_numeric(df["cost_price"], errors="coerce")
    df["discount_percent"] = pd.to_numeric(df["discount_percent"], errors="coerce")
    df["shipping_cost"] = pd.to_numeric(df["shipping_cost"], errors="coerce")

    # Drop rows missing the core revenue fields
    core = ["date", "category", "units_sold", "unit_price", "cost_price"]
    df = df.dropna(subset=core)

    # Impossible values
    df = df[df["units_sold"] > 0]
    df = df[df["unit_price"] > 0]
    df = df[df["cost_price"] >= 0]
    df = df[df["discount_percent"] >= 0]
    df["discount_percent"] = df["discount_percent"].clip(upper=1.0)

    # Derived metrics
    df["line_total"] = (df["units_sold"] * df["unit_price"]).round(2)
    df["gross_profit"] = (
        (df["unit_price"] - df["cost_price"]) * df["units_sold"]
    ).round(2)
    df["discount_amount"] = (df["line_total"] * df["discount_percent"]).round(2)

    # Drop duplicates
    df = df.drop_duplicates()

    # Sort chronologically
    df = df.sort_values("date").reset_index(drop=True)

    # Round derived numeric metric columns (datetime cannot be rounded this way)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].round(2)

    return df


def save_data(df: pd.DataFrame, raw_path: str, processed_path: str) -> None:
    df.to_csv(raw_path, index=False)
    # processed = derived metrics + clean subset
    clean = clean_data(df)
    clean.to_csv(processed_path, index=False)
