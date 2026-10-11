"""Feature engineering for the predictive models.

Adds time/seasonal/promo encodings and builds the modelling matrix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _season_name(month: int) -> str:
    return {
        1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn",
        11: "Autumn", 12: "Winter",
    }[month]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a feature-rich copy of the frame (does not mutate input)."""
    d = df.copy()
    d["year"] = d["date"].dt.year
    d["month"] = d["date"].dt.month
    d["day_of_week"] = d["date"].dt.dayofweek
    d["season"] = d["month"].map(_season_name)
    # ordered promo tier from discount share
    d["discount_tier"] = pd.cut(
        d["discount_percent"],
        bins=[-1.0, 0.08, 0.25, 1.0],
        labels=["low", "medium", "high"],
    )
    d["unit_revenue_after_discount"] = (
        d["unit_price"] * (1 - d["discount_percent"])
    ).round(2)
    d["profit_margin"] = (
        (d["unit_price"] - d["cost_price"]) / d["unit_price"]
    ).round(4)
    return d


def build_model_matrix(df: pd.DataFrame, target: str = "units_sold"):
    """Return (X, y) ready for scikit-learn.

    X contains numeric time features + one-hot encoded category/region/channel/
    segment + an ordinal discount tier.
    """
    d = engineer_features(df).copy()
    d["discount_tier_code"] = d["discount_tier"].map({"low": 0, "medium": 1, "high": 2})

    categorical = ["category", "region", "channel", "segment"]
    dummies = pd.get_dummies(d[categorical], prefix=categorical, drop_first=True)

    numeric = [
        "year", "month", "day_of_week", "discount_tier_code",
        "unit_revenue_after_discount", "shipping_cost", "profit_margin",
    ]
    X = pd.concat(
        [d[numeric].reset_index(drop=True), dummies.reset_index(drop=True)],
        axis=1,
    )
    y = d[target].reset_index(drop=True)
    return X, y
