"""Visualization helpers for the sales analytics project.

All plots are rendered to PNG files (Agg backend) and saved under plots/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def _save(fig: plt.Figure, out_path: Path) -> Path:
    """Render and close a figure, returning the saved path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------
def plot_revenue_over_time(df: pd.DataFrame, out_path: Path) -> Path:
    monthly = (
        df.groupby(df["date"].dt.to_period("M"))["line_total"].sum().reset_index()
    )
    monthly["date_str"] = monthly["date"].astype(str)
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.lineplot(data=monthly, x="date_str", y="line_total", marker="o", ax=ax)
    ax.set_title("Total Revenue Over Time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue ($)")
    _save(fig, out_path)


def plot_monthly_seasonality(df: pd.DataFrame, out_path: Path) -> Path:
    piv = (
        df.groupby([df["date"].dt.year, df["date"].dt.month])["line_total"]
        .sum()
        .unstack(level=0)
    )
    # Ensure all 12 months appear (some may be absent if a month has no rows)
    piv = piv.reindex(columns=range(1, 13), fill_value=0)
    piv.index = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    piv.plot(kind="bar", ax=ax)
    ax.set_title("Monthly Revenue (2023 vs 2024)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue ($)")
    ax.legend(title="Year")
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Categorical comparisons
# ---------------------------------------------------------------------------
def plot_revenue_by_region(df: pd.DataFrame, out_path: Path) -> Path:
    by_region = (
        df.groupby("region")["line_total"].sum().sort_values().reset_index()
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=by_region, x="region", y="line_total", hue="region",
                palette="viridis", legend=False, ax=ax)
    ax.set_title("Total Revenue by Region")
    ax.set_ylabel("Revenue ($)")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, out_path)


def plot_revenue_by_category(df: pd.DataFrame, out_path: Path) -> Path:
    by_cat = (
        df.groupby("category")["line_total"]
        .sum()
        .sort_values()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=by_cat, x="line_total", y="category", hue="category",
                palette="mako", legend=False, ax=ax)
    ax.set_title("Total Revenue by Category")
    ax.set_xlabel("Revenue ($)")
    _save(fig, out_path)


def plot_channel_segment_mix(df: pd.DataFrame, out_path: Path) -> Path:
    ct = pd.crosstab(df["segment"], df["channel"])
    fig, ax = plt.subplots(figsize=(9, 5))
    ct.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Transactions by Customer Segment and Channel")
    ax.set_xlabel("Customer Segment")
    ax.set_ylabel("Transactions")
    ax.legend(title="Channel")
    ax.tick_params(axis="x", rotation=0)
    _save(fig, out_path)


def plot_discount_distribution(df: pd.DataFrame, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df["discount_percent"], bins=30, kde=True, ax=ax)
    ax.set_title("Discount Percent Distribution")
    ax.set_xlabel("Discount (%)")
    ax.set_ylabel("Frequency")
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Relationship / composition
# ---------------------------------------------------------------------------
def plot_correlation_heatmap(df: pd.DataFrame, out_path: Path) -> Path:
    num = df.select_dtypes(include=[np.number])
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax
    )
    ax.set_title("Correlation Heatmap (numeric fields)")
    _save(fig, out_path)


def plot_scatter_units_vs_price(df: pd.DataFrame, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(
        data=df.sample(min(len(df), 1500), random_state=42),
        x="unit_price",
        y="units_sold",
        hue="category",
        alpha=0.6,
        legend=False,
        ax=ax,
    )
    ax.set_title("Units Sold vs Unit Price")
    ax.set_xlabel("Unit Price ($)")
    ax.set_ylabel("Units Sold")
    _save(fig, out_path)
