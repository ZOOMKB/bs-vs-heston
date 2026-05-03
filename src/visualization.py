from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator, griddata


def plot_volatility_smiles(
    iv_df: pd.DataFrame,
    x_col: str = "moneyness",
    title: str = "SPY Volatility Smile",
    columns: int = 3,
) -> tuple[plt.Figure, np.ndarray]:
    expiries = sorted(iv_df["expiry"].unique())
    rows = max(1, math.ceil(len(expiries) / columns))

    fig, axes = plt.subplots(rows, columns, figsize=(16, 5 * rows), squeeze=False)
    flat_axes = axes.flatten()

    for index, expiry in enumerate(expiries):
        ax = flat_axes[index]
        subset = iv_df[iv_df["expiry"] == expiry]
        T_value = subset["T"].iloc[0]

        calls = subset[subset["option_type"] == "call"]
        puts = subset[subset["option_type"] == "put"]

        ax.scatter(calls[x_col], calls["iv"], s=14, alpha=0.7, label="Calls", color="steelblue")
        ax.scatter(puts[x_col], puts["iv"], s=14, alpha=0.7, label="Puts", color="tomato")

        if x_col in {"moneyness", "forward_moneyness"}:
            ax.axvline(1.0, color="gray", ls="--", lw=0.8, label="ATM")
            ax.set_xlabel("Moneyness (K / S)" if x_col == "moneyness" else "Forward moneyness")
        else:
            ax.axvline(subset["S"].iloc[0], color="gray", ls="--", lw=0.8, label="Spot")
            ax.set_xlabel("Strike")

        ax.set_title(f"{expiry} (T = {T_value:.2f} yr)")
        ax.set_ylabel("Implied volatility")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for index in range(len(expiries), len(flat_axes)):
        flat_axes[index].set_visible(False)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig, axes


def plot_heston_smile_overlay(
    market_df: pd.DataFrame,
    heston_df: pd.DataFrame,
    market_iv_col: str = "market_iv",
    x_col: str = "moneyness",
    title: str = "Market vs Calibrated Heston Volatility Smile",
    columns: int = 3,
) -> tuple[plt.Figure, np.ndarray]:
    expiries = sorted(market_df["expiry"].unique())
    rows = max(1, math.ceil(len(expiries) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(16, 4.5 * rows), squeeze=False)
    flat_axes = axes.flatten()

    for index, expiry in enumerate(expiries):
        ax = flat_axes[index]
        market = market_df[market_df["expiry"] == expiry]
        heston = heston_df[heston_df["expiry"] == expiry].dropna(subset=["heston_iv"])
        T_value = market["T"].iloc[0]

        calls = market[market["option_type"] == "call"]
        puts = market[market["option_type"] == "put"]
        ax.scatter(calls[x_col], calls[market_iv_col], s=14, alpha=0.65, color="steelblue", label="Market calls")
        ax.scatter(puts[x_col], puts[market_iv_col], s=14, alpha=0.65, color="tomato", label="Market puts")
        ax.plot(heston[x_col], heston["heston_iv"], color="black", linewidth=2.0, label="Heston")
        ax.axvline(1.0, color="gray", ls="--", lw=0.8)
        ax.set_title(f"{expiry} (T = {T_value:.2f} yr)")
        ax.set_xlabel("Moneyness (K / S)" if x_col == "moneyness" else x_col)
        ax.set_ylabel("Implied volatility")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    for index in range(len(expiries), len(flat_axes)):
        flat_axes[index].set_visible(False)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig, axes


def plot_greek_comparison(
    greeks_df: pd.DataFrame,
    greek: str,
    title: str | None = None,
    columns: int = 3,
) -> tuple[plt.Figure, np.ndarray]:
    if greek not in {"delta", "gamma", "vega", "theta"}:
        raise ValueError("greek must be one of: delta, gamma, vega, theta.")

    expiries = sorted(greeks_df["expiry"].unique())
    rows = max(1, math.ceil(len(expiries) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(16, 4.5 * rows), squeeze=False)
    flat_axes = axes.flatten()
    bs_col = f"bs_{greek}"
    hs_col = f"heston_{greek}"

    for index, expiry in enumerate(expiries):
        ax = flat_axes[index]
        sub = greeks_df[greeks_df["expiry"] == expiry].sort_values("moneyness")
        T_value = sub["T"].iloc[0]
        ax.plot(sub["moneyness"], sub[bs_col], marker="o", markersize=3, linewidth=1.3, label="Black-Scholes")
        ax.plot(sub["moneyness"], sub[hs_col], marker="s", markersize=3, linewidth=1.3, label="Heston")
        ax.axvline(1.0, color="gray", ls="--", lw=0.8)
        ax.set_title(f"{expiry} (T = {T_value:.2f} yr)")
        ax.set_xlabel("Moneyness (K / S)")
        ax.set_ylabel(greek.title())
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    for index in range(len(expiries), len(flat_axes)):
        flat_axes[index].set_visible(False)

    fig.suptitle(title or f"{greek.title()} Comparison: Black-Scholes vs Heston", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig, axes


def plot_iv_surface_scatter(
    iv_df: pd.DataFrame,
    title: str = "SPY Implied Volatility Surface",
) -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        iv_df["moneyness"],
        iv_df["T"],
        iv_df["iv"],
        c=iv_df["iv"],
        cmap="RdYlBu_r",
        s=10,
        alpha=0.7,
    )

    ax.set_xlabel("Moneyness (K / S)")
    ax.set_ylabel("Time to maturity (years)")
    ax.set_zlabel("Implied volatility")
    ax.set_title(title)
    fig.colorbar(scatter, ax=ax, shrink=0.5, label="Implied volatility")
    ax.view_init(elev=25, azim=-50)
    fig.tight_layout()
    return fig, ax


def plot_iv_contour(
    iv_df: pd.DataFrame,
    title: str = "SPY Implied Volatility Surface (contour)",
    grid_size: int = 200,
    method: str = "linear",
) -> tuple[plt.Figure, plt.Axes]:
    m_grid = np.linspace(iv_df["moneyness"].min(), iv_df["moneyness"].max(), grid_size)
    t_grid = np.linspace(iv_df["T"].min(), iv_df["T"].max(), grid_size)
    m_mesh, t_mesh = np.meshgrid(m_grid, t_grid)

    iv_grid = griddata(
        points=(iv_df["moneyness"].values, iv_df["T"].values),
        values=iv_df["iv"].values,
        xi=(m_mesh, t_mesh),
        method=method,
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    contour = ax.contourf(m_mesh, t_mesh, iv_grid, levels=30, cmap="RdYlBu_r")
    fig.colorbar(contour, ax=ax, label="Implied volatility")
    ax.scatter(iv_df["moneyness"], iv_df["T"], c="black", s=3, alpha=0.3, label="Data")
    ax.set_xlabel("Moneyness (K / S)")
    ax.set_ylabel("Time to maturity (years)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_iv_surface_rbf(
    iv_df: pd.DataFrame,
    title: str = "SPY Implied Volatility Surface (interpolated)",
    grid_size: int = 120,
    smoothing: float = 0.001,
    clip: tuple[float, float] | None = (0.05, 0.35),
) -> tuple[plt.Figure, plt.Axes]:
    time_scale = max(float(iv_df["T"].max()), 1.0)
    points = np.column_stack([iv_df["moneyness"].values, iv_df["T"].values / time_scale])
    values = iv_df["iv"].values
    rbf = RBFInterpolator(points, values, kernel="thin_plate_spline", smoothing=smoothing)

    m_grid = np.linspace(iv_df["moneyness"].min(), iv_df["moneyness"].max(), grid_size)
    t_grid = np.linspace(iv_df["T"].min(), iv_df["T"].max(), grid_size)
    m_mesh, t_mesh = np.meshgrid(m_grid, t_grid)

    query = np.column_stack([m_mesh.ravel(), t_mesh.ravel() / time_scale])
    iv_smooth = rbf(query).reshape(m_mesh.shape)
    if clip is not None:
        iv_smooth = np.clip(iv_smooth, clip[0], clip[1])

    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        m_mesh,
        t_mesh,
        iv_smooth,
        cmap="RdYlBu_r",
        alpha=0.95,
        rstride=1,
        cstride=1,
        edgecolor="none",
        antialiased=True,
    )

    ax.set_xlabel("Moneyness (K / S)")
    ax.set_ylabel("Time to maturity (years)")
    ax.set_zlabel("Implied volatility")
    ax.set_title(title)
    fig.colorbar(surface, ax=ax, shrink=0.5, label="Implied volatility")
    ax.view_init(elev=25, azim=-50)
    fig.tight_layout()
    return fig, ax
