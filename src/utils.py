from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from src.black_scholes import bs_price, put_call_parity_rhs, validate_option_type


def option_mid_price(options_df: pd.DataFrame) -> pd.Series:
    return (options_df["bid"] + options_df["ask"]) / 2.0


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str,
    q: float,
    sigma_low: float = 1e-4,
    sigma_high: float = 5.0,
    tol: float = 1e-8,
) -> float:
    """Solve for Black-Scholes implied volatility with Brent's method."""

    option_type = validate_option_type(option_type)
    inputs = [market_price, S, K, T, r, q]
    if not np.all(np.isfinite(inputs)) or market_price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return np.nan

    low_price = bs_price(S, K, T, r, sigma_low, option_type=option_type, q=q)
    high_price = bs_price(S, K, T, r, sigma_high, option_type=option_type, q=q)

    if market_price < low_price - tol or market_price > high_price + tol:
        return np.nan
    if abs(market_price - low_price) <= tol:
        return sigma_low
    if abs(market_price - high_price) <= tol:
        return sigma_high

    def objective(sigma: float) -> float:
        return bs_price(S, K, T, r, sigma, option_type=option_type, q=q) - market_price

    try:
        return float(brentq(objective, sigma_low, sigma_high, xtol=tol))
    except ValueError:
        return np.nan


def add_option_features(options_df: pd.DataFrame, q: float | None = None) -> pd.DataFrame:
    """Add mid, moneyness, forward, and OTM flags without mutating the input."""

    df = options_df.copy()
    df["option_type"] = df["option_type"].str.lower()
    df["mid"] = option_mid_price(df)

    if "q" not in df.columns:
        if q is None:
            raise ValueError("Options data must contain a q column or q must be provided.")
        df["q"] = q

    df["moneyness"] = df["strike"] / df["S"]
    df["forward"] = df["S"] * np.exp((df["r"] - df["q"]) * df["T"])
    df["forward_moneyness"] = df["strike"] / df["forward"]
    df["is_otm"] = (
        ((df["option_type"] == "call") & (df["strike"] >= df["forward"]))
        | ((df["option_type"] == "put") & (df["strike"] <= df["forward"]))
    )
    return df


def prepare_iv_dataset(
    options_df: pd.DataFrame,
    q: float | None = None,
    min_mid: float = 0.10,
    moneyness_bounds: tuple[float, float] | None = (0.85, 1.15),
    iv_bounds: tuple[float, float] | None = (0.05, 0.35),
    use_otm_only: bool = False,
    max_relative_spread: float | None = None,
) -> pd.DataFrame:
    """Clean option quotes and calculate Black-Scholes implied volatilities."""

    df = add_option_features(options_df, q=q)
    required = ["bid", "ask", "mid", "strike", "S", "T", "r", "q"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    df = df[
        (df["bid"] >= 0)
        & (df["ask"] >= df["bid"])
        & (df["mid"] > min_mid)
        & (df["strike"] > 0)
        & (df["S"] > 0)
        & (df["T"] > 0)
    ].copy()

    if moneyness_bounds is not None:
        low, high = moneyness_bounds
        df = df[(df["moneyness"] >= low) & (df["moneyness"] <= high)].copy()

    if max_relative_spread is not None:
        relative_spread = (df["ask"] - df["bid"]) / df["mid"]
        df = df[relative_spread <= max_relative_spread].copy()

    if use_otm_only:
        df = df[df["is_otm"]].copy()

    df["iv"] = df.apply(
        lambda row: implied_vol(
            market_price=row["mid"],
            S=row["S"],
            K=row["strike"],
            T=row["T"],
            r=row["r"],
            option_type=row["option_type"],
            q=row["q"],
        ),
        axis=1,
    )
    df = df[df["iv"].notna()].copy()

    if iv_bounds is not None:
        low, high = iv_bounds
        df = df[(df["iv"] >= low) & (df["iv"] <= high)].copy()

    return df


def check_put_call_parity(
    options_df: pd.DataFrame,
    q: float | None = None,
    min_mid: float = 0.0,
) -> pd.DataFrame:
    """Match calls and puts by expiry/strike and compute put-call parity errors."""

    df = add_option_features(options_df, q=q)
    df = df[(df["mid"] > min_mid) & (df["T"] > 0)].copy()

    calls = df[df["option_type"] == "call"][
        ["expiry", "strike", "mid", "T", "S", "r", "q", "moneyness"]
    ].copy()
    puts = df[df["option_type"] == "put"][["expiry", "strike", "mid"]].copy()
    calls = calls.rename(columns={"mid": "C_mid"})
    puts = puts.rename(columns={"mid": "P_mid"})

    pairs = calls.merge(puts, on=["expiry", "strike"], how="inner")
    pairs = pairs[(pairs["C_mid"] > min_mid) & (pairs["P_mid"] > min_mid)].copy()
    pairs["lhs"] = pairs["C_mid"] - pairs["P_mid"]
    pairs["rhs"] = pairs.apply(
        lambda row: put_call_parity_rhs(row["S"], row["strike"], row["T"], row["r"], row["q"]),
        axis=1,
    )
    pairs["parity_error"] = pairs["lhs"] - pairs["rhs"]
    pairs["parity_error_pct"] = pairs["parity_error"].abs() / pairs["S"] * 100.0
    return pairs.sort_values(["expiry", "strike"]).reset_index(drop=True)


def summarize_put_call_parity(pairs: pd.DataFrame) -> dict[str, float]:
    return {
        "pairs": float(len(pairs)),
        "mean_abs_error": float(pairs["parity_error"].abs().mean()),
        "median_abs_error": float(pairs["parity_error"].abs().median()),
        "max_abs_error": float(pairs["parity_error"].abs().max()),
        "mean_error_pct_of_spot": float(pairs["parity_error_pct"].mean()),
    }


def summarize_put_call_parity_by_expiry(pairs: pd.DataFrame) -> pd.DataFrame:
    grouped = pairs.groupby("expiry", as_index=False).agg(
        pairs=("strike", "size"),
        mean_abs_error=("parity_error", lambda value: value.abs().mean()),
        median_abs_error=("parity_error", lambda value: value.abs().median()),
        mean_error_pct=("parity_error_pct", "mean"),
    )
    return grouped.sort_values("expiry").reset_index(drop=True)
