from __future__ import annotations

import numpy as np
from scipy.stats import norm

from src.black_scholes import bs_price, validate_option_type
from src.types import OptionGreeks


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float) -> tuple[float, float]:
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    return float(d1), float(d1 - sigma * sqrt_T)


def analytical_greeks_bs(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    q: float,
) -> OptionGreeks:
    """Closed-form Black-Scholes Greeks with continuous dividend yield."""

    option_type = validate_option_type(option_type)
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive.")
    if T <= 0 or sigma <= 0:
        return OptionGreeks(delta=np.nan, gamma=np.nan, vega=np.nan, theta=np.nan)

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    discounted_spot_weight = np.exp(-q * T)
    discounted_strike = K * np.exp(-r * T)

    if option_type == "call":
        delta = discounted_spot_weight * norm.cdf(d1)
        theta_per_year = (
            -S * discounted_spot_weight * norm.pdf(d1) * sigma / (2.0 * np.sqrt(T))
            - r * discounted_strike * norm.cdf(d2)
            + q * S * discounted_spot_weight * norm.cdf(d1)
        )
    else:
        delta = discounted_spot_weight * (norm.cdf(d1) - 1.0)
        theta_per_year = (
            -S * discounted_spot_weight * norm.pdf(d1) * sigma / (2.0 * np.sqrt(T))
            + r * discounted_strike * norm.cdf(-d2)
            - q * S * discounted_spot_weight * norm.cdf(-d1)
        )

    gamma = discounted_spot_weight * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * discounted_spot_weight * norm.pdf(d1) * np.sqrt(T)
    theta_per_day = theta_per_year / 365.0
    return OptionGreeks(float(delta), float(gamma), float(vega), float(theta_per_day))


def numerical_greeks_bs(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    q: float,
    spot_bump: float | None = None,
    sigma_bump: float = 1e-4,
    time_bump: float = 1.0 / 365.0,
) -> OptionGreeks:
    """Finite-difference Black-Scholes Greeks."""

    option_type = validate_option_type(option_type)
    if spot_bump is None:
        spot_bump = max(1e-4 * S, 1e-4)
    if sigma - sigma_bump <= 0:
        sigma_bump = sigma * 0.5

    mid = bs_price(S, K, T, r, sigma, option_type=option_type, q=q)
    up = bs_price(S + spot_bump, K, T, r, sigma, option_type=option_type, q=q)
    down = bs_price(S - spot_bump, K, T, r, sigma, option_type=option_type, q=q)
    delta = (up - down) / (2.0 * spot_bump)
    gamma = (up - 2.0 * mid + down) / spot_bump**2

    sigma_up = bs_price(S, K, T, r, sigma + sigma_bump, option_type=option_type, q=q)
    sigma_down = bs_price(S, K, T, r, sigma - sigma_bump, option_type=option_type, q=q)
    vega = (sigma_up - sigma_down) / (2.0 * sigma_bump)

    theta = np.nan
    if T > time_bump:
        shorter = bs_price(S, K, T - time_bump, r, sigma, option_type=option_type, q=q)
        theta = shorter - mid

    return OptionGreeks(float(delta), float(gamma), float(vega), float(theta))
