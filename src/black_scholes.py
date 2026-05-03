from __future__ import annotations

import numpy as np
from scipy.stats import norm


VALID_OPTION_TYPES = {"call", "put"}


def validate_option_type(option_type: str) -> str:
    normalized = option_type.lower()
    if normalized not in VALID_OPTION_TYPES:
        raise ValueError("option_type must be either 'call' or 'put'.")
    return normalized


def intrinsic_value(S: float, K: float, option_type: str) -> float:
    option_type = validate_option_type(option_type)
    if option_type == "call":
        return max(S - K, 0.0)
    return max(K - S, 0.0)


def forward_intrinsic_value(S: float, K: float, T: float, r: float, q: float, option_type: str) -> float:
    option_type = validate_option_type(option_type)
    discounted_spot = S * np.exp(-q * T)
    discounted_strike = K * np.exp(-r * T)
    if option_type == "call":
        return max(discounted_spot - discounted_strike, 0.0)
    return max(discounted_strike - discounted_spot, 0.0)


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float) -> tuple[float, float]:
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return float(d1), float(d2)


def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    q: float,
) -> float:
    """Black-Scholes price for a European option with continuous dividend yield."""

    option_type = validate_option_type(option_type)
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive.")
    if T <= 0:
        return intrinsic_value(S, K, option_type)
    if sigma <= 0:
        return forward_intrinsic_value(S, K, T, r, q, option_type)

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    discounted_spot = S * np.exp(-q * T)
    discounted_strike = K * np.exp(-r * T)

    if option_type == "call":
        return float(discounted_spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2))
    return float(discounted_strike * norm.cdf(-d2) - discounted_spot * norm.cdf(-d1))


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float, q: float) -> float:
    return bs_price(S=S, K=K, T=T, r=r, sigma=sigma, option_type="call", q=q)


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float, q: float) -> float:
    return bs_price(S=S, K=K, T=T, r=r, sigma=sigma, option_type="put", q=q)


def bs_vega(S: float, K: float, T: float, r: float, sigma: float, q: float) -> float:
    """Black-Scholes vega, expressed as price change per 1.00 volatility point."""

    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive.")
    if T <= 0 or sigma <= 0:
        return 0.0

    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return float(S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T))


def bs_delta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    q: float,
) -> float:
    """Black-Scholes delta for a European option with continuous dividend yield."""

    option_type = validate_option_type(option_type)
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive.")
    if T <= 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    if sigma <= 0:
        forward = S * np.exp((r - q) * T)
        discounted_spot_weight = np.exp(-q * T)
        if option_type == "call":
            return float(discounted_spot_weight if forward > K else 0.0)
        return float(-discounted_spot_weight if forward < K else 0.0)

    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    discounted_spot_weight = np.exp(-q * T)
    if option_type == "call":
        return float(discounted_spot_weight * norm.cdf(d1))
    return float(discounted_spot_weight * (norm.cdf(d1) - 1.0))


def put_call_parity_rhs(S: float, K: float, T: float, r: float, q: float) -> float:
    """Right-hand side of C - P = S exp(-qT) - K exp(-rT)."""

    return float(S * np.exp(-q * T) - K * np.exp(-r * T))


def put_call_parity_error(
    call_price: float,
    put_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
) -> float:
    return float(call_price - put_price - put_call_parity_rhs(S, K, T, r, q))
