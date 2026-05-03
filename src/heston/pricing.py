from __future__ import annotations

import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad

from src.black_scholes import intrinsic_value, put_call_parity_rhs, validate_option_type
from src.heston.characteristic import heston_char_func, heston_char_func_vec
from src.types import HestonParams


def default_phi_grid(max_phi: float = 200.0, n_points: int = 2000) -> np.ndarray:
    return np.linspace(1e-8, max_phi, n_points)


def _heston_integrand(
    phi: float,
    j: int,
    S: float,
    K: float,
    T: float,
    r: float,
    params: HestonParams,
    q: float,
) -> float:
    f_j = heston_char_func(phi=phi, S=S, T=T, r=r, params=params, j=j, q=q)
    value = np.exp(-1j * phi * np.log(K)) * f_j / (1j * phi)
    return float(np.real(value))


def heston_probability(
    j: int,
    S: float,
    K: float,
    T: float,
    r: float,
    params: HestonParams,
    q: float,
    lower: float = 1e-8,
    upper: float = np.inf,
    limit: int = 200,
    epsabs: float = 1e-10,
    epsrel: float = 1e-10,
) -> float:
    """Return Heston risk-neutral probability P_j from the Fourier integral."""

    if j not in {1, 2}:
        raise ValueError("j must be either 1 or 2.")
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive.")
    if T <= 0:
        if j == 1:
            return 1.0 if S > K else 0.0
        return 1.0 if S > K else 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        integral, _ = quad(
            _heston_integrand,
            lower,
            upper,
            args=(j, S, K, T, r, params, q),
            limit=limit,
            epsabs=epsabs,
            epsrel=epsrel,
        )

    return float(0.5 + integral / np.pi)


def heston_probabilities(
    S: float,
    K: float,
    T: float,
    r: float,
    params: HestonParams,
    q: float,
    **quad_kwargs,
) -> tuple[float, float]:
    return (
        heston_probability(1, S, K, T, r, params, q, **quad_kwargs),
        heston_probability(2, S, K, T, r, params, q, **quad_kwargs),
    )


def heston_call_price(
    S: float,
    K: float,
    T: float,
    r: float,
    params: HestonParams,
    q: float,
    **quad_kwargs,
) -> float:
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive.")
    if T <= 0:
        return intrinsic_value(S, K, "call")

    P1, P2 = heston_probabilities(S, K, T, r, params, q, **quad_kwargs)
    call_price = S * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2
    return float(max(call_price, 0.0))


def heston_price(
    S: float,
    K: float,
    T: float,
    r: float,
    params: HestonParams,
    option_type: str,
    q: float,
    **quad_kwargs,
) -> float:
    """European option price under the Heston model using adaptive quadrature."""

    option_type = validate_option_type(option_type)
    if T <= 0:
        return intrinsic_value(S, K, option_type)

    call_price = heston_call_price(S, K, T, r, params, q, **quad_kwargs)
    if option_type == "call":
        return call_price

    put_price = call_price - put_call_parity_rhs(S, K, T, r, q)
    return float(max(put_price, 0.0))


def heston_call_price_trapz(
    S: float,
    K: float,
    T: float,
    r: float,
    params: HestonParams,
    q: float,
    phi_grid: np.ndarray | None = None,
) -> float:
    """Fast Heston call price using a fixed phi-grid and trapezoidal integration."""

    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive.")
    if T <= 0:
        return intrinsic_value(S, K, "call")

    if phi_grid is None:
        phi_grid = default_phi_grid()

    log_strike = np.log(K)
    f1 = heston_char_func_vec(phi_grid, S, T, r, params, j=1, q=q)
    f2 = heston_char_func_vec(phi_grid, S, T, r, params, j=2, q=q)

    integrand_1 = np.real(np.exp(-1j * phi_grid * log_strike) * f1 / (1j * phi_grid))
    integrand_2 = np.real(np.exp(-1j * phi_grid * log_strike) * f2 / (1j * phi_grid))

    P1 = 0.5 + np.trapezoid(integrand_1, phi_grid) / np.pi
    P2 = 0.5 + np.trapezoid(integrand_2, phi_grid) / np.pi

    call_price = S * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2
    return float(max(call_price, 0.0))


def heston_price_trapz(
    S: float,
    K: float,
    T: float,
    r: float,
    params: HestonParams,
    option_type: str,
    q: float,
    phi_grid: np.ndarray | None = None,
) -> float:
    """Fast European Heston option price for calibration loops."""

    option_type = validate_option_type(option_type)
    if T <= 0:
        return intrinsic_value(S, K, option_type)

    call_price = heston_call_price_trapz(S, K, T, r, params, q, phi_grid=phi_grid)
    if option_type == "call":
        return call_price

    put_price = call_price - put_call_parity_rhs(S, K, T, r, q)
    return float(max(put_price, 0.0))
