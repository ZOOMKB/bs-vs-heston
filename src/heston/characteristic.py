from __future__ import annotations

import numpy as np

from src.types import HestonParams


def _heston_ab(params: HestonParams, j: int, lambda_: float = 0.0) -> tuple[float, float, float]:
    if j == 1:
        return params.kappa * params.theta, 0.5, params.kappa + lambda_ - params.rho * params.sigma_v
    if j == 2:
        return params.kappa * params.theta, -0.5, params.kappa + lambda_
    raise ValueError("j must be either 1 or 2.")


def heston_char_func(
    phi: float | np.ndarray,
    S: float,
    T: float,
    r: float,
    params: HestonParams,
    j: int,
    q: float,
    lambda_: float = 0.0,
) -> complex | np.ndarray:
    """Heston characteristic function f_j using the Little Heston Trap form."""

    if S <= 0:
        raise ValueError("S must be positive.")
    if T < 0:
        raise ValueError("T must be non-negative.")

    phi_arr = np.asarray(phi, dtype=float)
    scalar_input = np.isscalar(phi)

    x = np.log(S)
    a, u, b = _heston_ab(params, j, lambda_=lambda_)
    sigma2 = params.sigma_v**2
    i_phi = 1j * phi_arr

    d = np.sqrt((params.rho * params.sigma_v * i_phi - b) ** 2 - sigma2 * (2.0 * u * i_phi - phi_arr**2))
    numerator = b - params.rho * params.sigma_v * i_phi - d
    denominator = b - params.rho * params.sigma_v * i_phi + d
    g = numerator / denominator

    exp_neg_dT = np.exp(-d * T)
    one_minus_g_exp = 1.0 - g * exp_neg_dT
    one_minus_g = 1.0 - g

    D = numerator / sigma2 * (1.0 - exp_neg_dT) / one_minus_g_exp
    C = (
        (r - q) * i_phi * T
        + a
        / sigma2
        * (numerator * T - 2.0 * np.log(one_minus_g_exp / one_minus_g))
    )

    value = np.exp(C + D * params.v0 + i_phi * x)
    if scalar_input:
        return complex(np.asarray(value).item())
    return value


def heston_char_func_vec(
    phi: np.ndarray,
    S: float,
    T: float,
    r: float,
    params: HestonParams,
    j: int,
    q: float,
    lambda_: float = 0.0,
) -> np.ndarray:
    return np.asarray(
        heston_char_func(
            phi=phi,
            S=S,
            T=T,
            r=r,
            params=params,
            j=j,
            q=q,
            lambda_=lambda_,
        )
    )
