from __future__ import annotations

import numpy as np

from src.black_scholes import validate_option_type
from src.heston.pricing import default_phi_grid, heston_price_trapz
from src.types import HestonParams, OptionGreeks


def numerical_greeks_heston(
    S: float,
    K: float,
    T: float,
    r: float,
    params: HestonParams,
    option_type: str,
    q: float,
    phi_grid: np.ndarray | None = None,
    spot_bump: float | None = None,
    v0_bump: float = 1e-4,
    time_bump: float = 1.0 / 365.0,
    vega_scale: str = "vol0",
) -> OptionGreeks:
    """Finite-difference Heston Greeks.

    Vega is calculated with respect to v0 by finite difference. By default it is
    converted to sensitivity per 1.00 initial-volatility unit via d(v0)/d(vol0).
    Set ``vega_scale="v0"`` to keep raw variance sensitivity.
    """

    option_type = validate_option_type(option_type)
    if phi_grid is None:
        phi_grid = default_phi_grid()
    if spot_bump is None:
        spot_bump = max(1e-4 * S, 1e-4)

    def price(S_: float, T_: float, params_: HestonParams) -> float:
        return heston_price_trapz(
            S=S_,
            K=K,
            T=T_,
            r=r,
            params=params_,
            option_type=option_type,
            q=q,
            phi_grid=phi_grid,
        )

    mid = price(S, T, params)
    up = price(S + spot_bump, T, params)
    down = price(S - spot_bump, T, params)
    delta = (up - down) / (2.0 * spot_bump)
    gamma = (up - 2.0 * mid + down) / spot_bump**2

    lower_bump = min(v0_bump, params.v0 * 0.5)
    params_up = HestonParams(
        v0=params.v0 + v0_bump,
        kappa=params.kappa,
        theta=params.theta,
        sigma_v=params.sigma_v,
        rho=params.rho,
    )
    params_down = HestonParams(
        v0=params.v0 - lower_bump,
        kappa=params.kappa,
        theta=params.theta,
        sigma_v=params.sigma_v,
        rho=params.rho,
    )
    price_up = price(S, T, params_up)
    price_down = price(S, T, params_down)
    vega_v0 = (price_up - price_down) / (v0_bump + lower_bump)
    if vega_scale == "vol0":
        vega = vega_v0 * 2.0 * params.vol_0
    elif vega_scale == "v0":
        vega = vega_v0
    else:
        raise ValueError("vega_scale must be either 'vol0' or 'v0'.")

    theta = np.nan
    if T > time_bump:
        shorter = price(S, T - time_bump, params)
        theta = shorter - mid

    return OptionGreeks(float(delta), float(gamma), float(vega), float(theta))
