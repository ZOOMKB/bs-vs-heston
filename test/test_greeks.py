import numpy as np
import pandas as pd
import pytest

from src.greeks_bs import analytical_greeks_bs, numerical_greeks_bs
from src.heston.greeks_hs import numerical_greeks_heston
from src.heston.pricing import default_phi_grid
from src.heston.smile import (
    greeks_comparison_by_strike,
    greeks_comparison_on_grid,
    heston_implied_vol,
    heston_smile_surface,
    select_expiries_by_target_tenors,
)
from src.types import HestonParams, OptionGreeks


def test_numerical_bs_greeks_match_analytical_greeks():
    args = {
        "S": 100.0,
        "K": 105.0,
        "T": 0.75,
        "r": 0.04,
        "sigma": 0.22,
        "option_type": "call",
        "q": 0.01,
    }

    analytical = analytical_greeks_bs(**args)
    numerical = numerical_greeks_bs(**args)

    assert isinstance(analytical, OptionGreeks)
    assert numerical.delta == pytest.approx(analytical.delta, abs=1e-5)
    assert numerical.gamma == pytest.approx(analytical.gamma, abs=1e-6)
    assert numerical.vega == pytest.approx(analytical.vega, abs=1e-5)
    assert numerical.theta == pytest.approx(analytical.theta, abs=1e-4)


def test_heston_numerical_greeks_are_finite():
    params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.35, rho=-0.6)
    phi_grid = default_phi_grid(max_phi=120.0, n_points=600)

    greeks = numerical_greeks_heston(
        S=100.0,
        K=100.0,
        T=0.5,
        r=0.03,
        params=params,
        option_type="call",
        q=0.01,
        phi_grid=phi_grid,
    )

    assert isinstance(greeks, OptionGreeks)
    assert np.isfinite(list(greeks.to_dict().values())).all()
    assert greeks.gamma > 0
    assert greeks.vega > 0


def test_heston_implied_vol_matches_bs_limit():
    sigma = 0.20
    params = HestonParams(v0=sigma**2, kappa=5.0, theta=sigma**2, sigma_v=1e-6, rho=0.0)
    phi_grid = default_phi_grid(max_phi=150.0, n_points=1000)

    iv = heston_implied_vol(
        S=100.0,
        K=100.0,
        T=1.0,
        r=0.04,
        params=params,
        option_type="call",
        q=0.01,
        phi_grid=phi_grid,
    )

    assert iv == pytest.approx(sigma, abs=5e-3)


def test_heston_smile_surface_and_greeks_comparison_shapes():
    params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.35, rho=-0.6)
    phi_grid = default_phi_grid(max_phi=120.0, n_points=500)
    market = pd.DataFrame(
        {
            "expiry": ["2026-06-30", "2026-06-30", "2026-12-18", "2026-12-18"],
            "option_type": ["put", "call", "put", "call"],
            "strike": [95.0, 105.0, 92.0, 108.0],
            "moneyness": [0.95, 1.05, 0.92, 1.08],
            "market_iv": [0.22, 0.18, 0.24, 0.19],
            "S": [100.0] * 4,
            "T": [0.5, 0.5, 1.0, 1.0],
            "r": [0.03] * 4,
            "q": [0.01] * 4,
        }
    )

    smile = heston_smile_surface(market, params, n_strikes=5, phi_grid=phi_grid)
    greeks = greeks_comparison_by_strike(market, params, expiries=["2026-06-30"], phi_grid=phi_grid)

    assert len(smile) == 10
    assert {"heston_iv", "moneyness", "expiry"}.issubset(smile.columns)
    assert len(greeks) == 2
    assert {"bs_delta", "heston_delta", "bs_vega", "heston_vega"}.issubset(greeks.columns)


def test_greeks_comparison_on_grid_uses_target_expiries_and_common_grid():
    params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.35, rho=-0.6)
    phi_grid = default_phi_grid(max_phi=100.0, n_points=300)
    market = pd.DataFrame(
        {
            "expiry": ["short"] * 5 + ["mid"] * 5 + ["long"] * 5,
            "option_type": ["put", "put", "put", "call", "call"] * 3,
            "strike": [90.0, 95.0, 100.0, 105.0, 110.0] * 3,
            "moneyness": [0.90, 0.95, 1.00, 1.05, 1.10] * 3,
            "iv": [0.24, 0.22, 0.20, 0.18, 0.17] * 3,
            "S": [100.0] * 15,
            "T": [0.10] * 5 + [0.50] * 5 + [2.00] * 5,
            "r": [0.03] * 15,
            "q": [0.01] * 15,
        }
    )

    selected = select_expiries_by_target_tenors(market, target_tenors=(0.10, 0.50), include_longest=True)
    greeks = greeks_comparison_on_grid(
        market,
        params,
        expiries=selected,
        moneyness_range=(0.92, 1.08),
        n_strikes=5,
        phi_grid=phi_grid,
    )

    assert selected == ["short", "mid", "long"]
    assert greeks["expiry"].nunique() == 3
    assert {"interpolated_otm_spot"} == set(greeks["market_iv_source"])
    assert {"bs_delta", "heston_delta", "bs_vega", "heston_vega"}.issubset(greeks.columns)
    assert greeks.groupby("expiry").size().nunique() == 1
