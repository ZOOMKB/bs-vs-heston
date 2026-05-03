import numpy as np
import pytest

from src.black_scholes import bs_delta, bs_price, put_call_parity_rhs
from src.utils import implied_vol


def test_bs_atm_call_and_put_without_dividends():
    call = bs_price(100, 100, 1.0, 0.05, 0.20, option_type="call", q=0.0)
    put = bs_price(100, 100, 1.0, 0.05, 0.20, option_type="put", q=0.0)

    assert call == pytest.approx(10.4506, abs=1e-4)
    assert put == pytest.approx(5.5735, abs=1e-4)


def test_bs_expiry_returns_intrinsic_value():
    call = bs_price(105, 100, 0.0, 0.05, 0.20, option_type="call", q=0.0)
    put = bs_price(95, 100, 0.0, 0.05, 0.20, option_type="put", q=0.0)

    assert call == pytest.approx(5.0)
    assert put == pytest.approx(5.0)


def test_put_call_parity_with_dividend_yield():
    S, K, T, r, sigma, q = 100, 105, 0.75, 0.04, 0.22, 0.015
    call = bs_price(S, K, T, r, sigma, option_type="call", q=q)
    put = bs_price(S, K, T, r, sigma, option_type="put", q=q)

    assert call - put == pytest.approx(put_call_parity_rhs(S, K, T, r, q), abs=1e-10)


def test_bs_delta_with_dividend_yield():
    S, K, T, r, sigma, q = 100, 100, 1.0, 0.05, 0.20, 0.01

    call_delta = bs_delta(S, K, T, r, sigma, option_type="call", q=q)
    put_delta = bs_delta(S, K, T, r, sigma, option_type="put", q=q)

    assert call_delta > 0
    assert put_delta < 0
    assert call_delta - put_delta == pytest.approx(np.exp(-q * T), abs=1e-12)


def test_implied_vol_roundtrip():
    price = bs_price(100, 100, 1.0, 0.05, 0.35, option_type="put", q=0.0)
    iv = implied_vol(price, 100, 100, 1.0, 0.05, option_type="put", q=0.0)

    assert iv == pytest.approx(0.35, abs=1e-8)


def test_implied_vol_impossible_price_returns_nan():
    iv = implied_vol(0.001, 100, 90, 1.0, 0.05, option_type="call", q=0.0)

    assert np.isnan(iv)
