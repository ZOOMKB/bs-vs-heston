import numpy as np
import pytest

from src.black_scholes import bs_price, put_call_parity_rhs
from src.heston import HestonParams, default_phi_grid, heston_char_func, heston_price, heston_price_trapz


@pytest.fixture
def karlsson_params():
    return HestonParams(v0=0.16, kappa=10.0, theta=0.16, sigma_v=0.10, rho=-0.80)


def test_char_func_at_zero(karlsson_params):
    assert heston_char_func(0.0, 100.0, 1.0, 0.05, karlsson_params, j=1, q=0.0) == pytest.approx(1.0 + 0.0j)
    assert heston_char_func(0.0, 100.0, 1.0, 0.05, karlsson_params, j=2, q=0.0) == pytest.approx(1.0 + 0.0j)


def test_char_func_is_finite_on_grid(karlsson_params):
    phis = np.linspace(0.01, 200.0, 500)

    for j in (1, 2):
        values = heston_char_func(phis, 100.0, 1.0, 0.05, karlsson_params, j=j, q=0.0)
        assert np.all(np.isfinite(values.real))
        assert np.all(np.isfinite(values.imag))


def test_char_func_conjugate_symmetry(karlsson_params):
    phi = 5.0

    for j in (1, 2):
        f_pos = heston_char_func(phi, 100.0, 1.0, 0.05, karlsson_params, j=j, q=0.0)
        f_neg = heston_char_func(-phi, 100.0, 1.0, 0.05, karlsson_params, j=j, q=0.0)
        assert f_neg == pytest.approx(np.conj(f_pos), abs=1e-10)


def test_heston_fang_oosterlee_benchmark():
    params = HestonParams(
        v0=0.0175,
        kappa=1.5768,
        theta=0.0398,
        sigma_v=0.5751,
        rho=-0.5711,
    )

    price = heston_price(100.0, 100.0, 1.0, 0.0, params, option_type="call", q=0.0)

    assert price == pytest.approx(5.7854, abs=1e-3)


def test_heston_bs_limit():
    sigma = 0.20
    params = HestonParams(v0=sigma**2, kappa=5.0, theta=sigma**2, sigma_v=1e-6, rho=0.0)

    for K, T in [(100.0, 1.0), (90.0, 1.0), (110.0, 0.5)]:
        heston = heston_price(100.0, K, T, 0.05, params, option_type="call", q=0.0)
        black_scholes = bs_price(100.0, K, T, 0.05, sigma, option_type="call", q=0.0)
        assert heston == pytest.approx(black_scholes, abs=1e-2)


def test_heston_put_call_parity_with_dividend_yield(karlsson_params):
    S, K, T, r, q = 100.0, 110.0, 0.75, 0.04, 0.012

    call = heston_price(S, K, T, r, karlsson_params, option_type="call", q=q)
    put = heston_price(S, K, T, r, karlsson_params, option_type="put", q=q)

    assert call - put == pytest.approx(put_call_parity_rhs(S, K, T, r, q), abs=1e-8)


def test_heston_trapz_matches_quad(karlsson_params):
    phi_grid = default_phi_grid(max_phi=200.0, n_points=2000)

    for K, T, option_type in [(90.0, 1.0, "call"), (100.0, 0.5, "call"), (110.0, 1.0, "put")]:
        quad_price = heston_price(100.0, K, T, 0.05, karlsson_params, option_type=option_type, q=0.0)
        trapz_price = heston_price_trapz(
            100.0,
            K,
            T,
            0.05,
            karlsson_params,
            option_type=option_type,
            q=0.0,
            phi_grid=phi_grid,
        )
        assert trapz_price == pytest.approx(quad_price, abs=5e-2)


def test_heston_params_validation():
    with pytest.raises(ValueError):
        HestonParams(v0=-0.01, kappa=1.0, theta=0.04, sigma_v=0.3, rho=-0.5)

    with pytest.raises(ValueError):
        HestonParams(v0=0.04, kappa=1.0, theta=0.04, sigma_v=0.3, rho=-1.0)
