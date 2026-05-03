import numpy as np
import pandas as pd
import pytest

from src.heston.calibration import (
    calibration_objective,
    calibration_iv_proxy_rmse,
    calibration_rmse,
    evaluate_heston_fit,
    estimate_initial_params,
    prepare_calibration_data,
    summarize_calibration_errors,
    summarize_errors_by_expiry,
    summarize_errors_by_moneyness_bucket,
)
from src.heston.pricing import default_phi_grid, heston_price_trapz
from src.types import HestonParams


def _sample_options_frame():
    return pd.DataFrame(
        {
            "quote_id": [f"pass_{i}" for i in range(8)],
            "ticker": ["SPY"] * 8,
            "expiry": ["2026-06-30"] * 4 + ["2026-12-18"] * 4,
            "option_type": ["put", "put", "call", "call"] * 2,
            "strike": [90.0, 95.0, 105.0, 110.0, 88.0, 94.0, 110.0, 112.0],
            "bid": [1.0, 2.0, 2.0, 1.0, 3.0, 5.0, 5.0, 3.0],
            "ask": [1.1, 2.2, 2.2, 1.1, 3.3, 5.5, 5.5, 3.3],
            "openInterest": [100] * 8,
            "volume": [10] * 8,
            "impliedVolatility": [0.24, 0.22, 0.20, 0.21, 0.25, 0.23, 0.22, 0.23],
            "T": [0.25] * 4 + [0.75] * 4,
            "S": [100.0] * 8,
            "r": [0.04] * 8,
            "q": [0.01] * 8,
        }
    )


def test_prepare_calibration_data_filters_otm_and_adds_weights():
    options = _sample_options_frame()

    calib = prepare_calibration_data(options, selection="otm", moneyness_range=(0.85, 1.15))

    assert len(calib) == len(options)
    assert calib["is_otm"].all()
    assert (calib["mid_price"] > 0).all()
    assert (calib["bs_vega"] > 0).all()
    assert (calib["bid"] > 0).all()
    assert (calib["relative_spread"] <= 0.15).all()
    assert calib["T"].between(14 / 365, 1.5).all()
    assert calib["market_iv"].between(0.05, 1.00).all()
    assert calib["abs_delta"].between(0.05, 0.40).all()
    assert calib["weight"].sum() == pytest.approx(len(calib))
    assert calib.groupby(["expiry", "option_type"])["weight"].sum().nunique() == 1


def test_prepare_calibration_data_supports_vega_weights():
    options = _sample_options_frame()

    calib = prepare_calibration_data(
        options,
        selection="otm",
        moneyness_range=(0.85, 1.15),
        weight_scheme="vega",
    )

    assert calib["weight"].sum() == pytest.approx(len(calib))
    assert calib["weight"].std() > 0


def test_prepare_calibration_data_removes_low_quality_quotes():
    options = _sample_options_frame()
    noisy_rows = []

    def bad_row(quote_id, **overrides):
        row = options.iloc[0].to_dict()
        row["quote_id"] = quote_id
        row.update(overrides)
        noisy_rows.append(row)

    bad_row("zero_bid", bid=0.0, ask=1.0)
    bad_row("wide_spread", bid=1.0, ask=2.0)
    bad_row("short_expiry", T=3.0 / 365.0)
    bad_row("long_expiry", T=3.0)
    bad_row("low_iv", impliedVolatility=0.01)
    bad_row("high_iv", impliedVolatility=1.60)
    bad_row("low_open_interest", openInterest=0)
    bad_row("high_delta", option_type="call", strike=101.0, bid=4.0, ask=4.2, impliedVolatility=0.20)

    noisy_options = pd.concat([options, pd.DataFrame(noisy_rows)], ignore_index=True)
    calib = prepare_calibration_data(noisy_options)

    assert set(calib["quote_id"]) == set(options["quote_id"])


def test_estimate_initial_params_returns_valid_params():
    calib = prepare_calibration_data(_sample_options_frame(), selection="otm", moneyness_range=(0.85, 1.15))

    params = estimate_initial_params(calib)

    assert isinstance(params, HestonParams)
    assert params.v0 > 0
    assert params.theta > 0
    assert params.kappa > 0
    assert params.sigma_v > 0
    assert -1 < params.rho < 1


def test_calibration_objective_good_params_better_than_bad_params():
    true_params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.30, rho=-0.60)
    bad_params = HestonParams(v0=0.20, kappa=0.20, theta=0.20, sigma_v=1.50, rho=-0.05)
    phi_grid = default_phi_grid(max_phi=150.0, n_points=800)

    rows = []
    for K, T, option_type in [
        (90.0, 0.5, "put"),
        (95.0, 0.5, "put"),
        (105.0, 0.5, "call"),
        (110.0, 0.5, "call"),
        (90.0, 1.0, "put"),
        (110.0, 1.0, "call"),
    ]:
        price = heston_price_trapz(100.0, K, T, 0.04, true_params, option_type=option_type, q=0.01, phi_grid=phi_grid)
        rows.append(
            {
                "S": 100.0,
                "strike": K,
                "T": T,
                "r": 0.04,
                "q": 0.01,
                "option_type": option_type,
                "mid_price": price,
                "market_iv": 0.20,
                "bs_vega": 1.0,
                "weight": 1.0,
            }
        )
    market_data = pd.DataFrame(rows)

    err_true = calibration_objective(true_params.to_array(), market_data, phi_grid=phi_grid)
    err_bad = calibration_objective(bad_params.to_array(), market_data, phi_grid=phi_grid)
    rmse_true = calibration_rmse(true_params.to_array(), market_data, phi_grid=phi_grid)
    iv_proxy_true = calibration_iv_proxy_rmse(true_params.to_array(), market_data, phi_grid=phi_grid)
    iv_proxy_bad = calibration_objective(
        bad_params.to_array(),
        market_data,
        phi_grid=phi_grid,
        loss_type="iv_proxy_rmse",
    )

    assert err_true < err_bad
    assert rmse_true == pytest.approx(0.0, abs=1e-10)
    assert iv_proxy_true == pytest.approx(0.0, abs=1e-10)
    assert iv_proxy_true < iv_proxy_bad


def test_evaluate_heston_fit_and_summaries():
    true_params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.30, rho=-0.60)
    phi_grid = default_phi_grid(max_phi=150.0, n_points=800)
    rows = []
    for expiry, K, T, option_type in [
        ("2026-06-30", 95.0, 0.5, "put"),
        ("2026-06-30", 105.0, 0.5, "call"),
        ("2026-12-18", 90.0, 1.0, "put"),
        ("2026-12-18", 110.0, 1.0, "call"),
    ]:
        price = heston_price_trapz(100.0, K, T, 0.04, true_params, option_type=option_type, q=0.01, phi_grid=phi_grid)
        rows.append(
            {
                "expiry": expiry,
                "S": 100.0,
                "strike": K,
                "T": T,
                "r": 0.04,
                "q": 0.01,
                "option_type": option_type,
                "mid_price": price,
                "market_iv": 0.20,
                "moneyness": K / 100.0,
                "bs_vega": 1.0,
            }
        )
    market_data = pd.DataFrame(rows)

    fit = evaluate_heston_fit(market_data, true_params, phi_grid=phi_grid, include_model_iv=False)
    metrics = summarize_calibration_errors(fit)
    by_expiry = summarize_errors_by_expiry(fit, error_col="iv_proxy_error")
    by_moneyness = summarize_errors_by_moneyness_bucket(fit, error_col="iv_proxy_error")

    assert fit["price_error"].abs().max() == pytest.approx(0.0, abs=1e-10)
    assert metrics.loc[metrics["metric"] == "price_RMSE", "value"].iloc[0] == pytest.approx(0.0, abs=1e-10)
    assert len(by_expiry) == 2
    assert by_moneyness["count"].sum() == len(market_data)


def test_calibration_objective_rejects_invalid_params():
    market_data = pd.DataFrame(
        [
            {
                "S": 100.0,
                "strike": 100.0,
                "T": 1.0,
                "r": 0.04,
                "q": 0.01,
                "option_type": "call",
                "mid_price": 10.0,
                "weight": 1.0,
            }
        ]
    )

    assert calibration_objective(np.array([-0.01, 1.0, 0.04, 0.3, -0.5]), market_data) >= 1e9
