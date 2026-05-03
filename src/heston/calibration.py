from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, differential_evolution, minimize

from src.black_scholes import bs_delta, bs_vega
from src.heston.pricing import default_phi_grid, heston_price_trapz
from src.types import HestonParams
from src.utils import add_option_features, implied_vol


Selection = Literal["otm", "otm_wide", "all", "calls_only", "puts_only", "itm"]
LossType = Literal["relative_price", "relative_price_rmse", "price_rmse", "iv_proxy_rmse"]
WeightScheme = Literal["equal", "vega", "expiry", "option_type", "expiry_option_type"]

DEFAULT_BOUNDS: tuple[tuple[float, float], ...] = (
    (0.001, 0.25),
    (0.10, 15.0),
    (0.001, 0.25),
    (0.05, 2.0),
    (-0.99, 0.0),
)


@dataclass(frozen=True)
class CalibrationResult:
    params: HestonParams
    objective_value: float
    rmse: float
    price_rmse: float
    iv_proxy_rmse: float
    n_options: int
    elapsed_seconds: float
    method: str
    loss_type: LossType
    optimizer_result: OptimizeResult


def heston_bounds() -> tuple[tuple[float, float], ...]:
    return DEFAULT_BOUNDS


def params_from_array(params_arr: np.ndarray | list[float] | tuple[float, ...]) -> HestonParams | None:
    try:
        return HestonParams.from_array(params_arr)
    except (TypeError, ValueError):
        return None


def _market_iv_column(options_df: pd.DataFrame) -> str:
    if "iv" in options_df.columns:
        return "iv"
    if "impliedVolatility" in options_df.columns:
        return "impliedVolatility"
    raise ValueError("Options data must contain either iv or impliedVolatility.")


def prepare_calibration_data(
    options_df: pd.DataFrame,
    selection: Selection = "otm",
    moneyness_range: tuple[float, float] = (0.85, 1.15),
    min_price: float = 0.50,
    min_open_interest: int = 10,
    min_volume: float = 0.0,
    max_relative_spread: float | None = 0.15,
    t_bounds: tuple[float, float] | None = (14.0 / 365.0, 1.5),
    iv_bounds: tuple[float, float] | None = (0.05, 1.00),
    delta_abs_bounds: tuple[float, float] | None = (0.05, 0.40),
    require_positive_bid: bool = True,
    weight_scheme: WeightScheme = "expiry_option_type",
    max_options: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Prepare a clean option subset for Heston calibration without mutating input data."""

    df = add_option_features(options_df)
    iv_col = _market_iv_column(df)
    df["market_iv"] = pd.to_numeric(df[iv_col], errors="coerce")
    df["mid_price"] = df["mid"]
    df = df.replace([np.inf, -np.inf], np.nan)

    if selection == "otm_wide":
        selection = "otm"
        moneyness_range = (0.85, 1.15)

    if selection == "otm":
        df = df[df["is_otm"]]
    elif selection == "itm":
        df = df[~df["is_otm"]]
    elif selection == "calls_only":
        df = df[df["option_type"] == "call"]
    elif selection == "puts_only":
        df = df[df["option_type"] == "put"]
    elif selection == "all":
        pass
    else:
        raise ValueError(f"Unsupported selection: {selection}")

    low, high = moneyness_range
    bid_mask = df["bid"] > 0 if require_positive_bid else df["bid"] >= 0
    mask = (
        df["moneyness"].between(low, high)
        & (df["mid_price"] >= min_price)
        & bid_mask
        & (df["ask"] >= df["bid"])
        & (df["T"] > 0)
        & (df["market_iv"] > 0)
    )

    if t_bounds is not None:
        min_T, max_T = t_bounds
        mask &= df["T"].between(min_T, max_T)
    if iv_bounds is not None:
        min_iv, max_iv = iv_bounds
        mask &= df["market_iv"].between(min_iv, max_iv)
    if "openInterest" in df.columns:
        mask &= df["openInterest"].fillna(0) >= min_open_interest
    if "volume" in df.columns:
        mask &= df["volume"].fillna(0) >= min_volume

    calib = df[mask].copy()
    calib["relative_spread"] = (calib["ask"] - calib["bid"]) / calib["mid_price"]
    if max_relative_spread is not None:
        calib = calib[calib["relative_spread"] <= max_relative_spread].copy()

    calib["bs_delta"] = [
        bs_delta(
            S=float(row.S),
            K=float(row.strike),
            T=float(row.T),
            r=float(row.r),
            sigma=float(row.market_iv),
            option_type=str(row.option_type),
            q=float(row.q),
        )
        for row in calib.itertuples(index=False)
    ]
    calib["abs_delta"] = calib["bs_delta"].abs()
    if delta_abs_bounds is not None:
        min_delta, max_delta = delta_abs_bounds
        calib = calib[calib["abs_delta"].between(min_delta, max_delta)].copy()

    if max_options is not None and len(calib) > max_options:
        calib = calib.sample(n=max_options, random_state=random_state).copy()

    calib["bs_vega"] = [
        bs_vega(
            S=float(row.S),
            K=float(row.strike),
            T=float(row.T),
            r=float(row.r),
            sigma=float(row.market_iv),
            q=float(row.q),
        )
        for row in calib.itertuples(index=False)
    ]
    calib["bs_vega"] = calib["bs_vega"].clip(lower=1e-8)
    calib["weight"] = _calibration_weights(calib, weight_scheme)

    keep_first = [
        "ticker",
        "expiry",
        "option_type",
        "strike",
        "mid_price",
        "market_iv",
        "S",
        "T",
        "r",
        "q",
        "moneyness",
        "forward_moneyness",
        "is_otm",
        "relative_spread",
        "bs_delta",
        "abs_delta",
        "bs_vega",
        "weight",
    ]
    ordered = [col for col in keep_first if col in calib.columns]
    ordered += [col for col in calib.columns if col not in ordered]
    return calib[ordered].sort_values(["expiry", "strike", "option_type"]).reset_index(drop=True)


def _calibration_weights(calib: pd.DataFrame, weight_scheme: WeightScheme) -> pd.Series:
    """Return objective weights normalized to sum to the number of calibration rows."""

    if calib.empty:
        return pd.Series(dtype=float, index=calib.index)
    if weight_scheme == "equal":
        return pd.Series(1.0, index=calib.index)
    if weight_scheme == "vega":
        vega_sum = calib["bs_vega"].sum()
        if vega_sum > 0:
            return calib["bs_vega"] / vega_sum * len(calib)
        return pd.Series(1.0, index=calib.index)

    if weight_scheme == "expiry":
        group_cols = ["expiry"]
    elif weight_scheme == "option_type":
        group_cols = ["option_type"]
    elif weight_scheme == "expiry_option_type":
        group_cols = ["expiry", "option_type"]
    else:
        raise ValueError(f"Unsupported weight_scheme: {weight_scheme}")

    group_size = calib.groupby(group_cols, observed=False)["strike"].transform("size")
    n_groups = calib[group_cols].drop_duplicates().shape[0]
    weights = len(calib) / (n_groups * group_size)
    return weights / weights.mean()


def estimate_initial_params(calibration_data: pd.DataFrame) -> HestonParams:
    """Estimate a conservative data-driven starting point for Heston calibration."""

    if calibration_data.empty:
        raise ValueError("Cannot estimate initial params from an empty calibration dataset.")

    df = calibration_data.copy()
    df["atm_distance"] = (df["moneyness"] - 1.0).abs()

    short_T = df["T"].min()
    long_T = df["T"].max()
    short_atm = df[df["T"] == short_T].nsmallest(max(5, min(20, len(df))), "atm_distance")
    long_atm = df[df["T"] == long_T].nsmallest(max(5, min(20, len(df))), "atm_distance")

    v0 = float(np.nanmedian(short_atm["market_iv"]) ** 2)
    theta = float(np.nanmedian(long_atm["market_iv"]) ** 2)
    if not np.isfinite(theta) or theta <= 0:
        theta = float(np.nanmedian(df["market_iv"]) ** 2)

    kappa = 2.0
    mid_T = df["T"].median()
    mid_slice = df[df["T"].between(mid_T * 0.7, mid_T * 1.3)].copy()
    if len(mid_slice) < 20:
        mid_slice = df.copy()

    smile_width = float(np.nanstd(mid_slice["market_iv"]))
    sigma_v = float(np.clip(2.0 * smile_width * np.sqrt(kappa), 0.10, 2.0))

    rho = -0.70
    if len(mid_slice) >= 5 and mid_slice["moneyness"].nunique() >= 3:
        x = np.log(mid_slice["moneyness"].to_numpy(dtype=float))
        y = mid_slice["market_iv"].to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() >= 5 and np.nanstd(x[finite]) > 0:
            slope = np.polyfit(x[finite], y[finite], 1)[0]
            rho_estimate = slope * 2.0 * np.sqrt(max(v0, 1e-8)) / max(sigma_v, 1e-8)
            rho = float(np.clip(rho_estimate, -0.99, -0.10))

    bounds = heston_bounds()
    values = np.array([v0, kappa, theta, sigma_v, rho], dtype=float)
    clipped = np.array([np.clip(value, low, high) for value, (low, high) in zip(values, bounds)])
    return HestonParams.from_array(clipped)


def calibration_objective(
    params_arr: np.ndarray | list[float] | tuple[float, ...],
    market_data: pd.DataFrame,
    phi_grid: np.ndarray | None = None,
    loss_type: LossType = "relative_price",
    invalid_penalty: float = 1e10,
) -> float:
    """Scalar Heston calibration objective for scipy optimizers."""

    params = params_from_array(params_arr)
    if params is None:
        return invalid_penalty
    if market_data.empty:
        return invalid_penalty
    if phi_grid is None:
        phi_grid = default_phi_grid()
    if loss_type not in {"relative_price", "relative_price_rmse", "price_rmse", "iv_proxy_rmse"}:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    total_error = 0.0
    squared_errors = []
    weights = []
    failures = 0

    for row in market_data.itertuples(index=False):
        try:
            weight = _row_weight(row)
            model_price = heston_price_trapz(
                S=float(row.S),
                K=float(row.strike),
                T=float(row.T),
                r=float(row.r),
                params=params,
                option_type=str(row.option_type),
                q=float(row.q),
                phi_grid=phi_grid,
            )
            market_price = float(row.mid_price)
            price_error = model_price - market_price
            if loss_type == "relative_price":
                error = price_error / market_price
                total_error += weight * error**2
            elif loss_type == "relative_price_rmse":
                squared_errors.append((price_error / market_price) ** 2)
                weights.append(weight)
            elif loss_type == "iv_proxy_rmse":
                vega = _row_bs_vega(row)
                if not np.isfinite(vega) or vega <= 0:
                    failures += 1
                    squared_errors.append(1.0)
                    weights.append(weight)
                    continue
                squared_errors.append((price_error / vega) ** 2)
                weights.append(weight)
            else:
                squared_errors.append(price_error**2)
                weights.append(weight)
        except Exception:
            failures += 1
            if loss_type == "relative_price":
                total_error += 1.0
            else:
                squared_errors.append(1.0)
                weights.append(1.0)

    if failures > 0.10 * len(market_data):
        if loss_type == "relative_price":
            total_error += 1e6
        else:
            return invalid_penalty

    if loss_type == "relative_price":
        return float(total_error)
    if not squared_errors:
        return invalid_penalty
    return float(np.sqrt(np.average(squared_errors, weights=weights)))


def _row_weight(row) -> float:
    weight = getattr(row, "weight", 1.0)
    if np.isfinite(weight) and weight > 0:
        return float(weight)
    return 1.0


def _row_bs_vega(row) -> float:
    value = getattr(row, "bs_vega", np.nan)
    if np.isfinite(value) and value > 0:
        return float(value)

    market_iv = getattr(row, "market_iv", np.nan)
    if not np.isfinite(market_iv) or market_iv <= 0:
        return np.nan

    return bs_vega(
        S=float(row.S),
        K=float(row.strike),
        T=float(row.T),
        r=float(row.r),
        sigma=float(market_iv),
        q=float(row.q),
    )


def calibration_rmse(
    params_arr: np.ndarray | list[float] | tuple[float, ...],
    market_data: pd.DataFrame,
    phi_grid: np.ndarray | None = None,
) -> float:
    return calibration_objective(
        params_arr=params_arr,
        market_data=market_data,
        phi_grid=phi_grid,
        loss_type="price_rmse",
    )


def calibration_iv_proxy_rmse(
    params_arr: np.ndarray | list[float] | tuple[float, ...],
    market_data: pd.DataFrame,
    phi_grid: np.ndarray | None = None,
) -> float:
    return calibration_objective(
        params_arr=params_arr,
        market_data=market_data,
        phi_grid=phi_grid,
        loss_type="iv_proxy_rmse",
    )


def evaluate_heston_fit(
    market_data: pd.DataFrame,
    params: HestonParams,
    phi_grid: np.ndarray | None = None,
    include_model_iv: bool = True,
) -> pd.DataFrame:
    """Price calibration options and add price, relative, and IV error diagnostics."""

    if phi_grid is None:
        phi_grid = default_phi_grid()

    fit = market_data.copy()
    model_prices = []
    for row in fit.itertuples(index=False):
        model_prices.append(
            heston_price_trapz(
                S=float(row.S),
                K=float(row.strike),
                T=float(row.T),
                r=float(row.r),
                params=params,
                option_type=str(row.option_type),
                q=float(row.q),
                phi_grid=phi_grid,
            )
        )

    fit["model_price"] = model_prices
    fit["price_error"] = fit["model_price"] - fit["mid_price"]
    fit["relative_price_error"] = fit["price_error"] / fit["mid_price"]
    fit["iv_proxy_error"] = fit["price_error"] / fit["bs_vega"].clip(lower=1e-8)

    if include_model_iv:
        fit["model_iv"] = [
            implied_vol(
                market_price=float(row.model_price),
                S=float(row.S),
                K=float(row.strike),
                T=float(row.T),
                r=float(row.r),
                option_type=str(row.option_type),
                q=float(row.q),
            )
            for row in fit.itertuples(index=False)
        ]
        fit["iv_error"] = fit["model_iv"] - fit["market_iv"]

    return fit


def _rmse(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    return float(np.sqrt(np.mean(np.square(clean))))


def summarize_calibration_errors(fit_data: pd.DataFrame) -> pd.DataFrame:
    """Return headline diagnostics for a priced calibration dataset."""

    metrics = [
        {
            "metric": "price_RMSE",
            "value": _rmse(fit_data["price_error"]),
            "description": "RMSE of Heston price minus market mid price, in option price dollars.",
        },
        {
            "metric": "relative_price_RMSE",
            "value": _rmse(fit_data["relative_price_error"]),
            "description": "RMSE of relative price errors.",
        },
        {
            "metric": "IV_proxy_RMSE",
            "value": _rmse(fit_data["iv_proxy_error"]),
            "description": "RMSE of price error divided by Black-Scholes vega, in volatility points.",
        },
        {
            "metric": "price_bias",
            "value": float(fit_data["price_error"].mean()),
            "description": "Mean Heston price minus market mid price.",
        },
    ]
    if "iv_error" in fit_data.columns:
        metrics.extend(
            [
                {
                    "metric": "IV_RMSE",
                    "value": _rmse(fit_data["iv_error"]),
                    "description": "RMSE of model implied volatility minus market implied volatility.",
                },
                {
                    "metric": "IV_bias",
                    "value": float(fit_data["iv_error"].mean()),
                    "description": "Mean model implied volatility minus market implied volatility.",
                },
                {
                    "metric": "model_IV_success_rate",
                    "value": float(fit_data["model_iv"].notna().mean()),
                    "description": "Share of rows where model implied volatility was computed.",
                },
            ]
        )

    return pd.DataFrame(metrics)


def summarize_errors_by_expiry(fit_data: pd.DataFrame, error_col: str = "iv_error") -> pd.DataFrame:
    return _summarize_grouped_errors(fit_data, ["expiry"], error_col).sort_values("expiry").reset_index(drop=True)


def summarize_errors_by_option_type(fit_data: pd.DataFrame, error_col: str = "iv_error") -> pd.DataFrame:
    return _summarize_grouped_errors(fit_data, ["option_type"], error_col).sort_values("option_type").reset_index(drop=True)


def summarize_errors_by_moneyness_bucket(
    fit_data: pd.DataFrame,
    error_col: str = "iv_error",
    bins: tuple[float, ...] = (0.85, 0.92, 0.97, 1.03, 1.08, 1.15),
) -> pd.DataFrame:
    fit = fit_data.copy()
    fit["m_bucket"] = pd.cut(fit["moneyness"], bins=bins, include_lowest=True)
    return _summarize_grouped_errors(fit, ["m_bucket"], error_col).reset_index(drop=True)


def _summarize_grouped_errors(fit_data: pd.DataFrame, group_cols: list[str], error_col: str) -> pd.DataFrame:
    if error_col not in fit_data.columns:
        raise ValueError(f"{error_col} is not present in fit_data.")

    grouped = fit_data.groupby(group_cols, observed=False).agg(
        count=(error_col, "count"),
        bias=(error_col, "mean"),
        rmse=(error_col, _rmse),
        price_rmse=("price_error", _rmse),
        relative_price_rmse=("relative_price_error", _rmse),
        iv_proxy_rmse=("iv_proxy_error", _rmse),
    )
    return grouped.reset_index()


def run_calibration(
    market_data: pd.DataFrame,
    x0: HestonParams | np.ndarray | list[float] | None = None,
    bounds: tuple[tuple[float, float], ...] = DEFAULT_BOUNDS,
    phi_grid: np.ndarray | None = None,
    loss_type: LossType = "relative_price",
    maxiter_de: int = 30,
    popsize: int = 8,
    maxiter_nm: int = 250,
    seed: int = 42,
    polish_with_nelder_mead: bool = True,
) -> CalibrationResult:
    """Run Heston calibration. This can be slow; call explicitly from a notebook."""

    if phi_grid is None:
        phi_grid = default_phi_grid()
    if x0 is None:
        x0_arr = estimate_initial_params(market_data).to_array()
    elif isinstance(x0, HestonParams):
        x0_arr = x0.to_array()
    else:
        x0_arr = np.asarray(x0, dtype=float)

    start = perf_counter()
    de_result = differential_evolution(
        calibration_objective,
        bounds=bounds,
        args=(market_data, phi_grid, loss_type),
        seed=seed,
        maxiter=maxiter_de,
        popsize=popsize,
        tol=1e-6,
        mutation=(0.5, 1.5),
        recombination=0.8,
        x0=x0_arr,
        polish=False,
        updating="immediate",
    )

    best_result = de_result
    method = "differential_evolution"

    if polish_with_nelder_mead:
        nm_result = minimize(
            calibration_objective,
            x0=de_result.x,
            args=(market_data, phi_grid, loss_type),
            method="Nelder-Mead",
            options={"maxiter": maxiter_nm, "xatol": 1e-6, "fatol": 1e-8, "disp": False},
        )
        if nm_result.fun <= de_result.fun:
            best_result = nm_result
            method = "differential_evolution+nelder_mead"

    elapsed = perf_counter() - start
    params = HestonParams.from_array(best_result.x)
    price_rmse = calibration_rmse(params.to_array(), market_data, phi_grid)
    iv_proxy_rmse = calibration_iv_proxy_rmse(params.to_array(), market_data, phi_grid)
    return CalibrationResult(
        params=params,
        objective_value=float(best_result.fun),
        rmse=price_rmse,
        price_rmse=price_rmse,
        iv_proxy_rmse=iv_proxy_rmse,
        n_options=len(market_data),
        elapsed_seconds=elapsed,
        method=method,
        loss_type=loss_type,
        optimizer_result=best_result,
    )
