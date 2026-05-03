from __future__ import annotations

import numpy as np
import pandas as pd

from src.greeks_bs import analytical_greeks_bs
from src.heston.greeks_hs import numerical_greeks_heston
from src.heston.pricing import default_phi_grid, heston_price_trapz
from src.types import HestonParams
from src.utils import implied_vol


def heston_implied_vol(
    S: float,
    K: float,
    T: float,
    r: float,
    params: HestonParams,
    option_type: str,
    q: float,
    phi_grid: np.ndarray | None = None,
) -> float:
    """Convert a Heston model price into a Black-Scholes implied volatility."""

    if phi_grid is None:
        phi_grid = default_phi_grid()

    model_price = heston_price_trapz(
        S=S,
        K=K,
        T=T,
        r=r,
        params=params,
        option_type=option_type,
        q=q,
        phi_grid=phi_grid,
    )
    return implied_vol(
        market_price=model_price,
        S=S,
        K=K,
        T=T,
        r=r,
        option_type=option_type,
        q=q,
    )


def heston_smile_curve(
    S: float,
    T: float,
    r: float,
    q: float,
    params: HestonParams,
    strikes: np.ndarray,
    expiry: str | None = None,
    phi_grid: np.ndarray | None = None,
) -> pd.DataFrame:
    """Return Heston implied volatility across strikes for one maturity."""

    if phi_grid is None:
        phi_grid = default_phi_grid()
    forward = S * np.exp((r - q) * T)

    rows = []
    for K in strikes:
        option_type = "put" if K <= forward else "call"
        iv = heston_implied_vol(
            S=S,
            K=float(K),
            T=T,
            r=r,
            params=params,
            option_type=option_type,
            q=q,
            phi_grid=phi_grid,
        )
        rows.append(
            {
                "expiry": expiry,
                "strike": float(K),
                "moneyness": float(K / S),
                "forward_moneyness": float(K / forward),
                "option_type": option_type,
                "T": T,
                "S": S,
                "r": r,
                "q": q,
                "heston_iv": iv,
            }
        )

    return pd.DataFrame(rows)


def heston_smile_surface(
    market_data: pd.DataFrame,
    params: HestonParams,
    expiries: list[str] | None = None,
    moneyness_range: tuple[float, float] = (0.85, 1.15),
    n_strikes: int = 61,
    phi_grid: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build Heston smile curves for each selected market expiry."""

    if phi_grid is None:
        phi_grid = default_phi_grid()
    if expiries is None:
        expiries = sorted(market_data["expiry"].unique())

    curves = []
    for expiry in expiries:
        sub = market_data[market_data["expiry"] == expiry]
        if sub.empty:
            continue
        S = float(sub["S"].iloc[0])
        T = float(sub["T"].iloc[0])
        r = float(sub["r"].iloc[0])
        q = float(sub["q"].iloc[0])
        strikes = np.linspace(S * moneyness_range[0], S * moneyness_range[1], n_strikes)
        curves.append(heston_smile_curve(S, T, r, q, params, strikes, expiry=expiry, phi_grid=phi_grid))

    if not curves:
        return pd.DataFrame()
    return pd.concat(curves, ignore_index=True)


def select_representative_expiries(market_data: pd.DataFrame, n: int = 3) -> list[str]:
    """Select short/middle/long expiries by time to maturity."""

    expiries = market_data[["expiry", "T"]].drop_duplicates().sort_values("T").reset_index(drop=True)
    if len(expiries) <= n:
        return expiries["expiry"].tolist()
    indices = np.linspace(0, len(expiries) - 1, n, dtype=int)
    return expiries.iloc[indices]["expiry"].tolist()


def select_expiries_by_target_tenors(
    market_data: pd.DataFrame,
    target_tenors: tuple[float, ...] = (0.10, 0.50),
    include_longest: bool = True,
) -> list[str]:
    """Select expiries closest to explicit target tenors, optionally adding the longest expiry."""

    expiries = market_data[["expiry", "T"]].drop_duplicates().sort_values("T").reset_index(drop=True)
    if expiries.empty:
        return []

    selected: list[str] = []
    for target in target_tenors:
        idx = (expiries["T"] - target).abs().idxmin()
        expiry = str(expiries.loc[idx, "expiry"])
        if expiry not in selected:
            selected.append(expiry)

    if include_longest:
        expiry = str(expiries.iloc[-1]["expiry"])
        if expiry not in selected:
            selected.append(expiry)

    return selected


def _market_iv_column(market_data: pd.DataFrame) -> str:
    if "market_iv" in market_data.columns:
        return "market_iv"
    if "iv" in market_data.columns:
        return "iv"
    if "impliedVolatility" in market_data.columns:
        return "impliedVolatility"
    raise ValueError("market_data must contain market_iv, iv, or impliedVolatility.")


def _otm_mask(sub: pd.DataFrame, reference: str) -> pd.Series:
    if reference == "spot":
        return (
            ((sub["option_type"] == "call") & (sub["strike"] > sub["S"]))
            | ((sub["option_type"] == "put") & (sub["strike"] <= sub["S"]))
        )
    if reference == "forward":
        return (
            ((sub["option_type"] == "call") & (sub["strike"] >= sub["forward"]))
            | ((sub["option_type"] == "put") & (sub["strike"] <= sub["forward"]))
        )
    raise ValueError("otm_reference must be either 'spot' or 'forward'.")


def _grid_option_type(K: float, S: float, forward: float, reference: str) -> str:
    boundary = S if reference == "spot" else forward
    return "put" if K <= boundary else "call"


def greeks_comparison_on_grid(
    market_data: pd.DataFrame,
    params: HestonParams,
    expiries: list[str] | None = None,
    target_tenors: tuple[float, ...] = (0.10, 0.50),
    include_longest: bool = True,
    moneyness_range: tuple[float, float] = (0.88, 1.12),
    n_strikes: int = 61,
    phi_grid: np.ndarray | None = None,
    heston_vega_scale: str = "vol0",
    otm_reference: str = "spot",
) -> pd.DataFrame:
    """Compare BS and Heston Greeks on a common strike grid.

    Market IV is interpolated from the OTM side of the observed market smile, then used
    as the Black-Scholes volatility input on the same grid where Heston Greeks
    are calculated. This keeps the comparison paired strike-by-strike.
    """

    if phi_grid is None:
        phi_grid = default_phi_grid()
    if expiries is None:
        expiries = select_expiries_by_target_tenors(
            market_data,
            target_tenors=target_tenors,
            include_longest=include_longest,
        )

    iv_col = _market_iv_column(market_data)
    rows = []
    for expiry in expiries:
        sub = market_data[market_data["expiry"] == expiry].copy()
        if sub.empty:
            continue

        S = float(sub["S"].iloc[0])
        T = float(sub["T"].iloc[0])
        r = float(sub["r"].iloc[0])
        q = float(sub["q"].iloc[0])
        forward = float(S * np.exp((r - q) * T))

        source = sub[_otm_mask(sub, otm_reference)].copy()
        source = source.replace([np.inf, -np.inf], np.nan).dropna(subset=["moneyness", iv_col])
        source = source.sort_values("moneyness")
        if len(source) < 2:
            continue

        curve = source.groupby("moneyness", as_index=False)[iv_col].mean().sort_values("moneyness")
        x = curve["moneyness"].to_numpy(dtype=float)
        y = curve[iv_col].to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        if len(x) < 2:
            continue

        grid = np.linspace(moneyness_range[0], moneyness_range[1], n_strikes)
        grid = grid[(grid >= x.min()) & (grid <= x.max())]
        if len(grid) == 0:
            continue

        market_ivs = np.interp(grid, x, y)
        for moneyness, market_iv in zip(grid, market_ivs):
            K = float(S * moneyness)
            option_type = _grid_option_type(K, S, forward, otm_reference)
            bs = analytical_greeks_bs(
                S=S,
                K=K,
                T=T,
                r=r,
                sigma=float(market_iv),
                option_type=option_type,
                q=q,
            )
            hs = numerical_greeks_heston(
                S=S,
                K=K,
                T=T,
                r=r,
                params=params,
                option_type=option_type,
                q=q,
                phi_grid=phi_grid,
                vega_scale=heston_vega_scale,
            )
            rows.append(
                {
                    "expiry": expiry,
                    "strike": K,
                    "moneyness": float(moneyness),
                    "option_type": option_type,
                    "T": T,
                    "S": S,
                    "r": r,
                    "q": q,
                    "market_iv": float(market_iv),
                    "market_iv_source": f"interpolated_otm_{otm_reference}",
                    "bs_delta": bs.delta,
                    "bs_gamma": bs.gamma,
                    "bs_vega": bs.vega,
                    "bs_theta": bs.theta,
                    "heston_delta": hs.delta,
                    "heston_gamma": hs.gamma,
                    "heston_vega": hs.vega,
                    "heston_theta": hs.theta,
                }
            )

    return pd.DataFrame(rows)


def greeks_comparison_by_strike(
    market_data: pd.DataFrame,
    params: HestonParams,
    expiries: list[str] | None = None,
    moneyness_range: tuple[float, float] = (0.88, 1.12),
    phi_grid: np.ndarray | None = None,
    heston_vega_scale: str = "vol0",
) -> pd.DataFrame:
    """Calculate BS and Heston Delta/Vega across market strikes for selected expiries."""

    if phi_grid is None:
        phi_grid = default_phi_grid()
    if expiries is None:
        expiries = select_representative_expiries(market_data, n=3)

    iv_col = "market_iv" if "market_iv" in market_data.columns else "iv"
    rows = []
    for expiry in expiries:
        sub = market_data[
            (market_data["expiry"] == expiry)
            & market_data["moneyness"].between(moneyness_range[0], moneyness_range[1])
        ].copy()
        sub = sub.sort_values(["strike", "option_type"]).drop_duplicates(["strike", "option_type"])

        for row in sub.itertuples(index=False):
            market_iv = float(getattr(row, iv_col))
            bs = analytical_greeks_bs(
                S=float(row.S),
                K=float(row.strike),
                T=float(row.T),
                r=float(row.r),
                sigma=market_iv,
                option_type=str(row.option_type),
                q=float(row.q),
            )
            hs = numerical_greeks_heston(
                S=float(row.S),
                K=float(row.strike),
                T=float(row.T),
                r=float(row.r),
                params=params,
                option_type=str(row.option_type),
                q=float(row.q),
                phi_grid=phi_grid,
                vega_scale=heston_vega_scale,
            )
            rows.append(
                {
                    "expiry": expiry,
                    "strike": float(row.strike),
                    "moneyness": float(row.moneyness),
                    "option_type": str(row.option_type),
                    "T": float(row.T),
                    "market_iv": market_iv,
                    "bs_delta": bs.delta,
                    "bs_gamma": bs.gamma,
                    "bs_vega": bs.vega,
                    "bs_theta": bs.theta,
                    "heston_delta": hs.delta,
                    "heston_gamma": hs.gamma,
                    "heston_vega": hs.vega,
                    "heston_theta": hs.theta,
                }
            )

    return pd.DataFrame(rows)
