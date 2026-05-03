from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Iterable
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
import yfinance as yf
from scipy.interpolate import CubicSpline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_TICKER = "SPY"
RAW_OPTIONS_PATH = DATA_DIR / "spy_options_raw.csv"
ENRICHED_OPTIONS_PATH = DATA_DIR / "spy_options_enriched.csv"
TREASURY_REQUEST_TIMEOUT = 15

TREASURY_TENORS = {
    "1 Mo": 1 / 12,
    "1.5 Month": 1.5 / 12,
    "2 Mo": 2 / 12,
    "3 Mo": 3 / 12,
    "4 Mo": 4 / 12,
    "6 Mo": 6 / 12,
    "1 Yr": 1.0,
    "2 Yr": 2.0,
    "3 Yr": 3.0,
    "5 Yr": 5.0,
    "7 Yr": 7.0,
    "10 Yr": 10.0,
    "20 Yr": 20.0,
    "30 Yr": 30.0,
}


@dataclass(frozen=True)
class TreasuryCurve:
    """Daily U.S. Treasury par yield curve, with yields stored as decimals."""

    curve_date: date
    tenors: np.ndarray
    yields: np.ndarray

    def interpolate(self, time_to_expiry: float | Iterable[float]) -> float | np.ndarray:
        values = np.asarray(time_to_expiry, dtype=float)
        clipped = np.clip(values, self.tenors.min(), self.tenors.max())

        if len(self.tenors) >= 4:
            interpolator = CubicSpline(self.tenors, self.yields, bc_type="natural")
            result = interpolator(clipped)
        else:
            result = np.interp(clipped, self.tenors, self.yields)

        if np.isscalar(time_to_expiry):
            return float(result)
        return result


def _as_date(value: str | date | datetime | pd.Timestamp) -> date:
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()


def _ensure_parent(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _pick_evenly(items: list[str], count: int) -> list[str]:
    if count <= 0 or not items:
        return []
    if len(items) <= count:
        return items

    indices = np.linspace(0, len(items) - 1, count, dtype=int)
    return [items[i] for i in indices]


def _spot_from_history(history: pd.DataFrame) -> tuple[float, date]:
    if history.empty or "Close" not in history:
        raise ValueError("No spot price history returned by yfinance.")

    close = history["Close"].dropna()
    if close.empty:
        raise ValueError("No non-empty close prices returned by yfinance.")

    spot = float(close.iloc[-1])
    spot_date = close.index[-1].date()
    return spot, spot_date


def get_latest_spot(ticker: yf.Ticker) -> tuple[float, date]:
    """Return the latest available close and its trading date."""

    history = ticker.history(period="5d", auto_adjust=False)
    return _spot_from_history(history)


def get_spot_on_or_before(
    ticker: yf.Ticker,
    valuation_date: str | date | datetime | pd.Timestamp,
    lookback_days: int = 10,
) -> tuple[float, date]:
    """Return the latest underlying close on or before the valuation date."""

    target = _as_date(valuation_date)
    history = ticker.history(
        start=target - timedelta(days=lookback_days),
        end=target + timedelta(days=1),
        auto_adjust=False,
    )
    return _spot_from_history(history[history.index.date <= target])


def get_dividend_yield(
    ticker_symbol: str = DEFAULT_TICKER,
    as_of: str | date | datetime | pd.Timestamp | None = None,
    spot: float | None = None,
    method: str = "trailing",
) -> float:
    """Return a continuous dividend yield estimate for the underlying.

    The trailing method sums cash dividends paid in the previous 12 months,
    divides by the aligned spot price, and converts the resulting annual yield
    to a continuously compounded rate.
    """

    if method != "trailing":
        raise ValueError("Only method='trailing' is currently implemented.")

    ticker = yf.Ticker(ticker_symbol)
    target = _as_date(as_of) if as_of is not None else get_latest_spot(ticker)[1]
    aligned_spot = spot if spot is not None else get_spot_on_or_before(ticker, target)[0]
    if aligned_spot <= 0:
        raise ValueError("Spot price must be positive to compute dividend yield.")

    dividends = ticker.dividends
    if dividends.empty:
        return 0.0

    dividend_dates = pd.to_datetime(dividends.index).date
    start = target - timedelta(days=365)
    trailing_dividends = dividends[(dividend_dates > start) & (dividend_dates <= target)]
    annual_cash_dividend = float(trailing_dividends.sum())
    if annual_cash_dividend <= 0:
        return 0.0

    discrete_yield = annual_cash_dividend / aligned_spot
    return float(np.log1p(discrete_yield))


def select_expiration_dates(
    expiries: Iterable[str],
    valuation_date: str | date | datetime | pd.Timestamp,
    max_expiries: int | None = None,
    min_days: int = 7,
) -> list[str]:
    """Choose option expiries after the valuation date.

    By default all eligible expiries are returned. If max_expiries is set, an
    evenly spaced subset is returned to keep a maturity mix.
    """

    base_date = _as_date(valuation_date)
    short: list[str] = []
    mid: list[str] = []
    long: list[str] = []

    for expiry in sorted(expiries):
        days = (_as_date(expiry) - base_date).days
        if days < min_days:
            continue
        if days <= 45:
            short.append(expiry)
        elif days <= 180:
            mid.append(expiry)
        else:
            long.append(expiry)

    eligible = short + mid + long
    if max_expiries is None:
        selected = eligible
    else:
        selected = _pick_evenly(eligible, max_expiries)

    if not selected:
        raise ValueError("No option expirations matched the requested maturity buckets.")

    return selected


def fetch_options_data(
    ticker_symbol: str = DEFAULT_TICKER,
    output_path: str | Path = RAW_OPTIONS_PATH,
    valuation_date: str | date | datetime | pd.Timestamp | None = None,
    max_expiries: int | None = None,
    min_days: int = 7,
) -> pd.DataFrame:
    """
    Download selected full option chains from yfinance and save the raw dataset.

    The raw dataset keeps yfinance option-chain columns and adds enough metadata
    to process it later without downloading market data again.
    """

    ticker = yf.Ticker(ticker_symbol)
    if valuation_date is None:
        _, latest_spot_date = get_latest_spot(ticker)
        valuation_date = fetch_treasury_curve(
            latest_spot_date,
            allow_previous_date=True,
        ).curve_date

    spot, spot_date = get_spot_on_or_before(ticker, valuation_date)
    expiries = select_expiration_dates(
        ticker.options,
        valuation_date=spot_date,
        max_expiries=max_expiries,
        min_days=min_days,
    )

    downloaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    frames: list[pd.DataFrame] = []

    for expiry in expiries:
        chain = ticker.option_chain(expiry)
        days_to_expiry = (_as_date(expiry) - spot_date).days
        time_to_expiry = days_to_expiry / 365.0

        for option_type, chain_df in (("call", chain.calls), ("put", chain.puts)):
            if chain_df.empty:
                continue

            df = chain_df.copy()
            df["ticker"] = ticker_symbol
            df["option_type"] = option_type
            df["expiry"] = expiry
            df["days_to_expiry"] = days_to_expiry
            df["T"] = time_to_expiry
            df["S"] = spot
            df["spot_date"] = spot_date.isoformat()
            df["downloaded_at_utc"] = downloaded_at
            frames.append(df)

    if not frames:
        raise ValueError("No option-chain rows were returned by yfinance.")

    options_df = pd.concat(frames, ignore_index=True)

    preferred_cols = [
        "ticker",
        "expiry",
        "option_type",
        "strike",
        "lastPrice",
        "bid",
        "ask",
        "volume",
        "openInterest",
        "impliedVolatility",
        "days_to_expiry",
        "T",
        "S",
        "spot_date",
        "downloaded_at_utc",
        "contractSymbol",
    ]
    ordered_cols = [col for col in preferred_cols if col in options_df.columns]
    ordered_cols += [col for col in options_df.columns if col not in ordered_cols]
    options_df = options_df[ordered_cols]

    output_path = _ensure_parent(output_path)
    options_df.to_csv(output_path, index=False)
    return options_df


def treasury_csv_url(target_date: str | date | datetime | pd.Timestamp) -> str:
    target = _as_date(target_date)
    month = target.strftime("%Y%m")
    return (
        "https://home.treasury.gov/resource-center/data-chart-center/"
        "interest-rates/daily-treasury-rates.csv/all/"
        f"{month}?_format=csv&field_tdr_date_value_month={month}"
        "&page=&type=daily_treasury_yield_curve"
    )


def _read_treasury_csv(url: str) -> pd.DataFrame:
    try:
        with urlopen(url, timeout=TREASURY_REQUEST_TIMEOUT) as response:
            content = response.read().decode("utf-8")
    except (TimeoutError, URLError) as exc:
        raise ValueError(f"Could not fetch Treasury CSV: {url}") from exc

    if not content.strip():
        raise EmptyDataError("Treasury CSV response is empty.")

    return pd.read_csv(StringIO(content))


def fetch_treasury_curve(
    target_date: str | date | datetime | pd.Timestamp,
    allow_previous_date: bool = False,
    max_previous_months: int = 3,
) -> TreasuryCurve:
    """Fetch the U.S. Treasury par yield curve for the valuation date."""

    target = _as_date(target_date)
    monthly_frames: list[pd.DataFrame] = []

    for month_offset in range(max_previous_months + 1):
        month_date = pd.Timestamp(target).to_period("M") - month_offset
        lookup_date = month_date.to_timestamp().date()
        url = treasury_csv_url(lookup_date)
        try:
            monthly = _read_treasury_csv(url)
        except EmptyDataError:
            continue
        if monthly.empty:
            continue
        monthly["Date"] = pd.to_datetime(monthly["Date"]).dt.date
        monthly_frames.append(monthly)

        if not allow_previous_date:
            break
        if (monthly["Date"] <= target).any():
            break

    if not monthly_frames:
        raise ValueError(f"No Treasury yield-curve data found for {target:%Y-%m}.")

    treasury_df = pd.concat(monthly_frames, ignore_index=True)
    if allow_previous_date:
        candidates = treasury_df[treasury_df["Date"] <= target].sort_values("Date")
    else:
        candidates = treasury_df[treasury_df["Date"] == target]

    if candidates.empty:
        latest_available = treasury_df["Date"].max()
        raise ValueError(
            "Treasury curve is unavailable for "
            f"{target.isoformat()}. Latest date in fetched data: {latest_available}."
        )

    row = candidates.iloc[-1]
    available_columns = [col for col in TREASURY_TENORS if col in treasury_df.columns]
    yields_pct = pd.to_numeric(row[available_columns], errors="coerce")
    valid = yields_pct.notna()

    if valid.sum() < 2:
        raise ValueError(f"Not enough Treasury tenor data for {row['Date']}.")

    tenors = np.array([TREASURY_TENORS[col] for col in yields_pct.index[valid]], dtype=float)
    yields = yields_pct[valid].to_numpy(dtype=float) / 100.0

    sort_order = np.argsort(tenors)
    return TreasuryCurve(
        curve_date=row["Date"],
        tenors=tenors[sort_order],
        yields=yields[sort_order],
    )


def process_options_data(
    raw_path: str | Path = RAW_OPTIONS_PATH,
    output_path: str | Path = ENRICHED_OPTIONS_PATH,
    allow_previous_treasury_date: bool = False,
) -> pd.DataFrame:
    """Read the saved raw options dataset, add interpolated risk-free rates, and save."""

    raw_path = Path(raw_path)
    options_df = pd.read_csv(raw_path)
    if options_df.empty:
        raise ValueError(f"Raw options dataset is empty: {raw_path}")
    if "spot_date" not in options_df:
        raise ValueError("Raw options dataset must contain a spot_date column.")
    if "T" not in options_df:
        raise ValueError("Raw options dataset must contain a T column.")

    spot_dates = pd.to_datetime(options_df["spot_date"]).dt.date.unique()
    if len(spot_dates) != 1:
        raise ValueError(f"Expected one spot_date in raw data, found {len(spot_dates)}.")

    curve = fetch_treasury_curve(
        spot_dates[0],
        allow_previous_date=allow_previous_treasury_date,
    )
    ticker_symbol = (
        str(options_df["ticker"].iloc[0]) if "ticker" in options_df.columns else DEFAULT_TICKER
    )
    spot = float(options_df["S"].iloc[0])
    q = get_dividend_yield(ticker_symbol=ticker_symbol, as_of=spot_dates[0], spot=spot)

    enriched_df = options_df.copy()
    enriched_df["r"] = curve.interpolate(enriched_df["T"].to_numpy())
    enriched_df["q"] = q
    enriched_df["treasury_date"] = curve.curve_date.isoformat()

    output_path = _ensure_parent(output_path)
    enriched_df.to_csv(output_path, index=False)
    return enriched_df


def run_data_acquisition(
    ticker_symbol: str = DEFAULT_TICKER,
    raw_path: str | Path = RAW_OPTIONS_PATH,
    output_path: str | Path = ENRICHED_OPTIONS_PATH,
    max_expiries: int | None = None,
    allow_previous_treasury_date: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the two-stage data acquisition pipeline: raw download, then enrichment."""

    raw_df = fetch_options_data(
        ticker_symbol=ticker_symbol,
        output_path=raw_path,
        max_expiries=max_expiries,
    )
    enriched_df = process_options_data(
        raw_path=raw_path,
        output_path=output_path,
        allow_previous_treasury_date=allow_previous_treasury_date,
    )
    return raw_df, enriched_df


def summary_lines(options_df: pd.DataFrame) -> list[str]:
    ticker = options_df["ticker"].iloc[0] if "ticker" in options_df else DEFAULT_TICKER
    spot = float(options_df["S"].iloc[0])
    spot_date = options_df["spot_date"].iloc[0]
    treasury_date = options_df["treasury_date"].iloc[0] if "treasury_date" in options_df else "N/A"
    dividend_yield = options_df["q"].iloc[0] if "q" in options_df else np.nan

    lines = [
        "Options Data",
        f"Ticker: {ticker}",
        f"Total options: {len(options_df)}",
        f"Calls: {(options_df['option_type'] == 'call').sum()}",
        f"Puts: {(options_df['option_type'] == 'put').sum()}",
        f"Expirations: {options_df['expiry'].nunique()}",
        f"Strike range: ${options_df['strike'].min():.0f} - ${options_df['strike'].max():.0f}",
        f"T range: {options_df['T'].min():.4f} - {options_df['T'].max():.4f} years",
        "",
        "Per-expiry breakdown:",
        f"{'Expiry':<14} {'Calls':>5} {'Puts':>5} {'Total':>6}  {'T (years)':>10} {'r':>8}",
    ]

    for expiry in sorted(options_df["expiry"].unique()):
        sub = options_df[options_df["expiry"] == expiry]
        calls = (sub["option_type"] == "call").sum()
        puts = (sub["option_type"] == "put").sum()
        r_value = sub["r"].iloc[0] * 100 if "r" in sub else np.nan
        lines.append(
            f"{expiry:<14} {calls:>5} {puts:>5} {len(sub):>6}  "
            f"{sub['T'].iloc[0]:>10.4f} {r_value:>7.3f}%"
        )

    lines += [
        "",
        "Market Data",
        f"Spot price: ${spot:.2f}",
        f"Spot date: {spot_date}",
        f"Treasury date: {treasury_date}",
        f"Dividend yield q: {dividend_yield * 100:.3f}%",
        "Risk-free rate source: U.S. Treasury Daily Treasury Par Yield Curve Rates",
    ]
    return lines


def print_summary(options_df: pd.DataFrame) -> None:
    print("\n".join(summary_lines(options_df)))


if __name__ == "__main__":
    raw, enriched = run_data_acquisition()
    print(f"Raw dataset saved to {RAW_OPTIONS_PATH} ({len(raw)} rows)")
    print(f"Enriched dataset saved to {ENRICHED_OPTIONS_PATH} ({len(enriched)} rows)")
    print_summary(enriched)
