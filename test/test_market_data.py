from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.market_data import TreasuryCurve, get_dividend_yield, select_expiration_dates


def test_select_expiration_dates_returns_all_eligible_by_default():
    expiries = ["2026-05-01", "2026-05-10", "2026-06-20", "2027-01-17"]

    selected = select_expiration_dates(expiries, valuation_date="2026-05-01", min_days=7)

    assert selected == ["2026-05-10", "2026-06-20", "2027-01-17"]


def test_treasury_curve_interpolation_clips_boundaries():
    curve = TreasuryCurve(
        curve_date=date(2026, 4, 30),
        tenors=np.array([0.5, 1.0, 2.0]),
        yields=np.array([0.03, 0.04, 0.05]),
    )

    assert curve.interpolate(0.25) == pytest.approx(0.03)
    assert curve.interpolate(1.5) == pytest.approx(0.045)
    assert curve.interpolate(3.0) == pytest.approx(0.05)


def test_get_dividend_yield_trailing_with_mocked_yfinance(monkeypatch):
    class FakeTicker:
        dividends = pd.Series(
            [1.0, 1.2, 1.3],
            index=pd.to_datetime(["2025-06-01", "2025-12-01", "2026-03-01"]),
        )

        def __init__(self, ticker_symbol):
            self.ticker_symbol = ticker_symbol

    monkeypatch.setattr("src.market_data.yf.Ticker", FakeTicker)

    q = get_dividend_yield("SPY", as_of="2026-04-30", spot=100.0)

    assert q == pytest.approx(np.log1p(3.5 / 100.0))
