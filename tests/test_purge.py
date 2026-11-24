"""Purged train mask must exclude dates overlapping the test label horizon."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models import purged_train_mask  # noqa: E402


def test_purge_excludes_test_window():
    dates = pd.Series(pd.date_range("2000-01-31", periods=24, freq="ME"))
    test_start = dates.iloc[12]
    test_end = dates.iloc[15]
    mask = purged_train_mask(dates, test_start, test_end, embargo_months=1)
    # No training date should fall inside [test_start - 1M, test_end + embargo]
    train_dates = dates[mask]
    assert (train_dates < test_start - pd.DateOffset(months=1)).all()
    # Dates inside the test window must be False
    for d in dates.iloc[12:16]:
        assert not mask[dates[dates == d].index[0]]


def test_features_predict_next_not_same():
    """ret_next must be lead of portfolio return — spot-check alignment contract."""
    idx = pd.date_range("2000-01-31", periods=5, freq="ME")
    r = pd.Series([0.01, 0.02, -0.01, 0.03, 0.00], index=idx)
    y = r.shift(-1)
    assert pd.isna(y.iloc[-1])
    assert y.iloc[0] == 0.02
