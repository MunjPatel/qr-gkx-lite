"""Panel construction: lagged returns, rolling factor betas, static ranks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data import portfolio_static_chars


FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]


def _rolling_betas(port_excess: pd.Series, factors: pd.DataFrame, window: int = 36) -> pd.DataFrame:
    """Rolling OLS betas ending at t (inclusive) — no look-ahead into t+1."""
    cols = [c for c in FACTOR_COLS if c in factors.columns]
    y = port_excess.values.astype(float)
    X = factors[cols].values.astype(float)
    n, k = X.shape
    out = np.full((n, k), np.nan)
    # Prepend intercept column once per window via solve on [1|X]
    for i in range(window - 1, n):
        y_w = y[i - window + 1 : i + 1]
        X_w = X[i - window + 1 : i + 1]
        mask = np.isfinite(y_w) & np.isfinite(X_w).all(axis=1)
        if mask.sum() < window // 2:
            continue
        yy = y_w[mask]
        XX = np.column_stack([np.ones(mask.sum()), X_w[mask]])
        try:
            coef, _, rank, _ = np.linalg.lstsq(XX, yy, rcond=None)
            if rank >= k + 1:
                out[i] = coef[1:]
        except np.linalg.LinAlgError:
            continue
    return pd.DataFrame(out, index=port_excess.index, columns=[f"beta_{c}" for c in cols])


def build_panel(
    size_bm: pd.DataFrame,
    mom10: pd.DataFrame,
    ff5: pd.DataFrame,
    mom_fac: pd.Series,
    beta_window: int = 36,
) -> pd.DataFrame:
    """
    Cross-sectional panel indexed by (date, portfolio).
    Features at t predict ret_{t+1}.
    """
    factors = ff5.copy()
    factors["MOM"] = mom_fac.reindex(factors.index)
    factors = factors.dropna(subset=["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM", "RF"])

    ports = pd.concat([size_bm, mom10], axis=1)
    common = ports.index.intersection(factors.index)
    ports = ports.loc[common]
    factors = factors.loc[common]

    chars = portfolio_static_chars(ports)
    frames = []

    for col in ports.columns:
        excess = ports[col] - factors["RF"]
        betas = _rolling_betas(excess, factors, window=beta_window)
        ret_lag1 = ports[col].shift(1)
        ret_lag2 = ports[col].shift(2)
        ret_lag3 = ports[col].shift(3)
        # Target: next-month return (aligned so row at t has y = r_{t+1})
        y = ports[col].shift(-1)

        meta = chars.loc[col]
        part = pd.DataFrame(
            {
                "date": ports.index,
                "portfolio": col,
                "ret_next": y.values,
                "ret_lag1": ret_lag1.values,
                "ret_lag2": ret_lag2.values,
                "ret_lag3": ret_lag3.values,
                "size_q": meta["size_q"],
                "value_q": meta["value_q"],
                "mom_q": meta["mom_q"],
                "family": meta["family"],
            }
        )
        for c in betas.columns:
            part[c] = betas[c].values
        # Contemporaneous factor realizations must NOT enter as predictors of ret_next
        # (would be look-ahead if used for same-month). We only keep lagged factors.
        for fc in FACTOR_COLS:
            part[f"fac_lag_{fc}"] = factors[fc].shift(1).values
        frames.append(part)

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.dropna(subset=["ret_next", "ret_lag1"]).copy()
    # Fill static NaNs for cross-family (mom portfolios have no size/value)
    for c in ["size_q", "value_q", "mom_q"]:
        panel[c] = panel[c].fillna(0.0)
    return panel


FEATURE_COLS = [
    "ret_lag1",
    "ret_lag2",
    "ret_lag3",
    "size_q",
    "value_q",
    "mom_q",
    "beta_Mkt-RF",
    "beta_SMB",
    "beta_HML",
    "beta_RMW",
    "beta_CMA",
    "beta_MOM",
    "fac_lag_Mkt-RF",
    "fac_lag_SMB",
    "fac_lag_HML",
    "fac_lag_RMW",
    "fac_lag_CMA",
    "fac_lag_MOM",
]


BASELINE_COLS = ["ret_lag1", "fac_lag_Mkt-RF", "fac_lag_SMB", "fac_lag_HML", "fac_lag_RMW", "fac_lag_CMA", "fac_lag_MOM"]
