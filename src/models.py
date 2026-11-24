"""OLS / Elastic Net / optional LightGBM with purged walk-forward."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler

from features import BASELINE_COLS, FEATURE_COLS


@dataclass
class FoldResult:
    date: pd.Timestamp
    model: str
    oos_r2: float
    rank_ic: float
    n: int


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _rank_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 5:
        return float("nan")
    if np.nanstd(y_pred) < 1e-12 or np.nanstd(y_true) < 1e-12:
        return float("nan")
    a = pd.Series(y_true).rank()
    b = pd.Series(y_pred).rank()
    c = float(a.corr(b))
    return c if np.isfinite(c) else float("nan")


def purged_train_mask(
    dates: pd.Series,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    embargo_months: int = 1,
) -> np.ndarray:
    """
    AFML Ch.7-style purge: drop train rows whose label horizon overlaps the test window,
    plus an embargo after the test set.
    Labels are ret_next (1-month), so purge train dates in [test_start - 1M, test_end + embargo].
    Actually: a train observation at date t uses y = r_{t+1}. It overlaps test if t+1 falls in test.
    Simpler practical rule used here: exclude train dates in
      [test_start - purge, test_end + embargo] where purge=1 month.
    """
    purge_start = test_start - pd.DateOffset(months=1)
    embargo_end = test_end + pd.DateOffset(months=embargo_months)
    # Keep only dates strictly before purge_start (expanding history before test)
    return (dates < purge_start).values


def _fit_predict(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    scaler = StandardScaler()
    Xt = scaler.fit_transform(X_train)
    Xs = scaler.transform(X_test)
    if model_name == "ols":
        m = LinearRegression()
        m.fit(Xt, y_train)
        return m.predict(Xs)
    if model_name == "elastic_net":
        # Mild shrinkage: alpha=0.01 over-shrinks standardized monthly returns toward a flat cross-section
        m = ElasticNet(alpha=1e-4, l1_ratio=0.25, max_iter=8000, random_state=0)
        m.fit(Xt, y_train)
        return m.predict(Xs)
    if model_name == "lgbm":
        try:
            import lightgbm as lgb
        except ImportError:
            # Fallback: elastic net if lightgbm missing
            m = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000, random_state=0)
            m.fit(Xt, y_train)
            return m.predict(Xs)
        m = lgb.LGBMRegressor(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=0,
            verbosity=-1,
        )
        m.fit(Xt, y_train)
        return m.predict(Xs)
    raise ValueError(model_name)


def walk_forward_predict(
    panel: pd.DataFrame,
    model_name: str,
    feature_cols: list[str],
    min_train_months: int = 120,
    step_months: int = 12,
    embargo_months: int = 1,
) -> tuple[pd.DataFrame, list[FoldResult]]:
    """
    Expanding walk-forward: each year, train on purged history, predict next `step_months`.
    Returns prediction frame with columns date, portfolio, y_true, y_pred, model.
    """
    panel = panel.sort_values(["date", "portfolio"]).copy()
    dates = np.array(sorted(panel["date"].unique()))
    # Need enough history
    start_idx = None
    for i, d in enumerate(dates):
        if i >= min_train_months:
            start_idx = i
            break
    if start_idx is None:
        raise ValueError("Not enough history for walk-forward")

    preds = []
    folds: list[FoldResult] = []

    i = start_idx
    while i < len(dates):
        test_start = pd.Timestamp(dates[i])
        test_end_i = min(i + step_months - 1, len(dates) - 1)
        test_end = pd.Timestamp(dates[test_end_i])
        test_dates = set(dates[i : test_end_i + 1])

        train_mask = purged_train_mask(panel["date"], test_start, test_end, embargo_months)
        # Also require features present
        feat_ok = panel[feature_cols].notna().all(axis=1).values
        train = panel.loc[train_mask & feat_ok]
        test = panel.loc[panel["date"].isin(test_dates) & feat_ok]

        if len(train) < 500 or len(test) < 10:
            i += step_months
            continue

        X_tr = train[feature_cols].values.astype(float)
        y_tr = train["ret_next"].values.astype(float)
        X_te = test[feature_cols].values.astype(float)
        y_te = test["ret_next"].values.astype(float)

        y_hat = _fit_predict(model_name, X_tr, y_tr, X_te)

        part = test[["date", "portfolio", "ret_next", "family"]].copy()
        part["y_pred"] = y_hat
        part["model"] = model_name
        preds.append(part)

        # Cross-sectional IC averaged across test months
        ics = []
        for d, g in part.groupby("date"):
            ics.append(_rank_ic(g["ret_next"].values, g["y_pred"].values))
        ics_arr = np.asarray(ics, dtype=float)
        folds.append(
            FoldResult(
                date=test_start,
                model=model_name,
                oos_r2=_r2(y_te, y_hat),
                rank_ic=float(np.nanmean(ics_arr)) if np.isfinite(ics_arr).any() else float("nan"),
                n=len(test),
            )
        )
        i += step_months

    if not preds:
        raise ValueError(f"No folds produced for {model_name}")
    return pd.concat(preds, ignore_index=True), folds


def form_long_short(
    preds: pd.DataFrame,
    cost_bps: float = 10.0,
    top_frac: float = 0.2,
) -> pd.DataFrame:
    """Equal-weight long top / short bottom of predicted returns each month."""
    rows = []
    prev_w = {}
    for d, g in preds.groupby("date"):
        g = g.dropna(subset=["y_pred", "ret_next"])
        n = len(g)
        k = max(1, int(n * top_frac))
        g = g.sort_values("y_pred")
        short = g.head(k)
        long = g.tail(k)
        weights = {p: 0.0 for p in g["portfolio"]}
        for p in long["portfolio"]:
            weights[p] = 1.0 / k
        for p in short["portfolio"]:
            weights[p] = -1.0 / k
        gross = float(
            (long["ret_next"].mean() if len(long) else 0.0)
            - (short["ret_next"].mean() if len(short) else 0.0)
        )
        # Turnover vs previous weights
        all_p = set(weights) | set(prev_w)
        turnover = 0.5 * sum(abs(weights.get(p, 0.0) - prev_w.get(p, 0.0)) for p in all_p)
        cost = turnover * (cost_bps / 10000.0)
        rows.append(
            {
                "date": d,
                "gross": gross,
                "net": gross - cost,
                "turnover": turnover,
                "n": n,
            }
        )
        prev_w = weights
    return pd.DataFrame(rows).set_index("date").sort_index()


def run_horse_race(panel: pd.DataFrame, cost_bps: float = 10.0) -> dict:
    """OLS kitchen-sink vs Elastic Net vs LGBM vs FF5+MOM baseline features."""
    specs = [
        ("baseline_ols", BASELINE_COLS, "ols"),
        ("ols_full", FEATURE_COLS, "ols"),
        ("elastic_net", FEATURE_COLS, "elastic_net"),
        ("lgbm", FEATURE_COLS, "lgbm"),
    ]
    all_preds = {}
    fold_rows = []
    ls_rows = []
    summaries = []

    for label, cols, model in specs:
        # Drop cols that may be all-NaN
        use = [c for c in cols if c in panel.columns and panel[c].notna().any()]
        pred, folds = walk_forward_predict(panel, model, use)
        all_preds[label] = pred
        for f in folds:
            fold_rows.append(
                {
                    "fold_start": f.date,
                    "model": label,
                    "oos_r2": f.oos_r2,
                    "rank_ic": f.rank_ic,
                    "n": f.n,
                }
            )
        ls = form_long_short(pred, cost_bps=cost_bps)
        ls["model"] = label
        ls_rows.append(ls.reset_index())
        # Summary stats
        net = ls["net"]
        mu = float(net.mean())
        sd = float(net.std(ddof=1))
        sharpe = (mu / sd) * np.sqrt(12) if sd > 0 else float("nan")
        t_stat = mu / (sd / np.sqrt(len(net))) if sd > 0 else float("nan")
        r2s = np.array([f.oos_r2 for f in folds], dtype=float)
        ics = np.array([f.rank_ic for f in folds], dtype=float)
        summaries.append(
            {
                "model": label,
                "mean_oos_r2": float(np.nanmean(r2s)) if np.isfinite(r2s).any() else float("nan"),
                "mean_rank_ic": float(np.nanmean(ics)) if np.isfinite(ics).any() else float("nan"),
                "ls_sharpe_net": sharpe,
                "ls_t_stat": t_stat,
                "ls_ann_mean": mu * 12,
                "avg_turnover": float(ls["turnover"].mean()),
                "n_months": len(ls),
            }
        )

    return {
        "preds": all_preds,
        "folds": pd.DataFrame(fold_rows),
        "long_short": pd.concat(ls_rows, ignore_index=True),
        "summary": pd.DataFrame(summaries),
    }
