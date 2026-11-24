from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import load_10_momentum, load_25_size_bm, load_ff5, load_mom  # noqa: E402
from features import FEATURE_COLS, build_panel  # noqa: E402
from models import form_long_short, run_horse_race, walk_forward_predict  # noqa: E402


def main():
    out = ROOT / "outputs"
    out.mkdir(parents=True, exist_ok=True)

    panel_path = out / "panel.csv"
    if panel_path.exists():
        print("Loading cached panel...")
        panel = pd.read_csv(panel_path, parse_dates=["date"])
    else:
        print("Loading Ken French data...")
        ff5 = load_ff5()
        mom = load_mom()
        size_bm = load_25_size_bm()
        mom10 = load_10_momentum()

        print("Building panel (rolling betas — may take ~1–2 min)...")
        panel = build_panel(size_bm, mom10, ff5, mom, beta_window=36)
        # Restrict to post-1970 for denser factor coverage
        panel = panel.loc[panel["date"] >= "1970-01-01"].copy()
        panel.to_csv(panel_path, index=False)
    print(f"Panel rows: {len(panel):,}  portfolios: {panel['portfolio'].nunique()}  months: {panel['date'].nunique()}")

    print("Running horse race (OLS / Elastic Net / LGBM / baseline)...")
    race = run_horse_race(panel, cost_bps=10.0)
    race["summary"].to_csv(out / "model_summary.csv", index=False)
    race["folds"].to_csv(out / "fold_metrics.csv", index=False)
    race["long_short"].to_csv(out / "long_short_returns.csv", index=False)

    for label, pred in race["preds"].items():
        pred.to_csv(out / f"preds_{label}.csv", index=False)

    # Ablation: drop momentum-family portfolios
    print("Ablation: size-BM only (drop MOM deciles)...")
    panel_sb = panel.loc[panel["family"] == "size_bm"].copy()
    pred_en, folds_en = walk_forward_predict(panel_sb, "elastic_net", [c for c in FEATURE_COLS if c in panel_sb.columns])
    ls_sb = form_long_short(pred_en, cost_bps=10.0)
    abl = race["summary"].copy()
    net = ls_sb["net"]
    sd = float(net.std(ddof=1))
    mu = float(net.mean())
    r2s = np.array([f.oos_r2 for f in folds_en], dtype=float)
    ics = np.array([f.rank_ic for f in folds_en], dtype=float)
    abl_row = {
        "model": "elastic_net_sizebm_only",
        "mean_oos_r2": float(np.nanmean(r2s)) if np.isfinite(r2s).any() else float("nan"),
        "mean_rank_ic": float(np.nanmean(ics)) if np.isfinite(ics).any() else float("nan"),
        "ls_sharpe_net": (mu / sd) * np.sqrt(12) if sd > 0 else float("nan"),
        "ls_t_stat": mu / (sd / np.sqrt(len(net))) if sd > 0 else float("nan"),
        "ls_ann_mean": mu * 12,
        "avg_turnover": float(ls_sb["turnover"].mean()),
        "n_months": len(ls_sb),
    }
    abl = pd.concat([abl, pd.DataFrame([abl_row])], ignore_index=True)
    abl.to_csv(out / "model_summary.csv", index=False)
    ls_sb.assign(model="elastic_net_sizebm_only").reset_index().to_csv(
        out / "long_short_ablation_sizebm.csv", index=False
    )

    summary = abl.set_index("model")
    best = summary["ls_sharpe_net"].idxmax()
    en = summary.loc["elastic_net"]
    ols = summary.loc["ols_full"]
    base = summary.loc["baseline_ols"]

    meta = {
        "n_rows": int(len(panel)),
        "n_portfolios": int(panel["portfolio"].nunique()),
        "n_months": int(panel["date"].nunique()),
        "best_ls_sharpe_model": str(best),
        "elastic_net_minus_ols_sharpe": float(en["ls_sharpe_net"] - ols["ls_sharpe_net"]),
        "elastic_net_minus_baseline_sharpe": float(en["ls_sharpe_net"] - base["ls_sharpe_net"]),
        "elastic_net_mean_oos_r2": float(en["mean_oos_r2"]),
        "ols_full_mean_oos_r2": float(ols["mean_oos_r2"]),
        "hlz_note": "Judge long-short t-stats vs HLZ t>~3 after multiple tested models.",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    pd.Series(meta).to_csv(out / "summary.csv")

    memo = f"""# Research Memo — GKX-Lite: Purged ML Horse Race

## Hypothesis (Gu–Kelly–Xiu 2020)
Penalized / nonlinear models extract predictive structure that kitchen-sink OLS destroys out of sample.
We run a **public-data, portfolio-level** analogue on Ken French Size–BM and Momentum portfolios — not a stock-level GKX replication (no WRDS).

## Design
- **Cross-section:** 25 Size–BM + 10 Prior-return portfolios
- **Features at t → predict r_{{t+1}}:** lagged portfolio returns, static sort ranks, 36-month rolling FF5+MOM betas, lagged factor realizations
- **Models:** baseline OLS (lags + factors), full OLS, Elastic Net, LightGBM
- **Validation:** expanding walk-forward with 1-month purge + embargo (AFML Ch.7 spirit) — never random K-fold
- **Portfolio translation:** monthly long–short top/bottom 20% of predicted returns, 10 bps one-way costs

## Results (net of 10 bps)
| Model | Mean OOS R² | Rank IC | LS Sharpe | t-stat |
|-------|-------------|---------|-----------|--------|
"""
    for _, r in abl.iterrows():
        memo += (
            f"| {r['model']} | {r['mean_oos_r2']:.4f} | {r['mean_rank_ic']:.4f} | "
            f"{r['ls_sharpe_net']:.3f} | {r['ls_t_stat']:.2f} |\n"
        )

    memo += f"""
## Interpretation
- Elastic Net − full OLS Sharpe: **{meta['elastic_net_minus_ols_sharpe']:.3f}**
- Elastic Net − baseline Sharpe: **{meta['elastic_net_minus_baseline_sharpe']:.3f}**
- Best LS Sharpe model: **{meta['best_ls_sharpe_model']}**

Kitchen-sink OLS often posts weaker OOS R² / portfolio metrics than shrinkage (GKX lesson).
HLZ: with four model families tested, require t ≳ 3 before calling a “discovery.”

## Ablation
`elastic_net_sizebm_only` drops momentum-decile portfolios — shows how much the horse race leans on the momentum family.

## Limitations
- Portfolio-level proxies ≠ stock-level characteristic panel in Gu–Kelly–Xiu
- Elastic Net hyperparameters fixed (not inner purged CV) for MVP speed
- Betas estimated with rolling OLS; estimation error is real
- Public Ken French data only — no microstructure, no alternative data

## Sources
- Gu, Kelly & Xiu (2020), RFS — Empirical Asset Pricing via Machine Learning
- López de Prado (2018), AFML Ch.7 — Purged K-Fold & Embargo
- Duncombe, Nigro & Kay / Two Sigma (2019) — Forecasting Factor Returns
- Harvey, Liu & Zhu (2016) — multiple-testing hurdle
"""
    (out / "RESEARCH_MEMO.md").write_text(memo, encoding="utf-8")
    print("QR P3 GKX-lite complete ->", out)
    print(abl.to_string(index=False))


if __name__ == "__main__":
    main()
