# Research Memo — GKX-Lite: Purged ML Horse Race

## Hypothesis (Gu–Kelly–Xiu 2020)
Penalized / nonlinear models extract predictive structure that kitchen-sink OLS destroys out of sample.
We run a **public-data, portfolio-level** analogue on Ken French Size–BM and Momentum portfolios — not a stock-level GKX replication (no WRDS).

## Design
- **Cross-section:** 25 Size–BM + 10 Prior-return portfolios
- **Features at t → predict r_{t+1}:** lagged portfolio returns, static sort ranks, 36-month rolling FF5+MOM betas, lagged factor realizations
- **Models:** baseline OLS (lags + factors), full OLS, Elastic Net, LightGBM
- **Validation:** expanding walk-forward with 1-month purge + embargo (AFML Ch.7 spirit) — never random K-fold
- **Portfolio translation:** monthly long–short top/bottom 20% of predicted returns, 10 bps one-way costs

## Results (net of 10 bps)
| Model | Mean OOS R² | Rank IC | LS Sharpe | t-stat |
|-------|-------------|---------|-----------|--------|
| baseline_ols | -0.1009 | 0.0335 | 0.142 | 0.97 |
| ols_full | -0.1057 | 0.0442 | 0.238 | 1.63 |
| elastic_net | -0.1046 | 0.0442 | 0.289 | 1.97 |
| lgbm | -0.1852 | -0.0512 | -0.136 | -0.93 |
| elastic_net_sizebm_only | -0.1092 | 0.0305 | 0.266 | 1.82 |

## Interpretation
- Elastic Net − full OLS Sharpe: **0.050**
- Elastic Net − baseline Sharpe: **0.147**
- Best LS Sharpe model: **elastic_net**

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
