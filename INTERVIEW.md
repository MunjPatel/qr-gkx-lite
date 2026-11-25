# Project notes — GKX-Lite

I ran a public-data Gu–Kelly–Xiu–style horse race on Ken French portfolios: kitchen-sink OLS vs Elastic Net vs LightGBM under purged walk-forward validation, then translated forecasts into cost-aware long–short books.

**What I found:** Elastic Net led on net long–short Sharpe versus full OLS, but the edge does not clear a strict multiple-testing discovery bar. LightGBM underperformed in this portfolio-level setup.
