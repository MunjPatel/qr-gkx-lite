# Interview Narrative — GKX-Lite

## One-liner
I ran a Gu–Kelly–Xiu–style model horse race on public Ken French portfolios: kitchen-sink OLS vs Elastic Net vs LightGBM, with purged/embargoed walk-forward validation and turnover costs — showing why shrinkage beats naive high-dimensional OLS out of sample.

## What I would say in 90 seconds
1. **Hypothesis:** GKX show nonlinear/penalized ML recovers predictive content OLS loses when the feature set is large. I built a feasible public-data version at the *portfolio* level (no WRDS).
2. **Hygiene:** Features at month *t* only use information available at *t*; labels are *t+1* returns; walk-forward folds purge overlapping labels and add an embargo (AFML Ch.7).
3. **Translation:** Predictions become a long–short book with 10 bps costs — research that never hits a portfolio metric is incomplete.
4. **Honesty:** This is not stock-level GKX. Hyperparameters are fixed for the MVP. HLZ still applies when you race four models.

## Likely follow-ups
- Why not random K-fold? → Label overlap / leakage.
- Why portfolios not stocks? → Public reproducibility without CRSP/Compustat.
- What if Elastic Net loses? → Document it; regularization isn't magic if the feature set is weak.
- How would you harden this? → Inner purged CV for α/l1_ratio; expand characteristic sets; deflated Sharpe.
