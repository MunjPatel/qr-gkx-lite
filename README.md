# GKX-Lite — Purged ML Horse Race

**Role:** Quantitative Researcher  
**Sources:** Gu–Kelly–Xiu (2020); López de Prado AFML Ch.7; Two Sigma factor-premia forecasting; Harvey–Liu–Zhu (2016)

Public-data, **portfolio-level** analogue of GKX: OLS vs Elastic Net vs LightGBM on Ken French Size–BM + Momentum portfolios under purged walk-forward validation and cost-aware long–short translation.

## Reproduce

```bash
cd portfolio/projects
.venv\Scripts\activate
python 10_qr_gkx_lite/scripts/run_mvp.py
streamlit run 10_qr_gkx_lite/app/dashboard.py
```

Read `outputs/RESEARCH_MEMO.md` first. This is **not** a stock-level WRDS replication of GKX.



1. Push this repo to GitHub (already done if you cloned from MunjPatel/qr-gkx-lite).
2. Go to https://share.streamlit.io/ → **New app**
3. Select repo `MunjPatel/qr-gkx-lite`, branch `main`
4. Main file path: `app/dashboard.py`
5. Python version 3.11+ recommended → Deploy

The app reads artifacts under `outputs/`. Re-run `scripts/run_mvp.py` locally if you need to refresh them before pushing.

## Live dashboard

[https://qr-gkx-lite.streamlit.app/](https://qr-gkx-lite.streamlit.app/)

