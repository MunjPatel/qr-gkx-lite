from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # local shared/ for standalone + Streamlit Cloud
from shared.plot_theme import COLORS, apply_plotly_theme  # noqa: E402

st.set_page_config(page_title="GKX-Lite Horse Race", layout="wide")
st.markdown(
    f"<style>.stApp {{ background-color: {COLORS['bg']}; color: {COLORS['text']}; }}</style>",
    unsafe_allow_html=True,
)
st.title("GKX-Lite — Purged ML Horse Race")
out = ROOT / "outputs"
if not (out / "meta.json").exists():
    st.warning("Run `python 10_qr_gkx_lite/scripts/run_mvp.py` first.")
    st.stop()

meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
summary = pd.read_csv(out / "model_summary.csv")
folds = pd.read_csv(out / "fold_metrics.csv", parse_dates=["fold_start"])
ls = pd.read_csv(out / "long_short_returns.csv", parse_dates=["date"])

c1, c2, c3 = st.columns(3)
c1.metric("Best LS Sharpe model", str(meta["best_ls_sharpe_model"]))
c2.metric("EN − OLS Sharpe", f"{meta['elastic_net_minus_ols_sharpe']:.3f}")
c3.metric("EN mean OOS R²", f"{meta['elastic_net_mean_oos_r2']:.4f}")

tab1, tab2, tab3 = st.tabs(["Model comparison", "Fold IC / R²", "Memo"])
with tab1:
    fig = px.bar(summary, x="model", y="ls_sharpe_net", title="Long–short Sharpe (net 10 bps)")
    apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(summary, use_container_width=True)
    # Cumulative wealth by model
    curves = []
    for model, g in ls.groupby("model"):
        g = g.sort_values("date")
        curves.append(
            pd.DataFrame(
                {
                    "date": g["date"],
                    "model": model,
                    "wealth": (1 + g["net"]).cumprod(),
                }
            )
        )
    cum = pd.concat(curves, ignore_index=True)
    fig2 = px.line(cum, x="date", y="wealth", color="model", title="Cumulative net wealth")
    apply_plotly_theme(fig2)
    st.plotly_chart(fig2, use_container_width=True)
with tab2:
    fig = px.line(folds, x="fold_start", y="rank_ic", color="model", markers=True, title="Walk-forward rank IC")
    apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)
    fig = px.line(folds, x="fold_start", y="oos_r2", color="model", markers=True, title="Walk-forward OOS R²")
    apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)
with tab3:
    st.markdown((out / "RESEARCH_MEMO.md").read_text(encoding="utf-8"))
