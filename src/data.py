"""Ken French portfolio & factor loaders for GKX-lite."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CACHE = Path(__file__).resolve().parents[1] / "data" / "cache"


def _ensure():
    CACHE.mkdir(parents=True, exist_ok=True)


def _download_ff_zip(url: str) -> str:
    r = requests.get(url, timeout=90)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith((".csv", ".txt")))
        return zf.read(name).decode("utf-8", errors="ignore")


def _parse_monthly_block(raw: str, n_value_cols: int) -> pd.DataFrame:
    """Parse first YYYYMM monthly block with n_value_cols after date."""
    lines = raw.splitlines()
    start = None
    for i, line in enumerate(lines):
        tok = line.strip().split(",")[0].strip()
        if tok.isdigit() and len(tok) == 6:
            start = i
            break
    if start is None:
        raise ValueError("Could not find YYYYMM start")
    end = len(lines)
    for i in range(start + 1, len(lines)):
        tok = lines[i].strip().split(",")[0].strip()
        if not tok or not tok.isdigit():
            end = i
            break
    body = "\n".join(lines[start:end])
    df = pd.read_csv(io.StringIO(body), header=None)
    ncols = min(1 + n_value_cols, df.shape[1])
    df = df.iloc[:, :ncols]
    df[0] = df[0].astype(str)
    df["date"] = pd.to_datetime(df[0], format="%Y%m") + pd.offsets.MonthEnd(0)
    for c in range(1, ncols):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_ff5(force: bool = False) -> pd.DataFrame:
    _ensure()
    out = CACHE / "ff5.csv"
    if out.exists() and not force:
        return pd.read_csv(out, parse_dates=["date"]).set_index("date")
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
    raw = _download_ff_zip(url)
    df = _parse_monthly_block(raw, 6)
    df.columns = ["yyyymm", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF", "date"]
    for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]:
        df[c] = df[c] / 100.0
    df = df.set_index("date").sort_index()
    df.to_csv(out)
    return df


def load_mom(force: bool = False) -> pd.Series:
    _ensure()
    out = CACHE / "mom.csv"
    if out.exists() and not force:
        return pd.read_csv(out, parse_dates=["date"]).set_index("date")["MOM"]
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"
    raw = _download_ff_zip(url)
    df = _parse_monthly_block(raw, 1)
    df.columns = ["yyyymm", "MOM", "date"]
    df["MOM"] = df["MOM"] / 100.0
    s = df.set_index("date")["MOM"].sort_index()
    s.to_frame().to_csv(out)
    return s


def load_25_size_bm(force: bool = False) -> pd.DataFrame:
    _ensure()
    out = CACHE / "ff25_size_bm.csv"
    if out.exists() and not force:
        return pd.read_csv(out, parse_dates=["date"]).set_index("date")
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/25_Portfolios_5x5_CSV.zip"
    raw = _download_ff_zip(url)
    df = _parse_monthly_block(raw, 25)
    names = ["yyyymm"] + [f"S{s}BM{b}" for s in range(1, 6) for b in range(1, 6)]
    df.columns = names[: df.shape[1] - 1] + ["date"]
    for c in df.columns:
        if c not in ("yyyymm", "date"):
            df[c] = df[c] / 100.0
    df = df.set_index("date").drop(columns=["yyyymm"]).sort_index()
    df = df.replace(-0.9999, np.nan)
    df.to_csv(out)
    return df


def load_10_momentum(force: bool = False) -> pd.DataFrame:
    _ensure()
    out = CACHE / "ff10_mom.csv"
    if out.exists() and not force:
        return pd.read_csv(out, parse_dates=["date"]).set_index("date")
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/10_Portfolios_Prior_12_2_CSV.zip"
    raw = _download_ff_zip(url)
    df = _parse_monthly_block(raw, 10)
    names = ["yyyymm"] + [f"MOM{i}" for i in range(1, 11)]
    df.columns = names[: df.shape[1] - 1] + ["date"]
    for c in df.columns:
        if c not in ("yyyymm", "date"):
            df[c] = df[c] / 100.0
    df = df.set_index("date").drop(columns=["yyyymm"]).sort_index()
    df = df.replace(-0.9999, np.nan)
    df.to_csv(out)
    return df


def portfolio_static_chars(ports: pd.DataFrame) -> pd.DataFrame:
    """Map Size-BM portfolio id → size/value quintiles; MOM decile → mom rank."""
    rows = []
    for col in ports.columns:
        if col.startswith("S") and "BM" in col:
            # S{s}BM{b}
            s = int(col[1])
            b = int(col.split("BM")[1])
            rows.append({"portfolio": col, "size_q": s, "value_q": b, "mom_q": np.nan, "family": "size_bm"})
        elif col.startswith("MOM"):
            q = int(col.replace("MOM", ""))
            rows.append({"portfolio": col, "size_q": np.nan, "value_q": np.nan, "mom_q": q, "family": "mom"})
    return pd.DataFrame(rows).set_index("portfolio")
