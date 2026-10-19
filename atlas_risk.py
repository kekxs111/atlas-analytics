"""ATLAS data layer and risk engine."""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None

from atlas_config import ASSET_CLASS, BOND_DURATION, ticker_currency

# DATA LAYER
@st.cache_data(ttl=900, show_spinner=False)
def fetch_history(ticker: str, period: str = "5y") -> pd.DataFrame:
    try:
        df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True, actions=False)
        if df is None or df.empty or "Close" not in df.columns:
            return pd.DataFrame()
        df = df[["Close"]].dropna()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df[~df.index.duplicated(keep="last")].sort_index()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=900, show_spinner=False)
def fetch_returns_matrix(tickers: Tuple[str, ...], period: str = "3y") -> pd.DataFrame:
    frames = {}
    for t in tickers:
        h = fetch_history(t, period)
        if not h.empty and len(h) > 20:
            frames[t] = h["Close"].pct_change().dropna()
    if not frames:
        return pd.DataFrame()
    df = pd.DataFrame(frames).dropna(how="all")
    min_count = max(1, int(len(frames) * 0.7))
    df = df.dropna(thresh=min_count)
    return df.fillna(0.0)

@st.cache_data(ttl=900, show_spinner=False)
def single_metrics(ticker: str, name: str, kind: str) -> Optional[dict]:
    d = fetch_history(ticker, "max")
    if d.empty or len(d) < 10:
        return None
    p = d["Close"].astype(float)
    r = p.pct_change().dropna()
    if r.empty:
        return None
    years = max((p.index[-1] - p.index[0]).days / 365.25, 1 / 365.25)
    total = p.iloc[-1] / p.iloc[0] - 1
    ann = (p.iloc[-1] / p.iloc[0]) ** (1 / years) - 1 if p.iloc[-1] > 0 and p.iloc[0] > 0 else np.nan
    vol = r.std(ddof=1) * np.sqrt(252)
    dd = (p / p.cummax() - 1).min()
    mom = p.iloc[-1] / p.iloc[-min(len(p), 253)] - 1
    sharpe = (r.mean() * 252) / (r.std(ddof=1) * np.sqrt(252)) if r.std(ddof=1) > 0 else np.nan
    score = (
        50
        + 28 * np.tanh((ann if np.isfinite(ann) else 0) * 2)
        + 12 * np.tanh((mom if np.isfinite(mom) else 0) * 2)
        - 18 * min(float(vol) if np.isfinite(vol) else 1, 1.5)
        - 10 * min(abs(float(dd)) if np.isfinite(dd) else 1, 1.0)
    )
    risk = "Niedrig" if vol < 0.15 and abs(dd) < 0.20 else "Mittel" if vol < 0.30 and abs(dd) < 0.40 else "Hoch"
    return {
        "Ticker": ticker, "Anlage": name, "Typ": kind, "Kurs": float(p.iloc[-1]),
        "Seit Beginn %": total * 100, "Rendite p.a. %": ann * 100,
        "Volatilität %": vol * 100, "Max. DD %": dd * 100, "1J Rendite %": mom * 100,
        "Sharpe": sharpe, "Score": score, "Risiko": risk,
        "Währung": ticker_currency(ticker), "Asset Class": ASSET_CLASS.get(ticker, "Other"),
        "Letztes Datum": p.index[-1].date().isoformat(),
    }

def portfolio_returns(weights: Dict[str, float], period: str = "3y") -> Tuple[pd.Series, pd.DataFrame]:
    tickers = tuple(weights.keys())
    rets = fetch_returns_matrix(tickers, period)
    if rets.empty:
        return pd.Series(dtype=float), rets
    w = np.array([weights.get(t, 0.0) for t in rets.columns])
    w = w / w.sum() if w.sum() > 0 else w
    port = (rets * w).sum(axis=1)
    return port, rets

def historical_var_cvar(returns: pd.Series, alpha: float = 0.05) -> Tuple[float, float]:
    if returns.empty or len(returns) < 20:
        return np.nan, np.nan
    sorted_r = np.sort(returns.values)
    idx = max(0, int(np.floor(alpha * len(sorted_r))) - 1)
    var = -sorted_r[idx]
    cvar = -sorted_r[: idx + 1].mean() if idx >= 0 else var
    return float(var), float(cvar)

def parametric_var(returns: pd.Series, alpha: float = 0.05) -> float:
    if returns.empty or returns.std() == 0:
        return np.nan
    mu, sigma = returns.mean(), returns.std(ddof=1)
    if scipy_stats is not None:
        z = scipy_stats.norm.ppf(alpha)
    else:
        z = -1.64485 if alpha == 0.05 else -2.326 if alpha == 0.01 else -1.2816
    return float(-(mu + z * sigma))

def component_risk(weights: Dict[str, float], rets: pd.DataFrame) -> pd.DataFrame:
    if rets.empty or len(rets.columns) < 2:
        return pd.DataFrame()
    w = np.array([weights.get(t, 0.0) for t in rets.columns])
    w = w / w.sum() if w.sum() > 0 else w
    cov = rets.cov().values * 252
    port_var = float(w @ cov @ w)
    port_vol = np.sqrt(port_var) if port_var > 0 else 0.0
    if port_vol == 0:
        return pd.DataFrame()
    mcr = (cov @ w) / port_vol
    ccr = w * mcr
    pct = ccr / port_vol * 100
    out = pd.DataFrame({
        "Ticker": list(rets.columns),
        "Gewicht %": w * 100,
        "Marginal Risk": mcr,
        "Contribution to Risk": ccr,
        "Risk Contribution %": pct,
    }).sort_values("Risk Contribution %", ascending=False)
    return out

def stress_test(weights: Dict[str, float], shocks: Dict[str, float]) -> dict:
    impact = 0.0
    details = []
    for t, w in weights.items():
        shock = shocks.get(t)
        if shock is None:
            ac = ASSET_CLASS.get(t, "Other")
            shock = shocks.get(ac, 0.0)
        pnl = w * shock
        impact += pnl
        details.append({"Ticker": t, "Gewicht": w, "Shock %": shock * 100, "Beitrag %": pnl * 100})
    return {"portfolio_impact": impact, "details": pd.DataFrame(details)}

def monte_carlo_portfolio(returns: pd.Series, n_sims: int = 5000, horizon_days: int = 252) -> pd.DataFrame:
    if returns.empty or len(returns) < 30:
        return pd.DataFrame()
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    rng = np.random.default_rng(42)
    shocks = rng.normal(mu, sigma, size=(n_sims, horizon_days))
    paths = np.cumprod(1 + shocks, axis=1)
    final = paths[:, -1] - 1
    return pd.DataFrame({"Final Return": final, "Path": range(n_sims)})

def fixed_income_impact(weights: Dict[str, float], rate_shock_bp: float = 100) -> dict:
    total_impact = 0.0
    rows = []
    for t, w in weights.items():
        dur = BOND_DURATION.get(t)
        if dur is None:
            continue
        delta_y = rate_shock_bp / 10000.0
        impact = -dur * delta_y * w
        total_impact += impact
        rows.append({
            "Ticker": t, "Gewicht %": w * 100, "Duration": dur,
            "Rate Shock (bp)": rate_shock_bp, "Impact %": impact * 100,
        })
    return {"total_impact": total_impact, "details": pd.DataFrame(rows)}
