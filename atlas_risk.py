"""ATLAS data layer – Finnhub (optional) + Yahoo Chart-API + Batch."""
from __future__ import annotations
from typing import Dict, Iterable, Optional, Tuple
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import streamlit as st

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None

from atlas_config import ASSET_CLASS, BOND_DURATION, ticker_currency

try:
    import yfinance as yf
except Exception:
    yf = None

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def _http_json(url: str, timeout: float = 12.0) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": _UA,
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _finnhub_token() -> str:
    try:
        from atlas_cloud import finnhub_key
        return finnhub_key()
    except Exception:
        pass
    try:
        k = st.secrets.get("FINNHUB_API_KEY") or st.secrets.get("FINNHUB_KEY")  # type: ignore
        if k:
            return str(k).strip()
    except Exception:
        pass
    return str(st.session_state.get("finnhub_key") or "").strip()


def _yahoo_chart(ticker: str, range_: str = "5d", interval: str = "1d") -> Optional[dict]:
    sym = urllib.request.quote(ticker, safe=".=-")
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
        f"?range={range_}&interval={interval}&includePrePost=false&events=div%2Csplit"
    )
    for attempt in range(3):
        try:
            data = _http_json(url)
            results = (data.get("chart") or {}).get("result") or []
            if results:
                return results[0]
        except Exception:
            pass
        time.sleep(0.2 * (attempt + 1))
    return None


def _price_from_chart(chart: dict) -> float:
    if not chart:
        return float("nan")
    meta = chart.get("meta") or {}
    for key in ("regularMarketPrice", "previousClose"):
        px = meta.get(key)
        if px is not None:
            try:
                v = float(px)
                if v == v and v > 0:
                    return v
            except (TypeError, ValueError):
                pass
    try:
        closes = chart["indicators"]["quote"][0]["close"]
        closes = [c for c in closes if c is not None]
        if closes:
            v = float(closes[-1])
            if v == v and v > 0:
                return v
    except Exception:
        pass
    return float("nan")


def _strip_tz(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.to_datetime(index)
    if getattr(idx, "tz", None) is not None:
        try:
            return idx.tz_convert(None)
        except Exception:
            try:
                return idx.tz_localize(None)
            except Exception:
                return pd.DatetimeIndex([
                    pd.Timestamp(x).replace(tzinfo=None) if getattr(x, "tzinfo", None) else x for x in idx
                ])
    return idx


@st.cache_data(ttl=900, show_spinner=False)
def _cached_last_price(ticker: str, finnhub_tok: str = "") -> float:
    if finnhub_tok:
        try:
            syms = [ticker]
            if "." in ticker:
                syms.append(ticker.split(".")[0])
            for s in syms:
                try:
                    url = f"https://finnhub.io/api/v1/quote?symbol={urllib.request.quote(s)}&token={finnhub_tok}"
                    data = _http_json(url, timeout=8.0)
                    px = data.get("c") or data.get("pc")
                    if px is not None:
                        v = float(px)
                        if v == v and v > 0:
                            return v
                except Exception:
                    continue
        except Exception:
            pass
    for range_ in ("5d", "1mo", "3mo"):
        chart = _yahoo_chart(ticker, range_=range_, interval="1d")
        px = _price_from_chart(chart) if chart else float("nan")
        if px == px and px > 0:
            return px
    if yf is not None:
        try:
            df = yf.Ticker(ticker).history(period="5d", interval="1d", auto_adjust=True, actions=False)
            if df is not None and not df.empty and "Close" in df.columns:
                s = df["Close"].dropna()
                if not s.empty:
                    return float(s.iloc[-1])
        except Exception:
            pass
    return float("nan")


def get_last_price(ticker: str) -> float:
    if "price_cache" not in st.session_state:
        st.session_state.price_cache = {}
    tok = _finnhub_token()
    px = float("nan")
    try:
        px = float(_cached_last_price(ticker, tok))
    except Exception:
        px = float("nan")
    if px == px and px > 0:
        st.session_state.price_cache[ticker] = px
        return px
    fallback = st.session_state.price_cache.get(ticker)
    if fallback is not None and fallback == fallback and fallback > 0:
        return float(fallback)
    return float("nan")


def prefetch_prices(tickers: Iterable[str], max_workers: int = 6) -> Dict[str, float]:
    if "price_cache" not in st.session_state:
        st.session_state.price_cache = {}
    uniq = [t for t in dict.fromkeys(tickers) if t]
    out: Dict[str, float] = {}
    tok = _finnhub_token()

    def _one(t: str) -> Tuple[str, float]:
        try:
            return t, float(_cached_last_price(t, tok))
        except Exception:
            return t, float("nan")

    if not uniq:
        return out
    with ThreadPoolExecutor(max_workers=min(max_workers, len(uniq))) as pool:
        futs = {pool.submit(_one, t): t for t in uniq}
        for fut in as_completed(futs):
            t, px = fut.result()
            if px == px and px > 0:
                st.session_state.price_cache[t] = px
                out[t] = px
            else:
                fb = st.session_state.price_cache.get(t)
                if fb is not None and fb == fb and fb > 0:
                    out[t] = float(fb)
    return out


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_history(ticker: str, period: str = "5y") -> pd.DataFrame:
    range_map = {
        "1mo": "1mo", "3mo": "3mo", "6mo": "6mo",
        "1y": "1y", "2y": "2y", "5y": "5y", "10y": "10y",
        "max": "max", "ytd": "ytd",
    }
    range_ = range_map.get(period, period if period in range_map.values() else "5y")
    chart = _yahoo_chart(ticker, range_=range_, interval="1d")
    if not chart and range_ != "1y":
        chart = _yahoo_chart(ticker, range_="1y", interval="1d")
    if not chart:
        if yf is not None:
            try:
                df = yf.Ticker(ticker).history(
                    period=period if period != "max" else "max",
                    interval="1d", auto_adjust=True, actions=False,
                )
                if df is not None and not df.empty and "Close" in df.columns:
                    df = df[["Close"]].dropna()
                    df.index = _strip_tz(df.index)
                    return df[~df.index.duplicated(keep="last")].sort_index()
            except Exception:
                pass
        return pd.DataFrame()
    try:
        ts = chart.get("timestamp") or []
        closes = (chart.get("indicators") or {}).get("quote", [{}])[0].get("close") or []
        rows = []
        for t, c in zip(ts, closes):
            if c is None:
                continue
            rows.append((pd.to_datetime(t, unit="s"), float(c)))
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["Date", "Close"]).set_index("Date")
        df.index = _strip_tz(df.index)
        return df[~df.index.duplicated(keep="last")].sort_index()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
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


@st.cache_data(ttl=1800, show_spinner=False)
def single_metrics(ticker: str, name: str, kind: str) -> Optional[dict]:
    d = fetch_history(ticker, "max")
    if d.empty or len(d) < 10:
        d = fetch_history(ticker, "5y")
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
    return pd.DataFrame({
        "Ticker": list(rets.columns),
        "Gewicht %": w * 100,
        "Marginal Risk": mcr,
        "Contribution to Risk": ccr,
        "Risk Contribution %": pct,
    }).sort_values("Risk Contribution %", ascending=False)


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


def holdings_to_weights(holdings: dict, prices: dict) -> Dict[str, float]:
    values = {}
    for t, h in holdings.items():
        qty = float(h.get("qty", 0) or 0)
        px = prices.get(t, float("nan"))
        if qty > 0 and px == px and px > 0:
            values[t] = qty * px
    total = sum(values.values())
    if total <= 0:
        return {}
    return {t: v / total for t, v in values.items()}
