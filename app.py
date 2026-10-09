import os
from datetime import datetime
import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="ATLAS Analytics", page_icon="📈", layout="wide")

st.markdown("""
<style>
.stApp { background: #0b1220; color: #e7eef8; }
[data-testid="stMetric"] { background: #142238; padding: 16px; border-radius: 12px; }
div[data-testid="stDataFrame"] { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

st.title("📈 ATLAS Analytics")
st.caption("Marktdaten · Portfolio-Analyse · Risikoindikatoren")
st.info("Bildungs- und Analysewerkzeug. Daten können verzögert sein. Keine Anlageberatung.")

with st.sidebar:
    st.header("Einstellungen")
    api_key = st.text_input("Alpha Vantage API-Key", value=os.getenv("ALPHA_VANTAGE_API_KEY", ""), type="password")
    symbol = st.text_input("Ticker-Symbol", value="IBM").strip().upper()
    period = st.selectbox("Zeitraum", ["6 Monate", "1 Jahr", "2 Jahre", "5 Jahre"], index=1)
    st.caption("Beispiele: IBM, AAPL, MSFT, NVDA, SAP.DE (Verfügbarkeit abhängig vom Anbieter).")
    load = st.button("Kurse laden", type="primary", use_container_width=True)

period_map = {"6 Monate": 126, "1 Jahr": 252, "2 Jahre": 504, "5 Jahre": 1260}

@st.cache_data(ttl=900, show_spinner=False)
def fetch_daily(sym, key):
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": sym,
        "outputsize": "full",
        "apikey": key,
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    series_key = "Time Series (Daily)"
    if series_key not in payload:
        message = payload.get("Note") or payload.get("Information") or payload.get("Error Message") or str(payload)
        raise ValueError(message)
    df = pd.DataFrame.from_dict(payload[series_key], orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.rename(columns={
        "1. open": "Open", "2. high": "High", "3. low": "Low",
        "4. close": "Close", "5. adjusted close": "Adjusted Close",
        "6. volume": "Volume"
    })
    for col in ["Open", "High", "Low", "Close", "Adjusted Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def add_indicators(df):
    df = df.copy()
    price = df["Adjusted Close"] if "Adjusted Close" in df.columns else df["Close"]
    df["Price"] = price
    df["SMA20"] = price.rolling(20).mean()
    df["SMA50"] = price.rolling(50).mean()
    delta = price.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, float("nan"))
    df["RSI14"] = 100 - (100 / (1 + rs))
    df["Daily Return"] = price.pct_change()
    return df

if not api_key:
    st.markdown("### Starte mit echten historischen Börsendaten")
    st.write("1. Besorge dir einen API-Key beim Datenanbieter.\n2. Trage ihn links ein.\n3. Gib ein Börsensymbol ein und klicke auf **Kurse laden**.")
    st.markdown("[Alpha Vantage: API-Key und Dokumentation](https://www.alphavantage.co/support/#api-key)")
    st.warning("Der kostenlose Tarif hat Anfragegrenzen und US-Echtzeitdaten sind nicht automatisch kostenlos. Prüfe die aktuellen Tarif- und Nutzungsbedingungen.")
elif load:
    try:
        with st.spinner(f"Lade Marktdaten für {symbol} …"):
            raw = fetch_daily(symbol, api_key)
            df = add_indicators(raw).tail(period_map[period]).dropna(subset=["Price"])
        if df.empty:
            st.error("Keine Kursdaten erhalten. Prüfe das Symbol und die Verfügbarkeit beim Datenanbieter.")
            st.stop()

        latest = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else latest
        price_change = (latest["Price"] / previous["Price"] - 1) * 100 if previous["Price"] else 0
        returns = df["Daily Return"].dropna()
        total_return = (df["Price"].iloc[-1] / df["Price"].iloc[0] - 1) * 100
        volatility = returns.std() * (252 ** 0.5) * 100 if len(returns) > 1 else 0
        running_max = df["Price"].cummax()
        drawdown = (df["Price"] / running_max - 1) * 100
        max_drawdown = drawdown.min()

        st.caption(f"Symbol: {symbol} · Letzter verfügbarer Handelstag: {df.index[-1].date()} · Abruf: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Letzter Schlusskurs", f"{latest['Price']:.2f}", f"{price_change:+.2f}% zum Vortag")
        c2.metric(f"Rendite · {period}", f"{total_return:+.2f}%")
        c3.metric("Annualisierte Volatilität", f"{volatility:.2f}%")
        c4.metric("Max. Drawdown", f"{max_drawdown:.2f}%")

        st.subheader("Kursverlauf")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Price"], name="Kurs", line=dict(width=2)))
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA 20", line=dict(width=1.5)))
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA 50", line=dict(width=1.5)))
        fig.update_layout(template="plotly_dark", height=470, margin=dict(l=10,r=10,t=30,b=10), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        left, right = st.columns(2)
        with left:
            st.subheader("RSI (14 Tage)")
            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(x=df.index, y=df["RSI14"], name="RSI 14"))
            rsi_fig.add_hline(y=70, line_dash="dash")
            rsi_fig.add_hline(y=30, line_dash="dash")
            rsi_fig.update_layout(template="plotly_dark", height=300, yaxis_range=[0,100], margin=dict(l=10,r=10,t=20,b=10))
            st.plotly_chart(rsi_fig, use_container_width=True)
        with right:
            st.subheader("Drawdown")
            dd_fig = go.Figure()
            dd_fig.add_trace(go.Scatter(x=df.index, y=drawdown, fill="tozeroy", name="Drawdown"))
            dd_fig.update_layout(template="plotly_dark", height=300, margin=dict(l=10,r=10,t=20,b=10), yaxis_title="%")
            st.plotly_chart(dd_fig, use_container_width=True)

        st.subheader("Letzte Handelstage")
        display_cols = [c for c in ["Open", "High", "Low", "Close", "Adjusted Close", "Volume", "SMA20", "SMA50", "RSI14"] if c in df.columns]
        st.dataframe(df[display_cols].sort_index(ascending=False).head(20).round(3), use_container_width=True)

        csv = df.to_csv().encode("utf-8")
        st.download_button("Analyse-Daten als CSV herunterladen", csv, file_name=f"{symbol.lower()}_analysis.csv", mime="text/csv")
        st.caption("SMA = gleitender Durchschnitt. RSI ist ein technischer Momentum-Indikator. Volatilität und Drawdown beruhen auf historischen Daten und sagen die Zukunft nicht zuverlässig voraus.")
    except Exception as e:
        st.error(f"Die Daten konnten nicht geladen werden: {e}")
        st.write("Prüfe API-Key, Symbol, Internetverbindung und das Tageslimit des Datenanbieters.")
else:
    st.markdown("### Bereit zum Laden")
    st.write(f"Ausgewähltes Symbol: **{symbol or '—'}**")
    st.write("Klicke links auf **Kurse laden**, um historische Daten vom Anbieter abzurufen.")
