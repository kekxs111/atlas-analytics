import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(page_title="ATLAS Analytics", page_icon="📈", layout="wide")
st.title("📈 ATLAS Analytics")
st.caption("Globaler Aktien-Scanner · 12-Monats-Horizont · Wachstum mit Risikokontrolle")
st.warning("Hinweis: Kostenlose Yahoo-Finance-Daten können verzögert, unvollständig oder zeitweise nicht verfügbar sein. Das Ranking ist eine historische Kennzahlen-Auswertung, keine Kaufempfehlung.")

# Broad, curated cross-market universe; symbols use Yahoo Finance conventions.
UNIVERSE = {
    "USA": ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","AVGO","ORCL","CRM","AMD","LLY","V","MA","COST","NFLX","ADBE","QCOM","TXN","INTU","NOW","PANW","UBER","CAT","GE","JPM"],
    "Europa": ["ASML.AS","SAP.DE","SIE.DE","ALV.DE","DTE.DE","AIR.PA","MC.PA","OR.PA","SU.PA","SAN.PA","RMS.PA","AZN.L","SHEL.L","REL.L","RR.L","NOVO-B.CO","DSV.CO","NOKIA.HE","NESN.SW","ROG.SW","UBSG.SW","ABI.BR","ADYEN.AS","PRX.AS","IBE.MC"],
    "Asien": ["7203.T","6758.T","9984.T","7974.T","6861.T","6501.T","8035.T","005930.KS","000660.KS","005380.KS","0700.HK","9988.HK","3690.HK","9618.HK","2318.HK","1211.HK","RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS","603259.SS","600519.SS","300750.SZ","601318.SS"]
}

@st.cache_data(ttl=900, show_spinner=False)
def fetch_data(symbols):
    # Pull roughly 1 year plus buffer for moving averages
    raw = yf.download(
        tickers=symbols,
        period="2y",
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=True
    )
    closes = {}
    for symbol in symbols:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                series = raw[(symbol, "Close")].dropna()
            else:
                series = raw["Close"].dropna()
            if len(series) >= 130:
                closes[symbol] = series
        except Exception:
            continue
    return pd.DataFrame(closes).sort_index()

def max_drawdown(prices):
    peak = prices.cummax()
    dd = prices / peak - 1
    return float(dd.min())

with st.sidebar:
    st.header("Einstellungen")
    markets = st.multiselect("Märkte", list(UNIVERSE.keys()), default=list(UNIVERSE.keys()))
    max_results = st.slider("Anzahl angezeigter Aktien", 5, 30, 15)
    refresh = st.button("Daten neu laden")
    if refresh:
        st.cache_data.clear()

symbols = []
for market in markets:
    symbols.extend(UNIVERSE[market])
symbols = list(dict.fromkeys(symbols))

if not symbols:
    st.info("Bitte mindestens einen Markt auswählen.")
    st.stop()

with st.spinner(f"Lade historische Tageskurse für bis zu {len(symbols)} Aktien …"):
    try:
        prices = fetch_data(tuple(symbols))
    except Exception as exc:
        st.error("Die kostenlose Datenquelle ist gerade nicht erreichbar. Bitte später erneut versuchen.")
        st.caption(f"Technischer Hinweis: {type(exc).__name__}")
        st.stop()

if prices.empty:
    st.error("Es wurden keine Kursdaten empfangen. Prüfe die Internetverbindung oder versuche es später erneut.")
    st.stop()

rows = []
for ticker in prices.columns:
    s = prices[ticker].dropna()
    if len(s) < 130:
        continue
    last = float(s.iloc[-1])
    one_year_ago = s.iloc[-min(252, len(s))]
    six_month_ago = s.iloc[-min(126, len(s))]
    ret_1y = last / float(one_year_ago) - 1
    ret_6m = last / float(six_month_ago) - 1
    daily = s.pct_change().dropna()
    vol = float(daily.tail(252).std() * np.sqrt(252))
    dd = max_drawdown(s.tail(252))
    sma50 = float(s.tail(50).mean())
    sma200 = float(s.tail(200).mean()) if len(s) >= 200 else np.nan
    trend = 1 if last > sma50 else 0
    trend_long = 1 if pd.notna(sma200) and last > sma200 else 0
    # Transparent score: momentum/trend balanced against volatility and drawdown.
    # Clamp component scores to avoid a single extreme return dominating the list.
    momentum_score = np.clip((ret_1y + 0.20) / 0.70, 0, 1) * 25
    sixm_score = np.clip((ret_6m + 0.10) / 0.40, 0, 1) * 15
    trend_score = (trend * 5) + (trend_long * 5)
    vol_score = np.clip(1 - (vol / 0.65), 0, 1) * 20
    dd_score = np.clip(1 - (abs(dd) / 0.60), 0, 1) * 20
    penalty = 10 if ret_1y < 0 else 0
    score = float(np.clip(momentum_score + sixm_score + trend_score + vol_score + dd_score - penalty, 0, 100))
    risk = "Hoch" if vol >= 0.45 or dd <= -0.40 else ("Mittel" if vol >= 0.28 or dd <= -0.25 else "Niedrig")
    reasons = []
    reasons.append("positiver 1J-Trend" if ret_1y > 0 else "negative 1J-Rendite")
    reasons.append("Kurs über 50-Tage-Linie" if trend else "Kurs unter 50-Tage-Linie")
    if vol >= .45 or dd <= -.40:
        reasons.append("hohe Schwankung/Rückgang")
    elif vol < .28 and dd > -.25:
        reasons.append("historisch moderateres Risiko")
    else:
        reasons.append("mittleres historisches Risiko")
    market = next((m for m, ts in UNIVERSE.items() if ticker in ts), "Andere")
    rows.append({
        "Rang": 0, "Aktie": ticker, "Markt": market, "Score / 100": round(score, 1),
        "1J Rendite": ret_1y, "6M Rendite": ret_6m, "Volatilität": vol,
        "Max. Rückgang": dd, "Risiko": risk, "Bewertungsgrund": "; ".join(reasons),
        "Letzter Kurs": last, "Kursdatum": s.index[-1].date().isoformat()
    })

if not rows:
    st.error("Für die ausgewählten Aktien liegen nicht genügend Kursdaten vor.")
    st.stop()

result = pd.DataFrame(rows).sort_values(["Score / 100", "1J Rendite"], ascending=False).reset_index(drop=True)
result["Rang"] = np.arange(1, len(result) + 1)
st.subheader("Automatisch sortierte Rangliste")
st.caption(f"{len(result)} Aktien mit verfügbaren Daten · Letztes Kursdatum in der Tabelle · Aktualisierung höchstens alle 15 Minuten")
view = result.head(max_results).copy()
for col in ["1J Rendite", "6M Rendite", "Volatilität", "Max. Rückgang"]:
    view[col] = view[col].map(lambda x: f"{x:.1%}")
view["Letzter Kurs"] = view["Letzter Kurs"].map(lambda x: f"{x:,.2f}")
st.dataframe(view, use_container_width=True, hide_index=True)

selected = st.selectbox("Aktie genauer ansehen", result["Aktie"].tolist())
series = prices[selected].dropna()
st.subheader(f"{selected} · Kursverlauf")
st.line_chart(series.tail(252))
row = result[result["Aktie"] == selected].iloc[0]
c1,c2,c3,c4 = st.columns(4)
c1.metric("1-Jahres-Rendite", f"{row['1J Rendite']:.1%}")
c2.metric("Volatilität (annualisiert)", f"{row['Volatilität']:.1%}")
c3.metric("Max. Rückgang", f"{row['Max. Rückgang']:.1%}")
c4.metric("Risiko", row["Risiko"])
st.write("**Bewertungsgrund:**", row["Bewertungsgrund"])
st.write("**Kursdatum:**", row["Kursdatum"])
st.download_button("Rangliste als CSV herunterladen", result.to_csv(index=False).encode("utf-8-sig"), "atlas_rangliste.csv", "text/csv")

with st.expander("Wie wird der Score berechnet?"):
    st.markdown("""
    - 1-Jahres-Kursentwicklung: bis zu 25 Punkte
    - 6-Monats-Kursentwicklung: bis zu 15 Punkte
    - Kurs über 50- und 200-Tage-Durchschnitt: bis zu 10 Punkte
    - Niedrigere historische Volatilität: bis zu 20 Punkte
    - Geringerer maximaler Rückgang im letzten Jahr: bis zu 20 Punkte
    - Abzug von 10 Punkten bei negativer 1-Jahres-Rendite

    **Einschränkung:** Diese kostenlose Version nutzt historische Kurse als Wachstums-/Trend-Proxys. Sie enthält keine verlässliche vollständige weltweite Abdeckung und bewertet Umsatz- und Gewinnwachstum nicht. Die angezeigten Aktien sind eine kuratierte Auswahl, keine vollständige Suche über alle Börsen.
    """)
