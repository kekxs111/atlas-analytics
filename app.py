import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px

st.set_page_config(page_title="ATLAS Analytics", page_icon="📈", layout="wide")
st.title("📈 ATLAS Analytics")
st.caption("Aktien, ETFs und Rohstoffe · Kursentwicklung und historisches Risiko")

st.warning(
    "ATLAS ist ein Analyse- und Lernwerkzeug, keine Anlageberatung. "
    "Marktdaten können verzögert sein oder fehlen. Vergangene Entwicklung sagt zukünftige Ergebnisse nicht zuverlässig voraus."
)

# Ausgewählte Wertpapiere/Marktindikatoren. Keine vollständige Abdeckung des Weltmarkts.
UNIVERSE = {
    "Aktien": {
        "USA": {
            "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "AMZN": "Amazon",
            "GOOGL": "Alphabet", "META": "Meta Platforms", "AVGO": "Broadcom",
            "LLY": "Eli Lilly", "COST": "Costco", "NFLX": "Netflix", "AMD": "AMD",
            "CRM": "Salesforce", "ORCL": "Oracle", "ADBE": "Adobe", "QCOM": "Qualcomm",
            "JPM": "JPMorgan Chase", "V": "Visa", "MA": "Mastercard", "CAT": "Caterpillar",
            "GE": "GE Aerospace", "UBER": "Uber", "PANW": "Palo Alto Networks",
            "PLTR": "Palantir", "TSLA": "Tesla", "KO": "Coca-Cola",
        },
        "Europa": {
            "ASML.AS": "ASML", "SAP.DE": "SAP", "SIE.DE": "Siemens",
            "ALV.DE": "Allianz", "DTE.DE": "Deutsche Telekom", "AIR.PA": "Airbus",
            "MC.PA": "LVMH", "OR.PA": "L'Oréal", "SU.PA": "Schneider Electric",
            "SAN.PA": "Sanofi", "RMS.PA": "Hermès", "TTE.PA": "TotalEnergies",
            "NESN.SW": "Nestlé", "NOVN.SW": "Novartis", "ROG.SW": "Roche",
            "AZN.L": "AstraZeneca", "SHEL.L": "Shell", "HSBA.L": "HSBC",
            "ULVR.L": "Unilever", "REL.L": "RELX", "BAS.DE": "BASF",
            "BMW.DE": "BMW", "MBG.DE": "Mercedes-Benz", "IFX.DE": "Infineon",
            "ENEL.MI": "Enel",
        },
        "Asien": {
            "7203.T": "Toyota", "6758.T": "Sony", "9984.T": "SoftBank Group",
            "6501.T": "Hitachi", "8035.T": "Tokyo Electron", "7974.T": "Nintendo",
            "6861.T": "Keyence", "9432.T": "NTT", "8306.T": "Mitsubishi UFJ",
            "9988.HK": "Alibaba", "0700.HK": "Tencent", "3690.HK": "Meituan",
            "9618.HK": "JD.com", "1810.HK": "Xiaomi", "1211.HK": "BYD",
            "0005.HK": "HSBC Holdings", "RELIANCE.NS": "Reliance Industries",
            "TCS.NS": "Tata Consultancy Services", "INFY.NS": "Infosys",
            "HDFCBANK.NS": "HDFC Bank", "ICICIBANK.NS": "ICICI Bank",
            "005930.KS": "Samsung Electronics", "000660.KS": "SK Hynix",
            "035420.KS": "NAVER", "9983.T": "Fast Retailing",
        },
    },
    "ETFs": {
        "Welt & Regionen": {
            "VWCE.DE": "Vanguard FTSE All-World UCITS ETF",
            "IWDA.AS": "iShares Core MSCI World UCITS ETF",
            "EIMI.L": "iShares Core MSCI Emerging Markets IMI ETF",
            "VUSA.AS": "Vanguard S&P 500 UCITS ETF",
            "EXSA.DE": "iShares STOXX Europe 600 UCITS ETF",
            "CSPX.L": "iShares Core S&P 500 UCITS ETF",
            "QQQ": "Invesco QQQ (USA)",
            "VT": "Vanguard Total World Stock ETF (USA)",
            "VTI": "Vanguard Total Stock Market ETF (USA)",
            "BND": "Vanguard Total Bond Market ETF (USA)",
        },
    },
    "Rohstoffe": {
        "Rohstoff-Indikatoren": {
            "GC=F": "Gold-Futures",
            "SI=F": "Silber-Futures",
            "CL=F": "WTI-Öl-Futures",
            "BZ=F": "Brent-Öl-Futures",
            "NG=F": "Erdgas-Futures",
            "HG=F": "Kupfer-Futures",
            "ZC=F": "Mais-Futures",
            "ZW=F": "Weizen-Futures",
        },
        "Rohstoff-ETCs": {
            "SGLD.L": "Invesco Physical Gold ETC",
            "PHAG.L": "WisdomTree Physical Silver ETC",
        },
    },
}

@st.cache_data(ttl=900, show_spinner=False)
def download_history(tickers, period="2y"):
    raw = yf.download(
        tickers=list(tickers), period=period, interval="1d",
        auto_adjust=True, progress=False, group_by="column", threads=True,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"].copy()
        elif "Close" in raw.columns.get_level_values(-1):
            close = raw.xs("Close", axis=1, level=-1).copy()
        else:
            return pd.DataFrame()
    else:
        if "Close" not in raw.columns:
            return pd.DataFrame()
        close = raw[["Close"]].copy()
        close.columns = [list(tickers)[0]]

    if isinstance(close, pd.Series):
        close = close.to_frame(name=list(tickers)[0])

    close.index = pd.to_datetime(close.index, errors="coerce")
    try:
        if close.index.tz is not None:
            close.index = close.index.tz_localize(None)
    except (AttributeError, TypeError):
        pass
    close = close.loc[~close.index.isna()]
    close = close[~close.index.duplicated(keep="last")].sort_index()
    return close.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")


def calculate_metrics(prices):
    rows = []
    for ticker in prices.columns:
        s = prices[ticker].dropna().sort_index()
        if len(s) < 60:
            continue
        last_date = s.index[-1]
        year_cutoff = last_date - pd.Timedelta(days=365)
        half_cutoff = last_date - pd.Timedelta(days=182)
        year = s[s.index >= year_cutoff]
        half = s[s.index >= half_cutoff]
        if len(year) < 40:
            continue

        last_price = float(s.iloc[-1])
        ret1y = (last_price / float(year.iloc[0]) - 1) * 100
        ret6m = (last_price / float(half.iloc[0]) - 1) * 100 if len(half) >= 20 else np.nan
        daily = s.pct_change().dropna()
        recent = daily[daily.index >= year_cutoff]
        vol = float(recent.std() * np.sqrt(252) * 100) if len(recent) > 20 else np.nan
        year_prices = s[s.index >= year_cutoff]
        dd = (year_prices / year_prices.cummax() - 1) * 100
        max_dd = float(dd.min()) if not dd.empty else np.nan
        ma50 = float(s.tail(50).mean()) if len(s) >= 50 else np.nan
        ma200 = float(s.tail(200).mean()) if len(s) >= 200 else np.nan
        above50 = last_price > ma50 if pd.notna(ma50) else False
        above200 = last_price > ma200 if pd.notna(ma200) else False

        score = np.clip(ret1y, -50, 50) * 0.5
        if pd.notna(ret6m):
            score += np.clip(ret6m, -40, 40) * 0.25
        score += 8 if above50 else 0
        score += 8 if above200 else 0
        if pd.notna(vol):
            score += max(0, 20 - vol * 0.25)
        if pd.notna(max_dd):
            score += max(0, 20 + max_dd * 0.35)
        if ret1y < 0:
            score -= 10

        if pd.isna(vol) or pd.isna(max_dd):
            risk = "Nicht ausreichend"
        elif vol >= 45 or max_dd <= -40:
            risk = "Hoch"
        elif vol >= 28 or max_dd <= -25:
            risk = "Mittel"
        else:
            risk = "Niedriger"

        rows.append({
            "Ticker": ticker, "Letzter Kurs": last_price,
            "Kursdatum": last_date.strftime("%d.%m.%Y"),
            "1J Rendite (%)": ret1y, "6M Rendite (%)": ret6m,
            "Volatilität 1J (%)": vol, "Max. Rückgang 1J (%)": max_dd,
            "Trend 50 Tage": "Ja" if above50 else "Nein",
            "Trend 200 Tage": "Ja" if above200 else "Nein",
            "Risiko": risk, "Score": score,
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("Score", ascending=False).reset_index(drop=True)
    return result


with st.sidebar:
    st.header("Einstellungen")
    asset_types = st.multiselect("Anlageklassen", list(UNIVERSE.keys()), default=["Aktien", "ETFs", "Rohstoffe"])
    selected_markets = {}
    for asset in asset_types:
        st.subheader(asset)
        market_names = list(UNIVERSE[asset].keys())
        chosen = st.multiselect(
            f"{asset}: Märkte/Listen",
            options=market_names,
            default=market_names,
            key=f"markets_{asset}",
        )
        selected_markets[asset] = chosen

tickers = []
names = {}
types_by_ticker = {}
for asset in asset_types:
    for market in selected_markets.get(asset, []):
        for ticker, name in UNIVERSE[asset][market].items():
            tickers.append(ticker)
            names[ticker] = name
            types_by_ticker[ticker] = asset
tickers = list(dict.fromkeys(tickers))

if not tickers:
    st.info("Wähle mindestens eine Anlageklasse und eine Liste aus.")
    st.stop()

with st.spinner("Lade Kursdaten von Yahoo Finance …"):
    prices = download_history(tuple(tickers))
    metrics = calculate_metrics(prices)

if prices.empty or metrics.empty:
    st.error("Es konnten keine ausreichenden Kursdaten geladen werden. Bitte später erneut versuchen.")
    st.stop()

metrics.insert(1, "Anlageklasse", metrics["Ticker"].map(types_by_ticker).fillna("Unbekannt"))
metrics.insert(2, "Name", metrics["Ticker"].map(names).fillna(metrics["Ticker"]))
last_dates = pd.to_datetime(prices.index).dropna()
if len(last_dates):
    st.caption(f"Letztes vorhandenes Kursdatum: {last_dates.max().strftime('%d.%m.%Y')} · Zwischenspeicherung: bis zu 15 Minuten")

m1, m2, m3 = st.columns(3)
m1.metric("Instrumente mit Daten", len(metrics))
m2.metric("Niedrigeres historisches Risiko", int((metrics["Risiko"] == "Niedriger").sum()))
m3.metric("Hohes historisches Risiko", int((metrics["Risiko"] == "Hoch").sum()))

st.subheader("Rangliste")
st.caption("Vergleicht die ausgewählten Instrumente nach historischem Kursmomentum, Trend und Risiko. Kein Kauf-/Verkaufssignal.")
cols = ["Anlageklasse", "Ticker", "Name", "Kursdatum", "1J Rendite (%)", "6M Rendite (%)",
        "Volatilität 1J (%)", "Max. Rückgang 1J (%)", "Risiko", "Score"]
shown = metrics[cols].copy()
for c in ["1J Rendite (%)", "6M Rendite (%)", "Volatilität 1J (%)", "Max. Rückgang 1J (%)", "Score"]:
    shown[c] = shown[c].round(1)
st.dataframe(shown, use_container_width=True, hide_index=True)

st.subheader("Kursverlauf")
options = list(prices.columns)
default = metrics.iloc[0]["Ticker"] if not metrics.empty else options[0]
selected = st.selectbox("Instrument auswählen", options,
                        index=options.index(default) if default in options else 0,
                        format_func=lambda t: f"{t} — {names.get(t, t)}")
s = prices[selected].dropna().sort_index()
if not s.empty:
    chart = pd.DataFrame({"Datum": pd.to_datetime(s.index), "Kurs": s.to_numpy()})
    fig = px.line(chart, x="Datum", y="Kurs",
                  title=f"{names.get(selected, selected)} – Kursverlauf",
                  markers=False)
    fig.update_xaxes(type="date", tickformat="%b\n%Y", dtick="M1",
                     tickangle=-35, ticklabelmode="period", automargin=True)
    fig.update_layout(xaxis_title="Monat", yaxis_title="Kurs (Börsenwährung)",
                      margin=dict(l=10, r=10, t=55, b=25), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Für dieses Instrument sind keine Kursdaten verfügbar.")

with st.expander("So funktionieren Score und Risiko"):
    st.markdown("""
- **1J-/6M-Rendite:** historische Preisveränderung.
- **Volatilität:** annualisierte Schwankung täglicher Renditen.
- **Max. Rückgang:** größter Rückgang vom vorherigen Hoch innerhalb des Jahres.
- **Trend:** letzter Kurs gegenüber dem Durchschnitt der letzten 50 und 200 Handelstage.
- **Risiko:** grobe Einstufung anhand historischer Volatilität und maximalem Rückgang; keine Vorhersage.
- **Score:** eine vereinfachte Ranglistenkennzahl. Er berücksichtigt keine Fundamentaldaten, Kosten, Steuern, Fondsgebühren oder die besondere Struktur von Futures.
- **Rohstoffe:** Futures können durch Rollkosten und Laufzeiten anders reagieren als der Spotpreis. Diese Indikatoren sind nicht direkt mit Aktien oder ETFs gleichzusetzen.
""")
st.caption("Datenquelle: Yahoo Finance via yfinance. Instrumentenliste ist eine Auswahl, keine vollständige Abdeckung aller Märkte.")
