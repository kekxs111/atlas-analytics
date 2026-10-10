import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from datetime import date, timedelta

st.set_page_config(page_title="ATLAS Analytics", page_icon="📈", layout="wide")

st.title("📈 ATLAS Analytics")
st.caption("Internationaler Aktien-Scanner · Fokus auf Kursentwicklung und Risiko über etwa ein Jahr")
st.warning(
    "Hinweis: ATLAS ist ein Analyse- und Lernwerkzeug, keine Anlageberatung. "
    "Kursdaten können verzögert sein oder vorübergehend fehlen. Vergangene Kursentwicklung "
    "sagt die zukünftige Entwicklung nicht zuverlässig voraus."
)

# Eine ausgewählte Aktienliste – keine vollständige Erfassung aller Börsen weltweit.
MARKETS = {
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
}

@st.cache_data(ttl=900, show_spinner=False)
def download_history(tickers, period="2y"):
    raw = yf.download(
        tickers=list(tickers),
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()

    # Handle yfinance's different column layouts for one or multiple tickers.
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
    # Daily market data is normally indexed by trading date. Remove timezone without
    # shifting the calendar date, then sort chronologically.
    try:
        if close.index.tz is not None:
            close.index = close.index.tz_localize(None)
    except (AttributeError, TypeError):
        pass
    close = close.loc[~close.index.isna()]
    close = close[~close.index.duplicated(keep="last")].sort_index()
    close = close.apply(pd.to_numeric, errors="coerce")
    return close.dropna(axis=1, how="all")


def calc_metrics(prices):
    rows = []
    if prices.empty:
        return pd.DataFrame()

    for ticker in prices.columns:
        series = prices[ticker].dropna().sort_index()
        if len(series) < 60:
            continue
        last_date = series.index[-1]
        last_price = float(series.iloc[-1])
        year_cutoff = last_date - pd.Timedelta(days=365)
        half_cutoff = last_date - pd.Timedelta(days=182)
        year_series = series.loc[series.index >= year_cutoff]
        half_series = series.loc[series.index >= half_cutoff]

        if len(year_series) < 40:
            continue

        one_year_return = (last_price / float(year_series.iloc[0]) - 1) * 100
        six_month_return = (
            (last_price / float(half_series.iloc[0]) - 1) * 100
            if len(half_series) >= 20 else np.nan
        )

        daily_returns = series.pct_change().dropna()
        recent_returns = daily_returns.loc[daily_returns.index >= year_cutoff]
        volatility = float(recent_returns.std() * np.sqrt(252) * 100) if len(recent_returns) > 20 else np.nan

        year_prices = series.loc[series.index >= year_cutoff]
        running_peak = year_prices.cummax()
        drawdowns = (year_prices / running_peak - 1) * 100
        max_drawdown = float(drawdowns.min()) if not drawdowns.empty else np.nan

        ma50 = float(series.tail(50).mean()) if len(series) >= 50 else np.nan
        ma200 = float(series.tail(200).mean()) if len(series) >= 200 else np.nan
        above_ma50 = last_price > ma50 if pd.notna(ma50) else False
        above_ma200 = last_price > ma200 if pd.notna(ma200) else False

        # Transparent screening score, not a prediction or recommendation.
        score = 0.0
        score += np.clip(one_year_return, -50, 50) * 0.5
        if pd.notna(six_month_return):
            score += np.clip(six_month_return, -40, 40) * 0.25
        score += 8 if above_ma50 else 0
        score += 8 if above_ma200 else 0
        if pd.notna(volatility):
            score += max(0, 20 - volatility * 0.25)
        if pd.notna(max_drawdown):
            score += max(0, 20 + max_drawdown * 0.35)
        if one_year_return < 0:
            score -= 10

        if pd.isna(volatility) or pd.isna(max_drawdown):
            risk = "Nicht ausreichend"
        elif volatility >= 45 or max_drawdown <= -40:
            risk = "Hoch"
        elif volatility >= 28 or max_drawdown <= -25:
            risk = "Mittel"
        else:
            risk = "Niedriger"

        rows.append({
            "Ticker": ticker,
            "Letzter Kurs": last_price,
            "Kursdatum": pd.Timestamp(last_date).strftime("%d.%m.%Y"),
            "1J Rendite (%)": one_year_return,
            "6M Rendite (%)": six_month_return,
            "Volatilität 1J (%)": volatility,
            "Max. Rückgang 1J (%)": max_drawdown,
            "Über 50-Tage-Linie": "Ja" if above_ma50 else "Nein",
            "Über 200-Tage-Linie": "Ja" if above_ma200 else "Nein",
            "Risiko": risk,
            "Score": score,
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("Score", ascending=False).reset_index(drop=True)
    return result


with st.sidebar:
    st.header("Einstellungen")
    selected_markets = st.multiselect(
        "Märkte auswählen",
        options=list(MARKETS.keys()),
        default=list(MARKETS.keys()),
    )
    selected_tickers = []
    for market in selected_markets:
        selected_tickers.extend(MARKETS[market].keys())
    st.caption(f"{len(selected_tickers)} Aktien in der ausgewählten Liste")
    if st.button("Daten neu laden"):
        st.cache_data.clear()
        st.rerun()

if not selected_tickers:
    st.info("Bitte wähle mindestens einen Markt in der Seitenleiste aus.")
    st.stop()

with st.spinner("Lade Kursdaten von Yahoo Finance …"):
    prices = download_history(tuple(selected_tickers))
    metrics = calc_metrics(prices)

if prices.empty or metrics.empty:
    st.error("Es konnten keine ausreichenden Kursdaten geladen werden. Bitte später erneut versuchen.")
    st.stop()

# Attach readable company names.
name_map = {}
for market in selected_markets:
    name_map.update(MARKETS[market])
metrics.insert(1, "Unternehmen", metrics["Ticker"].map(name_map).fillna(metrics["Ticker"]))

latest_dates = pd.to_datetime(prices.index).dropna()
if len(latest_dates):
    st.caption(f"Letztes vorhandenes Kursdatum in den geladenen Daten: {latest_dates.max().strftime('%d.%m.%Y')} · Daten werden bis zu 15 Minuten zwischengespeichert.")

c1, c2, c3 = st.columns(3)
c1.metric("Aktien mit ausreichenden Daten", len(metrics))
c2.metric("Niedrigeres Risiko", int((metrics["Risiko"] == "Niedriger").sum()))
c3.metric("Hohes Risiko", int((metrics["Risiko"] == "Hoch").sum()))

st.subheader("Aktien-Rangliste")
st.caption("Der Score sortiert die ausgewählten Aktien nach Kursmomentum, Trend und historischen Risikokennzahlen. Er ist keine Prognose.")
display_cols = [
    "Ticker", "Unternehmen", "Kursdatum", "1J Rendite (%)", "6M Rendite (%)",
    "Volatilität 1J (%)", "Max. Rückgang 1J (%)", "Risiko", "Score"
]
table = metrics[display_cols].copy()
for col in ["1J Rendite (%)", "6M Rendite (%)", "Volatilität 1J (%)", "Max. Rückgang 1J (%)", "Score"]:
    table[col] = table[col].round(1)
st.dataframe(table, use_container_width=True, hide_index=True)

st.subheader("Kursverlauf")
available_tickers = list(prices.columns)
default_ticker = metrics.iloc[0]["Ticker"] if not metrics.empty else available_tickers[0]
chosen_ticker = st.selectbox(
    "Aktie auswählen",
    options=available_tickers,
    index=available_tickers.index(default_ticker) if default_ticker in available_tickers else 0,
    format_func=lambda t: f"{t} — {name_map.get(t, t)}",
)

series = prices[chosen_ticker].dropna().sort_index()
if not series.empty:
    chart_df = pd.DataFrame({
        # Keep actual pandas timestamps; converting to Python .date objects can
        # cause date values to be treated as categories in some chart setups.
        "Datum": pd.to_datetime(series.index),
        "Kurs": series.to_numpy(),
    }).sort_values("Datum")
    # Use a real continuous date axis with monthly ticks in chronological order.
    fig = px.line(
        chart_df,
        x="Datum",
        y="Kurs",
        title=f"{name_map.get(chosen_ticker, chosen_ticker)} – täglicher Schlusskurs (bereinigt)",
        markers=False,
    )
    fig.update_xaxes(
        type="date",
        tickformat="%b %Y",
        dtick="M1",
        tick0=chart_df["Datum"].min().strftime("%Y-%m-01"),
        tickangle=-35,
        ticklabelmode="instant",
        automargin=True,
    )
    fig.update_layout(
        xaxis_title="Monat",
        yaxis_title="Kurs (Währung der Börse)",
        margin=dict(l=10, r=10, t=55, b=25),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Für diese Aktie ist kein Kursverlauf verfügbar.")

with st.expander("Wie werden Risiko und Score berechnet?"):
    st.markdown(
        """
        - **1J-Rendite / 6M-Rendite:** Veränderung gegenüber dem ersten verfügbaren Handelstag im jeweiligen Zeitraum.
        - **Volatilität 1J:** annualisierte Schwankung der täglichen Renditen; höher bedeutet stärkere Schwankungen.
        - **Max. Rückgang 1J:** größter prozentualer Rückgang vom vorherigen Höchststand innerhalb des Zeitraums.
        - **Trend:** letzter Kurs wird mit dem Durchschnitt der letzten 50 bzw. 200 Handelstage verglichen.
        - **Risiko:** grobe Einteilung anhand historischer Volatilität und maximalem Rückgang. Das ist keine Garantie für zukünftiges Risiko.
        - **Score:** transparente Ranglisten-Kennzahl aus Rendite, Trend und Risikowerten. Sie berücksichtigt keine Unternehmensfundamentaldaten wie Umsatz, Gewinn oder Bewertung.
        """
    )

st.caption("Datenquelle: Yahoo Finance über yfinance. Die Aktienliste ist eine Auswahl und umfasst nicht den gesamten Weltmarkt.")
