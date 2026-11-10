from __future__ import annotations
import re, warnings
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
try:
    import feedparser
except Exception:
    feedparser = None
from atlas_config import STOCKS, ETFS, COMMODITIES, BOND_DURATION, ALL_UNIVERSE, ASSET_CLASS, ticker_currency
from atlas_risk import (
    fetch_history, single_metrics, portfolio_returns, historical_var_cvar,
    parametric_var, component_risk, stress_test, monte_carlo_portfolio,
    fixed_income_impact, get_last_price, holdings_to_weights,
)

warnings.filterwarnings("ignore", category=FutureWarning)
st.set_page_config(page_title="ATLAS", page_icon="\U0001f310", layout="wide", initial_sidebar_state="expanded")
TZ = ZoneInfo("Europe/Berlin")
NOW = datetime.now(TZ)

QUICK = ["SPY", "QQQ", "EUNL.DE", "VWCE.DE", "AGG", "GLD", "AAPL", "MSFT", "NVDA", "SAP"]

def init_state():
    if "holdings" not in st.session_state:
        st.session_state.holdings = {}
    if "cash" not in st.session_state:
        st.session_state.cash = 10000.0
    if "trades" not in st.session_state:
        st.session_state.trades = []
    if "books" in st.session_state:
        active = st.session_state.get("active_book", "Core")
        st.session_state.holdings = st.session_state.books.get(active, {})
        st.session_state.cash = float(st.session_state.cash.get(active, 10000.0)) if isinstance(st.session_state.cash, dict) else float(st.session_state.cash or 10000)
        del st.session_state.books

def portfolio_snapshot():
    cash = float(st.session_state.cash)
    rows, prices = [], {}
    invested = cost_total = 0.0
    for t, h in st.session_state.holdings.items():
        qty = float(h.get("qty", 0) or 0)
        avg = float(h.get("avg_cost", 0) or 0)
        if qty <= 0:
            continue
        px = get_last_price(t)
        prices[t] = px
        mv = qty * px if px == px else float("nan")
        cost = qty * avg
        pnl = (mv - cost) if mv == mv else float("nan")
        invested += mv if mv == mv else 0.0
        cost_total += cost
        rows.append({
            "Titel": t, "Name": ALL_UNIVERSE.get(t, t), "St\u00fcck": qty, "Einkauf": avg,
            "Kurs": px, "Wert \u20ac": mv, "Gewinn/Verlust \u20ac": pnl,
            "Gewinn/Verlust %": (pnl / cost * 100) if cost and pnl == pnl else float("nan"),
            "Anteil %": 0.0,
        })
    nav = cash + invested
    for r in rows:
        if nav > 0 and r["Wert \u20ac"] == r["Wert \u20ac"]:
            r["Anteil %"] = r["Wert \u20ac"] / nav * 100
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Wert \u20ac", ascending=False)
    return cash, df, invested, nav, cost_total, prices

def do_buy(ticker: str, qty: float, px: float):
    notional = qty * px
    cash = float(st.session_state.cash)
    if notional > cash + 1e-6:
        return False, f"Nicht genug Geld. Du hast {cash:,.2f} \u20ac, brauchst {notional:,.2f} \u20ac."
    h = st.session_state.holdings.get(ticker, {"qty": 0.0, "avg_cost": 0.0})
    old_q, old_c = float(h["qty"]), float(h["avg_cost"])
    new_q = old_q + qty
    new_avg = ((old_q * old_c) + notional) / new_q if new_q > 0 else px
    st.session_state.holdings[ticker] = {"qty": new_q, "avg_cost": new_avg}
    st.session_state.cash = cash - notional
    st.session_state.trades.append({
        "Zeit": NOW.strftime("%d.%m.%Y %H:%M"), "Art": "Kauf", "Titel": ticker,
        "St\u00fcck": qty, "Preis": px, "Betrag \u20ac": notional,
    })
    return True, f"Gekauft: {qty:.4g} \u00d7 {ticker} f\u00fcr {notional:,.2f} \u20ac"

def do_sell(ticker: str, qty: float, px: float):
    h = st.session_state.holdings.get(ticker, {"qty": 0.0, "avg_cost": 0.0})
    old_q = float(h["qty"])
    if qty > old_q + 1e-9:
        return False, f"Du hast nur {old_q:.4g} St\u00fcck von {ticker}."
    notional = qty * px
    new_q = old_q - qty
    if new_q < 1e-9:
        st.session_state.holdings.pop(ticker, None)
    else:
        st.session_state.holdings[ticker] = {"qty": new_q, "avg_cost": float(h["avg_cost"])}
    st.session_state.cash = float(st.session_state.cash) + notional
    st.session_state.trades.append({
        "Zeit": NOW.strftime("%d.%m.%Y %H:%M"), "Art": "Verkauf", "Titel": ticker,
        "St\u00fcck": qty, "Preis": px, "Betrag \u20ac": notional,
    })
    return True, f"Verkauft: {qty:.4g} \u00d7 {ticker} f\u00fcr {notional:,.2f} \u20ac"

init_state()
cash, pos_df, invested, nav, cost_total, prices = portfolio_snapshot()
upnl = invested - cost_total if cost_total else 0.0

with st.sidebar:
    st.title("\U0001f310 ATLAS")
    st.caption("Einfach \u00b7 Klar \u00b7 \u00dcbersichtlich")
    page = st.radio(
        "Men\u00fc",
        ["\U0001f3e0 Mein Depot", "\U0001f6d2 Kaufen / Verkaufen", "\U0001f4c9 Risiko", "\U0001f50d M\u00e4rkte", "\U0001f4f0 News & Mehr"],
        label_visibility="collapsed",
    )
    st.divider()
    st.metric("Gesamtwert", f"{nav:,.2f} \u20ac")
    st.caption(f"Bargeld: {cash:,.2f} \u20ac")
    if upnl:
        st.caption(f"Gewinn/Verlust: {upnl:+,.2f} \u20ac")
    if st.button("\U0001f504 Kurse aktualisieren"):
        st.cache_data.clear()
        st.rerun()
    st.caption(NOW.strftime("%d.%m.%Y %H:%M"))

if page == "\U0001f3e0 Mein Depot":
    st.header("Mein Depot")
    if not st.session_state.holdings and cash >= 9999:
        st.success("Willkommen! Du startest mit **10.000 \u20ac** Bargeld.")
        st.markdown("""
**So geht's in 3 Schritten:**
1. Gehe zu **\U0001f6d2 Kaufen / Verkaufen**
2. W\u00e4hle einen Titel (z. B. SPY oder EUNL.DE)
3. Gib an, wie viel Euro du anlegen willst \u2192 **Kaufen**
        """)
    a, b, c, d = st.columns(4)
    a.metric("Gesamtwert", f"{nav:,.2f} \u20ac")
    b.metric("Bargeld", f"{cash:,.2f} \u20ac")
    c.metric("Investiert", f"{invested:,.2f} \u20ac")
    d.metric("Gewinn/Verlust", f"{upnl:+,.2f} \u20ac", delta=f"{(upnl/cost_total*100) if cost_total else 0:.1f} %")
    st.subheader("Deine Positionen")
    if pos_df.empty:
        st.info("Noch keine Positionen. Kaufe deinen ersten Titel unter **Kaufen / Verkaufen**.")
    else:
        show = pos_df[["Titel", "Name", "St\u00fcck", "Kurs", "Wert \u20ac", "Gewinn/Verlust \u20ac", "Gewinn/Verlust %", "Anteil %"]].copy()
        st.dataframe(
            show.style.format({
                "St\u00fcck": "{:.4g}", "Kurs": "{:,.2f}", "Wert \u20ac": "{:,.2f}",
                "Gewinn/Verlust \u20ac": "{:+,.2f}", "Gewinn/Verlust %": "{:+.1f}", "Anteil %": "{:.1f}",
            }, na_rep="\u2014"),
            use_container_width=True, hide_index=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.pie(pos_df, values="Wert \u20ac", names="Titel", title="Aufteilung"), use_container_width=True)
        with col2:
            st.plotly_chart(px.bar(pos_df, x="Titel", y="Gewinn/Verlust \u20ac", title="Gewinn/Verlust je Titel", color="Gewinn/Verlust \u20ac", color_continuous_scale="RdYlGn"), use_container_width=True)
    with st.expander("Bargeld einzahlen oder abheben"):
        amt = st.number_input("Betrag in \u20ac", min_value=0.0, value=1000.0, step=100.0)
        x, y = st.columns(2)
        if x.button("\u2795 Einzahlen", use_container_width=True):
            st.session_state.cash = float(st.session_state.cash) + amt
            st.session_state.trades.append({"Zeit": NOW.strftime("%d.%m.%Y %H:%M"), "Art": "Einzahlung", "Titel": "\u2014", "St\u00fcck": 0, "Preis": 0, "Betrag \u20ac": amt})
            st.rerun()
        if y.button("\u2796 Abheben", use_container_width=True):
            if amt <= float(st.session_state.cash):
                st.session_state.cash = float(st.session_state.cash) - amt
                st.session_state.trades.append({"Zeit": NOW.strftime("%d.%m.%Y %H:%M"), "Art": "Auszahlung", "Titel": "\u2014", "St\u00fcck": 0, "Preis": 0, "Betrag \u20ac": amt})
                st.rerun()
            else:
                st.error("Nicht genug Bargeld.")
    if st.session_state.trades:
        with st.expander("Letzte Transaktionen"):
            st.dataframe(pd.DataFrame(st.session_state.trades[::-1][:20]), use_container_width=True, hide_index=True)

elif page == "\U0001f6d2 Kaufen / Verkaufen":
    st.header("Kaufen / Verkaufen")
    st.caption(f"Verf\u00fcgbares Bargeld: **{cash:,.2f} \u20ac**")
    mode = st.radio("Was m\u00f6chtest du tun?", ["Kaufen", "Verkaufen"], horizontal=True)
    if mode == "Kaufen":
        st.markdown("**Beliebte Titel \u2013 tippe zum Ausw\u00e4hlen:**")
        cols = st.columns(5)
        chosen = st.session_state.get("pick_ticker", "")
        for i, t in enumerate(QUICK):
            if cols[i % 5].button(t, key=f"q_{t}", use_container_width=True):
                st.session_state.pick_ticker = t
                chosen = t
                st.rerun()
        ticker = st.selectbox(
            "Oder Titel suchen",
            [""] + sorted(ALL_UNIVERSE.keys()),
            index=(sorted(ALL_UNIVERSE.keys()).index(chosen) + 1) if chosen in ALL_UNIVERSE else 0,
            format_func=lambda x: f"{x} \u2013 {ALL_UNIVERSE.get(x, '')}" if x else "Titel w\u00e4hlen \u2026",
            key="buy_ticker",
        )
        if ticker:
            px = get_last_price(ticker)
            if px != px or px <= 0:
                st.error("Kurs gerade nicht verf\u00fcgbar. Anderen Titel w\u00e4hlen oder sp\u00e4ter erneut versuchen.")
            else:
                st.info(f"**{ALL_UNIVERSE.get(ticker, ticker)}** \u00b7 aktueller Kurs: **{px:,.2f}** ({ticker_currency(ticker)})")
                how = st.radio("Eingabe", ["Betrag in \u20ac", "Anzahl St\u00fcck"], horizontal=True, key="buy_how")
                if how == "Betrag in \u20ac":
                    euro = st.number_input("Wie viel Euro anlegen?", min_value=1.0, value=min(1000.0, max(1.0, cash)), step=50.0)
                    qty = euro / px
                    st.caption(f"\u2248 **{qty:.4g} St\u00fcck** zu je {px:,.2f}")
                else:
                    qty = st.number_input("Anzahl St\u00fcck", min_value=0.01, value=1.0, step=1.0)
                    euro = qty * px
                    st.caption(f"\u2248 **{euro:,.2f} \u20ac**")
                if st.button("\u2713 Kaufen", type="primary", use_container_width=True):
                    ok, msg = do_buy(ticker, qty, px)
                    if ok:
                        st.success(msg)
                        st.balloons()
                        st.rerun()
                    else:
                        st.error(msg)
    else:
        owned = [t for t, h in st.session_state.holdings.items() if float(h.get("qty", 0)) > 0]
        if not owned:
            st.warning("Du hast noch nichts zum Verkaufen. Kaufe zuerst einen Titel.")
        else:
            ticker = st.selectbox(
                "Position verkaufen",
                owned,
                format_func=lambda x: f"{x} \u2013 {ALL_UNIVERSE.get(x, x)} ({st.session_state.holdings[x]['qty']:.4g} St\u00fcck)",
            )
            px = get_last_price(ticker)
            max_q = float(st.session_state.holdings[ticker]["qty"])
            if px != px:
                st.error("Kurs nicht verf\u00fcgbar.")
            else:
                st.info(f"Kurs: **{px:,.2f}** \u00b7 max. {max_q:.4g} St\u00fcck")
                qty = st.slider("St\u00fcck verkaufen", 0.01, float(max_q), float(max_q), 0.01)
                st.caption(f"Erl\u00f6s \u2248 **{qty * px:,.2f} \u20ac**")
                if st.button("\u2713 Verkaufen", type="primary", use_container_width=True):
                    ok, msg = do_sell(ticker, qty, px)
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

elif page == "\U0001f4c9 Risiko":
    st.header("Risiko-\u00dcbersicht")
    weights = holdings_to_weights(st.session_state.holdings, prices)
    if not weights:
        st.warning("Lege zuerst Positionen an, dann siehst du hier das Risiko.")
        st.stop()
    st.caption("Einfache Kennzahlen auf Basis der letzten 3 Jahre.")
    with st.spinner("Berechne \u2026"):
        port_rets, rets_mat = portfolio_returns(weights, "3y")
    if port_rets.empty:
        st.error("Zu wenig Kursdaten f\u00fcr die Berechnung.")
        st.stop()
    ann_ret = port_rets.mean() * 252
    ann_vol = port_rets.std(ddof=1) * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    hvar, hcvar = historical_var_cvar(port_rets, 0.05)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Erwartete Rendite / Jahr", f"{ann_ret*100:.1f} %")
    m2.metric("Schwankung / Jahr", f"{ann_vol*100:.1f} %")
    m3.metric("Sharpe-Ratio", f"{sharpe:.2f}")
    m4.metric("Tagesrisiko (VaR 95%)", f"{hvar*100:.2f} %")
    st.caption(f"Bei {nav:,.0f} \u20ac Gesamtwert: an einem schlechten Tag ca. **{hvar*nav:,.0f} \u20ac** Verlust m\u00f6glich (95 %-VaR).")
    eq = (1 + port_rets).cumprod()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Depot", fill="tozeroy"))
    fig.update_layout(title="Wertentwicklung (indexiert)", height=320, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Was passiert bei einem Crash? (Stress-Test)"):
        preset = st.selectbox("Szenario", ["B\u00f6rsencrash \u221240 %", "Tech-Ausverkauf", "Milde Rezession", "Zins +1,5 %"])
        shocks_map = {
            "B\u00f6rsencrash \u221240 %": {"Equity": -0.40, "Equity ETF": -0.38, "Fixed Income": 0.05, "Commodity": -0.20, "Real Estate": -0.35},
            "Tech-Ausverkauf": {"QQQ": -0.25, "NVDA": -0.35, "XLK": -0.22, "SMH": -0.30, "Equity": -0.08, "Equity ETF": -0.10},
            "Milde Rezession": {"Equity": -0.18, "Equity ETF": -0.16, "Fixed Income": 0.04, "Commodity": -0.10},
            "Zins +1,5 %": {},
        }
        shocks = shocks_map[preset]
        if preset == "Zins +1,5 %":
            fi = fixed_income_impact(weights, 150)
            st.metric("Auswirkung auf Anleihen", f"{fi['total_impact']*100:.2f} % \u2192 {fi['total_impact']*nav:,.0f} \u20ac")
        else:
            res = stress_test(weights, shocks)
            imp = res["portfolio_impact"]
            st.metric("Gesch\u00e4tzter Depot-Verlust", f"{imp*100:.1f} % \u2192 {imp*nav:,.0f} \u20ac")
            if not res["details"].empty:
                st.dataframe(res["details"].sort_values("Beitrag %"), use_container_width=True, hide_index=True)
    with st.expander("Monte-Carlo: m\u00f6gliche Ergebnisse in 1 Jahr"):
        mc = monte_carlo_portfolio(port_rets, 3000, 252)
        if not mc.empty:
            p5, p50, p95 = np.percentile(mc["Final Return"], [5, 50, 95])
            a, b, c = st.columns(3)
            a.metric("Schlechtes Jahr (5 %)", f"{nav*(1+p5):,.0f} \u20ac")
            b.metric("Mittleres Jahr", f"{nav*(1+p50):,.0f} \u20ac")
            c.metric("Gutes Jahr (95 %)", f"{nav*(1+p95):,.0f} \u20ac")

elif page == "\U0001f50d M\u00e4rkte":
    st.header("M\u00e4rkte ansehen")
    scope = st.radio("Bereich", ["Aktien", "ETFs", "Rohstoffe"], horizontal=True)
    uni = {"Aktien": STOCKS, "ETFs": ETFS, "Rohstoffe": COMMODITIES}[scope]
    limit = st.slider("Anzahl Titel", 10, 60, 25, 5)
    if st.button("Laden", type="primary") or "mkt_loaded" not in st.session_state:
        st.session_state.mkt_loaded = True
        rows, bar = [], st.progress(0, text="Lade Kurse \u2026")
        keys = list(uni.keys())[:limit]
        for i, t in enumerate(keys):
            m = single_metrics(t, uni[t], ASSET_CLASS.get(t, "Other"))
            if m:
                rows.append(m)
            bar.progress((i + 1) / len(keys), text=t)
        bar.empty()
        st.session_state.mkt_rows = rows
    rows = st.session_state.get("mkt_rows", [])
    if rows:
        df = pd.DataFrame(rows).sort_values("Score", ascending=False)
        st.dataframe(
            df[["Ticker", "Anlage", "Kurs", "1J Rendite %", "Volatilit\u00e4t %", "Sharpe", "Score", "Risiko"]],
            use_container_width=True, height=420, hide_index=True,
        )
        pick = st.selectbox("Chart anzeigen", df["Ticker"].tolist())
        h = fetch_history(pick, "2y")
        if not h.empty:
            st.plotly_chart(px.line(h, y="Close", title=f"{pick} \u2013 2 Jahre"), use_container_width=True)

else:
    st.header("News & Mehr")
    tab1, tab2, tab3 = st.tabs(["Nachrichten", "Report", "Zur\u00fccksetzen"])
    with tab1:
        @st.cache_data(ttl=300, show_spinner=False)
        def rss_news():
            if not feedparser:
                return pd.DataFrame()
            feeds = [
                ("BBC", "https://feeds.bbci.co.uk/news/world/rss.xml"),
                ("Fed", "https://www.federalreserve.gov/feeds/press_all.xml"),
                ("ECB", "https://www.ecb.europa.eu/rss/press.html"),
            ]
            rows = []
            cutoff = NOW - timedelta(days=2)
            for src, url in feeds:
                try:
                    feed = feedparser.parse(url)
                    for e in feed.entries[:15]:
                        title = getattr(e, "title", "")
                        link = getattr(e, "link", "")
                        raw = getattr(e, "published_parsed", None) or getattr(e, "updated_parsed", None)
                        dt = datetime(*raw[:6], tzinfo=timezone.utc).astimezone(TZ) if raw else None
                        if dt and dt >= cutoff:
                            rows.append({"Zeit": dt.strftime("%d.%m. %H:%M"), "Quelle": src, "Meldung": title, "Link": link})
                except Exception:
                    pass
            return pd.DataFrame(rows)
        news = rss_news()
        if news.empty:
            st.info("Keine aktuellen Meldungen geladen.")
        else:
            for _, r in news.head(20).iterrows():
                st.markdown(f"**{r['Meldung']}**  \n{r['Quelle']} \u00b7 {r['Zeit']} \u00b7 [Link]({r['Link']})")
                st.divider()
    with tab2:
        lines = [f"ATLAS Report \u2013 {NOW.strftime('%d.%m.%Y %H:%M')}", f"Gesamtwert: {nav:,.2f} \u20ac", f"Bargeld: {cash:,.2f} \u20ac", ""]
        if not pos_df.empty:
            for _, r in pos_df.iterrows():
                lines.append(f"{r['Titel']:10s}  {r['St\u00fcck']:8.2f} Stk  {r['Wert \u20ac']:10,.2f} \u20ac  G/V {r['Gewinn/Verlust \u20ac']:+,.2f} \u20ac")
        txt = "\n".join(lines)
        st.text_area("Report", txt, height=280)
        st.download_button("Als Textdatei speichern", txt, file_name=f"atlas_{NOW.strftime('%Y%m%d')}.txt")
    with tab3:
        st.warning("Setzt Depot, Bargeld und Historie zur\u00fcck (10.000 \u20ac Startkapital).")
        if st.button("Alles zur\u00fccksetzen", type="primary"):
            st.session_state.holdings = {}
            st.session_state.cash = 10000.0
            st.session_state.trades = []
            st.success("Zur\u00fcckgesetzt.")
            st.rerun()

st.divider()
st.caption("ATLAS \u00b7 Keine Anlageberatung \u00b7 Kurse von Yahoo Finance (verz\u00f6gert) \u00b7 Daten nur in dieser Browser-Session")
