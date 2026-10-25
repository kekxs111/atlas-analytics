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
st.set_page_config(page_title="ATLAS Analytics · Risk Platform", page_icon="🌐", layout="wide", initial_sidebar_state="expanded")
TZ = ZoneInfo("Europe/Berlin")
NOW = datetime.now(TZ)

def init_state():
    if "books" not in st.session_state:
        st.session_state.books = {"Core": {}}
    if "cash" not in st.session_state:
        st.session_state.cash = {"Core": 10000.0}
    if "active_book" not in st.session_state:
        st.session_state.active_book = "Core"
    if "trades" not in st.session_state:
        st.session_state.trades = []
    if "compliance_limits" not in st.session_state:
        st.session_state.compliance_limits = {"max_single_position": 0.25, "max_asset_class": 0.70}
    if "portfolio" in st.session_state and st.session_state.portfolio:
        for bname, wts in st.session_state.portfolio.items():
            if bname not in st.session_state.books:
                st.session_state.books[bname] = {}
                st.session_state.cash[bname] = 10000.0
        del st.session_state.portfolio

def book_nav(book_name: str):
    holdings = st.session_state.books.get(book_name, {})
    cash = float(st.session_state.cash.get(book_name, 0.0))
    rows, prices = [], {}
    invested = 0.0
    cost_total = 0.0
    for t, h in holdings.items():
        qty = float(h.get("qty", 0) or 0)
        avg = float(h.get("avg_cost", 0) or 0)
        if qty <= 0:
            continue
        px = get_last_price(t)
        prices[t] = px
        mv = qty * px if px == px else float("nan")
        cost = qty * avg
        pnl = (mv - cost) if mv == mv else float("nan")
        pnl_pct = (pnl / cost * 100) if cost and pnl == pnl else float("nan")
        invested += mv if mv == mv else 0.0
        cost_total += cost
        rows.append({
            "Ticker": t, "Name": ALL_UNIVERSE.get(t, t), "Stück": qty, "Ø Einkauf": avg,
            "Kurs": px, "Marktwert €": mv, "Kosten €": cost, "P&L €": pnl, "P&L %": pnl_pct,
            "Gewicht %": 0.0, "Asset Class": ASSET_CLASS.get(t, "—"), "Währung": ticker_currency(t),
        })
    nav = cash + invested
    for r in rows:
        if nav > 0 and r["Marktwert €"] == r["Marktwert €"]:
            r["Gewicht %"] = r["Marktwert €"] / nav * 100
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Marktwert €", ascending=False)
    return cash, df, invested, nav, cost_total, prices

def weights_from_book(book_name: str) -> dict:
    cash, df, invested, nav, _, prices = book_nav(book_name)
    holdings = st.session_state.books.get(book_name, {})
    return holdings_to_weights(holdings, prices)

init_state()

with st.sidebar:
    st.title("🌐 ATLAS")
    st.caption("Kapital · Risiko · Portfolio")
    page = st.radio("Modul", ["💰 Kapital & Portfolio", "📊 Market Scanner", "📉 Risk Analytics", "⚡ Stress Tests",
         "📜 Fixed Income", "💱 FX", "📋 Trade Journal", "✅ Compliance", "📄 Reports", "📰 News"], label_visibility="collapsed")
    st.divider()
    total_nav = 0.0
    total_cash = 0.0
    for bn in st.session_state.books:
        c, _, inv, nav, _, _ = book_nav(bn)
        total_nav += nav
        total_cash += c
    st.metric("Gesamt-NAV", f"{total_nav:,.2f} €")
    st.caption(f"davon Cash: {total_cash:,.2f} €")
    if st.button("🔄 Kurse & Cache aktualisieren"):
        st.cache_data.clear()
        st.rerun()
    st.caption(f"{NOW.strftime('%d.%m.%Y %H:%M')} · Yahoo Finance · Keine Anlageberatung")

if page == "💰 Kapital & Portfolio":
    st.header("Kapital & Portfolio")
    st.write("Echtes Geld: Cash, Stückzahlen, Marktwert, P&L. Käufe belasten Cash, Verkäufe erhöhen Cash.")
    books = list(st.session_state.books.keys())
    c1, c2 = st.columns([2, 1])
    with c1:
        active = st.selectbox("Buch", books, index=books.index(st.session_state.active_book) if st.session_state.active_book in books else 0)
        st.session_state.active_book = active
    with c2:
        nb = st.text_input("Neues Buch", placeholder="z. B. Satellite")
        if st.button("Buch anlegen") and nb and nb not in st.session_state.books:
            st.session_state.books[nb] = {}
            st.session_state.cash[nb] = 0.0
            st.session_state.active_book = nb
            st.rerun()
    st.subheader("Cash / Einzahlung")
    cc1, cc2, cc3 = st.columns([1, 1, 1])
    with cc1:
        deposit = st.number_input("Betrag €", min_value=0.0, value=1000.0, step=100.0, key="dep_amt")
    with cc2:
        st.write(""); st.write("")
        if st.button("➕ Einzahlen"):
            st.session_state.cash[active] = float(st.session_state.cash.get(active, 0)) + deposit
            st.session_state.trades.append({"Zeit": NOW.strftime("%Y-%m-%d %H:%M"), "Buch": active, "Ticker": "CASH", "Seite": "DEPOSIT", "Qty": 1, "Preis": deposit, "Notional": deposit, "Notiz": "Einzahlung"})
            st.rerun()
    with cc3:
        st.write(""); st.write("")
        if st.button("➖ Abheben") and deposit <= float(st.session_state.cash.get(active, 0)):
            st.session_state.cash[active] = float(st.session_state.cash.get(active, 0)) - deposit
            st.session_state.trades.append({"Zeit": NOW.strftime("%Y-%m-%d %H:%M"), "Buch": active, "Ticker": "CASH", "Seite": "WITHDRAW", "Qty": 1, "Preis": deposit, "Notional": deposit, "Notiz": "Auszahlung"})
            st.rerun()
    cash, pos_df, invested, nav, cost_total, prices = book_nav(active)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cash", f"{cash:,.2f} €")
    m2.metric("Investiert", f"{invested:,.2f} €")
    m3.metric("NAV (Gesamt)", f"{nav:,.2f} €")
    upnl = invested - cost_total if cost_total else 0.0
    m4.metric("Unreal. P&L", f"{upnl:,.2f} €", delta=f"{(upnl/cost_total*100) if cost_total else 0:.1f} %")
    st.subheader("Kaufen / Verkaufen")
    t1, t2, t3, t4, t5 = st.columns([2, 1, 1, 1, 1])
    with t1:
        ticker = st.selectbox("Ticker", [""] + sorted(ALL_UNIVERSE.keys()), format_func=lambda x: f"{x} – {ALL_UNIVERSE.get(x,'')}" if x else "—", key="trade_t")
    with t2:
        side = st.selectbox("Seite", ["BUY", "SELL"], key="trade_side")
    with t3:
        qty = st.number_input("Stück", min_value=0.01, value=1.0, step=1.0, key="trade_qty")
    with t4:
        use_mkt = st.checkbox("Marktpreis", value=True)
        manual_px = st.number_input("Preis", min_value=0.01, value=100.0, step=0.01, key="trade_px", disabled=use_mkt)
    with t5:
        st.write(""); st.write("")
        exec_btn = st.button("Order ausführen", type="primary")
    if exec_btn and ticker:
        px = get_last_price(ticker) if use_mkt else manual_px
        if px != px or px <= 0:
            st.error("Kein gültiger Preis verfügbar.")
        else:
            notional = qty * px
            holdings = st.session_state.books[active]
            cash_now = float(st.session_state.cash.get(active, 0))
            if side == "BUY":
                if notional > cash_now + 1e-6:
                    st.error(f"Nicht genug Cash ({cash_now:,.2f} €), benötigt {notional:,.2f} €.")
                else:
                    prev = holdings.get(ticker, {"qty": 0.0, "avg_cost": 0.0})
                    old_q, old_c = float(prev["qty"]), float(prev["avg_cost"])
                    new_q = old_q + qty
                    new_avg = ((old_q * old_c) + notional) / new_q if new_q > 0 else px
                    holdings[ticker] = {"qty": new_q, "avg_cost": new_avg}
                    st.session_state.cash[active] = cash_now - notional
                    st.session_state.trades.append({"Zeit": NOW.strftime("%Y-%m-%d %H:%M"), "Buch": active, "Ticker": ticker, "Seite": "BUY", "Qty": qty, "Preis": px, "Notional": notional, "Notiz": ""})
                    st.success(f"Gekauft: {qty} × {ticker} @ {px:.2f} = {notional:,.2f} €")
                    st.rerun()
            else:
                prev = holdings.get(ticker, {"qty": 0.0, "avg_cost": 0.0})
                old_q = float(prev["qty"])
                if qty > old_q + 1e-9:
                    st.error(f"Nur {old_q} Stück vorhanden.")
                else:
                    new_q = old_q - qty
                    if new_q < 1e-9:
                        holdings.pop(ticker, None)
                    else:
                        holdings[ticker] = {"qty": new_q, "avg_cost": float(prev["avg_cost"])}
                    st.session_state.cash[active] = cash_now + notional
                    st.session_state.trades.append({"Zeit": NOW.strftime("%Y-%m-%d %H:%M"), "Buch": active, "Ticker": ticker, "Seite": "SELL", "Qty": qty, "Preis": px, "Notional": notional, "Notiz": ""})
                    st.success(f"Verkauft: {qty} × {ticker} @ {px:.2f} = {notional:,.2f} €")
                    st.rerun()
    st.subheader(f"Positionen – {active}")
    if pos_df.empty:
        st.info("Keine Positionen. Zahle Cash ein und kaufe Titel.")
    else:
        fmt = {"Stück": "{:.4g}", "Ø Einkauf": "{:,.2f}", "Kurs": "{:,.2f}", "Marktwert €": "{:,.2f}", "Kosten €": "{:,.2f}", "P&L €": "{:,.2f}", "P&L %": "{:.1f}", "Gewicht %": "{:.1f}"}
        st.dataframe(pos_df.style.format(fmt, na_rep="—"), use_container_width=True, hide_index=True)
        st.plotly_chart(px.pie(pos_df, values="Marktwert €", names="Ticker", title=f"Allokation – {active}"), use_container_width=True)
        ac = pos_df.groupby("Asset Class")["Marktwert €"].sum().reset_index()
        st.plotly_chart(px.bar(ac, x="Asset Class", y="Marktwert €", title="Asset-Class"), use_container_width=True)
    if len(st.session_state.books) > 1 and active != "Core":
        if st.button("Buch löschen", type="secondary"):
            del st.session_state.books[active]
            st.session_state.cash.pop(active, None)
            st.session_state.active_book = "Core"
            st.rerun()

elif page == "📊 Market Scanner":
    st.header("Market Scanner")
    scope = st.radio("Universum", ["Aktien", "ETFs", "Rohstoffe", "Alles"], horizontal=True)
    uni = {"Aktien": STOCKS, "ETFs": ETFS, "Rohstoffe": COMMODITIES, "Alles": ALL_UNIVERSE}[scope]
    limit = st.slider("Max. Symbole", 20, min(120, len(uni)), 40, 10)
    tickers = list(uni.keys())[:limit]
    rows, prog = [], st.progress(0, text="Lade …")
    for i, t in enumerate(tickers):
        m = single_metrics(t, uni.get(t, t), ASSET_CLASS.get(t, "Other"))
        if m: rows.append(m)
        prog.progress((i + 1) / len(tickers), text=t)
    prog.empty()
    if rows:
        df = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
        st.dataframe(df, use_container_width=True, height=500)
        pick = st.selectbox("Chart", df["Ticker"].tolist())
        h = fetch_history(pick, "max")
        if not h.empty: st.plotly_chart(px.line(h, y="Close", title=pick), use_container_width=True)

elif page == "📉 Risk Analytics":
    st.header("Risk Analytics")
    book_name = st.selectbox("Buch", list(st.session_state.books.keys()), key="rb")
    weights = weights_from_book(book_name)
    if not weights:
        st.warning("Buch hat keine bewertbaren Positionen."); st.stop()
    period = st.selectbox("Historie", ["1y", "2y", "3y", "5y"], index=2)
    alpha = st.select_slider("VaR-Konfidenz", [0.01, 0.025, 0.05, 0.10], 0.05, format_func=lambda x: f"{int((1-x)*100)}%")
    _, _, _, nav, _, _ = book_nav(book_name)
    with st.spinner("Berechne …"):
        port_rets, rets_mat = portfolio_returns(weights, period)
    if port_rets.empty:
        st.error("Nicht genug Historie."); st.stop()
    ann_ret = port_rets.mean() * 252
    ann_vol = port_rets.std(ddof=1) * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    hvar, hcvar = historical_var_cvar(port_rets, alpha)
    pvar = parametric_var(port_rets, alpha)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rendite p.a.", f"{ann_ret*100:.1f}%"); c2.metric("Vol p.a.", f"{ann_vol*100:.1f}%")
    c3.metric("Sharpe", f"{sharpe:.2f}"); c4.metric("VaR 1d", f"{hvar*100:.2f}%"); c5.metric("CVaR 1d", f"{hcvar*100:.2f}%")
    st.caption(f"NAV {nav:,.2f} € → VaR ≈ {hvar*nav:,.2f} € · CVaR ≈ {hcvar*nav:,.2f} €")
    eq = (1 + port_rets).cumprod()
    fig = go.Figure(); fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Portfolio")); fig.update_layout(title="Wertentwicklung", height=340)
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Contribution to Risk")
    ctr = component_risk(weights, rets_mat)
    if not ctr.empty:
        st.dataframe(ctr, use_container_width=True)
        st.plotly_chart(px.bar(ctr, x="Ticker", y="Risk Contribution %"), use_container_width=True)
    if not rets_mat.empty and rets_mat.shape[1] >= 2:
        st.subheader("Korrelation")
        st.plotly_chart(px.imshow(rets_mat.corr(), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1), use_container_width=True)
    st.subheader("Monte-Carlo (1J)")
    mc = monte_carlo_portfolio(port_rets, 5000, 252)
    if not mc.empty:
        p5, p50, p95 = np.percentile(mc["Final Return"], [5, 50, 95])
        a, b, c = st.columns(3)
        a.metric("5%", f"{p5*100:.1f}% / {nav*(1+p5):,.0f} €")
        b.metric("Median", f"{p50*100:.1f}% / {nav*(1+p50):,.0f} €")
        c.metric("95%", f"{p95*100:.1f}% / {nav*(1+p95):,.0f} €")
        fig = px.histogram(mc, x="Final Return", nbins=50); fig.add_vline(x=p5, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

elif page == "⚡ Stress Tests":
    st.header("Stress Tests")
    book_name = st.selectbox("Buch", list(st.session_state.books.keys()), key="sb")
    weights = weights_from_book(book_name)
    _, _, _, nav, _, _ = book_nav(book_name)
    if not weights:
        st.warning("Buch leer."); st.stop()
    PRESETS = {"2008 Equity Crash": {"Equity": -0.40, "Equity ETF": -0.38, "Fixed Income": 0.08, "Commodity": -0.25, "Real Estate": -0.35}, "Inflation Spike": {"Equity": -0.12, "Fixed Income": -0.08, "Commodity": 0.25, "Real Estate": 0.05}, "Tech Sell-off": {"QQQ": -0.25, "NVDA": -0.35, "XLK": -0.22, "SMH": -0.30, "Equity": -0.08}, "Oil +50%": {"CL=F": 0.50, "USO": 0.48, "XLE": 0.20, "Equity": -0.05}, "Mild Recession": {"Equity": -0.18, "Equity ETF": -0.16, "Fixed Income": 0.04, "Commodity": -0.10}}
    preset = st.selectbox("Szenario", list(PRESETS.keys()) + ["Rate Shock", "Eigenes"])
    rate_bp = st.slider("Zins-Schock (bp)", -200, 300, 150, 10)
    if preset in ("Rate Shock", "Eigenes"):
        fi = fixed_income_impact(weights, rate_bp)
        st.metric(f"FI Impact {rate_bp} bp", f"{fi['total_impact']*100:.2f}% → {fi['total_impact']*nav:,.2f} €")
        if not fi["details"].empty: st.dataframe(fi["details"], use_container_width=True)
    if preset == "Eigenes":
        s_eq = st.number_input("Equity %", -50.0, 30.0, -15.0) / 100
        s_fi = st.number_input("Fixed Income %", -30.0, 20.0, -5.0) / 100
        s_com = st.number_input("Commodity %", -40.0, 50.0, 0.0) / 100
        shocks = {"Equity": s_eq, "Equity ETF": s_eq, "Fixed Income": s_fi, "Commodity": s_com, "Real Estate": s_eq * 0.7}
    elif preset in PRESETS: shocks = PRESETS[preset]
    else: shocks = {}
    if shocks:
        res = stress_test(weights, shocks)
        impact = res["portfolio_impact"]
        st.metric("Portfolio Impact", f"{impact*100:.2f}% → {impact*nav:,.2f} €")
        st.dataframe(res["details"].sort_values("Beitrag %"), use_container_width=True)

elif page == "📜 Fixed Income":
    st.header("Fixed Income")
    rows = []
    for t in BOND_DURATION:
        m = single_metrics(t, ETFS.get(t, t), "Fixed Income")
        if m:
            m["Duration"] = BOND_DURATION[t]; m["Impact +100bp %"] = -BOND_DURATION[t]; rows.append(m)
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df[["Ticker", "Anlage", "Kurs", "Duration", "Impact +100bp %", "Volatilität %", "1J Rendite %", "Sharpe"]], use_container_width=True)

elif page == "💱 FX":
    st.header("FX Exposure")
    for bname in st.session_state.books:
        cash, pos_df, inv, nav, _, _ = book_nav(bname)
        if pos_df.empty and cash <= 0: continue
        st.subheader(f"Buch: {bname}")
        ccy_exp = {"EUR": cash}
        if not pos_df.empty:
            for _, r in pos_df.iterrows():
                c = r["Währung"]; mv = r["Marktwert €"] if r["Marktwert €"] == r["Marktwert €"] else 0
                ccy_exp[c] = ccy_exp.get(c, 0) + mv
        cdf = pd.DataFrame([{"Währung": k, "Exposure €": v, "Anteil %": v / nav * 100 if nav else 0} for k, v in sorted(ccy_exp.items(), key=lambda x: -x[1]) if v])
        a, b = st.columns(2)
        with a: st.dataframe(cdf, use_container_width=True, hide_index=True)
        with b:
            if not cdf.empty: st.plotly_chart(px.pie(cdf, values="Exposure €", names="Währung"), use_container_width=True)

elif page == "📋 Trade Journal":
    st.header("Trade Journal")
    if st.session_state.trades:
        st.dataframe(pd.DataFrame(st.session_state.trades), use_container_width=True, hide_index=True)
        if st.button("Journal leeren"): st.session_state.trades = []; st.rerun()
    else: st.info("Noch keine Trades.")

elif page == "✅ Compliance":
    st.header("Compliance")
    lim = st.session_state.compliance_limits
    a, b = st.columns(2)
    lim["max_single_position"] = a.slider("Max Einzelposition", 0.05, 0.50, lim["max_single_position"], 0.01)
    lim["max_asset_class"] = b.slider("Max Asset Class", 0.20, 1.0, lim["max_asset_class"], 0.05)
    for bname in st.session_state.books:
        cash, pos_df, inv, nav, _, _ = book_nav(bname)
        if pos_df.empty: continue
        st.subheader(bname); breaches = []
        for _, r in pos_df.iterrows():
            w = r["Gewicht %"] / 100
            if w > lim["max_single_position"]: breaches.append(f"⚠️ {r['Ticker']}: {r['Gewicht %']:.1f}%")
        ac_w = pos_df.groupby("Asset Class")["Gewicht %"].sum()
        for ac, w in ac_w.items():
            if w / 100 > lim["max_asset_class"]: breaches.append(f"⚠️ {ac}: {w:.1f}%")
        if breaches:
            for x in breaches: st.error(x)
        else: st.success("Keine Verletzungen")

elif page == "📄 Reports":
    st.header("Reports")
    lines = [f"ATLAS Report – {NOW.strftime('%Y-%m-%d %H:%M')}", "=" * 50, ""]
    for bname in st.session_state.books:
        cash, pos_df, inv, nav, cost, _ = book_nav(bname)
        lines.append(f"BUCH: {bname}")
        lines.append(f"  Cash: {cash:,.2f} € | Investiert: {inv:,.2f} € | NAV: {nav:,.2f} €")
        lines.append("-" * 40)
        if not pos_df.empty:
            for _, r in pos_df.iterrows():
                lines.append(f"  {r['Ticker']:10s} {r['Stück']:8.2f} Stk  MW {r['Marktwert €']:10,.2f} €  P&L {r['P&L €']:8,.2f} €")
        lines.append("")
    txt = "\n".join(lines)
    st.text_area("Report", txt, height=360)
    st.download_button("Download .txt", txt, file_name=f"atlas_report_{NOW.strftime('%Y%m%d')}.txt")

elif page == "📰 News":
    st.header("News")
    @st.cache_data(ttl=300, show_spinner=False)
    def rss_news():
        if not feedparser: return pd.DataFrame(), "feedparser fehlt"
        feeds = [("Reuters", "https://feeds.reuters.com/Reuters/worldNews"), ("BBC", "https://feeds.bbci.co.uk/news/world/rss.xml"), ("Fed", "https://www.federalreserve.gov/feeds/press_all.xml"), ("ECB", "https://www.ecb.europa.eu/rss/press.html"), ("SEC", "https://www.sec.gov/news/pressreleases.rss")]
        rows, errs = [], []
        cutoff = NOW - timedelta(days=2)
        for src, url in feeds:
            try:
                feed = feedparser.parse(url)
                for e in feed.entries[:25]:
                    title = getattr(e, "title", ""); link = getattr(e, "link", "")
                    summary = re.sub("<[^>]+>", " ", getattr(e, "summary", getattr(e, "description", "")))
                    raw = getattr(e, "published_parsed", None) or getattr(e, "updated_parsed", None)
                    dt = datetime(*raw[:6], tzinfo=timezone.utc).astimezone(TZ) if raw else None
                    if dt and dt >= cutoff:
                        rows.append({"Zeit": dt.strftime("%d.%m.%Y %H:%M"), "Quelle": src, "Meldung": title, "Kurzinfo": summary.strip()[:350], "Link": link, "dt": dt})
            except Exception: errs.append(src)
        df = pd.DataFrame(rows)
        if not df.empty: df = df.sort_values("dt", ascending=False).drop_duplicates(subset=["Meldung"])
        return df, (", ".join(errs) if errs else "")
    news, status = rss_news()
    if status: st.caption("Nicht erreichbar: " + status)
    if news.empty: st.warning("Keine Meldungen")
    else:
        q = st.text_input("Filter", placeholder="Zinsen, KI, Öl …")
        if q: news = news[news.apply(lambda r: q.lower() in (str(r.Meldung) + str(r.Kurzinfo)).lower(), axis=1)]
        for _, r in news.head(35).iterrows():
            with st.container(border=True):
                st.markdown(f"**{r['Meldung']}**"); st.caption(f"{r['Quelle']} · {r['Zeit']}")
                if r["Kurzinfo"]: st.write(r["Kurzinfo"])
                if r["Link"]: st.markdown(f"[Quelle]({r['Link']})")

st.divider()
st.caption("ATLAS · Keine Anlageberatung · Kurse verzögert (Yahoo) · Session-lokal (nicht persistent auf Server)")
