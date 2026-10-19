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
from atlas_risk import (fetch_history, single_metrics, portfolio_returns, historical_var_cvar,
    parametric_var, component_risk, stress_test, monte_carlo_portfolio, fixed_income_impact)

warnings.filterwarnings("ignore", category=FutureWarning)
st.set_page_config(page_title="ATLAS Analytics · Risk Platform", page_icon="🌐", layout="wide", initial_sidebar_state="expanded")
TZ = ZoneInfo("Europe/Berlin"); NOW = datetime.now(TZ)

def init_state():
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = {"Core": {"SPY": 0.40, "EUNL.DE": 0.20, "AGG": 0.15, "GLD": 0.10, "QQQ": 0.15}}
    if "active_book" not in st.session_state: st.session_state.active_book = "Core"
    if "trades" not in st.session_state: st.session_state.trades = []
    if "compliance_limits" not in st.session_state:
        st.session_state.compliance_limits = {"max_single_position": 0.25, "max_asset_class": 0.70, "max_sector_equity": 0.40}
init_state()

with st.sidebar:
    st.title("🌐 ATLAS"); st.caption("Risk · Portfolio · Stress · Compliance")
    page = st.radio("Modul", ["📊 Market Scanner","💼 Portfolio Builder","📉 Risk Analytics","⚡ Stress Tests",
        "📜 Fixed Income","💱 FX & Multi-Currency","📋 Trade Journal","✅ Compliance","📄 Reports","📰 News"], label_visibility="collapsed")
    st.divider()
    if st.button("🔄 Cache leeren"): st.cache_data.clear(); st.rerun()
    st.caption(f"{NOW.strftime('%d.%m.%Y %H:%M')} · Yahoo Finance · Keine Anlageberatung")

if page == "📊 Market Scanner":
    st.header("Market Scanner")
    scope = st.radio("Universum", ["Aktien","ETFs","Rohstoffe","Alles"], horizontal=True)
    uni = {"Aktien": STOCKS, "ETFs": ETFS, "Rohstoffe": COMMODITIES, "Alles": ALL_UNIVERSE}[scope]
    limit = st.slider("Max. Symbole", 20, min(120, len(uni)), 40, 10)
    tickers = list(uni.keys())[:limit]
    rows, prog = [], st.progress(0, text="Lade …")
    for i, t in enumerate(tickers):
        m = single_metrics(t, uni.get(t, t), ASSET_CLASS.get(t, "Other"))
        if m: rows.append(m)
        prog.progress((i+1)/len(tickers), text=f"{t}")
    prog.empty()
    if rows:
        df = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
        st.dataframe(df, use_container_width=True, height=500)
        pick = st.selectbox("Chart", df["Ticker"].tolist())
        h = fetch_history(pick, "max")
        if not h.empty: st.plotly_chart(px.line(h, y="Close", title=pick), use_container_width=True)

elif page == "💼 Portfolio Builder":
    st.header("Portfolio Builder · Multi-Book")
    books = list(st.session_state.portfolio.keys())
    active = st.selectbox("Buch", books, index=books.index(st.session_state.active_book) if st.session_state.active_book in books else 0)
    st.session_state.active_book = active
    nb = st.text_input("Neues Buch"); 
    if st.button("Buch anlegen") and nb and nb not in st.session_state.portfolio:
        st.session_state.portfolio[nb] = {}; st.session_state.active_book = nb; st.rerun()
    book = st.session_state.portfolio[active]
    c1,c2,c3 = st.columns([2,1,1])
    with c1: add_t = st.selectbox("Ticker", [""]+sorted(ALL_UNIVERSE.keys()), format_func=lambda x: f"{x} – {ALL_UNIVERSE.get(x,'')}" if x else "—")
    with c2: add_w = st.number_input("Gewicht %", 0.1, 100.0, 5.0, 0.5)
    with c3:
        st.write(""); st.write("")
        if st.button("Hinzufügen") and add_t:
            book[add_t] = book.get(add_t,0)+add_w/100; s=sum(book.values())
            if s>0:
                for k in book: book[k]/=s
            st.rerun()
    if book:
        rows=[{"Ticker":t,"Name":ALL_UNIVERSE.get(t,t),"Gewicht %":round(w*100,2),"Asset Class":ASSET_CLASS.get(t,"—"),"Währung":ticker_currency(t)} for t,w in sorted(book.items(), key=lambda x:-x[1])]
        pdf=pd.DataFrame(rows); st.dataframe(pdf, use_container_width=True, hide_index=True)
        edit_t=st.selectbox("Gewicht anpassen", list(book.keys()))
        new_w=st.slider("Neues Gewicht %", 0.0, 100.0, float(book[edit_t]*100), 0.5)
        if st.button("Setzen"):
            if new_w==0: del book[edit_t]
            else: book[edit_t]=new_w/100
            s=sum(book.values())
            if s>0:
                for k in list(book.keys()): book[k]/=s
            st.rerun()
    st.plotly_chart(px.pie(pdf, values="Gewicht %", names="Ticker", title=f"Allokation – {active}"), use_container_width=True) if book else None
    if book:
        ac=pdf.groupby("Asset Class")["Gewicht %"].sum().reset_index()
        st.plotly_chart(px.bar(ac, x="Asset Class", y="Gewicht %", title="Asset-Class"), use_container_width=True)
    else: st.info("Buch leer.")

elif page == "📉 Risk Analytics":
    st.header("Risk Analytics")
    book_name=st.selectbox("Buch", list(st.session_state.portfolio.keys()), key="rb")
    weights=st.session_state.portfolio.get(book_name,{})
    if not weights: st.warning("Buch leer."); st.stop()
    period=st.selectbox("Historie", ["1y","2y","3y","5y"], index=2)
    alpha=st.select_slider("VaR-Konfidenz", [0.01,0.025,0.05,0.10], 0.05, format_func=lambda x:f"{int((1-x)*100)}%")
    with st.spinner("Berechne …"):
        port_rets, rets_mat = portfolio_returns(weights, period)
    if port_rets.empty: st.error("Nicht genug Historie."); st.stop()
    ann_ret=port_rets.mean()*252; ann_vol=port_rets.std(ddof=1)*np.sqrt(252)
    sharpe=ann_ret/ann_vol if ann_vol>0 else np.nan
    hvar,hcvar=historical_var_cvar(port_rets, alpha); pvar=parametric_var(port_rets, alpha)
    c1,c2,c3,c4,c5=st.columns(5)
    c1.metric("Rendite p.a.", f"{ann_ret*100:.1f}%"); c2.metric("Vol p.a.", f"{ann_vol*100:.1f}%")
    c3.metric("Sharpe", f"{sharpe:.2f}"); c4.metric(f"VaR 1d", f"{hvar*100:.2f}%"); c5.metric("CVaR 1d", f"{hcvar*100:.2f}%")
    st.caption(f"Param. VaR: {pvar*100:.2f}% · Hist. VaR 10d: {hvar*np.sqrt(10)*100:.2f}%")
    eq=(1+port_rets).cumprod()
    fig=go.Figure(); fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Portfolio")); fig.update_layout(title="Wertentwicklung", height=340)
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Contribution to Risk")
    ctr=component_risk(weights, rets_mat)
    if not ctr.empty:
        st.dataframe(ctr, use_container_width=True)
        st.plotly_chart(px.bar(ctr, x="Ticker", y="Risk Contribution %"), use_container_width=True)
    if not rets_mat.empty and rets_mat.shape[1]>=2:
        st.subheader("Korrelation"); st.plotly_chart(px.imshow(rets_mat.corr(), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1), use_container_width=True)
    st.subheader("Monte-Carlo (1J, 5000 Pfade)")
    mc=monte_carlo_portfolio(port_rets, 5000, 252)
    if not mc.empty:
        p5,p50,p95=np.percentile(mc["Final Return"],[5,50,95])
        c1,c2,c3=st.columns(3); c1.metric("5%", f"{p5*100:.1f}%"); c2.metric("Median", f"{p50*100:.1f}%"); c3.metric("95%", f"{p95*100:.1f}%")
        fig=px.histogram(mc, x="Final Return", nbins=50); fig.add_vline(x=p5, line_dash="dash", line_color="red"); st.plotly_chart(fig, use_container_width=True)

elif page == "⚡ Stress Tests":
    st.header("Stress Tests")
    book_name=st.selectbox("Buch", list(st.session_state.portfolio.keys()), key="sb")
    weights=st.session_state.portfolio.get(book_name,{})
    if not weights: st.warning("Buch leer."); st.stop()
    PRESETS={"2008 Equity Crash":{"Equity":-0.40,"Equity ETF":-0.38,"Fixed Income":0.08,"Commodity":-0.25,"Real Estate":-0.35},
        "Inflation Spike":{"Equity":-0.12,"Fixed Income":-0.08,"Commodity":0.25,"Real Estate":0.05},
        "Tech Sell-off":{"QQQ":-0.25,"NVDA":-0.35,"XLK":-0.22,"SMH":-0.30,"Equity":-0.08},
        "Oil +50%":{"CL=F":0.50,"USO":0.48,"XLE":0.20,"Equity":-0.05},
        "Mild Recession":{"Equity":-0.18,"Equity ETF":-0.16,"Fixed Income":0.04,"Commodity":-0.10}}
    preset=st.selectbox("Szenario", list(PRESETS.keys())+["Rate Shock","Eigenes"])
    rate_bp=st.slider("Zins-Schock (bp)", -200, 300, 150, 10)
    if preset=="Rate Shock" or preset=="Eigenes":
        fi=fixed_income_impact(weights, rate_bp)
        st.metric(f"FI Impact bei {rate_bp} bp", f"{fi['total_impact']*100:.2f}%")
        if not fi["details"].empty: st.dataframe(fi["details"], use_container_width=True)
    if preset=="Eigenes":
        s_eq=st.number_input("Equity %", -50.0, 30.0, -15.0)/100
        s_fi=st.number_input("Fixed Income %", -30.0, 20.0, -5.0)/100
        s_com=st.number_input("Commodity %", -40.0, 50.0, 0.0)/100
        shocks={"Equity":s_eq,"Equity ETF":s_eq,"Fixed Income":s_fi,"Commodity":s_com,"Real Estate":s_eq*0.7}
    elif preset in PRESETS: shocks=PRESETS[preset]
    else: shocks={}
    if shocks:
        res=stress_test(weights, shocks)
        st.metric("Portfolio Impact", f"{res['portfolio_impact']*100:.2f}%")
        st.dataframe(res["details"].sort_values("Beitrag %"), use_container_width=True)
    st.info("Lineare Näherungen. Convexity/Liquidität nicht modelliert.")

elif page == "📜 Fixed Income":
    st.header("Fixed Income Analytics")
    rows=[]
    for t in BOND_DURATION:
        m=single_metrics(t, ETFS.get(t,t), "Fixed Income")
        if m: m["Duration"]=BOND_DURATION[t]; m["Impact +100bp %"]=-BOND_DURATION[t]; rows.append(m)
    if rows:
        df=pd.DataFrame(rows)
        st.dataframe(df[["Ticker","Anlage","Kurs","Duration","Impact +100bp %","Volatilität %","1J Rendite %","Sharpe"]], use_container_width=True)
        bp=st.slider("Schock (bp)", -200, 300, 100, 10)
        impacts=[{"Ticker":t,"Duration":BOND_DURATION[t], f"Impact {bp}bp %":-BOND_DURATION[t]*(bp/10000)*100} for t in BOND_DURATION]
        st.plotly_chart(px.bar(pd.DataFrame(impacts), x="Ticker", y=f"Impact {bp}bp %", title=f"Impact bei {bp} bp"), use_container_width=True)
    for bname,wts in st.session_state.portfolio.items():
        fi_w={t:w for t,w in wts.items() if t in BOND_DURATION}
        if fi_w:
            st.write(f"**{bname}**"); fi=fixed_income_impact(fi_w,100)
            st.metric("Impact +100bp (FI-Anteil)", f"{fi['total_impact']*100:.2f}%")

elif page == "💱 FX & Multi-Currency":
    st.header("FX Exposure")
    for bname,wts in st.session_state.portfolio.items():
        if not wts: continue
        st.subheader(f"Buch: {bname}")
        ccy_exp={}
        for t,w in wts.items():
            c=ticker_currency(t); ccy_exp[c]=ccy_exp.get(c,0)+w
        cdf=pd.DataFrame([{"Währung":k,"Exposure %":v*100} for k,v in sorted(ccy_exp.items(), key=lambda x:-x[1])])
        c1,c2=st.columns(2)
        with c1: st.dataframe(cdf, use_container_width=True, hide_index=True)
        with c2: st.plotly_chart(px.pie(cdf, values="Exposure %", names="Währung"), use_container_width=True)

elif page == "📋 Trade Journal":
    st.header("Trade Journal")
    with st.form("tf"):
        c1,c2,c3,c4=st.columns(4)
        t_ticker=c1.text_input("Ticker","AAPL").upper(); t_side=c2.selectbox("Seite",["BUY","SELL"])
        t_qty=c3.number_input("Qty",0.01,value=10.0); t_px=c4.number_input("Preis",0.01,value=100.0)
        t_book=st.selectbox("Buch", list(st.session_state.portfolio.keys())); t_note=st.text_input("Notiz")
        if st.form_submit_button("Erfassen"):
            st.session_state.trades.append({"Zeit":NOW.strftime("%Y-%m-%d %H:%M"),"Buch":t_book,"Ticker":t_ticker,"Seite":t_side,"Qty":t_qty,"Preis":t_px,"Notional":t_qty*t_px,"Notiz":t_note})
            st.success("OK")
    if st.session_state.trades:
        st.dataframe(pd.DataFrame(st.session_state.trades), use_container_width=True, hide_index=True)
        if st.button("Leeren"): st.session_state.trades=[]; st.rerun()

elif page == "✅ Compliance":
    st.header("Compliance Limits")
    lim=st.session_state.compliance_limits
    c1,c2=st.columns(2)
    lim["max_single_position"]=c1.slider("Max Einzelposition",0.05,0.50,lim["max_single_position"],0.01)
    lim["max_asset_class"]=c2.slider("Max Asset Class",0.20,1.0,lim["max_asset_class"],0.05)
    for bname,wts in st.session_state.portfolio.items():
        if not wts: continue
        st.subheader(bname); breaches=[]
        for t,w in wts.items():
            if w>lim["max_single_position"]: breaches.append(f"⚠️ {t}: {w*100:.1f}% > {lim['max_single_position']*100:.0f}%")
        ac_w={}
        for t,w in wts.items():
            ac=ASSET_CLASS.get(t,"Other"); ac_w[ac]=ac_w.get(ac,0)+w
        for ac,w in ac_w.items():
            if w>lim["max_asset_class"]: breaches.append(f"⚠️ {ac}: {w*100:.1f}% > {lim['max_asset_class']*100:.0f}%")
        if breaches:
            for b in breaches: st.error(b)
        else: st.success("Keine Verletzungen")
        st.dataframe(pd.DataFrame([{"Ticker":t,"Gewicht %":w*100,"Asset Class":ASSET_CLASS.get(t,"—"),"Status":"BREACH" if w>lim["max_single_position"] else "OK"} for t,w in wts.items()]), use_container_width=True, hide_index=True)

elif page == "📄 Reports":
    st.header("Reports")
    lines=[f"ATLAS Report – {NOW.strftime('%Y-%m-%d %H:%M')}", "="*50, ""]
    for bname,wts in st.session_state.portfolio.items():
        lines.append(f"BUCH: {bname}"); lines.append("-"*30)
        for t,w in sorted(wts.items(), key=lambda x:-x[1]):
            lines.append(f"  {t:12s} {w*100:6.2f}%  {ALL_UNIVERSE.get(t,'')}")
        pr,_=portfolio_returns(wts,"2y")
        if not pr.empty:
            vol=pr.std(ddof=1)*np.sqrt(252); hv,hc=historical_var_cvar(pr,0.05)
            lines.append(f"  Vol: {vol*100:.1f}% | VaR95 1d: {hv*100:.2f}% | CVaR: {hc*100:.2f}%")
        lines.append("")
    txt="\n".join(lines); st.text_area("Report", txt, height=360)
    st.download_button("Download .txt", txt, file_name=f"atlas_report_{NOW.strftime('%Y%m%d')}.txt")
    rows=[{"Buch":b,"Ticker":t,"Name":ALL_UNIVERSE.get(t,""),"Gewicht":w,"Asset Class":ASSET_CLASS.get(t,""),"Währung":ticker_currency(t)} for b,wts in st.session_state.portfolio.items() for t,w in wts.items()]
    if rows: st.download_button("Portfolio CSV", pd.DataFrame(rows).to_csv(index=False), "atlas_portfolio.csv", "text/csv")

elif page == "📰 News":
    st.header("News")
    @st.cache_data(ttl=300, show_spinner=False)
    def rss_news():
        if not feedparser: return pd.DataFrame(), "feedparser fehlt"
        feeds=[("Reuters","https://feeds.reuters.com/Reuters/worldNews"),("BBC","https://feeds.bbci.co.uk/news/world/rss.xml"),
               ("Fed","https://www.federalreserve.gov/feeds/press_all.xml"),("ECB","https://www.ecb.europa.eu/rss/press.html"),
               ("SEC","https://www.sec.gov/news/pressreleases.rss")]
        rows,errs=[],[]
        cutoff=NOW-timedelta(days=2)
        for src,url in feeds:
            try:
                feed=feedparser.parse(url)
                for e in feed.entries[:25]:
                    title=getattr(e,"title",""); link=getattr(e,"link","")
                    summary=re.sub("<[^>]+>"," ",getattr(e,"summary",getattr(e,"description","")))
                    raw=getattr(e,"published_parsed",None) or getattr(e,"updated_parsed",None)
                    dt=datetime(*raw[:6],tzinfo=timezone.utc).astimezone(TZ) if raw else None
                    if dt and dt>=cutoff:
                        rows.append({"Zeit":dt.strftime("%d.%m.%Y %H:%M"),"Quelle":src,"Meldung":title,"Kurzinfo":summary.strip()[:350],"Link":link,"dt":dt})
            except Exception: errs.append(src)
        df=pd.DataFrame(rows)
        if not df.empty: df=df.sort_values("dt",ascending=False).drop_duplicates(subset=["Meldung"])
        return df, (", ".join(errs) if errs else "")
    news,status=rss_news()
    if status: st.caption("Nicht erreichbar: "+status)
    if news.empty: st.warning("Keine Meldungen")
    else:
        q=st.text_input("Filter", placeholder="Zinsen, KI, Öl …")
        if q: news=news[news.apply(lambda r: q.lower() in (str(r.Meldung)+str(r.Kurzinfo)).lower(), axis=1)]
        for _,r in news.head(35).iterrows():
            with st.container(border=True):
                st.markdown(f"**{r['Meldung']}**"); st.caption(f"{r['Quelle']} · {r['Zeit']}")
                if r["Kurzinfo"]: st.write(r["Kurzinfo"])
                if r["Link"]: st.markdown(f"[Quelle]({r['Link']})")

st.divider()
st.caption("ATLAS · Keine Anlageberatung · VaR/CVaR/Stress = Modellwerte · Duration-Näherungen · Multi-Book session-lokal")
