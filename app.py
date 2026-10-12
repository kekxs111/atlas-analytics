from __future__ import annotations
import math, re, warnings
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

try:
    import feedparser
except Exception:
    feedparser = None
try:
    import requests
except Exception:
    requests = None

st.set_page_config(page_title="ATLAS Analytics", page_icon="🌐", layout="wide")
TZ = ZoneInfo("Europe/Berlin")
now = datetime.now(TZ)

# Broad, curated universe; no free data source reliably covers every security worldwide.
STOCKS = {
# USA
"AAPL":"Apple","MSFT":"Microsoft","NVDA":"NVIDIA","AMZN":"Amazon","GOOGL":"Alphabet A","GOOG":"Alphabet C","META":"Meta Platforms","BRK-B":"Berkshire Hathaway B","LLY":"Eli Lilly","AVGO":"Broadcom","JPM":"JPMorgan Chase","V":"Visa","MA":"Mastercard","XOM":"Exxon Mobil","COST":"Costco","WMT":"Walmart","NFLX":"Netflix","ORCL":"Oracle","AMD":"AMD","CRM":"Salesforce","ADBE":"Adobe","QCOM":"Qualcomm","INTC":"Intel","TSLA":"Tesla","KO":"Coca-Cola","PEP":"PepsiCo","UNH":"UnitedHealth","CAT":"Caterpillar","GE":"GE Aerospace","IBM":"IBM","CSCO":"Cisco","AMAT":"Applied Materials","MU":"Micron","PLTR":"Palantir","UBER":"Uber","ABNB":"Airbnb","DIS":"Walt Disney","MCD":"McDonald's","BA":"Boeing","GS":"Goldman Sachs","BAC":"Bank of America","C":"Citigroup","WFC":"Wells Fargo","CVX":"Chevron","COP":"ConocoPhillips","PFE":"Pfizer","MRK":"Merck","TMO":"Thermo Fisher","ISRG":"Intuitive Surgical","NOW":"ServiceNow","PANW":"Palo Alto Networks","SNPS":"Synopsys","CRWD":"CrowdStrike","SHOP":"Shopify","DELL":"Dell Technologies","ARM":"Arm Holdings","SMCI":"Super Micro Computer","LRCX":"Lam Research","KLAC":"KLA","TXN":"Texas Instruments","BKNG":"Booking Holdings","SPOT":"Spotify","PYPL":"PayPal","T":"AT&T","VZ":"Verizon","NKE":"Nike","LOW":"Lowe's","HD":"Home Depot","UPS":"UPS","FDX":"FedEx","AXP":"American Express","BLK":"BlackRock","SCHW":"Charles Schwab","COIN":"Coinbase","MSTR":"Strategy","F":"Ford","GM":"General Motors",
# Europe
"SAP":"SAP","SIE.DE":"Siemens","ALV.DE":"Allianz","DTE.DE":"Deutsche Telekom","MUV2.DE":"Munich Re","AIR.PA":"Airbus","MC.PA":"LVMH","OR.PA":"L'Oreal","TTE.PA":"TotalEnergies","SAN.PA":"Sanofi","SU.PA":"Schneider Electric","ASML":"ASML","RHM.DE":"Rheinmetall","MBG.DE":"Mercedes-Benz","BMW.DE":"BMW","VOW3.DE":"Volkswagen","BAS.DE":"BASF","IFX.DE":"Infineon","ADS.DE":"Adidas","DBK.DE":"Deutsche Bank","ENR.DE":"Siemens Energy","BN.PA":"Danone","NOVO-B.CO":"Novo Nordisk","NVO":"Novo Nordisk ADR","AZN":"AstraZeneca","SHEL":"Shell","HSBA.L":"HSBC","BP.L":"BP","ULVR.L":"Unilever","RIO":"Rio Tinto","GSK":"GSK","REL.L":"RELX","LSEG.L":"London Stock Exchange Group","BHP":"BHP Group","NESN.SW":"Nestle","ROG.SW":"Roche","NOVN.SW":"Novartis","UBSG.SW":"UBS","7203.T":"Toyota","6758.T":"Sony","9984.T":"SoftBank Group","005930.KS":"Samsung Electronics","000660.KS":"SK Hynix","0700.HK":"Tencent","9988.HK":"Alibaba","3690.HK":"Meituan","TSM":"TSMC ADR","BABA":"Alibaba ADR","INFY":"Infosys ADR","RELIANCE.NS":"Reliance Industries","^BSESN":"BSE Sensex","^N225":"Nikkei 225",
# Additional US-listed equities across sectors
"ACN":"Accenture","ADP":"Automatic Data Processing","ADI":"Analog Devices","AIG":"American International Group","AMGN":"Amgen","APH":"Amphenol","APP":"AppLovin","APO":"Apollo Global Management","AXON":"Axon Enterprise","AXP":"American Express","BABA":"Alibaba ADR","BKR":"Baker Hughes","BMY":"Bristol Myers Squibb","BX":"Blackstone","CARR":"Carrier Global","CCL":"Carnival","CEG":"Constellation Energy","CHTR":"Charter Communications","CMCSA":"Comcast","CMG":"Chipotle Mexican Grill","CMI":"Cummins","COR":"Cencora","CPRT":"Copart","CRH":"CRH","CTAS":"Cintas","CVS":"CVS Health","DASH":"DoorDash","DDOG":"Datadog","DE":"Deere","DG":"Dollar General","DHR":"Danaher","DLR":"Digital Realty","DUK":"Duke Energy","EA":"Electronic Arts","EBAY":"eBay","ECL":"Ecolab","ELV":"Elevance Health","EMR":"Emerson Electric","ETN":"Eaton","FIS":"Fidelity National Information Services","FTNT":"Fortinet","GD":"General Dynamics","GILD":"Gilead Sciences","GMED":"Globus Medical","GPN":"Global Payments","HAL":"Halliburton","HON":"Honeywell","HPE":"Hewlett Packard Enterprise","HPQ":"HP Inc.","ICE":"Intercontinental Exchange","INTU":"Intuit","ITW":"Illinois Tool Works","JCI":"Johnson Controls","KDP":"Keurig Dr Pepper","KKR":"KKR","KHC":"Kraft Heinz","KMI":"Kinder Morgan","LHX":"L3Harris Technologies","LIN":"Linde","LMT":"Lockheed Martin","LULU":"Lululemon","MAR":"Marriott International","MCHP":"Microchip Technology","MDLZ":"Mondelez International","MET":"MetLife","MMM":"3M","MNST":"Monster Beverage","MO":"Altria","MRVL":"Marvell Technology","MS":"Morgan Stanley","MSI":"Motorola Solutions","NEM":"Newmont","NOC":"Northrop Grumman","ODFL":"Old Dominion Freight Line","ON":"ON Semiconductor","OXY":"Occidental Petroleum","PCAR":"PACCAR","PDD":"PDD Holdings","PGR":"Progressive","PH":"Parker Hannifin","PLD":"Prologis","PM":"Philip Morris International","PNC":"PNC Financial Services","PSA":"Public Storage","RBLX":"Roblox","REGN":"Regeneron Pharmaceuticals","ROP":"Roper Technologies","ROST":"Ross Stores","RTX":"RTX","SBUX":"Starbucks","SLB":"SLB","SNOW":"Snowflake","SO":"Southern Company","STX":"Seagate Technology","SYK":"Stryker","TGT":"Target","TJX":"TJX Companies","TMUS":"T-Mobile US","TRV":"Travelers","TT":"Trane Technologies","TTD":"The Trade Desk","TWLO":"Twilio","U":"Unity Software","VLO":"Valero Energy","VRTX":"Vertex Pharmaceuticals","WELL":"Welltower","WM":"Waste Management","WMB":"Williams Companies","WDAY":"Workday","XEL":"Xcel Energy","ZTS":"Zoetis",
# Additional international equities / ADRs
"7203.T":"Toyota Motor","6758.T":"Sony Group","8035.T":"Tokyo Electron","9983.T":"Fast Retailing","8306.T":"Mitsubishi UFJ Financial","6861.T":"Keyence","6501.T":"Hitachi","7974.T":"Nintendo","9432.T":"NTT","8058.T":"Mitsubishi Corporation","005380.KS":"Hyundai Motor","051910.KS":"LG Chem","005490.KS":"POSCO Holdings","035420.KS":"NAVER","066570.KS":"LG Electronics","000270.KS":"Kia","1810.HK":"Xiaomi","0941.HK":"China Mobile","1299.HK":"AIA Group","9618.HK":"JD.com","9999.HK":"NetEase","1024.HK":"Kuaishou","2318.HK":"Ping An Insurance","1398.HK":"ICBC","3988.HK":"Bank of China","D05.SI":"DBS Group","O39.SI":"OCBC","CBA.AX":"Commonwealth Bank of Australia","BHP.AX":"BHP Group Australia","CSL.AX":"CSL Limited","WES.AX":"Wesfarmers","NAB.AX":"National Australia Bank","RIO.AX":"Rio Tinto Australia","7203.T":"Toyota Motor","BARC.L":"Barclays","LLOY.L":"Lloyds Banking Group","DGE.L":"Diageo","PRU.L":"Prudential","III.L":"3i Group","AZN.L":"AstraZeneca London","BATS.L":"British American Tobacco","IBE.MC":"Iberdrola","ITX.MC":"Inditex","SAN.MC":"Banco Santander","BBVA.MC":"BBVA","ENEL.MI":"Enel","ISP.MI":"Intesa Sanpaolo","UCG.MI":"UniCredit","KER.PA":"Kering","CS.PA":"AXA","BNP.PA":"BNP Paribas","DG.PA":"Vinci","SAF.PA":"Safran","RACE.MI":"Ferrari","STLAM.MI":"Stellantis","ADYEN.AS":"Adyen","PHIA.AS":"Philips","PRX.AS":"Prosus","DSY.PA":"Dassault Systemes","BAYN.DE":"Bayer","HNR1.DE":"Hannover Rueck","HEI.DE":"Heidelberg Materials","FRE.DE":"Fresenius","BEI.DE":"Beiersdorf","ZAL.DE":"Zalando","RWE.DE":"RWE","VNA.DE":"Vonovia","MTX.DE":"MTU Aero Engines","HEN3.DE":"Henkel","PUM.DE":"Puma","CON.DE":"Continental","QIA.DE":"QIAGEN","DHL.DE":"DHL Group","DB1.DE":"Deutsche Boerse","VOW3.DE":"Volkswagen preferred","KNIN.SW":"Kuehne + Nagel","ZURN.SW":"Zurich Insurance","ABBN.SW":"ABB","NOVO-B.CO":"Novo Nordisk Copenhagen","EQNR.OL":"Equinor","DNB.OL":"DNB Bank","VOLV-B.ST":"Volvo Group","ERIC-B.ST":"Ericsson","NESTE.HE":"Neste","ENI.MI":"Eni","TCS.NS":"Tata Consultancy Services","HDFCBANK.NS":"HDFC Bank","ICICIBANK.NS":"ICICI Bank","BHARTIARTL.NS":"Bharti Airtel","LT.NS":"Larsen & Toubro","SBIN.NS":"State Bank of India","ITC.NS":"ITC Limited","TATAMOTORS.NS":"Tata Motors","INFY.NS":"Infosys","RELIANCE.NS":"Reliance Industries"
}
ETFS = {
"SPY":"SPDR S&P 500","VOO":"Vanguard S&P 500","QQQ":"Invesco Nasdaq-100","VTI":"Vanguard Total US Market","VT":"Vanguard Total World","ACWI":"iShares MSCI ACWI","VEA":"Vanguard Developed Markets","VWO":"Vanguard Emerging Markets","IEFA":"iShares Core MSCI EAFE","IEMG":"iShares Core MSCI Emerging Markets","EFA":"iShares MSCI EAFE","EEM":"iShares MSCI Emerging Markets","DIA":"SPDR Dow Jones Industrial Average","IWM":"iShares Russell 2000","SCHD":"Schwab US Dividend Equity","VIG":"Vanguard Dividend Appreciation","XLK":"Technology Select Sector","XLF":"Financial Select Sector","XLE":"Energy Select Sector","XLV":"Health Care Select Sector","XLI":"Industrial Select Sector","SMH":"VanEck Semiconductor","SOXX":"iShares Semiconductor","ARKK":"ARK Innovation","TLT":"iShares 20+ Year Treasury Bond","IEF":"iShares 7-10 Year Treasury Bond","SHY":"iShares 1-3 Year Treasury Bond","AGG":"iShares Core US Aggregate Bond","BND":"Vanguard Total Bond Market","LQD":"iShares Investment Grade Corporate Bond","HYG":"iShares High Yield Corporate Bond","TIP":"iShares TIPS Bond","GLD":"SPDR Gold Shares","IAU":"iShares Gold Trust","SLV":"iShares Silver Trust","PDBC":"Invesco Optimum Yield Diversified Commodity","DBC":"Invesco DB Commodity Index","USO":"United States Oil Fund","UNG":"United States Natural Gas Fund","VNQ":"Vanguard Real Estate","REET":"iShares Global REIT","EUNL.DE":"iShares Core MSCI World UCITS","IWDA.AS":"iShares Core MSCI World","VWCE.DE":"Vanguard FTSE All-World UCITS","VUSA.AS":"Vanguard S&P 500 UCITS","CSPX.L":"iShares Core S&P 500 UCITS","SXR8.DE":"iShares Core S&P 500 UCITS","EXSA.DE":"iShares STOXX Europe 600","IS3N.DE":"iShares Core MSCI EM IMI","XDWT.DE":"Xtrackers MSCI World Information Technology","IQQH.DE":"iShares Global Clean Energy UCITS","EIMI.L":"iShares Core MSCI EM IMI"
}
COMMODITIES = {
"GC=F":"Gold futures","SI=F":"Silver futures","PL=F":"Platinum futures","PA=F":"Palladium futures","HG=F":"Copper futures","CL=F":"WTI crude oil","BZ=F":"Brent crude oil","NG=F":"Natural gas","ALI=F":"Aluminium futures","ZC=F":"Corn futures","ZW=F":"Wheat futures","ZS=F":"Soybean futures","KC=F":"Coffee futures","SB=F":"Sugar futures","CT=F":"Cotton futures","CC=F":"Cocoa futures","LE=F":"Live cattle","HE=F":"Lean hogs","LBR=F":"Lumber futures"
}
PHYSICAL_METALS = {"GC=F":"Gold (exchange-traded futures proxy)","SI=F":"Silver (exchange-traded futures proxy)","PL=F":"Platinum (exchange-traded futures proxy)","PA=F":"Palladium (exchange-traded futures proxy)"}
BOND_DURATION = {"TLT":16.0,"IEF":7.5,"SHY":1.8,"AGG":6.0,"BND":6.0,"LQD":8.0,"HYG":3.5,"TIP":6.5}

@st.cache_data(ttl=900, show_spinner=False)
def fetch_history(ticker: str) -> pd.DataFrame:
    try:
        df = yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=True, actions=False)
        if df is None or df.empty or "Close" not in df: return pd.DataFrame()
        df = df[["Close"]].dropna()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df[~df.index.duplicated(keep="last")].sort_index()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=900, show_spinner=False)
def metrics(ticker, name, kind):
    d = fetch_history(ticker)
    if d.empty or len(d) < 3: return None
    p = d.Close.astype(float); r = p.pct_change().dropna()
    if r.empty: return None
    years = max((p.index[-1]-p.index[0]).days/365.25, 1/365.25)
    total = p.iloc[-1]/p.iloc[0]-1
    ann = (p.iloc[-1]/p.iloc[0])**(1/years)-1 if p.iloc[-1] > 0 and p.iloc[0] > 0 else np.nan
    vol = r.std(ddof=1)*np.sqrt(252)
    dd = (p/p.cummax()-1).min()
    ma50 = p.rolling(50).mean().iloc[-1] if len(p)>=50 else np.nan
    ma200 = p.rolling(200).mean().iloc[-1] if len(p)>=200 else np.nan
    momentum = p.iloc[-1]/p.iloc[-min(len(p),253)]-1
    sharpe = (r.mean()*252)/(r.std(ddof=1)*np.sqrt(252)) if r.std(ddof=1)>0 else np.nan
    # Transparent illustrative score: growth, trend, risk and drawdown are all represented.
    score = 50 + 28*np.tanh((ann if np.isfinite(ann) else 0)*2) + 12*np.tanh((momentum if np.isfinite(momentum) else 0)*2) - 18*min(float(vol) if np.isfinite(vol) else 1,1.5) - 10*min(abs(float(dd)) if np.isfinite(dd) else 1,1.0)
    risk = "Niedrig" if vol < .15 and abs(dd)<.20 else "Mittel" if vol < .30 and abs(dd)<.40 else "Hoch"
    return {"Ticker":ticker,"Anlage":name,"Typ":kind,"Kurs":p.iloc[-1],"Seit Beginn %":total*100,"Rendite p.a. %":ann*100,"Volatilität %":vol*100,"Max. Verlustphase %":dd*100,"1J Rendite %":momentum*100,"Sharpe":sharpe,"Score":score,"Risiko":risk,"Startdatum":p.index[0].date().isoformat(),"Letztes Datum":p.index[-1].date().isoformat(),"Datenpunkte":len(p)}

def get_table(universe, kind, selected=None):
    tickers = list(selected if selected is not None else universe.keys())
    out=[]; progress=st.progress(0, text=f"Lade {kind} …")
    for i,t in enumerate(tickers):
        row=metrics(t, universe.get(t,t), kind)
        if row: out.append(row)
        progress.progress((i+1)/max(len(tickers),1), text=f"Prüfe {t} ({i+1}/{len(tickers)})")
    progress.empty()
    if not out: return pd.DataFrame()
    return pd.DataFrame(out).sort_values("Score",ascending=False).reset_index(drop=True)

HIGH_BETTER = {"Seit Beginn %","Rendite p.a. %","1J Rendite %","Sharpe","Score"}
LOW_BETTER = {"Volatilität %","Max. Verlustphase %"}
def style_table(df):
    if df.empty: return df
    def colors(col):
        if col.name in HIGH_BETTER:
            best, worst = col.idxmax(), col.idxmin()
        elif col.name in LOW_BETTER:
            best, worst = col.idxmin(), col.idxmax()
        else: return [""]*len(col)
        return ["background-color: #c8f7c5; color: #123b12; font-weight: 600" if i==best else "background-color: #ffd2d2; color: #7d1010; font-weight: 600" if i==worst else "" for i in col.index]
    numeric=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    fmt={c:"{:,.2f}" for c in numeric}
    return df.style.apply(colors,axis=0).format(fmt, na_rep="—")

def rss_news():
    if not feedparser: return pd.DataFrame(), "Paket feedparser fehlt."
    feeds = [
      ("Reuters World","https://feeds.reuters.com/Reuters/worldNews"),
      ("BBC World","https://feeds.bbci.co.uk/news/world/rss.xml"),
      ("BBC Technology","https://feeds.bbci.co.uk/news/technology/rss.xml"),
      ("AP News","https://apnews.com/hub/world-news?output=rss"),
      ("NASA","https://www.nasa.gov/news-release/feed/"),
      ("European Central Bank","https://www.ecb.europa.eu/rss/press.html"),
      ("US Federal Reserve","https://www.federalreserve.gov/feeds/press_all.xml"),
      ("European Commission","https://ec.europa.eu/commission/presscorner/api/rss?language=en"),
      ("SEC","https://www.sec.gov/news/pressreleases.rss")]
    rows=[]; errors=[]; cutoff=now-timedelta(days=2)
    for source,url in feeds:
        try:
            feed=feedparser.parse(url)
            for e in feed.entries[:35]:
                title=getattr(e,"title",""); link=getattr(e,"link",""); summary=re.sub('<[^>]+>',' ',getattr(e,"summary",getattr(e,"description","")))
                raw=getattr(e,"published_parsed",None) or getattr(e,"updated_parsed",None)
                dt=datetime(*raw[:6],tzinfo=timezone.utc).astimezone(TZ) if raw else None
                if dt and dt >= cutoff:
                    rows.append({"Zeit (lokal)":dt.strftime("%d.%m.%Y %H:%M"),"Quelle":source,"Meldung":title,"Kurzinfo":summary.strip()[:450],"Link":link,"Zeitobjekt":dt})
        except Exception as ex: errors.append(source)
    df=pd.DataFrame(rows)
    if not df.empty: df=df.sort_values("Zeitobjekt",ascending=False).drop_duplicates(subset=["Meldung"])
    return df, ("Nicht erreichbar: "+", ".join(errors) if errors else "")

st.title("🌐 ATLAS Analytics")
st.caption("Markt-Screener · Langfristige Kursdaten · Risikoanalyse · Nachrichtenlage. Keine Anlageberatung.")
with st.sidebar:
    st.header("Einstellungen")
    refresh=st.button("🔄 Daten aktualisieren")
    if refresh:
        st.cache_data.clear()
        for key in ("commodities_df", "metals_df", "idea_universe_df", "idea_df", "idea_universe_version"):
            st.session_state.pop(key, None)
        st.rerun()
    st.caption("Kursdaten: Yahoo Finance, Aktualisierungscache 15 Minuten. Verfügbarkeit und Historie unterscheiden sich je Ticker.")
    st.divider()
    st.subheader("Investment-Szenario")
    money=st.number_input("Anlagebetrag (€)", min_value=10.0, max_value=10_000_000.0, value=1000.0, step=100.0, key="scenario_money")
    horizon=st.select_slider("Anlagezeitraum", options=["1 Monat","3 Monate","6 Monate","1 Jahr","3 Jahre","5 Jahre","10 Jahre"], value="5 Jahre", key="scenario_horizon")
    risk_pref=st.selectbox("Gewünschtes Risiko",["Niedrig","Mittel","Hoch","Alle"],index=1,key="scenario_risk")
    growth_pref=st.selectbox("Wachstumsziel",["Stabil","Ausgewogen","Hoch","Maximal"],index=1,key="scenario_growth")
    # Keep the ten suggestions unchanged until the user explicitly applies the scenario.
    if "applied_scenario" not in st.session_state:
        st.session_state.applied_scenario = {"money": 1000.0, "horizon": "5 Jahre", "risk": "Mittel", "growth": "Ausgewogen"}
    if st.button("Auf 10 Vorschläge anwenden", type="primary", key="apply_scenario_to_suggestions"):
        st.session_state.applied_scenario = {"money": float(money), "horizon": horizon, "risk": risk_pref, "growth": growth_pref}
        st.rerun()
    applied_scenario = st.session_state.applied_scenario
    applied_money = float(applied_scenario["money"])
    applied_horizon = applied_scenario["horizon"]
    applied_risk = applied_scenario["risk"]
    applied_growth = applied_scenario["growth"]
    st.caption(f"Zuletzt angewendet: {applied_money:,.2f} € · {applied_horizon} · Risiko {applied_risk} · Wachstum {applied_growth}.")
    st.caption("Vorschläge sind datenbasierte Beispiele, keine persönliche Finanzberatung oder Garantie.")

# Main universe
all_universe={**STOCKS,**ETFS,**COMMODITIES}
with st.spinner("Lade Marktübersichten …"):
    # The entire curated universe is screened, but users can reduce it for speed.
    universe_choice=st.radio("Umfang",["Komplette kuratierte Liste","Aktien","ETFs","Rohstoffe"],horizontal=True)
    if universe_choice=="Aktien": active={**STOCKS}
    elif universe_choice=="ETFs": active={**ETFS}
    elif universe_choice=="Rohstoffe": active={**COMMODITIES}
    else: active=all_universe
    st.caption(f"Auswahl: {len(active)} Symbole aus mehreren Märkten. Eine vollständige Liste jedes weltweit gelisteten Wertpapiers ist mit dieser kostenlosen Kursquelle nicht zuverlässig verfügbar.")
    st.caption("Die Rangliste lädt automatisch. Beim ersten Start kann die große weltweite Auswahl einige Minuten benötigen; Kursdaten werden anschließend 15 Minuten zwischengespeichert.")

# Automatic ranking load; no separate load button. Cached metric calculations avoid repeated price requests.
display_df = get_table(active, universe_choice)

# dedicated metals ranking: longest provider history available
st.header("Ranglisten")
st.subheader("📈 Aktien & ETFs / Gesamtmarkt")
st.write("Die Rangliste nutzt jeweils die **längste von der Datenquelle gelieferte Historie** (`period=max`). Die Score-Spalte ist eine vereinfachte Kennzahl, kein Kursziel.")
df=display_df
if not df.empty:
    types=st.multiselect("Anlagearten anzeigen",sorted(df.Typ.unique()),default=sorted(df.Typ.unique()),key="typefilter")
    view=df[df.Typ.isin(types)].copy()
    st.dataframe(style_table(view),use_container_width=True,height=540)
    pick=st.selectbox("Kursverlauf ansehen",view.Ticker.tolist(),key="chart_stock")
    h=fetch_history(pick)
    if not h.empty: st.plotly_chart(px.line(h.reset_index(),x=h.index.name or "Date",y="Close",title=f"{pick} – verfügbare Gesamthistorie"),use_container_width=True)
else: st.info("Für die ausgewählten Symbole sind derzeit keine ausreichenden Kursdaten abrufbar. Nutze bei Bedarf „Daten aktualisieren“.")
st.divider()
st.subheader("🛢️ Rohstoffe")
st.write("Rohstoffe werden überwiegend über börsengehandelte Futures abgebildet. Futures können Rollverluste, Laufzeitunterschiede und besondere Risiken aufweisen.")
if "commodities_df" not in st.session_state:
    st.session_state.commodities_df=get_table(COMMODITIES,"Rohstoffe")
cdf=st.session_state.get("commodities_df",pd.DataFrame())
if not cdf.empty: st.dataframe(style_table(cdf),use_container_width=True,height=520)
st.divider()
st.subheader("🥇 Physische Edelmetalle")
st.warning("Wichtig: Diese Liste verwendet Futures als Preis-Proxy für Edelmetalle, nicht den tatsächlichen Händlerpreis für Barren oder Münzen. Aufschläge, Lagerung und Kauf-/Verkaufspreise fehlen.")
if "metals_df" not in st.session_state:
    st.session_state.metals_df=get_table(PHYSICAL_METALS,"Physische Edelmetalle")
mdf=st.session_state.get("metals_df",pd.DataFrame())
if not mdf.empty:
    st.dataframe(style_table(mdf),use_container_width=True)
    mt=st.selectbox("Edelmetallkurs",mdf.Ticker.tolist(),key="metal_chart")
    mh=fetch_history(mt)
    if not mh.empty:
        fig=go.Figure(); fig.add_trace(go.Scatter(x=mh.index,y=mh.Close,mode="lines",name=mt)); fig.update_layout(title=f"{mt} – längste verfügbare Kursgeschichte",xaxis_title="Datum",yaxis_title="Preis (notierungsabhängig)")
        st.plotly_chart(fig,use_container_width=True)
st.divider()
st.subheader("🧮 Risiko & Portfolio")
st.subheader("Portfolio-Risiko-Rechner")
st.write("Füge Ticker und Gewichte ein. Die Gewichte werden für die Rechnung auf 100 % normiert.")
default_port=pd.DataFrame({"Ticker":["SPY","TLT","GLD"],"Gewicht %":[50.0,30.0,20.0]})
port=st.data_editor(default_port,num_rows="dynamic",use_container_width=True,key="portfolio_editor")
conf=st.selectbox("VaR-Konfidenz",["95 %","99 %"]); days=st.slider("VaR-Haltedauer (Tage)",1,252,1)
with st.expander("⚙️ Monte-Carlo-Formel einstellen", expanded=False):
    mc_sims = st.slider("Anzahl der Simulationen", 500, 20000, 5000, step=500, key="risk_mc_sims")
    mc_days = st.slider("Simulationszeitraum (Handelstage)", 21, 1260, 252, step=21, key="risk_mc_days")
    mc_mu_factor = st.slider("Rendite-Faktor μ", 0.0, 2.0, 1.0, step=0.1, key="risk_mc_mu")
    mc_sigma_factor = st.slider("Volatilitäts-Faktor σ", 0.0, 3.0, 1.0, step=0.1, key="risk_mc_sigma")
    mc_seed = st.number_input("Zufallsstartwert (Seed)", min_value=0, max_value=999999, value=42, step=1, key="risk_mc_seed")
    st.caption("Formel: Sₜ₊₁ = Sₜ · exp((μ − σ²/2)Δt + σ√Δt · Z). μ = historische mittlere Rendite, σ = historische Volatilität, Z = Zufallszahl.")
st.markdown(
    """**Kurzanleitung – Risiko & Portfolio (8 Punkte)**

    1. **Anzahl der Simulationen:** Mehr Durchläufe machen die Ergebnisse meist stabiler, brauchen aber länger.
    2. **Simulationszeitraum:** Legt fest, wie viele Handelstage simuliert werden.
    3. **Rendite-Faktor μ:** 0 blendet die historische Rendite aus; 1 nutzt sie unverändert; über 1 verstärkt sie sich.
    4. **Volatilitäts-Faktor σ:** 0 blendet Schwankungen aus; 1 nutzt die historische Volatilität; höhere Werte erhöhen das modellierte Risiko.
    5. **Seed:** Gleicher Wert und gleiche Daten liefern reproduzierbare Simulationen.
    6. **Monte-Carlo-Formel:** Erzeugt mögliche Kursverläufe aus Rendite, Volatilität und Zufallszahlen.
    7. **Risiko berechnen:** Startet die Analyse mit den eingestellten Parametern.
    8. **Hinweis:** Ergebnisse sind Szenarien, keine sichere Kursvorhersage."""
)
if st.button("Risiko berechnen",type="primary",key="risk_calc"):
    try:
        weights={str(r.Ticker).strip().upper():float(r["Gewicht %"])/100 for _,r in port.iterrows() if str(r.Ticker).strip() and pd.notna(r["Gewicht %"]) and float(r["Gewicht %"])>0}
        if not weights: st.error("Bitte mindestens einen Ticker mit positivem Gewicht eingeben.")
        else:
            hist={t:fetch_history(t).Close.pct_change().dropna().tail(756) for t in weights}
            rets=pd.concat(hist,axis=1).dropna(how="all").fillna(0)
            valid=[t for t in weights if t in rets and rets[t].notna().sum()>20]
            if not valid: st.error("Keine ausreichenden Kursdaten für die eingegebenen Ticker.")
            else:
                w=np.array([weights[t] for t in valid],float); w=w/w.sum(); R=rets[valid].tail(756); cov=R.cov().values
                daily_var=float(w.T@cov@w); daily_sigma=math.sqrt(max(daily_var,0)); z=1.65 if conf=="95 %" else 2.33
                var=z*daily_sigma*money*math.sqrt(days)
                st.metric(f"Parametrischer VaR ({conf}, {days} Tag(e))",f"€ {var:,.2f}",help="Modellschätzung: bei gewähltem Konfidenzniveau wird der Verlust in diesem Modell in den meisten Fällen nicht überschritten; Extremverluste können größer sein.")
                st.metric("Geschätzte jährliche Volatilität",f"{daily_sigma*math.sqrt(252)*100:.2f}%")
                st.metric("Summe der Portfolio-Gewichte",f"{sum(weights[t] for t in valid)/sum(weights[t] for t in valid)*100:.0f}% (normiert)")
                corr=R.corr(); st.write("**Korrelationsmatrix** (niedrigere Korrelation kann Diversifikation ermöglichen)"); st.dataframe(corr.style.format("{:.2f}").background_gradient(axis=None),use_container_width=True)
                st.caption("Formel: σₚ² = wᵀΣw. Parametrischer VaR: z · σ · V · √t. Die Rechnung nutzt historische tägliche Renditen und setzt für den VaR eine näherungsweise Normalverteilung voraus.")
                # factor model using broad market proxy SPY and bond proxy TLT if present
                st.write("**Faktormodell (vereinfachtes Beispiel)**")
                factors={"Markt (SPY)":"SPY","Anleihen (TLT)":"TLT","Gold (GLD)":"GLD"}
                factor_returns={label:fetch_history(t).Close.pct_change().rename(label) for label,t in factors.items() if t not in valid and not fetch_history(t).empty}
                if factor_returns:
                    fdf=pd.concat([R.mean(axis=1).rename("Portfolio").to_frame(),*factor_returns.values()],axis=1).dropna()
                    if len(fdf)>30:
                        y=fdf["Portfolio"].values; X=np.column_stack([np.ones(len(fdf)),fdf.drop(columns="Portfolio").values]); coef=np.linalg.lstsq(X,y,rcond=None)[0]
                        st.dataframe(pd.DataFrame({"Faktor":["Alpha"]+list(fdf.columns[1:]),"Koeffizient":coef}),use_container_width=True)
                # Monte Carlo, using daily historical mean and covariance
                rng=np.random.default_rng(int(mc_seed)); mu=R.mean().values * float(mc_mu_factor); covm=R.cov().values * float(mc_sigma_factor)**2
                try: sims=rng.multivariate_normal(mu,covm,size=(int(mc_sims),int(mc_days)),check_valid="ignore")
                except Exception: sims=rng.normal(mu,np.sqrt(np.maximum(np.diag(covm),1e-12)),size=(int(mc_sims),int(mc_days),len(valid)))
                port_daily=np.einsum("s t a,a->s t",sims,w); terminal=money*np.exp(np.log1p(np.clip(port_daily,-.99,None)).sum(axis=1))
                st.write(f"**Monte-Carlo: simulierte Portfolio-Werte nach {int(mc_days)} Handelstagen**")
                mcdf=pd.DataFrame({"Portfolio-Wert (€)":terminal}); st.plotly_chart(px.histogram(mcdf,x="Portfolio-Wert (€)",nbins=50,title=f"Simulierte Endwerte nach {int(mc_days)} Handelstagen (keine Prognose)"),use_container_width=True)
                q05=float(np.quantile(terminal,.05)); q50=float(np.quantile(terminal,.5)); q95=float(np.quantile(terminal,.95))
                st.write(f"5%-Quantil: **€ {q05:,.2f}** · Median: **€ {q50:,.2f}** · 95%-Quantil: **€ {q95:,.2f}**")
                st.caption("Monte-Carlo verwendet historische Mittelwerte/Kovarianzen. Ergebnisse hängen stark von Modellannahmen ab und sind keine Vorhersage.")
                # stress test: shocks on equity-like vs bond-like / commodity proxies
                shocks=[]
                for label,shock in [("Aktien −30 %",-.30),("Aktien +20 %",.20),("Zinsen +2 %-Punkte (Anleihen-Näherung)",None),("Gold −15 %",-.15),("Alle Anlagen −20 %",-.20)]:
                    change=0.0
                    for t,wi in zip(valid,w):
                        if label.startswith("Aktien"):
                            s=shock if t in {"SPY","VOO","QQQ","VTI","VT","ACWI","EUNL.DE","VWCE.DE","SXR8.DE"} else 0.0
                        elif label.startswith("Zinsen"):
                            dur=BOND_DURATION.get(t,0); s=-dur*.02 + .5*(dur**2)*(.02**2)*.25 if dur else 0
                        elif label.startswith("Gold"): s=-.15 if t in {"GLD","IAU","GC=F"} else 0
                        else: s=-.20
                        change+=wi*s
                    shocks.append({"Szenario":label,"Geschätzte Veränderung %":change*100,"Wert danach (€)":money*(1+change)})
                st.write("**Stresstest**"); st.dataframe(style_table(pd.DataFrame(shocks)),use_container_width=True)
                st.caption("Anleihen-Näherung: ΔP/P ≈ −Dₘₒd·Δy + ½·C·(Δy)². Durationen sind Näherungswerte je ETF; Konvexität ist hier grob modelliert. Für Nicht-Anleihen wird keine Duration angesetzt.")
    except Exception as e: st.error(f"Risikoanalyse fehlgeschlagen: {e}")

st.divider()
st.header("💡 10 gemischte Portfolio-Vorschläge")
st.write("Jeder Vorschlag besteht aus mehreren Anlagearten – zum Beispiel Aktien, ETFs und gegebenenfalls Rohstoffen. Die Gewichte werden auf 100 % verteilt. Das sind Modellportfolios, keine persönlichen Kaufempfehlungen.")
# Automatically calculate mixed suggestions as part of page loading; no load button.
if "idea_universe_df" not in st.session_state or st.session_state.get("idea_universe_version") != "global-universe-v2":
    sample = {**STOCKS, **ETFS, **COMMODITIES}
    with st.spinner("Lade Daten für die zehn gemischten Modellportfolios …"):
        frames=[]
        progress=st.progress(0.0, text="Analysiere Aktien, ETFs und Rohstoffe …")
        for i,(ticker,name) in enumerate(sample.items()):
            row=metrics(ticker,name, "Aktie" if ticker in STOCKS else "ETF" if ticker in ETFS else "Rohstoff")
            if row: frames.append(row)
            progress.progress((i+1)/len(sample), text=f"Prüfe {ticker} ({i+1}/{len(sample)})")
        progress.empty()
        st.session_state.idea_universe_df=pd.DataFrame(frames)
        st.session_state.idea_df=st.session_state.idea_universe_df.copy()
        st.session_state.idea_universe_version="global-universe-v2"

idea_source=st.session_state.get("idea_universe_df", pd.DataFrame())
if not idea_source.empty:
    eligible=idea_source.copy()
    # Use different category mixes in each model portfolio; avoid single-security 'portfolios'.
    low_pool=eligible[eligible["Risiko"]=="Niedrig"]
    med_pool=eligible[eligible["Risiko"].isin(["Niedrig","Mittel"])]
    if med_pool.empty: med_pool=eligible
    portfolios=[]
    # User's amount, risk preference, growth goal and horizon influence all ten model proposals.
    risk_order=["Niedrig","Mittel","Hoch"]
    growth_order=["Stabil","Ausgewogen","Hoch","Maximal"]
    risk_base={"Niedrig":0,"Mittel":1,"Hoch":2,"Alle":1}[applied_risk]
    growth_base={"Stabil":0,"Ausgewogen":1,"Hoch":2,"Maximal":3}[applied_growth]
    profile_specs=[
      ("Defensiv gemischt",-1,-1,.50,.30,.20),
      ("Ausgewogen global",0,0,.45,.40,.15),
      ("Weltweite Qualität",0,-1,.45,.45,.10),
      ("Technologie & Weltmarkt",1,1,.55,.35,.10),
      ("Dividenden & Stabilität",-1,-1,.30,.60,.10),
      ("Wachstum global",1,1,.60,.30,.10),
      ("Rohstoff-Beimischung",0,0,.40,.35,.25),
      ("USA + Europa + Asien",0,0,.50,.40,.10),
      ("Dynamisches Wachstum",1,2,.65,.25,.10),
      ("Maximales historisches Wachstum",2,2,.70,.20,.10),
    ]
    profiles=[]
    for label,rd,gd,sw,ew,cw in profile_specs:
        ri=min(2,max(0,risk_base+rd)); gi=min(3,max(0,growth_base+gd))
        profiles.append((label,risk_order[ri],growth_order[gi],sw,ew,cw))
    for idx,(label,risk,growth,sw,ew,cw) in enumerate(profiles,1):
        pool=eligible.copy()
        if risk=="Niedrig": pool=pool[pool.Risiko=="Niedrig"]
        elif risk=="Mittel": pool=pool[pool.Risiko.isin(["Niedrig","Mittel"])]
        if pool.empty: pool=eligible.copy()
        # Select multiple assets per category; rank by profile but diversify names.
        if growth=="Stabil": pool=pool.sort_values(["Volatilität %","Max. Verlustphase %"],ascending=[True,False])
        elif growth=="Ausgewogen": pool=pool.sort_values("Score",ascending=False)
        elif growth=="Hoch": pool=pool.sort_values(["Rendite p.a. %","Score"],ascending=False)
        else: pool=pool.sort_values(["1J Rendite %","Score"],ascending=False)
        stocks_pool=pool[pool.Typ=="Aktie"].head(3)
        etf_pool=pool[pool.Typ=="ETF"].head(3)
        comm_pool=pool[pool.Typ=="Rohstoff"].head(2)
        # If risk filtering removed a category, restore candidates from the full set.
        if len(stocks_pool)<2: stocks_pool=eligible[eligible.Typ=="Aktie"].sort_values("Score",ascending=False).head(3)
        if len(etf_pool)<2: etf_pool=eligible[eligible.Typ=="ETF"].sort_values("Score",ascending=False).head(3)
        if len(comm_pool)<1: comm_pool=eligible[eligible.Typ=="Rohstoff"].sort_values("Score",ascending=False).head(1)
        groups=[(stocks_pool,sw),(etf_pool,ew),(comm_pool,cw)]
        parts=[]
        for group,weight in groups:
            if not group.empty:
                each=weight/len(group)
                for _,r in group.iterrows(): parts.append({"Portfolio":label,"Anlage":r.Anlage,"Ticker":r.Ticker,"Typ":r.Typ,"Gewicht %":each*100,"Score":r.Score,"Risiko":r.Risiko,"Rendite p.a. %":r["Rendite p.a. %"],"Volatilität %":r["Volatilität %"]})
        for r in parts: portfolios.append(r)
    st.subheader("Übersicht: zehn gemischte Möglichkeiten")
    summary=[]
    for label, *_ in profiles:
        rows=[r for r in portfolios if r["Portfolio"]==label]
        types=sorted(set(r["Typ"] for r in rows))
        summary.append({"Nr.":len(summary)+1,"Portfolio":label,"Anzahl Anlagen":len(rows),"Anlagearten":", ".join(types),"Durchschnitts-Score":np.average([r["Score"] for r in rows],weights=[r["Gewicht %"] for r in rows]) if rows else np.nan,"Risiko-Profil":profiles[len(summary)][1],"Wachstumsziel":profiles[len(summary)][2]})
    st.dataframe(style_table(pd.DataFrame(summary)),use_container_width=True,hide_index=True)
    selected_port=st.selectbox("Portfolio genauer anschauen",[p[0] for p in profiles],index=1,key="mixed_port_select")
    detail=pd.DataFrame([r for r in portfolios if r["Portfolio"]==selected_port])
    if not detail.empty:
        st.subheader(f"Zusammensetzung: {selected_port}")
        st.dataframe(style_table(detail.drop(columns=["Portfolio"])),use_container_width=True,hide_index=True)
        st.write("**Beispielhafte Aufteilung deines Anlagebetrags**")
        allocation=detail[["Anlage","Ticker","Typ","Gewicht %"]].copy()
        allocation["Betrag (€)"]=allocation["Gewicht %"]*applied_money/100
        st.dataframe(allocation.style.format({"Gewicht %":"{:.1f}%","Betrag (€)":"€ {:,.2f}"}),use_container_width=True,hide_index=True)
        st.caption(f"Angewendete Einstellungen: {applied_horizon}, Risiko {applied_risk}, Wachstumsziel {applied_growth}, Betrag € {applied_money:,.2f}. Modellaufteilung, keine garantierte Rendite. Gebühren, Steuern, Währung und zukünftige Verluste sind nicht berücksichtigt.")
        # Automatic Monte Carlo for the selected suggested portfolio; no button required.
        with st.spinner("Monte-Carlo-Simulation für den ausgewählten Vorschlag läuft …"):
            try:
                tickers=detail.Ticker.astype(str).tolist()
                hist_parts={t:fetch_history(t).Close.pct_change().dropna().tail(756).rename(t) for t in tickers}
                hist_df=pd.concat(hist_parts.values(),axis=1).dropna(how="all").tail(756)
                weights=np.array(detail["Gewicht %"].astype(float).values/100.0)
                if not hist_df.empty and len(hist_df)>30:
                    hist_df=hist_df.reindex(columns=tickers).fillna(0.0)
                    mu=hist_df.mean().values; cov=hist_df.cov().values
                    rng=np.random.default_rng(2026)
                    sim_days={"1 Monat":21,"3 Monate":63,"6 Monate":126,"1 Jahr":252,"3 Jahre":756,"5 Jahre":1260,"10 Jahre":2520}.get(applied_horizon,252)
                    sim_count=1000
                    # Chunk simulations to keep memory use reasonable for long horizons.
                    ending_parts=[]
                    for start in range(0,sim_count,100):
                        batch=min(100,sim_count-start)
                        try:
                            simulated=rng.multivariate_normal(mu,cov,size=(batch,sim_days),check_valid="ignore")
                        except Exception:
                            simulated=rng.normal(mu,np.sqrt(np.maximum(np.diag(cov),1e-12)),size=(batch,sim_days,len(tickers)))
                        port_returns=np.einsum("sta,a->st",simulated,weights)
                        ending_parts.extend((applied_money*np.exp(np.log1p(np.clip(port_returns,-.99,None)).sum(axis=1))).tolist())
                    ending=np.asarray(ending_parts)
                    q05,q50,q95=np.quantile(ending,[.05,.50,.95])
                    c1,c2,c3=st.columns(3)
                    c1.metric("Monte-Carlo · schlechtes Szenario (5 %)",f"€ {q05:,.0f}")
                    c2.metric("Median-Szenario",f"€ {q50:,.0f}")
                    c3.metric("Gutes Szenario (95 %)",f"€ {q95:,.0f}")
                    st.plotly_chart(px.histogram(pd.DataFrame({"Simulierter Endwert (€)":ending}),x="Simulierter Endwert (€)",nbins=40,title=f"Automatische Monte-Carlo-Simulation · {applied_horizon} · {sim_count:,} Pfade"),use_container_width=True)
                    st.caption("Die Simulation nutzt historische Durchschnittsrenditen und Kovarianzen. Sie ist eine Modellspanne, keine verlässliche Vorhersage; bei langen Zeiträumen steigt die Unsicherheit erheblich.")
                else:
                    st.info("Für die automatische Monte-Carlo-Simulation sind derzeit nicht genügend gemeinsame historische Kursdaten verfügbar.")
            except Exception as mc_error:
                st.warning(f"Automatische Monte-Carlo-Simulation derzeit nicht verfügbar: {mc_error}")
else:
    st.info("Die zehn gemischten Vorschläge konnten nicht erstellt werden, weil aktuell zu wenige Kursreihen verfügbar sind. Bitte später Daten aktualisieren.")


st.header("🌍 Weltgeschehen: aktuelle Meldungen & anstehende Termine")
st.write("Der News-Tab versucht Meldungen der letzten 48 Stunden aus mehreren institutionellen bzw. etablierten RSS-Quellen zu laden. Das ist keine vollständige Echtzeit-Erfassung aller Ereignisse; Feeds können ausfallen oder verspätet sein.")
if feedparser is None: st.warning("Für Nachrichten bitte `feedparser` in requirements.txt installieren.")
else:
    if st.button("Nachrichten neu laden",key="news_refresh"): st.cache_data.clear()
    @st.cache_data(ttl=300,show_spinner=False)
    def cached_news(): return rss_news()
    news, status=cached_news()
    if status: st.caption(status)
    if news.empty: st.warning("Keine aktuellen RSS-Meldungen abrufbar. Bitte später erneut versuchen.")
    else:
        st.caption(f"{len(news)} Meldungen aus erreichbaren Feeds · geprüft am {now.strftime('%d.%m.%Y %H:%M')} (Europe/Berlin)")
        search=st.text_input("Nachrichten filtern",placeholder="z. B. Krieg, Zinsen, KI, Energie, Klima")
        if search: news=news[news.apply(lambda row: search.lower() in (str(row.Meldung)+" "+str(row.Kurzinfo)+" "+str(row.Quelle)).lower(),axis=1)]
        for _,r in news.head(50).iterrows():
            with st.container(border=True):
                st.markdown(f"**{r['Meldung']}**")
                st.caption(f"{r['Quelle']} · {r['Zeit (lokal)']}")
                if r['Kurzinfo']: st.write(r['Kurzinfo'])
                if r['Link']: st.markdown(f"[Originalquelle öffnen]({r['Link']})")
        st.subheader("Termine in den nächsten 48 Stunden")
        st.write("Die RSS-Feeds enthalten nicht immer strukturierte Kalenderdaten. Diese Liste zeigt nur eindeutig datierte Termine, die aus gelieferten Meldungen zuverlässig erkannt werden; wenn nichts erkannt wird, bedeutet das nicht, dass keine Termine stattfinden.")
        # Conservatively surface only entries explicitly referring to upcoming scheduled events.
        upcoming=re.compile(r"(tomorrow|next\s+(day|week|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|scheduled for|set to|will announce|due on|on (monday|tuesday|wednesday|thursday|friday|saturday|sunday)|upcoming|expected to announce|is expected to)",re.I)
        future=news[news.apply(lambda row: bool(upcoming.search(str(row.Meldung)+" "+str(row.Kurzinfo))),axis=1)]
        if future.empty: st.info("In den abgerufenen Meldungen wurden keine eindeutig erkennbaren angekündigten Termine gefunden.")
        else: st.dataframe(future[["Zeit (lokal)","Quelle","Meldung","Link"]].rename(columns={"Zeit (lokal":"Meldungszeit"}),use_container_width=True,hide_index=True)

st.divider()
st.caption("Datenhinweis: Yahoo Finance kann verspätete, unvollständige oder fehlerhafte Daten liefern. Kursreihen unterschiedlicher Instrumente beginnen zu unterschiedlichen Zeitpunkten. Futures und ETFs sind nicht identisch mit dem physischen Rohstoff. Alle Scores und Szenarien sind vereinfachte Modellwerte und keine Kaufempfehlung.")
