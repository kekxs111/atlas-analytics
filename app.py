import math, time, datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import feedparser

st.set_page_config(page_title='ATLAS Analytics', page_icon='📈', layout='wide')
st.title('📈 ATLAS Analytics')
st.caption('Historische Marktanalyse, Risiko-Szenarien und Portfolio-Werkzeuge. Keine Anlageberatung.')

ASSETS = {
    'Aktien USA': {'NVDA':'NVIDIA','MSFT':'Microsoft','AAPL':'Apple','AMZN':'Amazon','GOOGL':'Alphabet','META':'Meta Platforms','AVGO':'Broadcom','AMD':'AMD','TSLA':'Tesla','JPM':'JPMorgan Chase','BRK-B':'Berkshire Hathaway','LLY':'Eli Lilly','XOM':'Exxon Mobil','WMT':'Walmart','COST':'Costco'},
    'Aktien Europa': {'SAP.DE':'SAP','SIE.DE':'Siemens','ALV.DE':'Allianz','DTE.DE':'Deutsche Telekom','AIR.PA':'Airbus','MC.PA':'LVMH','ASML.AS':'ASML','NOVO-B.CO':'Novo Nordisk','NESN.SW':'Nestlé','RACE.MI':'Ferrari'},
    'Aktien Asien': {'6758.T':'Sony','7203.T':'Toyota','9984.T':'SoftBank Group','005930.KS':'Samsung Electronics','0700.HK':'Tencent','9988.HK':'Alibaba','RELIANCE.NS':'Reliance Industries','TCS.NS':'Tata Consultancy Services'},
    'ETFs': {'SPY':'S&P 500 ETF','QQQ':'Nasdaq-100 ETF','VT':'Weltaktien ETF','VTI':'US Gesamtmarkt ETF','VEA':'Aktien Industrieländer ex USA','VWO':'Schwellenländer ETF','IEFA':'Internationale Aktien','AGG':'US Anleihen ETF','TLT':'US Staatsanleihen lang','GLD':'Gold ETF','SLV':'Silber ETF','IBIT':'Bitcoin ETF (hohes Risiko)'},
    'Rohstoff-Futures': {'GC=F':'Gold-Future','SI=F':'Silber-Future','PL=F':'Platin-Future','PA=F':'Palladium-Future','CL=F':'WTI Rohöl-Future','BZ=F':'Brent Rohöl-Future','NG=F':'Erdgas-Future','HG=F':'Kupfer-Future','ZC=F':'Mais-Future','ZW=F':'Weizen-Future'},
    'Physische Rohstoffe (Preisindikatoren)': {'GC=F':'Gold (Barren-Preisindikator)','SI=F':'Silber (Barren-Preisindikator)','PL=F':'Platin (Preisindikator)','PA=F':'Palladium (Preisindikator)'}
}

with st.sidebar:
    st.header('Einstellungen')
    groups = st.multiselect('Anlageklassen', list(ASSETS), default=['Aktien USA','Aktien Europa','Aktien Asien','ETFs','Rohstoff-Futures'])
    countries = st.multiselect('Regionen / Länder', ['USA','Deutschland','Frankreich','Niederlande','Schweiz','Dänemark','Italien','Japan','Südkorea','Hongkong','Indien','Global'], default=['USA','Deutschland','Frankreich','Niederlande','Schweiz','Dänemark','Italien','Japan','Südkorea','Hongkong','Indien','Global'])
    period = st.selectbox('Historischer Zeitraum', ['Maximal verfügbar','20 Jahre','10 Jahre','5 Jahre','3 Jahre','1 Jahr'], index=0)
    horizon = st.slider('Prognosehorizont (Tage)', 30, 756, 252, 21)
    stale_hours = st.number_input('Daten gelten als veraltet nach (Stunden)', min_value=1, max_value=720, value=96)
    refresh = st.button('🔄 Daten aktualisieren', use_container_width=True)

region_map = {'USA':['Aktien USA','ETFs','Rohstoff-Futures','Physische Rohstoffe (Preisindikatoren)'], 'Deutschland':['Aktien Europa'], 'Frankreich':['Aktien Europa'], 'Niederlande':['Aktien Europa'], 'Schweiz':['Aktien Europa'], 'Dänemark':['Aktien Europa'], 'Italien':['Aktien Europa'], 'Japan':['Aktien Asien'], 'Südkorea':['Aktien Asien'], 'Hongkong':['Aktien Asien'], 'Indien':['Aktien Asien'], 'Global':['ETFs','Rohstoff-Futures','Physische Rohstoffe (Preisindikatoren)']}
selected = {}
for g in groups:
    for ticker, name in ASSETS[g].items():
        if any(g in region_map.get(c, []) for c in countries): selected[ticker] = name
if not selected:
    st.warning('Bitte wähle mindestens eine passende Anlageklasse und Region aus.')
    st.stop()

if refresh:
    st.cache_data.clear()

@st.cache_data(ttl=3600, show_spinner=False)
def load_history(ticker, period_label):
    p = {'Maximal verfügbar':'max','20 Jahre':'20y','10 Jahre':'10y','5 Jahre':'5y','3 Jahre':'3y','1 Jahr':'1y'}[period_label]
    df = yf.download(ticker, period=p, interval='1d', auto_adjust=True, progress=False, threads=False)
    if df is None or df.empty: return pd.DataFrame(), dt.datetime.now(dt.timezone.utc).isoformat()
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = df[['Close']].dropna(); df.index = pd.to_datetime(df.index)
    return df, dt.datetime.now(dt.timezone.utc).isoformat()

def metrics(df):
    r = df['Close'].pct_change().dropna()
    if len(r)<3: return {}
    close = df['Close']; peak = close.cummax(); dd = close/peak-1
    years = max((close.index[-1]-close.index[0]).days/365.25, 1/365.25)
    total = close.iloc[-1]/close.iloc[0]-1
    cagr = (close.iloc[-1]/close.iloc[0])**(1/years)-1 if close.iloc[-1]>0 and close.iloc[0]>0 else np.nan
    var95 = float(np.quantile(r, .05))
    es95 = float(r[r<=var95].mean()) if (r<=var95).any() else var95
    return {'Kurs':float(close.iloc[-1]),'Gesamtrendite':total,'Rendite p.a.':cagr,'Volatilität p.a.':float(r.std()*np.sqrt(252)),'Max. Drawdown':float(dd.min()),'VaR 95 % (Tag)':var95,'Expected Shortfall (Tag)':es95,'Letzter Kurs':close.index[-1].date().isoformat(),'Historie ab':close.index[0].date().isoformat(),'Beobachtungen':len(close)}

@st.cache_data(ttl=1800, show_spinner=False)
def load_news():
    feeds = [
      ('CNBC Markets','https://www.cnbc.com/id/100003114/device/rss/rss.html'),
      ('MarketWatch','https://feeds.marketwatch.com/marketwatch/topstories/'),
      ('Google News – KI Aktien','https://news.google.com/rss/search?q=AI+stocks+semiconductors+market&hl=en-US&gl=US&ceid=US:en'),
      ('Google News – Weltwirtschaft','https://news.google.com/rss/search?q=global+economy+inflation+interest+rates+markets&hl=en-US&gl=US&ceid=US:en')]
    rows=[]
    for source,url in feeds:
        try:
            f=feedparser.parse(url)
            for e in f.entries[:12]: rows.append({'Quelle':source,'Titel':e.get('title',''),'Link':e.get('link',''),'Veröffentlicht':e.get('published','')})
        except Exception: pass
    return pd.DataFrame(rows).drop_duplicates(subset=['Titel']) if rows else pd.DataFrame(columns=['Quelle','Titel','Link','Veröffentlicht'])

rows=[]; histories={}; fetch_times={}
progress=st.progress(0, text='Lade historische Kursdaten …')
items=list(selected.items())
with ThreadPoolExecutor(max_workers=6) as ex:
    futs={ex.submit(load_history,t,period): (t,n) for t,n in items}
    for i,f in enumerate(as_completed(futs),1):
        t,n=futs[f]
        try: df, fetched=f.result(); histories[t]=df; fetch_times[t]=fetched
        except Exception: histories[t]=pd.DataFrame(); fetch_times[t]=dt.datetime.now(dt.timezone.utc).isoformat()
        progress.progress(i/len(items), text=f'Kursdaten: {i}/{len(items)}')
progress.empty()
for t,n in items:
    df=histories[t]; m=metrics(df)
    if m:
        m.update({'Ticker':t,'Anlage':n,'Abgerufen (UTC)':fetch_times[t]})
        age=(dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(fetch_times[t])).total_seconds()/3600
        m['Datenstatus']='Prüfen' if age>stale_hours else 'Abruf aktuell; Kurs kann verzögert sein'
        rows.append(m)
results=pd.DataFrame(rows)

st.subheader('Marktübersicht und Rangliste')
if results.empty:
    st.error('Keine Kursdaten erhalten. Prüfe die Internetverbindung oder die Ticker-Symbole.')
else:
    sortby=st.selectbox('Rangliste sortieren nach', ['Rendite p.a.','Gesamtrendite','Volatilität p.a.','Max. Drawdown','Ticker'])
    ranked=results.sort_values(sortby, ascending=(sortby in ['Volatilität p.a.','Max. Drawdown','Ticker']), na_position='last')
    display=ranked.copy()
    for c in ['Gesamtrendite','Rendite p.a.','Volatilität p.a.','Max. Drawdown','VaR 95 % (Tag)','Expected Shortfall (Tag)']:
        display[c]=display[c].map(lambda x: f'{x:.2%}' if pd.notna(x) else '—')
    st.dataframe(display[['Ticker','Anlage','Kurs','Gesamtrendite','Rendite p.a.','Volatilität p.a.','Max. Drawdown','VaR 95 % (Tag)','Expected Shortfall (Tag)','Historie ab','Letzter Kurs','Abgerufen (UTC)','Datenstatus']], use_container_width=True, hide_index=True)
    st.caption('Kurse können verzögert sein. „Abgerufen“ ist der Abrufzeitpunkt, nicht zwingend der Zeitpunkt des letzten Börsenkurses.')

st.subheader('Ticker und Risiko berechnen')
st.caption('Ticker: Börsenkürzel, mit dem die Datenquelle eine Anlage erkennt. | Anlage: verständlicher Name des Wertpapiers oder Rohstoffindikators. | Historischer Zeitraum: bestimmt, wie viele verfügbare Kursdaten ausgewertet werden. | Rendite: misst die bisherige Kursentwicklung und ist keine Garantie für die Zukunft. | Volatilität: misst die Schwankungsbreite der Renditen. | Max. Drawdown: größter historischer Rückgang vom vorherigen Höchststand. | VaR: historischer 95-%-Verlustschwellenwert pro Tag; keine Verlustobergrenze. | Expected Shortfall: durchschnittliche Tagesrendite in den schlechtesten 5 % der beobachteten Tage. | Datenstatus/Zeitstempel: zeigt Abrufzeit und letzten Kurszeitpunkt; Börsenkurse können verzögert sein. | Risiko berechnen: wählt eine Anlage aus und zeigt Kennzahlen sowie mögliche Szenarien auf Basis historischer Daten.')
if not results.empty:
    chosen=st.selectbox('Ticker für Detailanalyse', results['Ticker'].tolist())
    df=histories[chosen]
    if not df.empty:
        st.line_chart(df['Close'], height=260)
        st.write(f'**{chosen}** · Historie: {df.index[0].date()} bis {df.index[-1].date()} · Datenabruf UTC: {fetch_times[chosen]}')
        if st.button('Risiko berechnen', key='riskcalc'):
            m=metrics(df)
            a,b,c=st.columns(3); a.metric('Volatilität p.a.',f"{m['Volatilität p.a.']:.1%}"); b.metric('Max. Drawdown',f"{m['Max. Drawdown']:.1%}"); c.metric('VaR 95 % (Tag)',f"{m['VaR 95 % (Tag)']:.2%}")

st.subheader('Monte-Carlo-Simulation – selbst einstellen')
left,right=st.columns(2)
with left:
    mc_ticker=st.selectbox('Anlage für Simulation', list(histories), key='mc_ticker')
    initial=st.number_input('Startkapital (€)', min_value=100.0, max_value=100000000.0, value=10000.0, step=500.0)
    sim_count=st.slider('Anzahl Simulationen', 100, 10000, 2000, 100)
    days=st.slider('Prognosehorizont (Handelstage)', 21, 756, int(min(horizon,756)), 21)
with right:
    dfm=histories[mc_ticker]
    rr=dfm['Close'].pct_change().dropna() if not dfm.empty else pd.Series(dtype=float)
    default_mu=float(rr.mean()*252) if len(rr)>2 else .07
    default_sigma=float(rr.std()*np.sqrt(252)) if len(rr)>2 else .25
    mu=st.number_input('Erwartete Rendite p.a. (%)', min_value=-100.0, max_value=300.0, value=float(np.clip(default_mu*100,-100,300)), step=1.0)/100
    sigma=st.number_input('Volatilität p.a. (%)', min_value=0.1, max_value=300.0, value=float(np.clip(default_sigma*100,.1,300)), step=1.0)/100
    seed=st.number_input('Zufallsstartwert (Seed)', min_value=0, max_value=9999999, value=42, step=1)
st.markdown('**Geometrische Brownsche Bewegung:**')
st.latex(r'S_{t+\Delta t}=S_t\exp\left[(\mu-\tfrac{1}{2}\sigma^2)\Delta t+\sigma\sqrt{\Delta t}Z\right]')
st.caption('μ = erwartete annualisierte Rendite, σ = annualisierte Volatilität, Z = standardnormaler Zufallswert. Ergebnisse hängen stark von den Annahmen ab und sind keine Vorhersagegarantie.')
if st.button('Monte-Carlo-Simulation starten'):
    rng=np.random.default_rng(int(seed)); dt_step=1/252
    shocks=rng.normal(size=(int(days),int(sim_count)))
    log_steps=(mu-.5*sigma*sigma)*dt_step+sigma*np.sqrt(dt_step)*shocks
    paths=initial*np.exp(np.vstack([np.zeros((1,int(sim_count))),np.cumsum(log_steps,axis=0)]))
    final=paths[-1,:]
    q=np.quantile(final,[.05,.25,.5,.75,.95])
    st.line_chart(pd.DataFrame({'5 %':np.quantile(paths,.05,axis=1),'Median':np.quantile(paths,.5,axis=1),'95 %':np.quantile(paths,.95,axis=1)}))
    st.dataframe(pd.DataFrame({'Kennzahl':['5-%-Szenario','25-%-Szenario','Median','75-%-Szenario','95-%-Szenario'],'Endwert (€)':q,'Gewinn/Verlust (€)':q-initial}),use_container_width=True,hide_index=True)

st.subheader('Zehn gemischte Anlagekombinationen')
st.caption('Beispielhafte, regelbasierte Musterportfolios – keine persönlichen Empfehlungen. Gewichte sind illustrative und ergeben jeweils 100 %.')
portfolio_templates=[('Ausgewogen global', [('VT',50),('AGG',25),('GLD',10),('QQQ',10),('GC=F',5)]),('Wachstum global',[('QQQ',30),('VT',30),('VWO',10),('AGG',10),('GLD',10),('GC=F',10)]),('Defensiv',[('AGG',40),('VT',25),('VEA',10),('GLD',15),('GC=F',10)]),('Technologie mit Puffer',[('NVDA',15),('MSFT',15),('QQQ',25),('VT',20),('AGG',15),('GLD',10)]),('Rohstoffe & Aktien',[('VT',35),('AGG',20),('GLD',15),('SI=F',5),('HG=F',10),('XOM',15)]),('USA-lastig',[('VTI',40),('QQQ',20),('AGG',20),('GLD',10),('GC=F',10)]),('International',[('VT',40),('VEA',20),('VWO',15),('AGG',15),('GLD',10)]),('Dividenden/Qualität',[('BRK-B',20),('JPM',10),('WMT',10),('VT',25),('AGG',20),('GLD',15)]),('Halbleiter-Sensitivität',[('NVDA',15),('AMD',10),('AVGO',10),('ASML.AS',10),('QQQ',20),('VT',20),('AGG',10),('GLD',5)]),('Rohstoff-Schutz',[('VT',35),('AGG',25),('GLD',15),('GC=F',10),('SI=F',5),('XOM',10)])]
for i in range(0,10,2):
    cols=st.columns(2)
    for col,(name,holdings) in zip(cols,portfolio_templates[i:i+2]):
        with col:
            st.markdown(f'**{i+1 if name==portfolio_templates[i][0] else i+2}. {name}**')
            st.dataframe(pd.DataFrame([{'Ticker':t,'Gewicht':f'{w}%','Verfügbar in aktueller Auswahl':'Ja' if t in histories else 'Nein – ggf. Auswahl erweitern'} for t,w in holdings]),use_container_width=True,hide_index=True)

st.subheader('KI-Blasencheck')
st.caption('Ein heuristischer Warnindikator, kein Nachweis einer Blase. Er betrachtet starke historische Kursanstiege, Volatilität und verfügbare Bewertungskennzahlen; fehlende Kennzahlen werden nicht erfunden.')
ai_tickers=[t for t in ['NVDA','AMD','AVGO','MSFT','GOOGL','META','AMZN','ASML.AS','TSLA'] if t in histories]
if ai_tickers:
    bubble_rows=[]
    for t in ai_tickers:
        d=histories[t]; m=metrics(d)
        gain=d['Close'].iloc[-1]/d['Close'].iloc[max(0,len(d)-253)]-1 if len(d)>2 else np.nan
        score=0; reasons=[]
        if pd.notna(gain):
            if gain>.8: score+=2; reasons.append('starker 1-Jahres-Anstieg')
            elif gain>.4: score+=1; reasons.append('erhöhter 1-Jahres-Anstieg')
        vol=m.get('Volatilität p.a.',np.nan)
        if pd.notna(vol) and vol>.65: score+=2; reasons.append('hohe Volatilität')
        elif pd.notna(vol) and vol>.4: score+=1; reasons.append('erhöhte Volatilität')
        try:
            info=yf.Ticker(t).get_info(); pe=info.get('trailingPE'); fpe=info.get('forwardPE'); ps=info.get('priceToSalesTrailing12Months')
            if pe and pe>60: score+=1; reasons.append('hohes KGV')
            if fpe and pe and fpe>pe*1.2: score+=1; reasons.append('hohes erwartetes KGV relativ zum aktuellen KGV')
            if ps and ps>20: score+=1; reasons.append('hohes Kurs-Umsatz-Verhältnis')
        except Exception: pe=fpe=ps=None
        bubble_rows.append({'Ticker':t,'1J-Kursänderung':gain,'Volatilität p.a.':vol,'KGV':pe,'Forward-KGV':fpe,'Kurs/Umsatz':ps,'Warnscore (0–6+)':score,'Hinweise':', '.join(reasons) if reasons else 'Keine der einfachen Warnregeln ausgelöst'})
    bd=pd.DataFrame(bubble_rows)
    for c in ['1J-Kursänderung','Volatilität p.a.']:
        bd[c]=bd[c].map(lambda x:f'{x:.1%}' if pd.notna(x) else '—')
    st.dataframe(bd.sort_values('Warnscore (0–6+)',ascending=False),use_container_width=True,hide_index=True)
else: st.info('Wähle mindestens eine KI-/Halbleiteraktie in den Anlageklassen und Regionen aus.')

st.subheader('Aktuelle Marktnachrichten')
news=load_news()
if news.empty: st.warning('Nachrichten konnten nicht geladen werden. Prüfe die Verbindung oder öffne die Quellen später erneut.')
else:
    for _,r in news.head(24).iterrows():
        st.markdown(f"- **{r['Quelle']}** · [{r['Titel']}]({r['Link']})  \n  <small>{r['Veröffentlicht']}</small>",unsafe_allow_html=True)
    st.caption('RSS-Feeds können ausfallen und haben unterschiedliche Aktualisierungsintervalle. Veröffentlichung und Abrufzeit sind nicht dasselbe.')

st.subheader('Datenquellen und Grenzen')
st.markdown('- Historische Kurse: Yahoo Finance über `yfinance`; Verfügbarkeit, Tiefe und Verzögerung hängen von Börse und Symbol ab.\n- Nachrichten: öffentliche RSS-Feeds von CNBC, MarketWatch und Google News.\n- Wirtschaftsdaten: in dieser Version noch nicht als einheitliche, verlässlich aktuelle Makrodatenbank integriert; News-Suche ist kein Ersatz für offizielle Datenreihen.\n- Physische Rohstoffe: Futures sind nur Preisindikatoren. Sie entsprechen nicht dem Kaufpreis physischer Barren (Aufgeld, Lagerung, Steuern und Händlerpreise fehlen).\n- Die Rangliste umfasst einen kuratierten Ticker-Katalog und nicht automatisch jedes weltweit gelistete Wertpapier.\n- Dieses Tool dient Lern- und Analysezwecken und ist keine Finanzberatung.')
