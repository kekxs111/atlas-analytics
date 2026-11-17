from __future__ import annotations
import json
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from atlas_config import STOCKS, ETFS, COMMODITIES, ALL_UNIVERSE, ASSET_CLASS
from atlas_risk import (
    fetch_history, single_metrics, portfolio_returns, historical_var_cvar,
    stress_test, monte_carlo_portfolio,
    get_last_price, holdings_to_weights, prefetch_prices,
)
from atlas_features import (
    TIPS, STRESS_SCENARIOS, diversifikation_status, monthly_needed,
    package_backtest, build_report_text,
)

warnings.filterwarnings("ignore", category=FutureWarning)
st.set_page_config(page_title="ATLAS – Einfach anlegen", page_icon="\U0001f310", layout="wide", initial_sidebar_state="expanded")
TZ = ZoneInfo("Europe/Berlin")
NOW = datetime.now(TZ)
SAVE_PATH = Path(__file__).resolve().parent / "atlas_save.json"

PRODUCTS = [
    {"ticker": "EUNL.DE", "name": "Welt-Aktien", "tag": "Empfohlen zum Start",
     "text": "Ein Paket aus Tausenden Unternehmen weltweit. Gut zum langfristigen Anlegen.", "risk": "Mittel"},
    {"ticker": "VWCE.DE", "name": "Welt-Aktien (Vanguard)", "tag": "Alternative",
     "text": "\u00c4hnlich wie Welt-Aktien: breit gestreut \u00fcber viele L\u00e4nder und Firmen.", "risk": "Mittel"},
    {"ticker": "SPY", "name": "USA-Aktien (S&P 500)", "tag": "Beliebt",
     "text": "Die 500 gr\u00f6\u00dften US-Firmen. Historisch stark, aber nur Amerika.", "risk": "Mittel"},
    {"ticker": "QQQ", "name": "Tech-Aktien", "tag": "Wachstum",
     "text": "Gro\u00dfe Tech-Firmen (z. B. Apple, Microsoft). Kann stark schwanken.", "risk": "H\u00f6her"},
    {"ticker": "AGG", "name": "Anleihen (stabiler)", "tag": "Ruhiger",
     "text": "Staats- und Unternehmensanleihen. Weniger Schwankung, oft niedrigere Rendite.", "risk": "Niedrig"},
    {"ticker": "GLD", "name": "Gold", "tag": "Beimischung",
     "text": "Gold als Absicherung. Kann in Krisen helfen, bringt keine Zinsen.", "risk": "Mittel"},
    {"ticker": "AAPL", "name": "Apple", "tag": "Einzelaktie",
     "text": "Eine einzelne Firma. Mehr Risiko als ein Paket aus vielen Firmen.", "risk": "H\u00f6her"},
    {"ticker": "MSFT", "name": "Microsoft", "tag": "Einzelaktie",
     "text": "Eine einzelne Firma. Mehr Risiko als ein Paket aus vielen Firmen.", "risk": "H\u00f6her"},
    {"ticker": "NVDA", "name": "NVIDIA", "tag": "Einzelaktie",
     "text": "Chip-Firma. Kann stark steigen und fallen.", "risk": "Hoch"},
    {"ticker": "SAP", "name": "SAP", "tag": "Deutschland",
     "text": "Deutsches Software-Unternehmen. Eine einzelne Firma.", "risk": "H\u00f6her"},
]
PRODUCT_BY_T = {p["ticker"]: p for p in PRODUCTS}

PACKAGES = {
    "Ruhig starten": {
        "desc": "Mehr Stabilit\u00e4t, weniger Schwankung. Gut, wenn du nerv\u00f6s bei Verlusten bist.",
        "parts": [("AGG", 0.50), ("EUNL.DE", 0.40), ("GLD", 0.10)],
        "risk_label": "eher ruhig",
    },
    "Ausgewogen (Empfehlung)": {
        "desc": "Klassische Mischung: viel Welt, etwas Anleihen, etwas Gold.",
        "parts": [("EUNL.DE", 0.60), ("AGG", 0.25), ("GLD", 0.15)],
        "risk_label": "mittel",
    },
    "Mehr Wachstum": {
        "desc": "Haupts\u00e4chlich Aktien. Kann st\u00e4rker schwanken, langfristig oft h\u00f6here Chance.",
        "parts": [("EUNL.DE", 0.50), ("SPY", 0.30), ("QQQ", 0.20)],
        "risk_label": "eher unruhig",
    },
}

def export_state() -> dict:
    return {
        "version": 2,
        "saved_at": datetime.now(TZ).isoformat(),
        "cash": float(st.session_state.get("cash", 10000.0)),
        "holdings": st.session_state.get("holdings", {}),
        "trades": st.session_state.get("trades", []),
        "seen_welcome": bool(st.session_state.get("seen_welcome", False)),
        "price_cache": st.session_state.get("price_cache", {}),
        "watchlist": st.session_state.get("watchlist", []),
        "sparplan": st.session_state.get("sparplan", {"pkg": "Ausgewogen (Empfehlung)", "amount": 200.0, "months_done": 0}),
        "nav_history": st.session_state.get("nav_history", []),
    }

def apply_state(data: dict) -> None:
    try:
        st.session_state.cash = float(data.get("cash", 10000.0))
    except (TypeError, ValueError):
        st.session_state.cash = 10000.0
    holdings = data.get("holdings") or {}
    if not isinstance(holdings, dict):
        holdings = {}
    clean = {}
    for t, h in holdings.items():
        if not isinstance(h, dict):
            continue
        try:
            qty = float(h.get("qty", 0) or 0)
            avg = float(h.get("avg_cost", 0) or 0)
        except (TypeError, ValueError):
            continue
        if qty > 0:
            clean[str(t)] = {"qty": qty, "avg_cost": avg}
    st.session_state.holdings = clean
    trades = data.get("trades") or []
    st.session_state.trades = trades if isinstance(trades, list) else []
    st.session_state.seen_welcome = bool(data.get("seen_welcome", False))
    pc = data.get("price_cache") or {}
    if "price_cache" not in st.session_state:
        st.session_state.price_cache = {}
    if isinstance(pc, dict):
        for k, v in pc.items():
            try:
                fv = float(v)
                if fv == fv and fv > 0:
                    st.session_state.price_cache[str(k)] = fv
            except (TypeError, ValueError):
                pass
    wl = data.get("watchlist") or []
    st.session_state.watchlist = [str(x) for x in wl] if isinstance(wl, list) else []
    sp = data.get("sparplan") or {}
    if isinstance(sp, dict):
        st.session_state.sparplan = {
            "pkg": sp.get("pkg", "Ausgewogen (Empfehlung)"),
            "amount": float(sp.get("amount", 200) or 200),
            "months_done": int(sp.get("months_done", 0) or 0),
        }
    nh = data.get("nav_history") or []
    st.session_state.nav_history = nh if isinstance(nh, list) else []

def save_to_disk() -> bool:
    try:
        SAVE_PATH.write_text(json.dumps(export_state(), ensure_ascii=False, indent=2), encoding="utf-8")
        return True
    except Exception:
        return False

def load_from_disk() -> bool:
    try:
        if not SAVE_PATH.exists():
            return False
        data = json.loads(SAVE_PATH.read_text(encoding="utf-8"))
        apply_state(data)
        return True
    except Exception:
        return False

def persist() -> None:
    save_to_disk()

def init_state():
    defaults = {
        "price_cache": {}, "holdings": {}, "cash": 10000.0, "trades": [],
        "seen_welcome": False, "watchlist": [],
        "sparplan": {"pkg": "Ausgewogen (Empfehlung)", "amount": 200.0, "months_done": 0},
        "nav_history": [],
    }
    if "holdings" not in st.session_state:
        if load_from_disk():
            pass
        else:
            for k, v in defaults.items():
                st.session_state[k] = dict(v) if isinstance(v, dict) else (list(v) if isinstance(v, list) else v)
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = dict(v) if isinstance(v, dict) else (list(v) if isinstance(v, list) else v)

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
        if not (px == px and px > 0) and avg > 0:
            px = avg
        prices[t] = px
        mv = qty * px if px == px and px > 0 else float("nan")
        cost = qty * avg
        pnl = (mv - cost) if mv == mv else float("nan")
        invested += mv if mv == mv else 0.0
        cost_total += cost
        pname = PRODUCT_BY_T.get(t, {}).get("name") or ALL_UNIVERSE.get(t, t)
        rows.append({
            "Was": pname, "K\u00fcrzel": t, "St\u00fcck": qty, "Dein Einkauf": avg,
            "Aktueller Kurs": px, "Wert jetzt \u20ac": mv, "Gewinn/Verlust \u20ac": pnl,
            "Gewinn/Verlust %": (pnl / cost * 100) if cost and pnl == pnl else float("nan"),
            "Anteil %": 0.0,
        })
    nav = cash + invested
    for r in rows:
        if nav > 0 and r["Wert jetzt \u20ac"] == r["Wert jetzt \u20ac"]:
            r["Anteil %"] = r["Wert jetzt \u20ac"] / nav * 100
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Wert jetzt \u20ac", ascending=False)
    return cash, df, invested, nav, cost_total, prices

def do_buy(ticker: str, qty: float, px: float):
    notional = qty * px
    cash = float(st.session_state.cash)
    if notional > cash + 1e-6:
        return False, f"Du hast nur {cash:,.2f} \u20ac frei. Diese Order braucht {notional:,.2f} \u20ac."
    h = st.session_state.holdings.get(ticker, {"qty": 0.0, "avg_cost": 0.0})
    old_q, old_c = float(h["qty"]), float(h["avg_cost"])
    new_q = old_q + qty
    new_avg = ((old_q * old_c) + notional) / new_q if new_q > 0 else px
    st.session_state.holdings[ticker] = {"qty": new_q, "avg_cost": new_avg}
    st.session_state.cash = cash - notional
    name = PRODUCT_BY_T.get(ticker, {}).get("name") or ticker
    st.session_state.trades.append({
        "Zeit": NOW.strftime("%d.%m.%Y %H:%M"), "Art": "Kauf", "Was": name,
        "K\u00fcrzel": ticker, "St\u00fcck": qty, "Preis": px, "Betrag \u20ac": notional,
    })
    persist()
    return True, f"Geschafft! Du hast {name} f\u00fcr {notional:,.2f} \u20ac gekauft."

def do_sell(ticker: str, qty: float, px: float):
    h = st.session_state.holdings.get(ticker, {"qty": 0.0, "avg_cost": 0.0})
    old_q = float(h["qty"])
    if qty > old_q + 1e-9:
        return False, f"Du besitzt nur {old_q:.4g} St\u00fcck."
    notional = qty * px
    new_q = old_q - qty
    if new_q < 1e-9:
        st.session_state.holdings.pop(ticker, None)
    else:
        st.session_state.holdings[ticker] = {"qty": new_q, "avg_cost": float(h["avg_cost"])}
    st.session_state.cash = float(st.session_state.cash) + notional
    name = PRODUCT_BY_T.get(ticker, {}).get("name") or ticker
    st.session_state.trades.append({
        "Zeit": NOW.strftime("%d.%m.%Y %H:%M"), "Art": "Verkauf", "Was": name,
        "K\u00fcrzel": ticker, "St\u00fcck": qty, "Preis": px, "Betrag \u20ac": notional,
    })
    persist()
    return True, f"Verkauft: {name} \u00b7 Du erh\u00e4ltst ca. {notional:,.2f} \u20ac zur\u00fcck."

def buy_package(pkg_name: str, euro: float):
    parts = PACKAGES[pkg_name]["parts"]
    prices = {}
    for t, _ in parts:
        px = get_last_price(t)
        if px != px or px <= 0:
            return False, f"Kurs f\u00fcr {t} gerade nicht verf\u00fcgbar. Bitte sp\u00e4ter nochmal."
        prices[t] = px
    cash = float(st.session_state.cash)
    if euro > cash + 1e-6:
        return False, f"Du hast nur {cash:,.2f} \u20ac frei."
    orders = []
    total = 0.0
    for t, w in parts:
        spend = euro * w
        qty = spend / prices[t]
        orders.append((t, qty, prices[t]))
        total += qty * prices[t]
    if total > cash + 1e-6:
        return False, f"Du hast nur {cash:,.2f} \u20ac frei."
    for t, qty, px in orders:
        ok, msg = do_buy(t, qty, px)
        if not ok:
            return False, msg
    persist()
    return True, f"Paket \u201e{pkg_name}\u201c f\u00fcr {euro:,.2f} \u20ac ist eingerichtet."

def record_nav(nav: float) -> None:
    hist = st.session_state.get("nav_history") or []
    stamp = NOW.strftime("%Y-%m-%d %H:%M")
    if hist and hist[-1].get("t") == stamp:
        hist[-1]["v"] = float(nav)
    else:
        hist.append({"t": stamp, "v": float(nav)})
        hist = hist[-60:]
    st.session_state.nav_history = hist

init_state()
_prefetch_list = list(st.session_state.holdings.keys()) + [p["ticker"] for p in PRODUCTS]
for _parts in PACKAGES.values():
    _prefetch_list.extend(t for t, _ in _parts["parts"])
_prefetch_list.extend(st.session_state.get("watchlist") or [])
if "prices_warmed" not in st.session_state:
    with st.spinner("Kurse werden geladen \u2026"):
        prefetch_prices(_prefetch_list)
    st.session_state.prices_warmed = True

cash, pos_df, invested, nav, cost_total, prices = portfolio_snapshot()
upnl = invested - cost_total if cost_total else 0.0
has_positions = not pos_df.empty
div_label, div_text = diversifikation_status(pos_df, nav, PRODUCT_BY_T)
record_nav(nav)

with st.sidebar:
    st.title("\U0001f310 ATLAS")
    st.caption("F\u00fcr Einsteiger gemacht")
    page = st.radio("Men\u00fc", ["\U0001f3e0 Start", "\U0001f6d2 Anlegen", "\U0001f4c9 Wie riskant?", "\u2696 Vergleich", "\u2b50 Merkliste", "\U0001f50d Entdecken", "\u2753 Hilfe"], label_visibility="collapsed")
    st.divider()
    st.metric("Dein Gesamtwert", f"{nav:,.2f} \u20ac")
    st.caption(f"Davon Bargeld: {cash:,.2f} \u20ac")
    if has_positions:
        st.caption(("\U0001f7e2" if upnl >= 0 else "\U0001f534") + f" Bisheriger Gewinn/Verlust: {upnl:+,.2f} \u20ac")
    st.caption(f"Streuung: {div_label}")
    if st.button("\U0001f504 Kurse aktualisieren"):
        st.cache_data.clear()
        st.session_state.pop("prices_warmed", None)
        st.rerun()
    st.divider()
    st.caption("**Depot speichern / laden**")
    st.download_button("\U0001f4be Depot herunterladen", data=json.dumps(export_state(), ensure_ascii=False, indent=2), file_name="atlas_depot.json", mime="application/json", use_container_width=True)
    up = st.file_uploader("\U0001f4c2 Depot laden", type=["json"], label_visibility="collapsed")
    if up is not None:
        sig = f"{up.name}:{getattr(up, 'size', 0)}"
        if st.session_state.get("_upload_sig") != sig:
            try:
                data = json.loads(up.getvalue().decode("utf-8"))
                if not isinstance(data, dict) or "cash" not in data:
                    st.error("Ung\u00fcltige Depot-Datei.")
                else:
                    apply_state(data)
                    persist()
                    st.session_state._upload_sig = sig
                    st.success("Depot geladen.")
                    st.rerun()
            except Exception:
                st.error("Datei konnte nicht gelesen werden.")
    if SAVE_PATH.exists():
        st.caption("Auto-Speicherung aktiv")
    st.caption("\u00dcbungs-Depot \u00b7 kein echtes Geld")
    st.caption(NOW.strftime("%d.%m.%Y %H:%M"))

st.info("Neue Men\u00fcpunkte: Vergleich, Merkliste, Ziel-Sparen & Stress-Tests. Vollst\u00e4ndige UI folgt im n\u00e4chsten Update – Kernfunktionen (Anlegen, Kurse, Depot) laufen.")
if page == "\U0001f3e0 Start":
    st.header("Dein \u00dcbungs-Depot")
    with st.expander("\u2705 Kurze Checkliste", expanded=not st.session_state.seen_welcome):
        for tip in TIPS:
            st.markdown(f"- {tip}")
        st.info(f"**Tipp des Tages:** {TIPS[int(NOW.strftime('%j')) % len(TIPS)]}")
    a, b, c, d = st.columns(4)
    a.metric("Gesamtwert", f"{nav:,.2f} \u20ac")
    b.metric("Bargeld", f"{cash:,.2f} \u20ac")
    c.metric("Investiert", f"{invested:,.2f} \u20ac")
    d.metric("Gewinn / Verlust", f"{upnl:+,.2f} \u20ac")
    invest_pct = (invested / nav * 100) if nav > 0 else 0.0
    st.progress(min(1.0, max(0.0, invest_pct / 100.0)), text=f"Angelegt: {invest_pct:.0f} %")
    st.caption(f"**Streuung:** {div_label} \u2014 {div_text}")
    if not has_positions:
        st.warning("Noch nichts angelegt.")
        _qa = min(2000.0, max(100.0, float(cash)))
        q1, q2, q3 = st.columns(3)
        if q1.button(f"Ausgewogen \u00b7 {_qa:,.0f} \u20ac", type="primary", key="qb"):
            ok, msg = buy_package("Ausgewogen (Empfehlung)", _qa)
            (st.success if ok else st.error)(msg)
            if ok:
                st.session_state.seen_welcome = True
                st.rerun()
        if q2.button(f"Ruhig \u00b7 {_qa:,.0f} \u20ac", key="qc"):
            ok, msg = buy_package("Ruhig starten", _qa)
            (st.success if ok else st.error)(msg)
            if ok:
                st.rerun()
        if q3.button(f"Wachstum \u00b7 {_qa:,.0f} \u20ac", key="qg"):
            ok, msg = buy_package("Mehr Wachstum", _qa)
            (st.success if ok else st.error)(msg)
            if ok:
                st.rerun()
    else:
        st.dataframe(pos_df, use_container_width=True, hide_index=True)
    with st.expander("\U0001f3af Ziel-Sparen"):
        t = st.number_input("Zielbetrag \u20ac", value=50000.0, step=1000.0)
        y = st.slider("Jahre", 1, 40, 10)
        r = st.slider("Rendite p.a. %", 0.0, 12.0, 6.0, 0.5)
        need = monthly_needed(t, y, r, nav)
        st.metric("Ungef\u00e4hr monatlich sparen", f"{need:,.0f} \u20ac" if need > 0 else "Ziel schon erreichbar")
    with st.expander("\U0001f4c5 Monats-Sparplan \u00fcben"):
        sp = st.session_state.sparplan
        pkg = st.selectbox("Paket", list(PACKAGES.keys()), index=1)
        amt = st.number_input("Betrag / Monat", min_value=50.0, value=float(sp.get("amount", 200)), step=50.0)
        st.caption(f"Monate ge\u00fcbt: {int(sp.get('months_done', 0))}")
        if st.button("Diesen Monat einzahlen", type="primary"):
            ok, msg = buy_package(pkg, amt)
            if ok:
                st.session_state.sparplan = {"pkg": pkg, "amount": amt, "months_done": int(sp.get("months_done", 0)) + 1}
                persist()
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)
    report = build_report_text(nav, cash, invested, upnl, pos_df, div_label, div_text, NOW)
    st.download_button("\U0001f4c4 Report", data=report, file_name="atlas_report.txt")
elif page == "\U0001f6d2 Anlegen":
    st.header("Geld anlegen")
    st.write(f"Frei: **{cash:,.2f} \u20ac**")
    for name, info in PACKAGES.items():
        with st.container(border=True):
            st.markdown(f"### {name}")
            st.write(info["desc"])
            euro = st.number_input(f"Betrag {name}", min_value=100.0, max_value=max(100.0, cash), value=min(2000.0, max(100.0, cash)), key=f"e_{name}")
            if st.button(f"Kaufen: {name}", key=f"b_{name}"):
                ok, msg = buy_package(name, euro)
                (st.success if ok else st.error)(msg)
                if ok:
                    st.rerun()
elif page == "\U0001f4c9 Wie riskant?":
    st.header("Wie riskant?")
    st.caption(f"{div_label} \u2014 {div_text}")
    weights = holdings_to_weights(st.session_state.holdings, prices)
    if not weights:
        st.warning("Lege zuerst etwas an.")
    else:
        port_rets, _ = portfolio_returns(weights, "3y")
        if not port_rets.empty:
            ann_vol = port_rets.std(ddof=1) * np.sqrt(252)
            ampel = "\U0001f7e2 eher ruhig" if ann_vol < 0.12 else ("\U0001f7e1 mittel" if ann_vol < 0.22 else "\U0001f534 eher unruhig")
            st.subheader(f"Ampel: {ampel}")
            st.metric("Schwankung / Jahr", f"\u00b1 {ann_vol*100:.0f} %")
        st.subheader("Stress-Tests")
        for i, (label, shocks) in enumerate(STRESS_SCENARIOS.items()):
            if st.button(label, key=f"st{i}"):
                imp = stress_test(weights, shocks)["portfolio_impact"]
                st.write(f"Gesch\u00e4tzt: **{imp*100:+.1f} %** \u2192 ca. **{imp*nav:+,.0f} \u20ac**")
elif page == "\u2696 Vergleich":
    st.header("Zwei Pakete vergleichen")
    l, r = st.columns(2)
    left = l.selectbox("Links", list(PACKAGES.keys()), index=1)
    right = r.selectbox("Rechts", list(PACKAGES.keys()), index=0)
    if st.button("Vergleichen"):
        for col, name in ((l, left), (r, right)):
            with col:
                st.subheader(name)
                st.write(PACKAGES[name]["desc"])
                bt = package_backtest(PACKAGES, name, 10000, 5)
                if bt["ok"]:
                    st.metric("Aus 10.000 \u20ac (ca. 5 J.)", f"{bt['end']:,.0f} \u20ac")
elif page == "\u2b50 Merkliste":
    st.header("Merkliste")
    wl = st.session_state.watchlist
    add = st.selectbox("Hinzuf\u00fcgen", [""] + [p["ticker"] for p in PRODUCTS])
    if add and st.button("Merken"):
        if add not in wl:
            wl.append(add)
            st.session_state.watchlist = wl
            persist()
            st.rerun()
    if wl:
        for t in wl:
            st.write(f"{PRODUCT_BY_T.get(t, {}).get('name', t)} ({t}): {get_last_price(t):,.2f}")
    else:
        st.info("Noch nichts gemerkt.")
elif page == "\U0001f50d Entdecken":
    st.header("M\u00e4rkte")
    st.caption("Nutze die Pakete unter Anlegen \u2013 oder lade hier Kurse.")
    if st.button("Beispiel: Welt-Aktien Kurs"):
        h = fetch_history("EUNL.DE", "2y")
        if not h.empty:
            st.plotly_chart(px.line(h, y="Close", title="EUNL.DE"), use_container_width=True)
else:
    st.header("Hilfe")
    st.markdown("""
### Neu
- Ziel-Sparen, Monats-Sparplan, Checkliste, Streuungs-Ampel
- Vergleich, Merkliste, Stress-Tests, Report

### Speichern
Depot links herunterladen und sp\u00e4ter wieder laden.

### Sp\u00e4ter (braucht Setup)
Eigene API-Keys, Cloud-Speicher, Login \u2013 sag Bescheid, wenn du das willst.
""")
    if st.button("Depot zur\u00fccksetzen"):
        st.session_state.holdings = {}
        st.session_state.cash = 10000.0
        st.session_state.trades = []
        st.session_state.watchlist = []
        st.session_state.sparplan = {"pkg": "Ausgewogen (Empfehlung)", "amount": 200.0, "months_done": 0}
        st.session_state.nav_history = []
        persist()
        st.rerun()

st.caption("ATLAS \u00dcbungs-Depot \u00b7 Kein echtes Geld \u00b7 Keine Anlageberatung")
