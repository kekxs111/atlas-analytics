from __future__ import annotations
import base64
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

warnings.filterwarnings("ignore", category=FutureWarning)
st.set_page_config(page_title="ATLAS – Einfach anlegen", page_icon="\U0001f310", layout="wide", initial_sidebar_state="expanded")
TZ = ZoneInfo("Europe/Berlin")
NOW = datetime.now(TZ)
SAVE_PATH = Path(__file__).resolve().parent / "atlas_save.json"

PRODUCTS = [
    {"ticker": "EUNL.DE", "name": "Welt-Aktien", "tag": "Empfohlen",
     "text": "Viele Firmen weltweit in einem Paket. Gut zum Start.", "risk": "Mittel"},
    {"ticker": "VWCE.DE", "name": "Welt-Aktien (Vanguard)", "tag": "Alternative",
     "text": "\u00c4hnlich wie Welt-Aktien, anderer Anbieter.", "risk": "Mittel"},
    {"ticker": "SPY", "name": "USA-Aktien (S&P 500)", "tag": "Beliebt",
     "text": "Die 500 gr\u00f6\u00dften US-Firmen.", "risk": "Mittel"},
    {"ticker": "QQQ", "name": "Tech-Aktien", "tag": "Wachstum",
     "text": "Gro\u00dfe Tech-Firmen. Kann stark schwanken.", "risk": "H\u00f6her"},
    {"ticker": "AGG", "name": "Anleihen", "tag": "Ruhiger",
     "text": "Weniger Schwankung, oft niedrigere Rendite.", "risk": "Niedrig"},
    {"ticker": "GLD", "name": "Gold", "tag": "Beimischung",
     "text": "Kann in Krisen helfen, bringt keine Zinsen.", "risk": "Mittel"},
    {"ticker": "AAPL", "name": "Apple", "tag": "Einzelaktie",
     "text": "Eine Firma \u2013 riskanter als ein Paket.", "risk": "H\u00f6her"},
    {"ticker": "MSFT", "name": "Microsoft", "tag": "Einzelaktie",
     "text": "Eine Firma \u2013 riskanter als ein Paket.", "risk": "H\u00f6her"},
    {"ticker": "NVDA", "name": "NVIDIA", "tag": "Einzelaktie",
     "text": "Kann stark steigen und fallen.", "risk": "Hoch"},
    {"ticker": "SAP", "name": "SAP", "tag": "Deutschland",
     "text": "Deutsches Software-Unternehmen.", "risk": "H\u00f6her"},
]
PRODUCT_BY_T = {p["ticker"]: p for p in PRODUCTS}

PACKAGES = {
    "Ruhig starten": {
        "desc": "Mehr Stabilit\u00e4t. Gut, wenn du Verluste ungern siehst.",
        "parts": [("AGG", 0.50), ("EUNL.DE", 0.40), ("GLD", 0.10)],
    },
    "Ausgewogen (Empfehlung)": {
        "desc": "Klassisch: viel Welt, etwas Anleihen, etwas Gold.",
        "parts": [("EUNL.DE", 0.60), ("AGG", 0.25), ("GLD", 0.15)],
    },
    "Mehr Wachstum": {
        "desc": "Haupts\u00e4chlich Aktien. Kann st\u00e4rker schwanken.",
        "parts": [("EUNL.DE", 0.50), ("SPY", 0.30), ("QQQ", 0.20)],
    },
}


def export_state() -> dict:
    return {
        "version": 3,
        "saved_at": datetime.now(TZ).isoformat(),
        "cash": float(st.session_state.get("cash", 10000.0)),
        "holdings": st.session_state.get("holdings", {}),
        "trades": st.session_state.get("trades", []),
        "seen_welcome": bool(st.session_state.get("seen_welcome", False)),
        "price_cache": st.session_state.get("price_cache", {}),
        "watchlist": st.session_state.get("watchlist", []),
        "sparplan": st.session_state.get("sparplan", {"pkg": "Ausgewogen (Empfehlung)", "amount": 200.0, "months_done": 0}),
    }


def apply_state(data: dict) -> None:
    try:
        st.session_state.cash = float(data.get("cash", 10000.0))
    except (TypeError, ValueError):
        st.session_state.cash = 10000.0
    holdings = data.get("holdings") or {}
    clean = {}
    if isinstance(holdings, dict):
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
    if "price_cache" not in st.session_state:
        st.session_state.price_cache = {}
    pc = data.get("price_cache") or {}
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


def state_to_code() -> str:
    raw = json.dumps(export_state(), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def code_to_state(code: str):
    try:
        raw = base64.urlsafe_b64decode(code.strip().encode("ascii"))
        data = json.loads(raw.decode("utf-8"))
        return data if isinstance(data, dict) and "cash" in data else None
    except Exception:
        return None


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
        apply_state(json.loads(SAVE_PATH.read_text(encoding="utf-8")))
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
    }
    if "holdings" not in st.session_state:
        if not load_from_disk():
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
            "Was": pname, "K\u00fcrzel": t, "St\u00fcck": qty,
            "Kurs": px, "Wert \u20ac": mv, "G/V \u20ac": pnl,
            "G/V %": (pnl / cost * 100) if cost and pnl == pnl else float("nan"),
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
        return False, f"Nur {cash:,.2f} \u20ac frei \u2013 braucht {notional:,.2f} \u20ac."
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
    return True, f"Gekauft: {name} f\u00fcr {notional:,.2f} \u20ac."


def do_sell(ticker: str, qty: float, px: float):
    h = st.session_state.holdings.get(ticker, {"qty": 0.0, "avg_cost": 0.0})
    old_q = float(h["qty"])
    if qty > old_q + 1e-9:
        return False, f"Du hast nur {old_q:.4g} St\u00fcck."
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
    return True, f"Verkauft: {name} \u00b7 ca. {notional:,.2f} \u20ac zur\u00fcck."


def buy_package(pkg_name: str, euro: float):
    parts = PACKAGES[pkg_name]["parts"]
    prices = {}
    for t, _ in parts:
        px = get_last_price(t)
        if px != px or px <= 0:
            return False, f"Kurs f\u00fcr {t} gerade nicht da. Sp\u00e4ter nochmal."
        prices[t] = px
    cash = float(st.session_state.cash)
    if euro > cash + 1e-6:
        return False, f"Nur {cash:,.2f} \u20ac frei."
    orders = []
    total = 0.0
    for t, w in parts:
        spend = euro * w
        qty = spend / prices[t]
        orders.append((t, qty, prices[t]))
        total += qty * prices[t]
    if total > cash + 1e-6:
        return False, f"Nur {cash:,.2f} \u20ac frei."
    for t, qty, px in orders:
        ok, msg = do_buy(t, qty, px)
        if not ok:
            return False, msg
    persist()
    return True, f"Paket \u201e{pkg_name}\u201c f\u00fcr {euro:,.2f} \u20ac eingerichtet."


def diversifikation(pos_df, nav):
    if pos_df is None or pos_df.empty or nav <= 0:
        return "\u26aa noch nichts", "Lege zuerst etwas an."
    max_share = float(pos_df["Anteil %"].max())
    n = len(pos_df)
    if max_share >= 80:
        return "\U0001f534 zu einseitig", "\u00dcber 80 % in einer Sache \u2013 besser streuen."
    if max_share >= 50 and n <= 2:
        return "\U0001f7e1 etwas einseitig", "Ein gro\u00dfer Brocken. Ein Welt-Paket hilft."
    if n >= 3 and max_share < 50:
        return "\U0001f7e2 gut gestreut", "Auf mehrere Bausteine verteilt."
    return "\U0001f7e1 okay", "Geht in Ordnung."


init_state()
_pref = list(st.session_state.holdings.keys()) + [p["ticker"] for p in PRODUCTS]
for _p in PACKAGES.values():
    _pref.extend(t for t, _ in _p["parts"])
if "prices_warmed" not in st.session_state:
    with st.spinner("Kurse laden \u2026"):
        prefetch_prices(_pref)
    st.session_state.prices_warmed = True

cash, pos_df, invested, nav, cost_total, prices = portfolio_snapshot()
upnl = invested - cost_total if cost_total else 0.0
has_pos = not pos_df.empty
div_l, div_t = diversifikation(pos_df, nav)

with st.sidebar:
    st.title("\U0001f310 ATLAS")
    st.caption("Einfach \u00fcben \u2013 kein echtes Geld")
    page = st.radio(
        "Men\u00fc",
        ["\U0001f3e0 Start", "\U0001f6d2 Anlegen", "\U0001f4c9 Risiko", "\u2696 Vergleich", "\u2753 Hilfe"],
        label_visibility="collapsed",
    )
    st.divider()
    st.metric("Gesamtwert", f"{nav:,.2f} \u20ac")
    st.caption(f"Bargeld: {cash:,.2f} \u20ac")
    if has_pos:
        st.caption(("\U0001f7e2" if upnl >= 0 else "\U0001f534") + f" G/V: {upnl:+,.2f} \u20ac")
    st.caption(f"Streuung: {div_l}")
    if st.button("\U0001f504 Kurse aktualisieren"):
        st.cache_data.clear()
        st.session_state.pop("prices_warmed", None)
        st.rerun()

    st.divider()
    st.markdown("### \U0001f4be Speichern")
    st.caption("Damit dein Depot nach dem Schlie\u00dfen erhalten bleibt.")
    st.download_button(
        "Datei speichern",
        data=json.dumps(export_state(), ensure_ascii=False, indent=2),
        file_name="mein_atlas_depot.json",
        mime="application/json",
        use_container_width=True,
        type="primary",
    )
    up = st.file_uploader("Datei wieder laden", type=["json"], label_visibility="collapsed")
    if up is not None:
        sig = f"{up.name}:{getattr(up, 'size', 0)}"
        if st.session_state.get("_upload_sig") != sig:
            try:
                data = json.loads(up.getvalue().decode("utf-8"))
                if isinstance(data, dict) and "cash" in data:
                    apply_state(data)
                    persist()
                    st.session_state._upload_sig = sig
                    st.success("Depot geladen.")
                    st.rerun()
                else:
                    st.error("Ung\u00fcltige Datei.")
            except Exception:
                st.error("Datei konnte nicht gelesen werden.")

    with st.expander("Oder: Code kopieren"):
        st.caption("Code speichern (Notizen/Handy). Sp\u00e4ter wieder einf\u00fcgen.")
        st.text_area("Dein Code (zum Kopieren)", value=state_to_code(), height=80, key="code_out")
        paste = st.text_area("Code hier einf\u00fcgen zum Laden", height=80, key="code_in")
        if st.button("Code laden") and paste.strip():
            data = code_to_state(paste)
            if data:
                apply_state(data)
                persist()
                st.success("Geladen.")
                st.rerun()
            else:
                st.error("Code ung\u00fcltig.")

    st.caption(NOW.strftime("%d.%m.%Y %H:%M"))

if page == "\U0001f3e0 Start":
    st.header("Dein \u00dcbungs-Depot")
    st.write("Du startest mit **10.000 \u20ac Spielgeld**. Nichts hier ist echt.")

    a, b, c, d = st.columns(4)
    a.metric("Gesamtwert", f"{nav:,.2f} \u20ac")
    b.metric("Bargeld", f"{cash:,.2f} \u20ac")
    c.metric("Investiert", f"{invested:,.2f} \u20ac")
    d.metric("Gewinn/Verlust", f"{upnl:+,.2f} \u20ac")

    pct = (invested / nav * 100) if nav > 0 else 0.0
    st.progress(min(1.0, max(0.0, pct / 100.0)), text=f"Angelegt: {pct:.0f} %")
    st.caption(f"**Streuung:** {div_l} \u2014 {div_t}")

    if not has_pos:
        st.info("Noch nichts angelegt. Ein Klick gen\u00fcgt:")
        amt = min(2000.0, max(100.0, cash))
        c1, c2, c3 = st.columns(3)
        if c1.button(f"Ausgewogen \u00b7 {amt:,.0f} \u20ac", type="primary", use_container_width=True):
            ok, msg = buy_package("Ausgewogen (Empfehlung)", amt)
            (st.success if ok else st.error)(msg)
            if ok:
                st.session_state.seen_welcome = True
                st.balloons()
                st.rerun()
        if c2.button(f"Ruhig \u00b7 {amt:,.0f} \u20ac", use_container_width=True):
            ok, msg = buy_package("Ruhig starten", amt)
            (st.success if ok else st.error)(msg)
            if ok:
                st.rerun()
        if c3.button(f"Wachstum \u00b7 {amt:,.0f} \u20ac", use_container_width=True):
            ok, msg = buy_package("Mehr Wachstum", amt)
            (st.success if ok else st.error)(msg)
            if ok:
                st.rerun()
    else:
        st.subheader("Was du besitzt")
        st.dataframe(pos_df, use_container_width=True, hide_index=True)
        if len(pos_df) > 0:
            st.plotly_chart(
                px.pie(pos_df, values="Wert \u20ac", names="Was", title="Verteilung"),
                use_container_width=True,
            )

    with st.expander("\U0001f3af Ziel-Sparen"):
        ziel = st.number_input("Ziel in \u20ac", min_value=1000.0, value=50000.0, step=1000.0)
        jahre = st.slider("In Jahren", 1, 40, 10)
        rendite = st.slider("Erwartete Rendite % p.a.", 0.0, 12.0, 6.0, 0.5)
        r_m = rendite / 100 / 12
        n = jahre * 12
        fv0 = nav * ((1 + rendite / 100) ** jahre)
        need = max(0.0, ziel - fv0)
        if need <= 0:
            st.success("Mit dem aktuellen Depot und der Rendite erreichst du das Ziel voraussichtlich ohne Extra-Sparen.")
        else:
            monat = need / n if r_m <= 0 else need * r_m / ((1 + r_m) ** n - 1)
            st.metric("Ungef\u00e4hr monatlich sparen", f"{monat:,.0f} \u20ac")
        st.caption("Nur Orientierung \u2013 keine Garantie.")

    with st.expander("\U0001f4c5 Monats-Sparplan \u00fcben"):
        sp = st.session_state.sparplan
        pkg = st.selectbox("Paket", list(PACKAGES.keys()), index=1)
        betrag = st.number_input("Betrag \u20ac", min_value=50.0, value=float(sp.get("amount", 200)), step=50.0)
        st.caption(f"Schon ge\u00fcbt: {int(sp.get('months_done', 0))} Monat(e)")
        if st.button("Diesen Monat einzahlen", type="primary"):
            ok, msg = buy_package(pkg, betrag)
            if ok:
                st.session_state.sparplan = {"pkg": pkg, "amount": betrag, "months_done": int(sp.get("months_done", 0)) + 1}
                persist()
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    with st.expander("Spielgeld \u00e4ndern"):
        amt = st.number_input("Betrag \u20ac", min_value=0.0, value=1000.0, step=100.0, key="cash_chg")
        x, y = st.columns(2)
        if x.button("\u2795 Erh\u00f6hen"):
            st.session_state.cash = float(st.session_state.cash) + amt
            persist()
            st.rerun()
        if y.button("\u2796 Verringern"):
            if amt <= float(st.session_state.cash):
                st.session_state.cash = float(st.session_state.cash) - amt
                persist()
                st.rerun()
            else:
                st.error("Nicht genug Bargeld.")

    if st.session_state.trades:
        with st.expander("Letzte K\u00e4ufe/Verk\u00e4ufe"):
            st.dataframe(pd.DataFrame(st.session_state.trades[::-1][:20]), use_container_width=True, hide_index=True)

elif page == "\U0001f6d2 Anlegen":
    st.header("Geld anlegen")
    st.write(f"Frei: **{cash:,.2f} \u20ac**")
    tab1, tab2, tab3 = st.tabs(["Fertiges Paket", "Einzeltitel", "Verkaufen"])

    with tab1:
        if cash < 100:
            st.warning("Mindestens 100 \u20ac Bargeld n\u00f6tig.")
            euro = 0.0
        else:
            euro = st.number_input("Betrag f\u00fcrs Paket \u20ac", min_value=100.0, max_value=float(cash),
                                   value=min(2000.0, float(cash)), step=100.0)
        for name, info in PACKAGES.items():
            with st.container(border=True):
                st.markdown(f"**{name}**")
                st.write(info["desc"])
                parts = " + ".join(f"{int(w*100)}% {PRODUCT_BY_T.get(t,{}).get('name',t)}" for t, w in info["parts"])
                st.caption(parts)
                if cash >= 100 and st.button(f"Kaufen: {name}", key=f"pkg_{name}",
                                            type="primary" if "Empfehlung" in name else "secondary"):
                    ok, msg = buy_package(name, euro)
                    (st.success if ok else st.error)(msg)
                    if ok:
                        st.balloons()
                        st.rerun()

    with tab2:
        st.caption("Tipp: Ein Paket ist oft sinnvoller als eine einzelne Aktie.")
        for p in PRODUCTS:
            with st.container(border=True):
                st.markdown(f"**{p['name']}** `{p['ticker']}` \u00b7 Risiko: {p['risk']}")
                st.write(p["text"])
                if st.button("Ausw\u00e4hlen", key=f"pick_{p['ticker']}"):
                    st.session_state.pick = p["ticker"]
                    st.rerun()
        t = st.session_state.get("pick")
        if t:
            px = get_last_price(t)
            name = PRODUCT_BY_T.get(t, {}).get("name") or t
            if px != px or px <= 0:
                st.error("Kurs gerade nicht verf\u00fcgbar.")
            else:
                st.success(f"{name} \u00b7 ca. {px:,.2f}")
                euro = st.number_input("Wie viel \u20ac?", min_value=1.0, max_value=max(1.0, cash),
                                      value=min(500.0, max(1.0, cash)), step=50.0, key="buy_e")
                if st.button("\u2713 Kaufen", type="primary"):
                    ok, msg = do_buy(t, euro / px, px)
                    (st.success if ok else st.error)(msg)
                    if ok:
                        st.rerun()

    with tab3:
        owned = [t for t, h in st.session_state.holdings.items() if float(h.get("qty", 0)) > 0]
        if not owned:
            st.info("Noch nichts zum Verkaufen.")
        else:
            t = st.selectbox(
                "Anlage", owned,
                format_func=lambda x: f"{PRODUCT_BY_T.get(x,{}).get('name',x)} ({st.session_state.holdings[x]['qty']:.4g})",
            )
            px = get_last_price(t)
            max_q = float(st.session_state.holdings[t]["qty"])
            if px != px:
                st.error("Kurs fehlt.")
            else:
                st.write(f"Kurs: **{px:,.2f}**")
                step = 0.01 if max_q >= 0.02 else max(max_q / 10, 1e-6)
                qty = st.slider("St\u00fcck verkaufen", float(step), float(max_q), float(max_q), float(step))
                st.caption(f"Ca. **{qty * px:,.2f} \u20ac** zur\u00fcck")
                if st.button("\u2713 Verkaufen", type="primary"):
                    ok, msg = do_sell(t, qty, px)
                    (st.success if ok else st.error)(msg)
                    if ok:
                        st.rerun()

elif page == "\U0001f4c9 Risiko":
    st.header("Wie riskant?")
    st.caption(f"{div_l} \u2014 {div_t}")
    weights = holdings_to_weights(st.session_state.holdings, prices)
    if not weights:
        st.warning("Lege zuerst etwas an.")
    else:
        with st.spinner("Rechne \u2026"):
            port_rets, _ = portfolio_returns(weights, "3y")
        if port_rets.empty:
            st.error("Zu wenig Kursdaten.")
        else:
            vol = port_rets.std(ddof=1) * np.sqrt(252)
            hvar, _ = historical_var_cvar(port_rets, 0.05)
            ampel = "\U0001f7e2 eher ruhig" if vol < 0.12 else ("\U0001f7e1 mittel" if vol < 0.22 else "\U0001f534 eher unruhig")
            st.subheader(f"Ampel: {ampel}")
            m1, m2 = st.columns(2)
            m1.metric("Typische Schwankung / Jahr", f"\u00b1 {vol*100:.0f} %")
            m2.metric("Schlechter Tag (Sch\u00e4tzung)", f"\u2212 {hvar*nav:,.0f} \u20ac")

            st.subheader("Was w\u00e4re wenn \u2026")
            scenarios = {
                "B\u00f6rse \u221220 %": {"Equity": -0.20, "Equity ETF": -0.20, "Fixed Income": 0.02, "Commodity": -0.05, "Real Estate": -0.15},
                "Crash \u221235 %": {"Equity": -0.35, "Equity ETF": -0.33, "Fixed Income": 0.05, "Commodity": -0.15, "Real Estate": -0.30},
                "Zinsen steigen": {"Equity": -0.05, "Equity ETF": -0.05, "Fixed Income": -0.08, "Commodity": 0.0, "Real Estate": -0.10},
                "Gold +15 %": {"Equity": 0.0, "Equity ETF": 0.0, "Fixed Income": 0.0, "Commodity": 0.15, "Real Estate": 0.0},
            }
            cols = st.columns(2)
            for i, (label, shocks) in enumerate(scenarios.items()):
                with cols[i % 2]:
                    if st.button(label, key=f"s{i}", use_container_width=True):
                        imp = stress_test(weights, shocks)["portfolio_impact"]
                        st.write(f"**{imp*100:+.1f} %** \u2192 ca. **{imp*nav:+,.0f} \u20ac**")

elif page == "\u2696 Vergleich":
    st.header("Zwei Pakete vergleichen")
    l, r = st.columns(2)
    left = l.selectbox("Links", list(PACKAGES.keys()), index=1)
    right = r.selectbox("Rechts", list(PACKAGES.keys()), index=0)
    if st.button("Vergleichen", type="primary"):
        for col, name in ((l, left), (r, right)):
            with col:
                st.subheader(name)
                st.write(PACKAGES[name]["desc"])
                end = 0.0
                ok = True
                for t, w in PACKAGES[name]["parts"]:
                    h = fetch_history(t, "5y")
                    if h.empty or len(h) < 30:
                        ok = False
                        break
                    p = h["Close"].astype(float)
                    end += 10000 * w * (p.iloc[-1] / p.iloc[0])
                if ok:
                    st.metric("Aus 10.000 \u20ac (ca. 5 Jahre)", f"{end:,.0f} \u20ac")
                else:
                    st.warning("Historie gerade nicht ladbar.")
        st.caption("Vergangenheit \u2260 Zukunft.")

else:
    st.header("Hilfe")
    st.markdown(
        """
### So speicherst du dein Depot
1. Links auf **Datei speichern** klicken
2. Die Datei landet auf deinem Ger\u00e4t
3. Sp\u00e4ter: **Datei wieder laden**

**Alternative:** Unter \u201eOder: Code kopieren\u201c den Textcode in Notizen speichern und sp\u00e4ter wieder einf\u00fcgen.

### Tipps
- Starte mit dem Paket **Ausgewogen**
- Streue \u2013 nicht alles in eine Aktie
- Das ist ein **\u00dcbungsprogramm**, keine Anlageberatung

### Kurse
Kommen automatisch. Manchmal kurz verz\u00f6gert \u2013 dann **Kurse aktualisieren** tippen.
        """
    )
    if st.button("Alles zur\u00fccksetzen (10.000 \u20ac)", type="primary"):
        st.session_state.holdings = {}
        st.session_state.cash = 10000.0
        st.session_state.trades = []
        st.session_state.seen_welcome = False
        st.session_state.price_cache = {}
        st.session_state.watchlist = []
        st.session_state.sparplan = {"pkg": "Ausgewogen (Empfehlung)", "amount": 200.0, "months_done": 0}
        st.session_state.pop("prices_warmed", None)
        persist()
        st.success("Zur\u00fcckgesetzt.")
        st.rerun()

st.divider()
st.caption("ATLAS \u00b7 \u00dcbungs-Depot \u00b7 Kein echtes Geld \u00b7 Keine Beratung")
