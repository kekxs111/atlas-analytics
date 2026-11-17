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

warnings.filterwarnings("ignore", category=FutureWarning)
st.set_page_config(page_title="ATLAS – Einfach anlegen", page_icon="🌐", layout="wide", initial_sidebar_state="expanded")
TZ = ZoneInfo("Europe/Berlin")
NOW = datetime.now(TZ)
SAVE_PATH = Path(__file__).resolve().parent / "atlas_save.json"

PRODUCTS = [
    {"ticker": "EUNL.DE", "name": "Welt-Aktien", "tag": "Empfohlen zum Start",
     "text": "Ein Paket aus Tausenden Unternehmen weltweit. Gut zum langfristigen Anlegen.", "risk": "Mittel"},
    {"ticker": "VWCE.DE", "name": "Welt-Aktien (Vanguard)", "tag": "Alternative",
     "text": "Ähnlich wie Welt-Aktien: breit gestreut über viele Länder und Firmen.", "risk": "Mittel"},
    {"ticker": "SPY", "name": "USA-Aktien (S&P 500)", "tag": "Beliebt",
     "text": "Die 500 größten US-Firmen. Historisch stark, aber nur Amerika.", "risk": "Mittel"},
    {"ticker": "QQQ", "name": "Tech-Aktien", "tag": "Wachstum",
     "text": "Große Tech-Firmen (z. B. Apple, Microsoft). Kann stark schwanken.", "risk": "Höher"},
    {"ticker": "AGG", "name": "Anleihen (stabiler)", "tag": "Ruhiger",
     "text": "Staats- und Unternehmensanleihen. Weniger Schwankung, oft niedrigere Rendite.", "risk": "Niedrig"},
    {"ticker": "GLD", "name": "Gold", "tag": "Beimischung",
     "text": "Gold als Absicherung. Kann in Krisen helfen, bringt keine Zinsen.", "risk": "Mittel"},
    {"ticker": "AAPL", "name": "Apple", "tag": "Einzelaktie",
     "text": "Eine einzelne Firma. Mehr Risiko als ein Paket aus vielen Firmen.", "risk": "Höher"},
    {"ticker": "MSFT", "name": "Microsoft", "tag": "Einzelaktie",
     "text": "Eine einzelne Firma. Mehr Risiko als ein Paket aus vielen Firmen.", "risk": "Höher"},
    {"ticker": "NVDA", "name": "NVIDIA", "tag": "Einzelaktie",
     "text": "Chip-Firma. Kann stark steigen und fallen.", "risk": "Hoch"},
    {"ticker": "SAP", "name": "SAP", "tag": "Deutschland",
     "text": "Deutsches Software-Unternehmen. Eine einzelne Firma.", "risk": "Höher"},
]
PRODUCT_BY_T = {p["ticker"]: p for p in PRODUCTS}

PACKAGES = {
    "Ruhig starten": {
        "desc": "Mehr Stabilität, weniger Schwankung. Gut, wenn du nervös bei Verlusten bist.",
        "parts": [("AGG", 0.50), ("EUNL.DE", 0.40), ("GLD", 0.10)],
    },
    "Ausgewogen (Empfehlung)": {
        "desc": "Klassische Mischung: viel Welt, etwas Anleihen, etwas Gold.",
        "parts": [("EUNL.DE", 0.60), ("AGG", 0.25), ("GLD", 0.15)],
    },
    "Mehr Wachstum": {
        "desc": "Hauptsächlich Aktien. Kann stärker schwanken, langfristig oft höhere Chance.",
        "parts": [("EUNL.DE", 0.50), ("SPY", 0.30), ("QQQ", 0.20)],
    },
}

def export_state() -> dict:
    return {
        "version": 1,
        "saved_at": datetime.now(TZ).isoformat(),
        "cash": float(st.session_state.get("cash", 10000.0)),
        "holdings": st.session_state.get("holdings", {}),
        "trades": st.session_state.get("trades", []),
        "seen_welcome": bool(st.session_state.get("seen_welcome", False)),
        "price_cache": st.session_state.get("price_cache", {}),
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
    if "price_cache" not in st.session_state:
        st.session_state.price_cache = {}
    if "holdings" not in st.session_state:
        if load_from_disk():
            pass
        else:
            st.session_state.holdings = {}
            st.session_state.cash = 10000.0
            st.session_state.trades = []
            st.session_state.seen_welcome = False
    if "cash" not in st.session_state:
        st.session_state.cash = 10000.0
    if "trades" not in st.session_state:
        st.session_state.trades = []
    if "seen_welcome" not in st.session_state:
        st.session_state.seen_welcome = False
    if "books" in st.session_state:
        active = st.session_state.get("active_book", "Core")
        st.session_state.holdings = st.session_state.books.get(active, {})
        c = st.session_state.cash
        st.session_state.cash = float(c.get(active, 10000.0)) if isinstance(c, dict) else float(c or 10000)
        del st.session_state.books
        persist()

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
            "Was": pname, "Kürzel": t, "Stück": qty, "Dein Einkauf": avg,
            "Aktueller Kurs": px, "Wert jetzt €": mv, "Gewinn/Verlust €": pnl,
            "Gewinn/Verlust %": (pnl / cost * 100) if cost and pnl == pnl else float("nan"),
            "Anteil %": 0.0,
        })
    nav = cash + invested
    for r in rows:
        if nav > 0 and r["Wert jetzt €"] == r["Wert jetzt €"]:
            r["Anteil %"] = r["Wert jetzt €"] / nav * 100
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Wert jetzt €", ascending=False)
    return cash, df, invested, nav, cost_total, prices

def do_buy(ticker: str, qty: float, px: float):
    notional = qty * px
    cash = float(st.session_state.cash)
    if notional > cash + 1e-6:
        return False, f"Du hast nur {cash:,.2f} € frei. Diese Order braucht {notional:,.2f} €."
    h = st.session_state.holdings.get(ticker, {"qty": 0.0, "avg_cost": 0.0})
    old_q, old_c = float(h["qty"]), float(h["avg_cost"])
    new_q = old_q + qty
    new_avg = ((old_q * old_c) + notional) / new_q if new_q > 0 else px
    st.session_state.holdings[ticker] = {"qty": new_q, "avg_cost": new_avg}
    st.session_state.cash = cash - notional
    name = PRODUCT_BY_T.get(ticker, {}).get("name") or ticker
    st.session_state.trades.append({
        "Zeit": NOW.strftime("%d.%m.%Y %H:%M"), "Art": "Kauf", "Was": name,
        "Kürzel": ticker, "Stück": qty, "Preis": px, "Betrag €": notional,
    })
    persist()
    return True, f"Geschafft! Du hast {name} für {notional:,.2f} € gekauft."

def do_sell(ticker: str, qty: float, px: float):
    h = st.session_state.holdings.get(ticker, {"qty": 0.0, "avg_cost": 0.0})
    old_q = float(h["qty"])
    if qty > old_q + 1e-9:
        return False, f"Du besitzt nur {old_q:.4g} Stück."
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
        "Kürzel": ticker, "Stück": qty, "Preis": px, "Betrag €": notional,
    })
    persist()
    return True, f"Verkauft: {name} · Du erhältst ca. {notional:,.2f} € zurück."

def buy_package(pkg_name: str, euro: float):
    parts = PACKAGES[pkg_name]["parts"]
    prices = {}
    for t, _ in parts:
        px = get_last_price(t)
        if px != px or px <= 0:
            return False, f"Kurs für {t} gerade nicht verfügbar. Bitte später nochmal."
        prices[t] = px
    cash = float(st.session_state.cash)
    if euro > cash + 1e-6:
        return False, f"Du hast nur {cash:,.2f} € frei."
    orders = []
    total = 0.0
    for t, w in parts:
        spend = euro * w
        qty = spend / prices[t]
        orders.append((t, qty, prices[t]))
        total += qty * prices[t]
    if total > cash + 1e-6:
        return False, f"Du hast nur {cash:,.2f} € frei."
    for t, qty, px in orders:
        ok, msg = do_buy(t, qty, px)
        if not ok:
            return False, msg
    persist()
    return True, f"Paket „{pkg_name}“ für {euro:,.2f} € ist eingerichtet."

init_state()

_prefetch_list = list(st.session_state.holdings.keys()) + [p["ticker"] for p in PRODUCTS]
for _parts in PACKAGES.values():
    _prefetch_list.extend(t for t, _ in _parts["parts"])
if "prices_warmed" not in st.session_state:
    with st.spinner("Kurse werden geladen …"):
        prefetch_prices(_prefetch_list)
    st.session_state.prices_warmed = True

cash, pos_df, invested, nav, cost_total, prices = portfolio_snapshot()
upnl = invested - cost_total if cost_total else 0.0
has_positions = not pos_df.empty

with st.sidebar:
    st.title("🌐 ATLAS")
    st.caption("Für Einsteiger gemacht")
    page = st.radio(
        "Menü",
        ["🏠 Start", "🛒 Anlegen", "📉 Wie riskant?", "🔍 Entdecken", "❓ Hilfe"],
        label_visibility="collapsed",
    )
    st.divider()
    st.metric("Dein Gesamtwert", f"{nav:,.2f} €")
    st.caption(f"Davon Bargeld: {cash:,.2f} €")
    if has_positions:
        color = "🟢" if upnl >= 0 else "🔴"
        st.caption(f"{color} Bisheriger Gewinn/Verlust: {upnl:+,.2f} €")
    if st.button("🔄 Kurse aktualisieren"):
        st.cache_data.clear()
        st.session_state.pop("prices_warmed", None)
        st.rerun()
    st.divider()
    st.caption("**Depot speichern / laden**")
    st.download_button(
        "💾 Depot herunterladen",
        data=json.dumps(export_state(), ensure_ascii=False, indent=2),
        file_name="atlas_depot.json",
        mime="application/json",
        use_container_width=True,
        help="Speichert Bargeld, Positionen und letzte Kurse auf deinem Gerät.",
    )
    up = st.file_uploader("📂 Depot laden", type=["json"], label_visibility="collapsed")
    if up is not None:
        sig = f"{up.name}:{getattr(up, 'size', 0)}"
        if st.session_state.get("_upload_sig") != sig:
            try:
                data = json.loads(up.getvalue().decode("utf-8"))
                if not isinstance(data, dict) or "cash" not in data:
                    st.error("Ungültige Depot-Datei.")
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
    st.caption("Übungs-Depot · kein echtes Geld")
    st.caption(NOW.strftime("%d.%m.%Y %H:%M"))

if page == "🏠 Start":
    st.header("Dein Übungs-Depot")
    st.markdown(
        "Hier kannst du **ohne echtes Geld** üben, wie Anlegen funktioniert. "
        "Du startest mit **10.000 € Spielgeld**."
    )
    if not has_positions and not st.session_state.seen_welcome:
        st.info(
            "**Willkommen!** Du brauchst kein Vorwissen.\n\n"
            "1. Lies die Erklärung unten.\n"
            "2. Nutze den **Schnellstart** oder gehe zu **🛒 Anlegen**.\n"
            "3. Schau danach hier, was passiert ist."
        )
        with st.expander("Was ist das hier eigentlich? (1 Minute lesen)", expanded=True):
            st.markdown(
                """
**Bargeld** = Geld, das noch nicht investiert ist.\n\n
**Anlegen** = Du tauschst Bargeld gegen Anteile an Firmen oder Paketen (ETFs).  
Der Wert kann steigen oder fallen.\n\n
**Gesamtwert** = Bargeld + aktueller Wert deiner Anteile.\n\n
**Gewinn/Verlust** = Wie viel mehr oder weniger deine Anteile wert sind als beim Kauf.\n\n
Das ist ein **Übungsprogramm**. Es kauft nichts Echtes.\n\n
Dein Depot wird **automatisch gespeichert** und du kannst es links unter
„Depot herunterladen“ auf dein Gerät sichern und später wieder laden.
                """
            )
        if st.button("Verstanden – weiter zum Anlegen", type="primary"):
            st.session_state.seen_welcome = True
            persist()
            st.rerun()
    a, b, c, d = st.columns(4)
    a.metric("Gesamtwert", f"{nav:,.2f} €", help="Bargeld + Wert deiner Anlagen")
    b.metric("Bargeld", f"{cash:,.2f} €", help="Noch nicht investiertes Geld")
    c.metric("Investiert", f"{invested:,.2f} €", help="Aktueller Marktwert deiner Anteile")
    d.metric("Gewinn / Verlust", f"{upnl:+,.2f} €",
             delta=f"{(upnl/cost_total*100) if cost_total else 0:.1f} %",
             help="Plus = im Plus seit dem Kauf, Minus = im Minus")
    invest_pct = (invested / nav * 100) if nav > 0 else 0.0
    st.progress(min(1.0, max(0.0, invest_pct / 100.0)), text=f"Angelegt: {invest_pct:.0f} % deines Gesamtwerts")
    if invest_pct < 5 and cash > 500:
        st.info("Tipp: Du hast noch viel Bargeld. Unter **Anlegen** startest du am einfachsten mit einem fertigen Paket.")
    st.subheader("Was du besitzt")
    if not has_positions:
        st.warning("Noch nichts angelegt.")
        st.markdown("**Schnellstart** – ein Klick genügt:")
        _qa = min(2000.0, max(100.0, float(cash)))
        q1, q2, q3 = st.columns(3)
        if q1.button(f"Ausgewogen · {_qa:,.0f} €", type="primary", use_container_width=True, key="quick_bal"):
            ok, msg = buy_package("Ausgewogen (Empfehlung)", _qa)
            if ok:
                st.session_state.seen_welcome = True
                st.success(msg)
                st.balloons()
                st.rerun()
            else:
                st.error(msg)
        if q2.button(f"Ruhig · {_qa:,.0f} €", use_container_width=True, key="quick_calm"):
            ok, msg = buy_package("Ruhig starten", _qa)
            if ok:
                st.session_state.seen_welcome = True
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)
        if q3.button(f"Wachstum · {_qa:,.0f} €", use_container_width=True, key="quick_growth"):
            ok, msg = buy_package("Mehr Wachstum", _qa)
            if ok:
                st.session_state.seen_welcome = True
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)
    else:
        show = pos_df[["Was", "Kürzel", "Stück", "Aktueller Kurs", "Wert jetzt €", "Gewinn/Verlust €", "Gewinn/Verlust %", "Anteil %"]]
        st.dataframe(
            show.style.format({
                "Stück": "{:.4g}", "Aktueller Kurs": "{:,.2f}", "Wert jetzt €": "{:,.2f}",
                "Gewinn/Verlust €": "{:+,.2f}", "Gewinn/Verlust %": "{:+.1f}", "Anteil %": "{:.0f}",
            }, na_rep="—"),
            use_container_width=True, hide_index=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.pie(pos_df, values="Wert jetzt €", names="Was", title="So ist dein Geld verteilt"), use_container_width=True)
        with c2:
            st.plotly_chart(
                px.bar(pos_df, x="Was", y="Gewinn/Verlust €", title="Gewinn oder Verlust je Anlage",
                       color="Gewinn/Verlust €", color_continuous_scale="RdYlGn"),
                use_container_width=True,
            )
    with st.expander("Mehr Spielgeld holen oder abheben"):
        st.caption("Nur im Übungs-Depot – kein echtes Konto.")
        amt = st.number_input("Betrag in €", min_value=0.0, value=1000.0, step=100.0, key="cash_amt")
        x, y = st.columns(2)
        if x.button("➕ Bargeld erhöhen", use_container_width=True):
            st.session_state.cash = float(st.session_state.cash) + amt
            persist()
            st.rerun()
        if y.button("➖ Bargeld verringern", use_container_width=True):
            if amt <= float(st.session_state.cash):
                st.session_state.cash = float(st.session_state.cash) - amt
                persist()
                st.rerun()
            else:
                st.error("So viel Bargeld hast du nicht.")
    with st.expander("📈 Sparplan-Rechner (Was wäre wenn …)"):
        st.caption("Einfache Hochrechnung – keine Garantie, nur Orientierung.")
        s1, s2, s3 = st.columns(3)
        monthly = s1.number_input("Monatlich sparen (€)", min_value=0.0, value=200.0, step=50.0, key="sp_month")
        years = s2.slider("Jahre", 1, 40, 10, key="sp_years")
        rate = s3.slider("Erwartete Rendite p.a. %", 0.0, 12.0, 6.0, 0.5, key="sp_rate")
        r_m = rate / 100.0 / 12.0
        n = int(years * 12)
        if r_m > 0:
            fv_contrib = monthly * (((1 + r_m) ** n - 1) / r_m)
        else:
            fv_contrib = monthly * n
        fv_start = nav * ((1 + rate / 100.0) ** years)
        total_fv = fv_start + fv_contrib
        paid = monthly * n
        st.metric(f"Geschätzter Wert in {years} Jahren", f"{total_fv:,.0f} €")
        st.caption(f"Eingezahlt aus Sparplan: {paid:,.0f} € · Aktuelles Depot (geschätzt): {fv_start:,.0f} €")
    if st.session_state.trades:
        with st.expander("Deine letzten Käufe und Verkäufe"):
            st.dataframe(pd.DataFrame(st.session_state.trades[::-1][:15]), use_container_width=True, hide_index=True)

elif page == "🛒 Anlegen":
    st.header("Geld anlegen")
    st.markdown(f"Du hast **{cash:,.2f} €** frei. Wähle einen Weg:")
    tab_pkg, tab_single, tab_sell = st.tabs(["① Fertiges Paket (einfach)", "② Einzelnen Titel wählen", "③ Etwas verkaufen"])
    with tab_pkg:
        st.subheader("Ein Klick – fertig gemischt")
        st.caption("Wir kaufen für dich mehrere Bausteine auf einmal.")
        if cash < 100:
            st.warning("Für ein Paket brauchst du mindestens 100 € Bargeld.")
            euro = 0.0
        else:
            euro = st.number_input(
                "Wie viel Euro soll das Paket kosten?",
                min_value=100.0, max_value=float(cash),
                value=min(2000.0, float(cash)), step=100.0, key="pkg_euro",
            )
        for name, info in PACKAGES.items():
            with st.container(border=True):
                st.markdown(f"### {name}")
                st.write(info["desc"])
                parts_txt = " + ".join(
                    f"{int(w*100)} % {PRODUCT_BY_T.get(t, {}).get('name', t)}" for t, w in info["parts"]
                )
                st.caption(f"Enthält: {parts_txt}")
                if euro >= 100 and st.button(f"Dieses Paket für {euro:,.0f} € kaufen", key=f"pkg_{name}", type="primary" if "Empfehlung" in name else "secondary"):
                    ok, msg = buy_package(name, euro)
                    if ok:
                        st.session_state.seen_welcome = True
                        st.success(msg)
                        st.balloons()
                        st.rerun()
                    else:
                        st.error(msg)
    with tab_single:
        st.subheader("Einen Titel selbst wählen")
        st.caption("Tipp: Lieber ein **Paket** (Welt-Aktien) als eine einzelne Firma.")
        for p in PRODUCTS:
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"**{p['name']}** · `{p['ticker']}`")
                    st.caption(f"{p['tag']} · Risiko: {p['risk']}")
                    st.write(p["text"])
                with c2:
                    if st.button("Auswählen", key=f"sel_{p['ticker']}", use_container_width=True):
                        st.session_state.pick_ticker = p["ticker"]
                        st.rerun()
        ticker = st.session_state.get("pick_ticker", "")
        if ticker:
            px = get_last_price(ticker)
            pname = PRODUCT_BY_T.get(ticker, {}).get("name") or ALL_UNIVERSE.get(ticker, ticker)
            if px != px or px <= 0:
                st.error("Kurs gerade nicht da. Bitte später nochmal.")
            else:
                st.success(f"Ausgewählt: **{pname}** · Kurs ca. **{px:,.2f}**")
                euro = st.number_input(
                    "Wie viel Euro investieren?",
                    min_value=1.0, max_value=max(1.0, cash),
                    value=min(500.0, max(1.0, cash)), step=50.0, key="single_euro",
                )
                qty = euro / px
                st.caption(f"Das sind ungefähr {qty:.4g} Anteile.")
                if st.button("✓ Jetzt kaufen", type="primary", use_container_width=True):
                    ok, msg = do_buy(ticker, qty, px)
                    if ok:
                        st.session_state.seen_welcome = True
                        st.success(msg)
                        st.balloons()
                        st.rerun()
                    else:
                        st.error(msg)
        with st.expander("Andere Kürzel suchen (Fortgeschritten)"):
            t2 = st.selectbox(
                "Suche", [""] + sorted(ALL_UNIVERSE.keys()),
                format_func=lambda x: f"{x} – {ALL_UNIVERSE.get(x, '')}" if x else "—",
                key="adv_ticker",
            )
            if t2 and st.button("Dieses Kürzel auswählen"):
                st.session_state.pick_ticker = t2
                st.rerun()
    with tab_sell:
        owned = [t for t, h in st.session_state.holdings.items() if float(h.get("qty", 0)) > 0]
        if not owned:
            st.info("Du besitzt noch nichts.")
        else:
            ticker = st.selectbox(
                "Deine Anlage", owned,
                format_func=lambda x: f"{PRODUCT_BY_T.get(x, {}).get('name', ALL_UNIVERSE.get(x, x))} ({st.session_state.holdings[x]['qty']:.4g} Stück)",
            )
            px = get_last_price(ticker)
            max_q = float(st.session_state.holdings[ticker]["qty"])
            if px != px:
                st.error("Kurs nicht verfügbar.")
            else:
                st.write(f"Aktueller Kurs: **{px:,.2f}**")
                step = 0.01 if max_q >= 0.02 else max(max_q / 10, 1e-6)
                qty = st.slider("Wie viele Anteile verkaufen?", float(step), float(max_q), float(max_q), float(step))
                st.caption(f"Du bekommst ungefähr **{qty * px:,.2f} €** zurück.")
                if st.button("✓ Verkaufen", type="primary", use_container_width=True):
                    ok, msg = do_sell(ticker, qty, px)
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

elif page == "📉 Wie riskant?":
    st.header("Wie riskant ist dein Depot?")
    weights = holdings_to_weights(st.session_state.holdings, prices)
    if not weights:
        st.warning("Lege zuerst etwas an.")
        st.stop()
    st.markdown("Schätzung anhand der letzten Jahre – keine Vorhersage.")
    with st.spinner("Einen Moment …"):
        port_rets, _ = portfolio_returns(weights, "3y")
    if port_rets.empty:
        st.error("Zu wenig Daten.")
        st.stop()
    ann_ret = port_rets.mean() * 252
    ann_vol = port_rets.std(ddof=1) * np.sqrt(252)
    hvar, _ = historical_var_cvar(port_rets, 0.05)
    if ann_vol < 0.12:
        ampel, text = "🟢 eher ruhig", "Dein Mix schwankt eher wenig."
    elif ann_vol < 0.22:
        ampel, text = "🟡 mittel", "Normale Schwankungen."
    else:
        ampel, text = "🔴 eher unruhig", "Stärkere Auf und Abs möglich."
    st.subheader(f"Risiko-Ampel: {ampel}")
    st.write(text)
    m1, m2, m3 = st.columns(3)
    m1.metric("Geschätzte Rendite / Jahr", f"{ann_ret*100:.0f} %")
    m2.metric("Typische Schwankung / Jahr", f"± {ann_vol*100:.0f} %")
    m3.metric("Schlechter Tag (Schätzung)", f"− {hvar*nav:,.0f} €")
    eq = (1 + port_rets).cumprod()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Depot", fill="tozeroy"))
    fig.update_layout(title="Wertentwicklung (indexiert)", height=300, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Was wäre bei einem Börsencrash?"):
        res = stress_test(weights, {"Equity": -0.35, "Equity ETF": -0.33, "Fixed Income": 0.05, "Commodity": -0.15, "Real Estate": -0.30})
        imp = res["portfolio_impact"]
        st.metric("Geschätzter Verlust bei −35 % Aktien", f"{imp*100:.0f} %  →  ca. {imp*nav:,.0f} €")
    with st.expander("Mögliche Werte in einem Jahr"):
        mc = monte_carlo_portfolio(port_rets, 2500, 252)
        if not mc.empty:
            p5, p50, p95 = np.percentile(mc["Final Return"], [5, 50, 95])
            a, b, c = st.columns(3)
            a.metric("Ungünstiges Jahr", f"{nav*(1+p5):,.0f} €")
            b.metric("Mittleres Jahr", f"{nav*(1+p50):,.0f} €")
            c.metric("Günstiges Jahr", f"{nav*(1+p95):,.0f} €")

elif page == "🔍 Entdecken":
    st.header("Märkte ansehen")
    scope = st.radio("Was anschauen?", ["Aktien", "ETFs (Pakete)", "Rohstoffe"], horizontal=True)
    uni = {"Aktien": STOCKS, "ETFs (Pakete)": ETFS, "Rohstoffe": COMMODITIES}[scope]
    limit = st.slider("Wie viele anzeigen?", 10, 40, 20, 5)
    if st.button("Liste laden", type="primary") or "mkt_loaded" not in st.session_state:
        st.session_state.mkt_loaded = True
        rows, bar = [], st.progress(0, text="Lade …")
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
        nice = df.rename(columns={
            "Ticker": "Kürzel", "Anlage": "Name", "1J Rendite %": "1 Jahr %",
            "Volatilität %": "Schwankung %",
        })
        st.dataframe(
            nice[["Kürzel", "Name", "Kurs", "1 Jahr %", "Schwankung %", "Risiko", "Score"]],
            use_container_width=True, height=400, hide_index=True,
        )
        pick = st.selectbox("Kursverlauf anzeigen", df["Ticker"].tolist())
        h = fetch_history(pick, "2y")
        if not h.empty:
            st.plotly_chart(px.line(h, y="Close", title=f"{pick} – letzte 2 Jahre"), use_container_width=True)

else:
    st.header("Hilfe & Erklärungen")
    st.markdown(
        """
### Neu in dieser Version
- **Schnellstart** auf der Startseite (3 Pakete, ein Klick)
- **Sparplan-Rechner**: Was wäre, wenn du monatlich sparst?
- **Schnellere Kurse** (parallel geladen)
- Fortschrittsbalken: wie viel deines Geldes ist angelegt

### Speichern
- Links: **Depot herunterladen** speichert alles als Datei auf deinem Gerät.
- **Depot laden**: dieselbe Datei wieder hochladen nach einem Reload.

### Tipps
- Starte mit dem Paket **„Ausgewogen“**.
- Das ist **kein** echtes Depot und **keine** Anlageberatung.
        """
    )
    if st.button("Übungs-Depot auf 10.000 € zurücksetzen", type="primary"):
        st.session_state.holdings = {}
        st.session_state.cash = 10000.0
        st.session_state.trades = []
        st.session_state.seen_welcome = False
        st.session_state.price_cache = {}
        st.session_state.pop("prices_warmed", None)
        persist()
        st.success("Zurückgesetzt.")
        st.rerun()
    with st.expander("Kurz-Report speichern"):
        lines = [f"ATLAS Report – {NOW.strftime('%d.%m.%Y %H:%M')}", f"Gesamtwert: {nav:,.2f} €", f"Bargeld: {cash:,.2f} €", ""]
        if has_positions:
            for _, r in pos_df.iterrows():
                lines.append(f"{r['Was']}: {r['Wert jetzt €']:,.2f} € (G/V {r['Gewinn/Verlust €']:+,.2f} €)")
        txt = "\n".join(lines)
        st.download_button("Als Textdatei herunterladen", txt, file_name=f"atlas_{NOW.strftime('%Y%m%d')}.txt")

st.divider()
st.caption("ATLAS Übungs-Depot · Kein echtes Geld · Keine Anlageberatung · Kurse: Yahoo Chart-API + Cache · Depot speicherbar")
