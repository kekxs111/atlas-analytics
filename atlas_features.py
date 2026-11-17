"""ATLAS Hilfsfunktionen – Ziele, Streuung, Report, Backtest."""
from __future__ import annotations
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from atlas_config import ALL_UNIVERSE, ASSET_CLASS
from atlas_risk import fetch_history, single_metrics

TZ = ZoneInfo("Europe/Berlin")

TIPS = [
    "Speichere dein Depot links unter \u201eDepot herunterladen\u201c, damit nichts verloren geht.",
    "Lege nicht alles in eine einzelne Aktie \u2013 streue besser mit einem Paket (ETF).",
    "Anlegen ist langfristig gedacht. Kurzfristige Schwankungen sind normal.",
    "Nur Geld anlegen, das du nicht sofort brauchst.",
    "Vergleiche ruhig zwei Pakete, bevor du kaufst \u2013 unter \u201eVergleich\u201c.",
]

STRESS_SCENARIOS = {
    "B\u00f6rse f\u00e4llt stark (\u221220 %)": {"Equity": -0.20, "Equity ETF": -0.20, "Fixed Income": 0.02, "Commodity": -0.05, "Real Estate": -0.15},
    "B\u00f6rsencrash (\u221235 %)": {"Equity": -0.35, "Equity ETF": -0.33, "Fixed Income": 0.05, "Commodity": -0.15, "Real Estate": -0.30},
    "Zinsen steigen (Anleihen fallen)": {"Equity": -0.05, "Equity ETF": -0.05, "Fixed Income": -0.08, "Commodity": 0.0, "Real Estate": -0.10},
    "Gold steigt stark (+15 %)": {"Equity": 0.0, "Equity ETF": 0.0, "Fixed Income": 0.0, "Commodity": 0.15, "Real Estate": 0.0},
    "Ruhiger Bullenmarkt (+15 % Aktien)": {"Equity": 0.15, "Equity ETF": 0.15, "Fixed Income": 0.01, "Commodity": 0.02, "Real Estate": 0.08},
}


def diversifikation_status(pos_df: pd.DataFrame, nav: float, product_by_t: dict) -> tuple[str, str]:
    if pos_df is None or pos_df.empty or nav <= 0:
        return "\u26aa noch nichts", "Lege zuerst etwas an \u2013 am besten ein fertiges Paket."
    max_share = float(pos_df["Anteil %"].max()) if "Anteil %" in pos_df.columns else 100.0
    n = len(pos_df)
    if max_share >= 80:
        return "\U0001f534 zu einseitig", "Mehr als 80 % stecken in einer einzigen Sache. Besser streuen."
    if max_share >= 50 and n <= 2:
        return "\U0001f7e1 etwas einseitig", "Ein gro\u00dfer Brocken dominiert. Ein Welt-Paket hilft beim Streuen."
    if n >= 3 and max_share < 50:
        return "\U0001f7e2 gut gestreut", "Dein Geld ist auf mehrere Bausteine verteilt \u2013 das ist sinnvoll."
    return "\U0001f7e1 okay", "Geht in Ordnung. Mit einem fertigen Paket wird es noch einfacher."


def monthly_needed(target: float, years: float, rate_pct: float, start: float) -> float:
    if years <= 0:
        return max(0.0, target - start)
    r = rate_pct / 100.0 / 12.0
    n = int(years * 12)
    fv_start = start * ((1 + rate_pct / 100.0) ** years)
    need = target - fv_start
    if need <= 0:
        return 0.0
    if r <= 0:
        return need / n
    return need * r / ((1 + r) ** n - 1)


def package_backtest(packages: dict, pkg_name: str, euro: float = 10000.0, years: int = 5) -> dict:
    parts = packages[pkg_name]["parts"]
    end_vals, vols = [], []
    for t, w in parts:
        h = fetch_history(t, "5y")
        if h.empty or len(h) < 30:
            continue
        p = h["Close"].astype(float)
        total = p.iloc[-1] / p.iloc[0] - 1
        r = p.pct_change().dropna()
        vol = float(r.std(ddof=1) * np.sqrt(252)) if len(r) > 5 else float("nan")
        end_vals.append(euro * w * (1 + total))
        if vol == vol:
            vols.append(vol * w)
    if not end_vals:
        return {"end": float("nan"), "vol": float("nan"), "ok": False}
    return {"end": sum(end_vals), "vol": sum(vols) if vols else float("nan"), "ok": True}


def build_report_text(nav, cash, invested, upnl, pos_df, div_label, div_text, now=None) -> str:
    now = now or datetime.now(TZ)
    lines = [
        "ATLAS \u00dcbungs-Report",
        f"Stand: {now.strftime('%d.%m.%Y %H:%M')}",
        "",
        f"Gesamtwert:     {nav:,.2f} \u20ac",
        f"Bargeld:        {cash:,.2f} \u20ac",
        f"Investiert:     {invested:,.2f} \u20ac",
        f"Gewinn/Verlust: {upnl:+,.2f} \u20ac",
        f"Streuung:       {div_label} \u2013 {div_text}",
        "",
        "Positionen:",
    ]
    if pos_df is not None and not pos_df.empty:
        for _, r in pos_df.iterrows():
            lines.append(
                f"  - {r['Was']}: {r['Wert jetzt \u20ac']:,.2f} \u20ac "
                f"(Anteil {r['Anteil %']:.0f} %, G/V {r['Gewinn/Verlust \u20ac']:+,.2f} \u20ac)"
            )
    else:
        lines.append("  (noch keine Positionen)")
    lines += ["", "Hinweis: \u00dcbungs-Depot, kein echtes Geld, keine Anlageberatung."]
    return "\n".join(lines)
