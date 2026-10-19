# ATLAS Analytics · Risk Platform

Streamlit-basiertes Multi-Asset Risk- und Portfolio-Tool (inspiriert von institutionellen Plattformen wie Aladdin).

## Module

| Modul | Funktion |
|-------|----------|
| **Market Scanner** | Kuratierte globale Aktien/ETFs/Rohstoffe, Ranking, Score, Charts |
| **Portfolio Builder** | Multi-Book (Core, Satellite, …), Gewichte, Asset-Class-Allokation |
| **Risk Analytics** | Historischer & parametrischer VaR/CVaR, Contribution to Risk, Korrelation, Monte-Carlo |
| **Stress Tests** | Vordefinierte Szenarien (2008, Rate Shock, Tech Sell-off, …) + eigene Schocks |
| **Fixed Income** | Duration, approximierte Zins-Sensitivität, Bond-ETF-Übersicht |
| **FX & Multi-Currency** | Währungs-Exposure über alle Bücher |
| **Trade Journal** | Einfaches Order-/Trade-Log (Session) |
| **Compliance** | Konzentrations- und Asset-Class-Limits |
| **Reports** | Text- und CSV-Export |
| **News** | RSS (Reuters, BBC, Fed, ECB, SEC) |

## Lokal starten

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Hinweise

- Kursdaten: Yahoo Finance (verzögert, nicht garantiert vollständig).
- VaR, CVaR, Stress-Tests und Duration sind **Modellnäherungen**, keine Vorhersagen.
- Multi-Book und Trade-Journal leben nur in der Streamlit-Session (nicht persistent).
- Keine Anlageberatung.
