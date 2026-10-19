# ATLAS Analytics · Risk Platform

Streamlit Multi-Asset Risk- und Portfolio-Tool (inspiriert von institutionellen Plattformen wie Aladdin).

## Module

| Modul | Funktion |
|-------|----------|
| **Market Scanner** | Globale Aktien / ETFs / Rohstoffe, Ranking, Score, Charts |
| **Portfolio Builder** | Multi-Book (Core, Satellite, …), Gewichte, Asset-Class-Allokation |
| **Risk Analytics** | Hist. & parametrischer VaR / CVaR, Contribution to Risk, Korrelation, Monte-Carlo |
| **Stress Tests** | 2008, Rate Shock, Tech Sell-off, Inflation, eigene Schocks |
| **Fixed Income** | Duration, approximierte Zins-Sensitivität |
| **FX & Multi-Currency** | Währungs-Exposure über alle Bücher |
| **Trade Journal** | Order-/Trade-Log (Session) |
| **Compliance** | Konzentrations- und Asset-Class-Limits |
| **Reports** | Text- und CSV-Export |
| **News** | RSS (Reuters, BBC, Fed, ECB, SEC) |

## Dateien

- `app.py` – UI und Navigation
- `atlas_config.py` – Universum, Währungen, Asset Classes, Bond-Duration
- `atlas_risk.py` – Datenlayer + Risikomodelle (VaR, CVaR, Stress, Monte-Carlo)
- `requirements.txt`

## Start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Hinweise

- Kursdaten: Yahoo Finance (verzögert).
- VaR, CVaR, Stress und Duration sind **Modellnäherungen**.
- Multi-Book und Trade-Journal nur in der Streamlit-Session (nicht persistent).
- Keine Anlageberatung.
