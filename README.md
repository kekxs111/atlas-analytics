# ATLAS Analytics

Streamlit-Dashboard für historische Kursdaten, Risiko-Kennzahlen, Monte-Carlo-Szenarien, News-RSS, einen heuristischen KI-Blasencheck und Beispiel-Portfolios.

## Lokal starten

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Hinweise
- Die Kursdaten werden über Yahoo Finance / `yfinance` abgerufen. Die Daten sind nicht garantiert echtzeitfähig; der letzte Abrufzeitpunkt und die letzte beobachtete Kurszeile werden getrennt angezeigt.
- Das Symboluniversum ist kuratiert und nicht die vollständige globale Börse.
- Monte-Carlo-Ergebnisse hängen von Rendite- und Volatilitätsannahmen ab und sind keine Vorhersagegarantie.
- Futures sind keine direkten Händlerpreise für physische Edelmetallbarren.
- Die Musterportfolios sind illustrative Beispiele, keine Anlageberatung.
