# ATLAS Analytics

Ein kleines Dashboard zur Analyse echter historischer Börsendaten.

## Enthaltene Funktionen
- Historische tägliche Kursdaten über Alpha Vantage
- Kursdiagramm mit gleitenden Durchschnitten (SMA 20/50)
- RSI (14 Tage)
- Annualisierte historische Volatilität
- Maximaler Drawdown
- CSV-Export

## Starten (Windows / macOS / Linux)
1. Python 3.10 oder neuer installieren.
2. Terminal im Projektordner öffnen.
3. Abhängigkeiten installieren:
   `python -m pip install -r requirements.txt`
4. Dashboard starten:
   `streamlit run app.py`
5. Browser-Adresse öffnen, die im Terminal angezeigt wird.

## API-Key
Im Dashboard links den Alpha-Vantage-Key eintragen. Du kannst alternativ vor dem Start eine Umgebungsvariable `ALPHA_VANTAGE_API_KEY` setzen.

Offizielle Seite:
https://www.alphavantage.co/support/#api-key

## Hinweise
- Das Programm nutzt tägliche historische Daten; es ist kein garantiertes Echtzeit-Feed.
- Gratis-Tarife haben Nutzungsgrenzen. Prüfe Anbieterbedingungen, Symbolabdeckung und Datenlizenz.
- API-Key nicht veröffentlichen oder in Dateien einchecken.
- Kennzahlen sind historische Messungen und keine Vorhersagen oder Anlageberatung.
