# L & G Consulting — Portfolio Monte Carlo Lab

Applicazione Streamlit per confrontare portafogli, simulare scenari Monte Carlo, analizzare stress test e scaricare report PDF.

## Contenuto

- `app.py`: interfaccia Streamlit e flusso principale.
- `simulation.py`: download prezzi Yahoo Finance, rendimenti, bootstrap e metriche.
- `report.py`: generazione report PDF.
- `logo.svg`: logo vettoriale L & G Consulting.
- `sample_config.json`: esempio importabile di configurazione.
- `requirements.txt`: dipendenze Python.

## Avvio locale

Richiede Python 3.10 o successivo.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Streamlit mostrerà un URL locale nel terminale.

## Pubblicazione su Streamlit Community Cloud

1. Crea un repository GitHub e carica tutti i file del progetto.
2. Apri Streamlit Community Cloud e collega il repository.
3. Seleziona `app.py` come file principale.
4. Avvia il deploy.
5. Quando GitHub viene aggiornato, Streamlit può ricostruire l'app dal repository.

## Pubblicazione su Replit

1. Crea un nuovo progetto Python su Replit.
2. Carica tutti i file mantenendo la struttura delle cartelle.
3. Installa le dipendenze da `requirements.txt`.
4. Avvia con `streamlit run app.py --server.address 0.0.0.0 --server.port 3000`.
5. Configura la porta pubblica secondo le impostazioni correnti di Replit.

Le interfacce dei provider possono cambiare; i nomi esatti dei pulsanti possono quindi differire.

## Avvertenze metodologiche importanti

- `yfinance` usa dati Yahoo Finance per finalità di ricerca/analisi; non è un feed istituzionale né una garanzia di disponibilità.
- I ticker iniziali sono esempi: verificare ticker, borsa, valuta, data di lancio e disponibilità su Yahoo Finance.
- Il motore ricampiona righe giornaliere sincronizzate per mantenere la dipendenza storica tra gli asset. Non è un modello predittivo e può sottostimare rischi fuori dal campione.
- Gli eventi di stress sono shock sintetici applicati casualmente per classe di attivo. Non costituiscono ricostruzioni dei rendimenti reali dei singoli ETF alle date indicate.
- Il VaR/CVaR sono misurati sulla distribuzione dei rendimenti terminali dell'orizzonte, non sulle perdite giornaliere.
- La copertura put è un overlay didattico con costo e payout parametrizzati; non usa una catena di opzioni reali, strike, scadenze o volatilità implicite. Non va interpretata come backtest di una strategia negoziabile.
- L'indicatore underwater è espresso in giorni di mercato simulati e in anni convenzionali da 252 sedute.
- Prima di utilizzare l'app per decisioni reali, validare il codice, le fonti, le ipotesi e i risultati con test indipendenti.

## Miglioramenti consigliati prima dell'uso professionale

1. Aggiungere test automatici unitari e casi di riferimento.
2. Implementare block bootstrap o modelli multivariati con volatilità condizionale.
3. Distinguere VaR/CVaR a un giorno, a un anno e sull'intero orizzonte.
4. Migliorare il modello di ribilanciamento con commissioni, spread, imposte e flussi.
5. Per le put, integrare prezzi storici di opzioni e regole esplicite di strike/scadenza/rollover.
6. Aggiungere autenticazione e limiti di utilizzo se l'app non deve essere pubblica a tutti.
