from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from simulation import download_prices, historical_returns, simulate_portfolio, BLACK_SWAN_EVENTS, classify_asset
from report import build_pdf

APP_DIR = Path(__file__).parent
st.set_page_config(page_title="L & G Consulting | Portfolio Lab", page_icon="◇", layout="wide")

# --- Branding ---
st.markdown("""
<style>
.stApp { background: #f6f7f9; }
section[data-testid="stSidebar"] { background: #101820; }
section[data-testid="stSidebar"] * { color: #f5f5f5; }
section[data-testid="stSidebar"] { background: #101820; }
section[data-testid="stSidebar"] * { color: #f5f5f5; }

section[data-testid="stSidebar"] input {
    color: #101820 !important;
}
.lg-header { background: #101820; color: white; padding: 1.2rem 1.5rem; border-radius: 12px;
  border-bottom: 4px solid #D5B36A; margin-bottom: 1rem; }
.lg-kicker { color: #D5B36A; letter-spacing: .22em; font-size: .76rem; font-weight: 700; }
.metric-note { color: #59636e; font-size: .8rem; }
</style>
""", unsafe_allow_html=True)
col_logo, col_title = st.columns([1, 7])
with col_logo:
    if (APP_DIR / "logo.svg").exists():
        st.image(str(APP_DIR / "logo.svg"), width=90)
with col_title:
    st.markdown('<div class="lg-header"><div class="lg-kicker">L &amp; G CONSULTING</div><h1 style="margin:.2rem 0">Portfolio Monte Carlo Lab</h1><div>Simulazione probabilistica · Stress test · Tail risk</div></div>', unsafe_allow_html=True)

DEFAULT_CONFIG = {
    "schema_version": 1,
    "simulation": {"initial_value": 100000, "years": 10, "simulations": 3000,
        "history_years": 10, "confidence_level": 0.95, "inflation_rate": 0.02,
        "rebalance_frequency": "Annuale", "include_black_swans": True,
        "include_put_hedge": False, "put_annual_cost_pct": 0.75,
        "put_trigger_drawdown_pct": 15.0, "random_seed": 42},
    "portfolios": [
        {"name": "Portafoglio A - Bilanciato", "assets": [
            {"ticker": "VWCE.DE", "weight": 60.0, "asset_class": "Azionario globale"},
            {"ticker": "AGGH.L", "weight": 40.0, "asset_class": "Obbligazionario globale"}]},
        {"name": "Portafoglio B - Prudente", "assets": [
            {"ticker": "VWCE.DE", "weight": 35.0, "asset_class": "Azionario globale"},
            {"ticker": "AGGH.L", "weight": 55.0, "asset_class": "Obbligazionario globale"},
            {"ticker": "SGLD.MI", "weight": 10.0, "asset_class": "Oro"}]},
        {"name": "Portafoglio C - Crescita", "assets": [
            {"ticker": "VWCE.DE", "weight": 80.0, "asset_class": "Azionario globale"},
            {"ticker": "AGGH.L", "weight": 10.0, "asset_class": "Obbligazionario globale"},
            {"ticker": "SGLD.MI", "weight": 10.0, "asset_class": "Oro"}]}
    ]
}

def init_state():
    if "config" not in st.session_state:
        st.session_state.config = json.loads(json.dumps(DEFAULT_CONFIG))
    if "results" not in st.session_state:
        st.session_state.results = None
    if "prices" not in st.session_state:
        st.session_state.prices = None

init_state()
config = st.session_state.config
sim = config["simulation"]

with st.sidebar:
    st.markdown("## Parametri")
    initial_value = st.number_input("Capitale iniziale (€)", min_value=1000.0, max_value=100000000.0,
                                    value=float(sim["initial_value"]), step=5000.0)
    years = st.slider("Orizzonte (anni)", 1, 40, int(sim["years"]))
    n_sims = st.select_slider("Numero simulazioni", options=[500, 1000, 2000, 3000, 5000, 10000],
                              value=int(sim["simulations"]) if int(sim["simulations"]) in [500,1000,2000,3000,5000,10000] else 3000)
    history_years = st.slider("Anni di dati storici", 2, 20, int(sim["history_years"]))
    confidence = st.select_slider("Livello di confidenza VaR", options=[0.90, 0.95, 0.975, 0.99],
                                  value=float(sim["confidence_level"]))
    inflation_pct = st.slider("Inflazione annua ipotizzata (%)", 0.0, 8.0, float(sim["inflation_rate"]) * 100, 0.5)
inflation = inflation_pct / 100    
    rebalance = st.selectbox("Ribilanciamento", ["Mai", "Mensile", "Trimestrale", "Annuale"],
                             index=["Mai", "Mensile", "Trimestrale", "Annuale"].index(sim["rebalance_frequency"])
                             if sim["rebalance_frequency"] in ["Mai", "Mensile", "Trimestrale", "Annuale"] else 3)
    include_swans = st.toggle("Includi shock 'cigno nero'", value=bool(sim["include_black_swans"]))
    include_put = st.toggle("Attiva overlay put parametrico", value=bool(sim["include_put_hedge"]))
    put_cost = st.number_input("Costo annuo ipotetico put (% capitale)", min_value=0.0, max_value=10.0,
                               value=float(sim["put_annual_cost_pct"]), step=0.1, disabled=not include_put)
    put_trigger = st.slider("Trigger put: drawdown (%)", 5, 40, int(sim["put_trigger_drawdown_pct"]),
                            disabled=not include_put)
    seed = st.number_input("Seed casuale", min_value=0, max_value=999999, value=int(sim["random_seed"]), step=1)
    run_btn = st.button("▶ Esegui simulazione", type="primary", use_container_width=True)

st.markdown("### 1. Configurazione dei portafogli")
st.caption("Inserisci ticker Yahoo Finance e pesi percentuali. I pesi di ciascun portafoglio vengono normalizzati durante la simulazione se non sommano esattamente al 100%.")
portfolios = config["portfolios"]
tabs = st.tabs([p.get("name", f"Portafoglio {i+1}") for i, p in enumerate(portfolios[:3])] + ["➕ Gestione"])
for i, p in enumerate(portfolios[:3]):
    with tabs[i]:
        p["name"] = st.text_input("Nome portafoglio", value=p.get("name", f"Portafoglio {i+1}"), key=f"name_{i}")
        df = pd.DataFrame(p["assets"])
        if df.empty:
            df = pd.DataFrame([{"ticker": "VWCE.DE", "weight": 100.0, "asset_class": "Azionario globale"}])
        edited = st.data_editor(df, num_rows="dynamic", use_container_width=True, key=f"assets_{i}",
            column_config={
                "ticker": st.column_config.TextColumn("Ticker Yahoo", required=True),
                "weight": st.column_config.NumberColumn("Peso (%)", min_value=0.0, max_value=100.0, step=1.0, format="%.1f"),
                "asset_class": st.column_config.SelectboxColumn("Classe attivo", options=[
                    "Azionario globale", "Azionario USA", "Azionario sviluppati", "Emergenti",
                    "Obbligazionario globale", "Obbligazionario governativo", "Obbligazionario inflation-linked",
                    "Oro", "Liquidità", "Altro"], required=True)
            })
        p["assets"] = edited.fillna({"ticker": "", "weight": 0.0, "asset_class": "Altro"}).to_dict("records")
        weight_sum = sum(float(a.get("weight", 0) or 0) for a in p["assets"])
        st.caption(f"Somma pesi: **{weight_sum:.1f}%**" + (" · normalizzati in simulazione" if abs(weight_sum-100)>0.05 else ""))
with tabs[3]:
    st.markdown("#### Gestione configurazione")
    if st.button("Aggiungi portafoglio", disabled=len(portfolios) >= 6):
        portfolios.append({"name": f"Portafoglio {len(portfolios)+1}", "assets": [
            {"ticker": "VWCE.DE", "weight": 60.0, "asset_class": "Azionario globale"},
            {"ticker": "AGGH.L", "weight": 40.0, "asset_class": "Obbligazionario globale"}]})
        st.rerun()
    if len(portfolios) > 3:
        remove_name = st.selectbox("Portafoglio da rimuovere", [p["name"] for p in portfolios[3:]])
        if st.button("Rimuovi selezionato"):
            portfolios[:] = [p for p in portfolios if p["name"] != remove_name]
            st.rerun()
    st.markdown("#### Importa / esporta JSON")
    json_bytes = json.dumps(config, indent=2, ensure_ascii=False).encode("utf-8")
    st.download_button("⬇ Esporta configurazione JSON", data=json_bytes, file_name="lg_portfolios_config.json",
                       mime="application/json", use_container_width=True)
    uploaded = st.file_uploader("Importa configurazione JSON", type=["json"])
    if uploaded is not None and st.button("Carica configurazione importata"):
        try:
            imported = json.load(uploaded)
            if not isinstance(imported.get("portfolios"), list) or len(imported["portfolios"]) < 1:
                raise ValueError("Il JSON deve contenere una lista 'portfolios'.")
            if "simulation" not in imported:
                imported["simulation"] = DEFAULT_CONFIG["simulation"]
            st.session_state.config = imported
            st.session_state.results = None
            st.rerun()
        except Exception as exc:
            st.error(f"Configurazione non valida: {exc}")

st.markdown("### 2. Eventi di stress disponibili")
with st.expander("Mostra i sette eventi e le ipotesi di shock", expanded=False):
    event_rows = []
    for name, event in BLACK_SWAN_EVENTS.items():
        event_rows.append({"Evento": name, "Data di riferimento": event["date"],
                           "Azioni": f"{event['equity']:.0%}", "Obbligazioni": f"{event['bonds']:.0%}",
                           "Oro": f"{event['gold']:.0%}", "Altri": f"{event['other']:.0%}"})
    st.dataframe(pd.DataFrame(event_rows), hide_index=True, use_container_width=True)
    st.info("Questi sono shock sintetici per classe di attivo, campionati casualmente nel percorso futuro. Non sono una ricostruzione dei rendimenti giornalieri reali dell'evento.")

# Save sidebar values to the live config on each run.
sim.update({"initial_value": initial_value, "years": years, "simulations": n_sims,
            "history_years": history_years, "confidence_level": confidence,
            "inflation_rate": inflation, "rebalance_frequency": rebalance,
            "include_black_swans": include_swans, "include_put_hedge": include_put,
            "put_annual_cost_pct": put_cost, "put_trigger_drawdown_pct": put_trigger,
            "random_seed": int(seed)})

if run_btn:
    try:
        all_tickers = sorted({str(a.get("ticker", "")).strip().upper()
                              for p in portfolios for a in p.get("assets", [])
                              if str(a.get("ticker", "")).strip()})
        if len(portfolios) < 3:
            st.warning("Sono consigliati almeno tre portafogli per il confronto richiesto.")
        if not all_tickers:
            raise ValueError("Inserire almeno un ticker.")
        with st.spinner("Scarico i prezzi da Yahoo Finance e preparo i rendimenti..."):
            prices = download_prices(all_tickers, period=f"{max(history_years, 2)}y")
            rets = historical_returns(prices, years=history_years)
        st.session_state.prices = prices
        results = {}
        progress = st.progress(0)
        for idx, p in enumerate(portfolios):
            valid_assets = [a for a in p.get("assets", []) if str(a.get("ticker", "")).strip()]
            if not valid_assets:
                st.warning(f"{p['name']}: nessun asset, saltato.")
                continue
            weights = [float(a.get("weight", 0) or 0) for a in valid_assets]
            if sum(weights) <= 0:
                st.warning(f"{p['name']}: somma pesi zero, saltato.")
                continue
            results[p["name"]] = simulate_portfolio(
                returns=rets, assets=valid_assets, weights_pct=weights,
                initial_value=initial_value, years=years, n_sims=n_sims,
                confidence=confidence, inflation_rate=inflation, rebalance_frequency=rebalance,
                include_black_swans=include_swans, include_put_hedge=include_put,
                put_annual_cost_pct=put_cost, put_trigger_drawdown_pct=put_trigger, seed=int(seed)+idx
            )
            progress.progress((idx + 1) / len(portfolios))
        st.session_state.results = results
        st.session_state.run_settings = {"initial_value": initial_value, "years": years, "n_sims": n_sims,
            "confidence": confidence, "inflation_rate": inflation, "include_black_swans": include_swans,
            "include_put_hedge": include_put, "put_annual_cost_pct": put_cost,
            "put_trigger_drawdown_pct": put_trigger, "rebalance_frequency": rebalance,
            "history_years": history_years}
        st.success(f"Simulazione completata: {len(results)} portafogli analizzati.")
    except Exception as exc:
        st.error(f"Simulazione non riuscita: {exc}")
        st.caption("Controlla i ticker Yahoo Finance, la connessione e la disponibilità di dati storici comuni.")

results = st.session_state.results
if results:
    settings = st.session_state.get("run_settings", {})
    st.markdown("---")
    st.markdown("## 3. Risultati e metriche di rischio")
    metric_rows = []
    for name, r in results.items():
        metric_rows.append({"Portafoglio": name, "Valore finale mediano (€)": r["median_terminal"],
            "5° percentile (€)": r["p5_terminal"], "95° percentile (€)": r["p95_terminal"],
            "VaR terminale": r["var"], "CVaR terminale": r["cvar"],
            "Max drawdown mediano": r["max_drawdown_median"],
            "Max drawdown 95° percentile": r["max_drawdown_p95"],
            "Underwater mediano (anni)": r["underwater_days_median"]/252,
            "Valore reale mediano (€)": r["median_real_terminal"]})
    metrics = pd.DataFrame(metric_rows)
    c1,c2,c3,c4 = st.columns(4)
    first = next(iter(results.values()))
    c1.metric("Valore mediano finale", f"€{first['median_terminal']:,.0f}")
    c2.metric("VaR terminale", f"{first['var']:.1%}")
    c3.metric("Max drawdown mediano", f"{first['max_drawdown_median']:.1%}")
    c4.metric("Underwater mediano", f"{first['underwater_days_median']/252:.1f} anni")
    st.caption("Le metriche sintetiche sopra si riferiscono al primo portafoglio. La tabella confronta tutti i portafogli.")
    st.dataframe(metrics.style.format({
        "Valore finale mediano (€)": "€{:,.0f}", "5° percentile (€)": "€{:,.0f}",
        "95° percentile (€)": "€{:,.0f}", "VaR terminale": "{:.1%}", "CVaR terminale": "{:.1%}",
        "Max drawdown mediano": "{:.1%}", "Max drawdown 95° percentile": "{:.1%}",
        "Underwater mediano (anni)": "{:.1f}", "Valore reale mediano (€)": "€{:,.0f}"
    }), use_container_width=True)

    st.markdown("### Fan chart: traiettorie nominali")
    fig_fan = go.Figure()
    for name, r in results.items():
        paths = r["paths"]
        x_years = np.arange(paths.shape[1]) / 252
        qs = np.quantile(paths, [0.05, 0.25, 0.50, 0.75, 0.95], axis=0)
        fig_fan.add_trace(go.Scatter(x=x_years, y=qs[4], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig_fan.add_trace(go.Scatter(x=x_years, y=qs[0], mode="lines", line=dict(width=0), fill="tonexty",
            fillcolor="rgba(120,140,160,0.10)", name=f"{name} · 5–95%"))
        fig_fan.add_trace(go.Scatter(x=x_years, y=qs[2], mode="lines", line=dict(width=2), name=f"{name} · mediana"))
    fig_fan.update_layout(height=520, xaxis_title="Anni", yaxis_title="Valore portafoglio (€)",
                          legend_title="", template="plotly_white")
    st.plotly_chart(fig_fan, use_container_width=True)

    st.markdown("### Distribuzione dei valori finali")
    fig_hist = go.Figure()
    for name, r in results.items():
        fig_hist.add_trace(go.Histogram(x=r["terminal"], name=name, opacity=0.55, nbinsx=55))
    fig_hist.update_layout(barmode="overlay", template="plotly_white", xaxis_title="Valore finale nominale (€)",
                          yaxis_title="Frequenza", height=420)
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("### Rendimento storico e correlazioni")
    prices = st.session_state.prices
    if prices is not None:
        ret_view = prices.pct_change().dropna()
        corr = ret_view.corr()
        col_a, col_b = st.columns([1,1])
        with col_a:
            st.markdown("**Matrice di correlazione storica**")
            st.plotly_chart(px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
                                      zmin=-1, zmax=1), use_container_width=True)
        with col_b:
            st.markdown("**Prezzi normalizzati (base 100)**")
            normalized = prices / prices.iloc[0] * 100
            st.plotly_chart(px.line(normalized, x=normalized.index, y=normalized.columns,
                                    labels={"value":"Indice base 100","variable":"Ticker","x":"Data"},
                                    template="plotly_white"), use_container_width=True)

    st.markdown("### 4. Esporta report")
    figures = {"Fan chart dei portafogli": fig_fan, "Distribuzione dei valori finali": fig_hist}
    try:
        pdf_bytes = build_pdf(results, settings, config, figures)
        st.download_button("⬇ Scarica report PDF L & G Consulting", data=pdf_bytes,
                           file_name="LG_Consulting_Monte_Carlo_Report.pdf", mime="application/pdf",
                           type="primary")
    except Exception as exc:
        st.warning(f"PDF non generato: {exc}. Le tabelle e i grafici restano disponibili nell'app.")
    st.download_button("⬇ Esporta risultati CSV", data=metrics.to_csv(index=False).encode("utf-8"),
                       file_name="LG_Consulting_Monte_Carlo_Metrics.csv", mime="text/csv")
else:
    st.markdown("### 3. Esegui una prima simulazione")
    st.info("Configura i portafogli e premi **Esegui simulazione** nella barra laterale. La prima esecuzione scarica i prezzi da Yahoo Finance.")
    st.markdown("**Nota:** i ticker predefiniti sono esempi da verificare su Yahoo Finance. Le disponibilità e le valute possono differire.")

st.markdown("---")
st.caption("L & G Consulting · Strumento quantitativo sperimentale. I risultati dipendono dalle ipotesi e dai dati storici; non costituiscono consulenza finanziaria.")
