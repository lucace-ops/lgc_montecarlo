from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from simulation import download_prices,historical_returns,simulate_portfolio,BLACK_SWAN_EVENTS
from report import build_pdf

APP_DIR=Path(__file__).parent
st.set_page_config(page_title="L & G Consulting | Portfolio Lab",page_icon="◇",layout="wide")
st.markdown('''<style>.stApp{background:#f6f7f9}section[data-testid="stSidebar"]{background:#101820}section[data-testid="stSidebar"] *{color:#f5f5f5}section[data-testid="stSidebar"] input{color:#101820!important}.lg-header{background:#101820;color:white;padding:1.2rem 1.5rem;border-radius:12px;border-bottom:4px solid #D5B36A;margin-bottom:1rem}.lg-kicker{color:#D5B36A;letter-spacing:.22em;font-size:.76rem;font-weight:700}</style>''',unsafe_allow_html=True)
c0,c1=st.columns([1,7])
with c0:
    if (APP_DIR/"logo.svg").exists(): st.image(str(APP_DIR/"logo.svg"),width=90)
with c1: st.markdown('<div class="lg-header"><div class="lg-kicker">L &amp; G CONSULTING</div><h1 style="margin:.2rem 0">Portfolio Monte Carlo Lab</h1><div>Simulazione probabilistica · PAC · Stress test · Tail risk · Convexity</div></div>',unsafe_allow_html=True)

DEFAULT={"simulation":{"initial_value":100000,"years":10,"simulations":3000,"history_years":10,"inflation_rate":.02},"portfolios":[{"name":"Portafoglio A","assets":[{"ticker":"VWCE.DE","weight":60.,"asset_class":"Azionario globale"},{"ticker":"AGGH.L","weight":40.,"asset_class":"Obbligazionario globale"}]}]}
if "config" not in st.session_state: st.session_state.config=json.loads(json.dumps(DEFAULT))
if "results" not in st.session_state: st.session_state.results=None
if "prices" not in st.session_state: st.session_state.prices=None
config=st.session_state.config; sim=config.setdefault("simulation",{})

with st.sidebar:
    st.markdown("## Parametri base")
    initial=st.number_input("Capitale iniziale (€)",1000.,100000000.,float(sim.get("initial_value",100000)),5000.)
    contribution=st.number_input("Versamento PAC (€)",0.,1000000.,float(sim.get("contribution",800)),50.)
    contribution_frequency=st.selectbox("Frequenza PAC",["Mensile","Annuale"],index=0 if sim.get("contribution_frequency","Mensile")=="Mensile" else 1)
    years=st.slider("Orizzonte (anni)",1,40,int(sim.get("years",10)))
    n_sims=st.select_slider("Numero simulazioni",options=[500,1000,2000,3000,5000,10000],value=int(sim.get("simulations",3000)) if int(sim.get("simulations",3000)) in [500,1000,2000,3000,5000,10000] else 3000)
    history_years=st.slider("Anni di dati storici",2,20,int(sim.get("history_years",10)))
    inflation=st.slider("Inflazione annua ipotizzata (%)",0.,8.,float(sim.get("inflation_rate",.02))*100,.5)/100
    with st.expander("⚙️ Impostazioni avanzate",expanded=False):
        method=st.selectbox("Motore Monte Carlo",["Bootstrap storico","Approccio parametrico"])
        distribution="Student-t multivariata"; dof=6; block=3
        if method=="Bootstrap storico": block=st.slider("Lunghezza blocco bootstrap (mesi)",1,12,3)
        else:
            distribution=st.selectbox("Distribuzione",["Student-t multivariata","Normale multivariata"])
            if distribution=="Student-t multivariata": dof=st.slider("Gradi di libertà Student-t",3,30,6)
        risk_free=st.number_input("Tasso privo di rischio (%)",-5.,30.,2.5,.25)/100
        ruin=st.slider("Soglia di rovina (% del totale versato)",1,100,50)/100
        include_put=st.toggle("Copertura put teorica",False)
        moneyness=.20; maturity=12; iv=.25; hedge_fraction=1.
        if include_put:
            moneyness=st.selectbox("Strike OTM",[.10,.20,.30],index=1,format_func=lambda x:f"{x:.0%} OTM")
            maturity=st.selectbox("Scadenza put (mesi)",[3,6,12],index=2)
            iv=st.slider("Volatilità implicita (%)",5,120,25)/100
            hedge_fraction=st.slider("Quota coperta (%)",10,100,100)/100
        custom_enabled=st.toggle("Shock personalizzato",False)
        custom=None
        if custom_enabled:
            sm=st.slider("Mese di inizio shock",1,max(1,years*12),min(12,years*12)); sd=st.slider("Durata shock (mesi)",1,min(60,years*12),min(6,years*12)); si=st.slider("Impatto cumulato shock (%)",-95,100,-30)/100; sv=st.slider("Moltiplicatore volatilità",1.,5.,2.,.25); custom={"start_month":sm,"duration":sd,"impact":si,"vol_multiplier":sv}
    seed=st.number_input("Seed casuale",0,999999,42,1)
    run=st.button("▶ Esegui simulazione",type="primary",use_container_width=True)

st.markdown("### 1. Portafogli")
portfolios=config.setdefault("portfolios",DEFAULT["portfolios"])
tabs=st.tabs([p.get("name",f"Portafoglio {i+1}") for i,p in enumerate(portfolios)]+["➕ Gestione"])
classes=["Azionario globale","Azionario USA","Azionario sviluppati","Emergenti","Obbligazionario globale","Obbligazionario governativo","Obbligazionario inflation-linked","Oro","Liquidità","Altro"]
for i,p in enumerate(portfolios):
    with tabs[i]:
        p["name"]=st.text_input("Nome portafoglio",p.get("name",f"Portafoglio {i+1}"),key=f"n{i}")
        df=pd.DataFrame(p.get("assets",[]))
        if df.empty: df=pd.DataFrame([{"ticker":"VWCE.DE","weight":100.,"asset_class":"Azionario globale"}])
        ed=st.data_editor(df,num_rows="dynamic",width="stretch",key=f"a{i}",column_config={"ticker":st.column_config.TextColumn("Ticker Yahoo",required=True),"weight":st.column_config.NumberColumn("Peso (%)",min_value=0.,max_value=100.,step=1.,format="%.1f"),"asset_class":st.column_config.SelectboxColumn("Classe attivo",options=classes,required=True)})
        p["assets"]=ed.fillna({"ticker":"","weight":0.,"asset_class":"Altro"}).to_dict("records")
        if p["assets"]:
            ridx=st.selectbox("Ticker da eliminare",range(len(p["assets"])),format_func=lambda j:str(p["assets"][j].get("ticker","")).strip() or f"Riga {j+1}",key=f"r{i}")
            if st.button("🗑 Elimina ticker selezionato",key=f"rb{i}",use_container_width=True): p["assets"].pop(ridx); st.session_state.results=None; st.rerun()
        ws=sum(float(a.get("weight",0) or 0) for a in p["assets"]); st.caption(f"Somma pesi: **{ws:.1f}%**"+(" · normalizzati in simulazione" if abs(ws-100)>.05 else ""))
with tabs[len(portfolios)]:
    if st.button("➕ Aggiungi portafoglio",disabled=len(portfolios)>=6,use_container_width=True): portfolios.append({"name":f"Portafoglio {len(portfolios)+1}","assets":[{"ticker":"VWCE.DE","weight":60.,"asset_class":"Azionario globale"},{"ticker":"AGGH.L","weight":40.,"asset_class":"Obbligazionario globale"}]}); st.rerun()
    if len(portfolios)>1:
        pi=st.selectbox("Portafoglio da eliminare",range(len(portfolios)),format_func=lambda j:portfolios[j]["name"])
        if st.button("🗑 Elimina portafoglio",use_container_width=True): portfolios.pop(pi); st.session_state.results=None; st.rerun()
    st.caption(f"Portafogli attivi: {len(portfolios)} / 6")
    blob=json.dumps(config,indent=2,ensure_ascii=False).encode(); st.download_button("⬇ Esporta configurazione JSON",blob,"lg_portfolios_config.json","application/json",use_container_width=True)
    up=st.file_uploader("Importa configurazione JSON",type=["json"])
    if up is not None and st.button("Carica configurazione"):
        try:
            imp=json.load(up)
            if not isinstance(imp.get("portfolios"),list) or not imp["portfolios"]: raise ValueError("Serve almeno un portafoglio")
            if len(imp["portfolios"])>6: raise ValueError("Massimo 6 portafogli")
            st.session_state.config=imp; st.session_state.results=None; st.rerun()
        except Exception as e: st.error(str(e))

sim.update({"initial_value":initial,"contribution":contribution,"contribution_frequency":contribution_frequency,"years":years,"simulations":n_sims,"history_years":history_years,"inflation_rate":inflation})

st.markdown("### 2. Metodologia")
st.info("Il motore può usare bootstrap storico a blocchi oppure un modello parametrico multivariato normale/Student-t. Il PAC viene inserito nel percorso; la copertura put è prezzata teoricamente con Black–Scholes. Gli shock personalizzati modificano impatto e volatilità in una finestra scelta.")

if run:
    try:
        tickers=sorted({str(a.get("ticker","")).strip().upper() for p in portfolios for a in p.get("assets",[]) if str(a.get("ticker","")).strip()})
        if not tickers: raise ValueError("Inserire almeno un ticker.")
        with st.spinner("Scarico i dati e simulo i percorsi..."):
            prices=download_prices(tickers,period=f"{max(history_years,2)}y"); rets=historical_returns(prices,history_years); st.session_state.prices=prices; results={}; bar=st.progress(0)
            for idx,p in enumerate(portfolios):
                assets=[a for a in p.get("assets",[]) if str(a.get("ticker","")).strip()]; weights=[float(a.get("weight",0) or 0) for a in assets]
                if not assets or sum(weights)<=0: continue
                results[p["name"]]=simulate_portfolio(rets,assets,weights,initial,years,n_sims,.95,inflation,"Annuale",False,include_put,0,15,int(seed)+idx,contribution,contribution_frequency,method,distribution,dof,block,risk_free,ruin,moneyness,maturity,iv,hedge_fraction,custom)
                bar.progress((idx+1)/len(portfolios))
        st.session_state.results=results; st.success(f"Simulazione completata: {len(results)} portafogli.")
    except Exception as e: st.error(f"Simulazione non riuscita: {e}")

results=st.session_state.results
if results:
    st.markdown("---"); st.markdown("## 3. Risultati")
    rows=[]
    for name,r in results.items(): rows.append({"Portafoglio":name,"Totale versato (€)":r["total_invested"],"Mediana finale (€)":r["median_terminal"],"5° percentile (€)":r["p5_terminal"],"95° percentile (€)":r["p95_terminal"],"Valore reale mediano (€)":r["median_real_terminal"],"Prob. perdita":r["probability_loss"],"Prob. rovina":r["probability_ruin"],"VaR 95% mensile":r["var95_monthly"],"CVaR 95% mensile":r["cvar95_monthly"],"VaR 99% mensile":r["var99_monthly"],"CVaR 99% mensile":r["cvar99_monthly"],"Sharpe":r["sharpe"],"Sortino":r["sortino"],"Max DD mediano":r["max_drawdown_median"]})
    metrics=pd.DataFrame(rows); st.dataframe(metrics.style.format({"Totale versato (€)":"€{:,.0f}","Mediana finale (€)":"€{:,.0f}","5° percentile (€)":"€{:,.0f}","95° percentile (€)":"€{:,.0f}","Valore reale mediano (€)":"€{:,.0f}","Prob. perdita":"{:.1%}","Prob. rovina":"{:.1%}","VaR 95% mensile":"{:.1%}","CVaR 95% mensile":"{:.1%}","VaR 99% mensile":"{:.1%}","CVaR 99% mensile":"{:.1%}","Sharpe":"{:.2f}","Sortino":"{:.2f}","Max DD mediano":"{:.1%}"}),width="stretch")
    st.markdown("### Fan chart")
    fig=go.Figure()
    for name,r in results.items():
        paths=r["paths"]; x=np.arange(paths.shape[1])/12; q=np.quantile(paths,[.05,.5,.95],axis=0); fig.add_trace(go.Scatter(x=x,y=q[2],mode="lines",line=dict(width=0),showlegend=False)); fig.add_trace(go.Scatter(x=x,y=q[0],mode="lines",line=dict(width=0),fill="tonexty",name=f"{name} · 5–95%")); fig.add_trace(go.Scatter(x=x,y=q[1],mode="lines",name=f"{name} · mediana"))
    fig.update_layout(template="plotly_white",height=520,xaxis_title="Anni",yaxis_title="Valore (€)"); st.plotly_chart(fig,width="stretch")
    st.markdown("### Distribuzione valori finali"); hist=go.Figure()
    for name,r in results.items(): hist.add_trace(go.Histogram(x=r["terminal"],name=name,opacity=.55,nbinsx=55))
    hist.update_layout(barmode="overlay",template="plotly_white",height=420); st.plotly_chart(hist,width="stretch")
    if include_put:
        st.markdown("### Copertura put teorica")
        for name,r in results.items(): st.write(f"**{name}** — premio medio cumulato: €{r['hedging']['premium_paid']:,.0f} · payoff medio ricevuto: €{r['hedging']['payoff_received']:,.0f}")
    st.download_button("⬇ Esporta risultati CSV",metrics.to_csv(index=False).encode(),"LG_Consulting_Monte_Carlo_Metrics.csv","text/csv")
else:
    st.markdown("### 3. Esegui una simulazione"); st.info("Configura portafogli e parametri, poi premi **Esegui simulazione**.")
st.markdown("---"); st.caption("L & G Consulting · Strumento quantitativo sperimentale. I risultati dipendono da ipotesi e dati storici e non costituiscono consulenza finanziaria.")