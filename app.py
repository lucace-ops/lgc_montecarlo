from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from market_data import load_monthly_eur_returns, load_monthly_eur_return_series, MarketDataError
from proxy_suggestions import suggested_proxies, normalize_proxy_ticker
from simulation import simulate_portfolio, HISTORICAL_SHOCKS
from report import build_pdf

APP_DIR=Path(__file__).parent
st.set_page_config(page_title="L & G Consulting | Portfolio Lab",page_icon="◇",layout="wide")
st.markdown('''<style>.stApp{background:#f6f7f9}section[data-testid="stSidebar"]{background:#101820}section[data-testid="stSidebar"] *{color:#f5f5f5}section[data-testid="stSidebar"] input{color:#101820!important}.lg-header{background:#101820;color:white;padding:1.2rem 1.5rem;border-radius:12px;border-bottom:4px solid #D5B36A;margin-bottom:1rem}.lg-kicker{color:#D5B36A;letter-spacing:.22em;font-size:.76rem;font-weight:700}</style>''',unsafe_allow_html=True)

c0,c1=st.columns([1,7])
with c0:
    if (APP_DIR/"logo.svg").exists(): st.image(str(APP_DIR/"logo.svg"),width=90)
with c1: st.markdown('<div class="lg-header"><div class="lg-kicker">L &amp; G CONSULTING</div><h1 style="margin:.2rem 0">Portfolio Monte Carlo Lab</h1><div>Monte Carlo · PAC · Fiscalità · Ribilanciamento · Stress · Convexity</div></div>',unsafe_allow_html=True)

DEFAULT={"simulation":{"initial_value":100000,"years":10,"simulations":3000,"history_years":10,"inflation_rate":.02,"contribution":800,"contribution_frequency":"Mensile"},"portfolios":[{"name":"Portafoglio A","assets":[{"ticker":"VWCE.DE","weight":60.,"asset_class":"Azionario globale","ter":.22,"tax_rate":.26},{"ticker":"AGGH.L","weight":40.,"asset_class":"Obbligazionario globale","ter":.10,"tax_rate":.26}]}],"historical_proxies":{}}
if "config" not in st.session_state: st.session_state.config=json.loads(json.dumps(DEFAULT))
if "results" not in st.session_state: st.session_state.results=None
if "run_settings" not in st.session_state: st.session_state.run_settings={}
config=st.session_state.config; sim=config.setdefault("simulation",{})

@st.cache_data(ttl=3600,show_spinner=False)
def cached_common_history(tickers:tuple[str,...],start:str): return load_monthly_eur_returns(list(tickers),start=start)
@st.cache_data(ttl=3600,show_spinner=False)
def cached_series(ticker:str): return load_monthly_eur_return_series(ticker,start="1970-01-01")

with st.sidebar:
    st.markdown("## Modalità")
    mode=st.radio("Livello di controllo",["Basic","Avanzata"],horizontal=True,label_visibility="collapsed")
    advanced=mode=="Avanzata"
    st.caption("Basic: pochi parametri essenziali. Avanzata: costi, fiscalità, ribilanciamento, stress e coperture.")
    st.markdown("## Parametri base")
    initial=st.number_input("Capitale iniziale (€)",100.,100000000.,float(sim.get("initial_value",100000)),100.)
    contribution=st.number_input("Versamento PAC (€)",0.,1000000.,float(sim.get("contribution",800)),50.)
    years=st.slider("Orizzonte (anni)",1,40,int(sim.get("years",10)))
    n_sims=st.select_slider("Numero simulazioni",options=[500,1000,2000,3000,5000,10000],value=int(sim.get("simulations",3000)) if int(sim.get("simulations",3000)) in [500,1000,2000,3000,5000,10000] else 3000)
    inflation=st.slider("Inflazione annua ipotizzata (%)",0.,8.,float(sim.get("inflation_rate",.02))*100,.5)/100

    contribution_frequency="Mensile"; contribution_growth=0.; contribution_stop=None; extraordinary_month=None; extraordinary_amount=0.
    history_years=int(sim.get("history_years",10)); method="Bootstrap storico"; distribution="Student-t multivariata"; dof=6; block=3; risk_free=.025; ruin=.5; seed=42
    rebalance_mode="Annuale"; rebalance_band=.05; contributions_first=True
    taxation_enabled=True; fee_fixed=0.; fee_pct=0.; platform_fee=0.
    include_put=False; moneyness=.20; maturity=12; iv=.25; hedge_fraction=1.; custom=None

    if advanced:
        with st.expander("💶 PAC dinamico",expanded=False):
            contribution_frequency=st.selectbox("Frequenza PAC",["Mensile","Annuale"],index=0 if sim.get("contribution_frequency","Mensile")=="Mensile" else 1)
            contribution_growth=st.slider("Crescita annua PAC (%)",-10.,20.,0.,.5)/100
            if st.toggle("Interrompi PAC dopo un certo numero di anni",False): contribution_stop=st.slider("Anni di versamenti",1,years,max(1,years//2))
            if st.toggle("Versamento straordinario",False):
                extraordinary_month=st.slider("Mese del versamento straordinario",1,years*12,min(12,years*12)); extraordinary_amount=st.number_input("Importo straordinario (€)",0.,10000000.,10000.,500.)
        with st.expander("🔄 Ribilanciamento",expanded=False):
            rebalance_mode=st.selectbox("Strategia",["Mai","Semestrale","Annuale","Bande"],index=2)
            if rebalance_mode=="Bande": rebalance_band=st.slider("Soglia deviazione (%)",1.,25.,5.,.5)/100
            contributions_first=st.toggle("Usa prima i nuovi versamenti per correggere i pesi",True)
        with st.expander("🇮🇹 Fiscalità e costi",expanded=False):
            taxation_enabled=st.toggle("Applica fiscalità italiana semplificata",True)
            st.caption("26% ordinario; 12,5% per titoli di Stato/white list quando impostato sul singolo asset. Modello indicativo, non fiscale-certificativo.")
            fee_fixed=st.number_input("Commissione fissa per operazione (€)",0.,100.,0.,.5)
            fee_pct=st.number_input("Commissione variabile per operazione (%)",0.,5.,0.,.01)
            platform_fee=st.number_input("Costo piattaforma annuo (%)",0.,5.,0.,.05)
        with st.expander("🧮 Motore Monte Carlo",expanded=False):
            history_years=st.slider("Anni per calibrare il Monte Carlo",2,20,int(sim.get("history_years",10)))
            method=st.selectbox("Motore",["Bootstrap storico","Approccio parametrico"])
            if method=="Bootstrap storico": block=st.slider("Lunghezza blocco (mesi)",1,12,3)
            else:
                distribution=st.selectbox("Distribuzione",["Student-t multivariata","Normale multivariata"])
                if distribution=="Student-t multivariata": dof=st.slider("Gradi di libertà Student-t",3,30,6)
            risk_free=st.number_input("Tasso privo di rischio (%)",-5.,30.,2.5,.25)/100
            ruin=st.slider("Soglia di rovina (% del totale versato)",1,100,50)/100
            seed=st.number_input("Seed casuale",0,999999,42,1)
        with st.expander("🛡️ Copertura e shock personalizzato",expanded=False):
            include_put=st.toggle("Copertura put teorica",False)
            if include_put:
                moneyness=st.selectbox("Strike OTM",[.10,.20,.30],index=1,format_func=lambda x:f"{x:.0%} OTM"); maturity=st.selectbox("Scadenza put (mesi)",[3,6,12],index=2); iv=st.slider("Volatilità implicita (%)",5,120,25)/100; hedge_fraction=st.slider("Quota coperta (%)",10,100,100)/100
            if st.toggle("Shock personalizzato",False):
                sm=st.slider("Mese inizio shock",1,max(1,years*12),min(12,years*12)); sd=st.slider("Durata shock (mesi)",1,min(60,years*12),min(6,years*12)); si=st.slider("Impatto cumulato (%)",-95,100,-30)/100; sv=st.slider("Moltiplicatore volatilità",1.,5.,2.,.25); custom={"start_month":sm,"duration":sd,"impact":si,"vol_multiplier":sv}
    run=st.button("▶ Esegui simulazione",type="primary",use_container_width=True)

st.markdown("### 1. Portafogli")
portfolios=config.setdefault("portfolios",DEFAULT["portfolios"])
tabs=st.tabs([p.get("name",f"Portafoglio {i+1}") for i,p in enumerate(portfolios)]+["➕ Gestione"])
classes=["Azionario globale","Azionario USA","Azionario sviluppati","Emergenti","Obbligazionario globale","Obbligazionario governativo","Obbligazionario inflation-linked","Oro","Liquidità","Altro"]
for i,p in enumerate(portfolios):
    with tabs[i]:
        p["name"]=st.text_input("Nome portafoglio",p.get("name",f"Portafoglio {i+1}"),key=f"n{i}")
        df=pd.DataFrame(p.get("assets",[]))
        if df.empty: df=pd.DataFrame([{"ticker":"VWCE.DE","weight":100.,"asset_class":"Azionario globale","ter":.20,"tax_rate":.26}])
        if "ter" not in df: df["ter"]=0.
        if "tax_rate" not in df: df["tax_rate"]=.26
        df["tax_profile"]=df["tax_rate"].apply(lambda x:"12,5% Stato/white list" if abs(float(x)-.125)<.001 else "26% ordinaria")
        order=["ticker","weight","asset_class","ter","tax_profile"] if advanced else ["ticker","weight","asset_class"]
        ed=st.data_editor(df,num_rows="dynamic",width="stretch",key=f"a{i}",column_order=order,column_config={"ticker":st.column_config.TextColumn("Ticker Yahoo",required=True),"weight":st.column_config.NumberColumn("Peso (%)",min_value=0.,max_value=100.,step=1.,format="%.1f"),"asset_class":st.column_config.SelectboxColumn("Classe attivo",options=classes,required=True),"ter":st.column_config.NumberColumn("TER annuo (%)",min_value=0.,max_value=5.,step=.01,format="%.2f"),"tax_profile":st.column_config.SelectboxColumn("Fiscalità",options=["26% ordinaria","12,5% Stato/white list"],required=True)})
        records=ed.fillna({"ticker":"","weight":0.,"asset_class":"Altro","ter":0.,"tax_profile":"26% ordinaria"}).to_dict("records")
        for a in records: a["tax_rate"]=.125 if a.get("tax_profile")=="12,5% Stato/white list" else .26
        p["assets"]=records
        if p["assets"]:
            ridx=st.selectbox("Ticker da eliminare",range(len(p["assets"])),format_func=lambda j:str(p["assets"][j].get("ticker","")).strip() or f"Riga {j+1}",key=f"r{i}")
            if st.button("🗑 Elimina ticker selezionato",key=f"rb{i}",use_container_width=True): p["assets"].pop(ridx); st.session_state.results=None; st.rerun()
        ws=sum(float(a.get("weight",0) or 0) for a in p["assets"]); st.caption(f"Somma pesi: **{ws:.1f}%**"+(" · normalizzati in simulazione" if abs(ws-100)>.05 else ""))

with tabs[len(portfolios)]:
    if st.button("➕ Aggiungi portafoglio",disabled=len(portfolios)>=6,use_container_width=True): portfolios.append({"name":f"Portafoglio {len(portfolios)+1}","assets":[{"ticker":"VWCE.DE","weight":60.,"asset_class":"Azionario globale","ter":.20,"tax_rate":.26},{"ticker":"AGGH.L","weight":40.,"asset_class":"Obbligazionario globale","ter":.10,"tax_rate":.26}]}); st.session_state.results=None; st.rerun()
    if len(portfolios)>1:
        pi=st.selectbox("Portafoglio da eliminare",range(len(portfolios)),format_func=lambda j:portfolios[j]["name"])
        if st.button("🗑 Elimina portafoglio",use_container_width=True): portfolios.pop(pi); st.session_state.results=None; st.rerun()
    st.caption(f"Portafogli attivi: {len(portfolios)} / 6")
    st.download_button("⬇ Esporta configurazione JSON",json.dumps(config,indent=2,ensure_ascii=False).encode(),"lg_portfolios_config.json","application/json",use_container_width=True)

sim.update({"initial_value":initial,"contribution":contribution,"contribution_frequency":contribution_frequency,"years":years,"simulations":n_sims,"history_years":history_years,"inflation_rate":inflation})
unique_assets={}
for p in portfolios:
    for a in p.get("assets",[]):
        t=str(a.get("ticker","")).strip().upper()
        if t: unique_assets.setdefault(t,str(a.get("asset_class","Altro")))
historical_selected=[]; historical_start=1; proxy_mappings={}
if advanced:
    st.markdown("### 2. Stress storici e proxy")
    historical_selected=st.multiselect("Eventi storici da applicare",options=list(HISTORICAL_SHOCKS),default=[])
    if historical_selected: historical_start=st.slider("Mese di inizio del primo stress",1,max(1,years*12),min(24,years*12))
    if historical_selected and unique_assets:
        with st.expander("Proxy per ETF con storico insufficiente",expanded=False):
            for ticker,asset_class in unique_assets.items():
                suggestions=suggested_proxies(ticker,asset_class); symbols=[s for s,_ in suggestions]; reasons={s:r for s,r in suggestions}
                selected=st.selectbox(f"{ticker} · {asset_class}",["Nessun proxy"]+symbols+["Altro ticker…"],key=f"proxy_{ticker}",format_func=lambda x:f"{x} — {reasons[x]}" if x in reasons else x)
                if selected=="Altro ticker…":
                    cp=st.text_input(f"Ticker proxy personalizzato per {ticker}",key=f"proxy_custom_{ticker}")
                    if cp.strip():
                        try:
                            npy=normalize_proxy_ticker(cp)
                            if npy!=ticker: proxy_mappings[ticker]=npy
                        except Exception as exc: st.warning(f"{ticker}: {exc}")
                elif selected!="Nessun proxy": proxy_mappings[ticker]=selected
    config["historical_proxies"]=proxy_mappings
else:
    st.markdown("### 2. Modalità Basic")
    st.info("Bootstrap storico, PAC mensile, ribilanciamento annuale e fiscalità italiana semplificata. Passa ad **Avanzata** per TER, commissioni, bande, stress, proxy e coperture.")

st.markdown("### 3. Metodologia")
st.caption("La fiscalità è una stima semplificata: aliquote per asset sulle plusvalenze realizzate nei ribilanciamenti e alla liquidazione finale. Non sostituisce il calcolo fiscale dell'intermediario o del commercialista.")

if run:
    try:
        tickers=sorted(unique_assets)
        if not tickers: raise ValueError("Inserire almeno un ticker.")
        start_date=str((pd.Timestamp.today().normalize()-pd.DateOffset(years=history_years+1)).date())
        with st.spinner("Scarico prezzi, cambi e rendimenti mensili in EUR..."):
            market=cached_common_history(tuple(tickers),start_date); rets=market.asset_returns; cutoff=rets.index.max()-pd.DateOffset(years=history_years); rets=rets.loc[rets.index>=cutoff]
            ah={}; ph={}
            if historical_selected:
                for t in tickers:
                    try: ah[t]=cached_series(t)
                    except Exception: pass
                for px in sorted(set(proxy_mappings.values())):
                    try: ph[px]=cached_series(px)
                    except Exception: pass
            events=[{"name":name,**({"start_month":historical_start} if i==0 else {})} for i,name in enumerate(historical_selected)]
            results={}; bar=st.progress(0)
            for i,p in enumerate(portfolios):
                assets=[a for a in p.get("assets",[]) if str(a.get("ticker","")).strip()]; weights=[float(a.get("weight",0) or 0) for a in assets]
                if not assets or sum(weights)<=0: continue
                results[p["name"]]=simulate_portfolio(returns=rets,assets=assets,weights_pct=weights,initial_value=initial,years=years,n_sims=n_sims,inflation_rate=inflation,seed=int(seed)+i,contribution=contribution,contribution_frequency=contribution_frequency,method=method,distribution=distribution,degrees_of_freedom=dof,block_length=block,risk_free_rate=risk_free,ruin_threshold=ruin,include_put_hedge=include_put,put_moneyness=moneyness,put_maturity_months=maturity,put_implied_volatility=iv,put_hedge_fraction=hedge_fraction,custom_shock=custom,historical_events=events,historical_asset_returns=ah,historical_proxies=proxy_mappings,proxy_histories=ph,rebalance_mode=rebalance_mode,rebalance_band=rebalance_band,contributions_first=contributions_first,contribution_growth_rate=contribution_growth,contribution_stop_after_years=contribution_stop,extraordinary_month=extraordinary_month,extraordinary_amount=extraordinary_amount,taxation_enabled=taxation_enabled,transaction_fee_fixed=fee_fixed,transaction_fee_pct=fee_pct,annual_platform_fee_pct=platform_fee)
                bar.progress((i+1)/len(portfolios))
        settings={"mode":mode,"initial_value":initial,"contribution":contribution,"contribution_frequency":contribution_frequency,"contribution_growth_rate":contribution_growth,"contribution_stop_after_years":contribution_stop,"extraordinary_month":extraordinary_month,"extraordinary_amount":extraordinary_amount,"years":years,"n_sims":n_sims,"history_years":history_years,"inflation_rate":inflation,"method":method,"distribution":distribution,"degrees_of_freedom":dof,"block_length":block,"risk_free_rate":risk_free,"ruin_threshold":ruin,"rebalance_mode":rebalance_mode,"rebalance_band":rebalance_band,"contributions_first":contributions_first,"taxation_enabled":taxation_enabled,"transaction_fee_fixed":fee_fixed,"transaction_fee_pct":fee_pct,"annual_platform_fee_pct":platform_fee,"include_put_hedge":include_put,"historical_events":historical_selected,"historical_start_month":historical_start,"proxy_mappings":proxy_mappings,"market_warnings":market.warnings,"currencies":market.currencies}
        st.session_state.results=results; st.session_state.run_settings=settings; st.success(f"Simulazione completata: {len(results)} portafogli.")
    except MarketDataError as e: st.error(f"Dati di mercato non disponibili: {e}")
    except Exception as e: st.error(f"Simulazione non riuscita: {e}")

results=st.session_state.results
if results:
    settings=st.session_state.run_settings; st.markdown("---"); st.markdown("## 4. Risultati")
    rows=[]
    for name,r in results.items(): rows.append({"Portafoglio":name,"Totale versato (€)":r["total_invested"],"Lordo finale mediano (€)":r["median_terminal_gross"],"Netto finale mediano (€)":r["median_terminal"],"Netto reale mediano (€)":r["median_real_terminal"],"Tasse medie (€)":r["taxes_paid_mean"],"Costi medi (€)":r["fees_paid_mean"],"5° percentile netto (€)":r["p5_terminal"],"95° percentile netto (€)":r["p95_terminal"],"Prob. perdita":r["probability_loss"],"Prob. rovina":r["probability_ruin"],"Sharpe":r["sharpe"],"Sortino":r["sortino"],"Max DD mediano":r["max_drawdown_median"]})
    metrics=pd.DataFrame(rows); fmts={c:"€{:,.0f}" for c in metrics.columns if "(€)" in c}; fmts.update({"Prob. perdita":"{:.1%}","Prob. rovina":"{:.1%}","Sharpe":"{:.2f}","Sortino":"{:.2f}","Max DD mediano":"{:.1%}"}); st.dataframe(metrics.style.format(fmts),width="stretch")
    first=next(iter(results.values())); a,b,c,d=st.columns(4); a.metric("Netto mediano",f"€{first['median_terminal']:,.0f}"); b.metric("Netto reale",f"€{first['median_real_terminal']:,.0f}"); c.metric("Tasse medie",f"€{first['taxes_paid_mean']:,.0f}"); d.metric("Costi medi",f"€{first['fees_paid_mean']:,.0f}")
    st.markdown("### Fan chart netto"); fig=go.Figure()
    for name,r in results.items():
        paths=r["paths"]; x=np.arange(paths.shape[1])/12; q=np.quantile(paths,[.05,.5,.95],axis=0); fig.add_trace(go.Scatter(x=x,y=q[2],mode="lines",line=dict(width=0),showlegend=False,hoverinfo="skip")); fig.add_trace(go.Scatter(x=x,y=q[0],mode="lines",line=dict(width=0),fill="tonexty",name=f"{name} · 5–95%")); fig.add_trace(go.Scatter(x=x,y=q[1],mode="lines",name=f"{name} · mediana"))
    fig.update_layout(template="plotly_white",height=520,xaxis_title="Anni",yaxis_title="Valore (€)"); st.plotly_chart(fig,width="stretch")
    st.markdown("### Distribuzione valori finali netti"); hist=go.Figure()
    for name,r in results.items(): hist.add_trace(go.Histogram(x=r["terminal_net"],name=name,opacity=.55,nbinsx=55))
    hist.update_layout(barmode="overlay",template="plotly_white",height=420,xaxis_title="Valore finale netto (€)",yaxis_title="Frequenza"); st.plotly_chart(hist,width="stretch")
    warnings=[(name,w) for name,r in results.items() for w in r.get("warnings",[])]; market_warnings=settings.get("market_warnings",[])
    if warnings or market_warnings:
        with st.expander("Dettagli stress, proxy e dati di mercato",expanded=False):
            for w in market_warnings: st.caption("• "+w)
            for name,w in warnings: st.warning(f"{name}: {w}") if str(w).startswith("PROXY_RAPPRESENTATIVO:") else st.caption(f"• {name}: {w}")
    st.markdown("### 5. Esporta"); cp,cc=st.columns(2)
    with cp:
        try:
            pdf=build_pdf(results,settings,config,{"Fan chart netto dei portafogli":fig,"Distribuzione dei valori finali netti":hist}); st.download_button("⬇ Scarica report PDF completo",pdf,"LG_Consulting_Portfolio_Lab_Report.pdf","application/pdf",type="primary",use_container_width=True)
        except Exception as exc: st.warning(f"PDF non generato: {exc}")
    with cc: st.download_button("⬇ Esporta risultati CSV",metrics.to_csv(index=False).encode(),"LG_Consulting_Monte_Carlo_Metrics.csv","text/csv",use_container_width=True)
else:
    st.markdown("### 4. Esegui una simulazione"); st.info("Configura portafogli e parametri, poi premi **Esegui simulazione**.")
st.markdown("---"); st.caption("L & G Consulting · Strumento quantitativo sperimentale. Fiscalità e costi sono stime semplificate; i risultati non costituiscono consulenza finanziaria o fiscale.")