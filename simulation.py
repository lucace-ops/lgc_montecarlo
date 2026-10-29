"""Core calculations for the L & G Consulting Monte Carlo portfolio simulator."""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd

BLACK_SWAN_EVENTS = {
    "Lunedì nero (1987)": {"date":"1987-10-19","equity":-0.22,"bonds":0.01,"gold":0.04,"other":-0.12},
    "Crisi asiatica / LTCM (1998)": {"date":"1998-08-31","equity":-0.12,"bonds":0.03,"gold":-0.02,"other":-0.08},
    "Bolla dot-com (2000-2002)": {"date":"2001-09-21","equity":-0.18,"bonds":0.04,"gold":0.03,"other":-0.12},
    "Crisi finanziaria globale (2008)": {"date":"2008-10-10","equity":-0.20,"bonds":0.02,"gold":-0.05,"other":-0.15},
    "Crisi Euro / downgrade USA (2011)": {"date":"2011-08-08","equity":-0.10,"bonds":0.03,"gold":0.05,"other":-0.08},
    "Sell-off COVID (2020)": {"date":"2020-03-16","equity":-0.12,"bonds":-0.02,"gold":-0.03,"other":-0.10},
    "Shock inflazione / tassi (2022)": {"date":"2022-06-13","equity":-0.05,"bonds":-0.04,"gold":-0.02,"other":-0.05},
}
ASSET_CLASS_HINTS = {
    "equity": ("equity","azion","world","sp500","s&p","nasdaq","msci","stock"),
    "bonds": ("bond","obblig","treasury","govt","aggregate","agg","gilt"),
    "gold": ("gold","oro","sgld","4gl"),
}
def classify_asset(asset: dict[str, Any]) -> str:
    haystack = str(asset.get("asset_class","")).lower()+" "+str(asset.get("ticker","")).lower()
    for cls,hints in ASSET_CLASS_HINTS.items():
        if any(h in haystack for h in hints): return cls
    return "other"

def download_prices(tickers:list[str], period:str="10y")->pd.DataFrame:
    import yfinance as yf
    cleaned=list(dict.fromkeys(t.strip().upper() for t in tickers if t.strip()))
    if not cleaned: raise ValueError("Inserire almeno un ticker valido.")
    raw=yf.download(tickers=cleaned,period=period,auto_adjust=True,progress=False,group_by="ticker",threads=True)
    if raw.empty: raise ValueError("Yahoo Finance non ha restituito dati. Verifica ticker e connessione.")
    if isinstance(raw.columns,pd.MultiIndex):
        frames={}
        for ticker in cleaned:
            if ticker in raw.columns.get_level_values(0):
                block=raw[ticker]; col="Close" if "Close" in block.columns else block.columns[0]; frames[ticker]=block[col]
            elif ticker in raw.columns.get_level_values(-1):
                block=raw.xs(ticker,axis=1,level=-1); col="Close" if "Close" in block.columns else block.columns[0]; frames[ticker]=block[col]
        prices=pd.DataFrame(frames)
    else:
        col="Close" if "Close" in raw.columns else raw.columns[0]; prices=raw[[col]].rename(columns={col:cleaned[0]})
    prices=prices.sort_index().apply(pd.to_numeric,errors="coerce").dropna(axis=1,how="all").ffill().dropna(how="all")
    if prices.empty: raise ValueError("Non è stato possibile estrarre prezzi validi.")
    return prices

def historical_returns(prices:pd.DataFrame,years:int=10)->pd.DataFrame:
    if prices.empty: raise ValueError("La serie dei prezzi è vuota.")
    cutoff=prices.index.max()-pd.Timedelta(days=int(years*365.25))
    r=prices.loc[prices.index>=cutoff].pct_change().replace([np.inf,-np.inf],np.nan).dropna(how="all").dropna(axis=1,how="all")
    if len(r)<60: raise ValueError("Dati insufficienti: servono almeno circa 60 osservazioni giornaliere.")
    return r

def _portfolio_daily_returns(returns,assets,weights):
    tickers=[a["ticker"] for a in assets]; usable=[t for t in tickers if t in returns.columns]
    if not usable: raise ValueError("Nessuno dei ticker del portafoglio è presente nei dati scaricati.")
    wm={a["ticker"]:float(w) for a,w in zip(assets,weights)}; w=np.array([wm[t] for t in usable]); w/=w.sum()
    return returns[usable].dropna().dot(w)

def max_drawdown(path):
    peaks=np.maximum.accumulate(path); return float(np.nanmin(path/np.maximum(peaks,1e-12)-1))
def underwater_duration(path):
    peaks=np.maximum.accumulate(path); u=path<peaks*(1-1e-10); longest=current=0
    for flag in u:
        if flag: current+=1; longest=max(longest,current)
        else: current=0
    return int(longest)

def simulate_portfolio(returns,assets,weights_pct,initial_value=100_000,years=10,n_sims=5000,
 confidence=0.95,inflation_rate=0.02,rebalance_frequency="Annuale",include_black_swans=True,
 include_put_hedge=False,put_annual_cost_pct=0.75,put_trigger_drawdown_pct=15.0,seed=42):
    tickers=[str(a["ticker"]).strip().upper() for a in assets]
    original_map={ticker:float(w) for ticker,w in zip(tickers,weights_pct)}
    norm=[]
    for asset,ticker in zip(assets,tickers):
        if ticker in returns.columns:
            x=dict(asset); x["ticker"]=ticker; norm.append(x)
    assets=norm
    if not assets: raise ValueError("I ticker del portafoglio non sono disponibili.")
    weights=np.array([original_map[a["ticker"]] for a in assets],dtype=float)
    if np.any(weights<0) or weights.sum()<=0: raise ValueError("I pesi devono essere non negativi e avere somma positiva.")
    weights/=weights.sum()
    data=returns[[a["ticker"] for a in assets]].dropna().to_numpy(dtype=float)
    if data.shape[0]<30: raise ValueError("Dati storici comuni insufficienti per simulare.")
    years=max(1,int(years)); n_sims=max(1,int(n_sims)); steps=years*252; rng=np.random.default_rng(int(seed))
    paths=np.empty((n_sims,steps+1),dtype=np.float32); paths[:,0]=initial_value
    holdings=np.broadcast_to(float(initial_value)*weights,(n_sims,len(weights))).copy()
    value=np.full(n_sims,float(initial_value)); peak=value.copy(); mdd=np.zeros(n_sims)
    uwcur=np.zeros(n_sims,dtype=np.int32); uwlong=np.zeros(n_sims,dtype=np.int32); paid=np.zeros(n_sims,dtype=bool)
    daily_cost=max(0,float(put_annual_cost_pct)/100)/252 if include_put_hedge else 0
    reb={"Mensile":21,"Trimestrale":63,"Annuale":252}.get(rebalance_frequency,steps+1)
    events=list(BLACK_SWAN_EVENTS.values()); classes=[classify_asset(a) for a in assets]
    shocks=np.array([[e.get(c,e["other"]) for c in classes] for e in events]); prob=min(.035,7/252) if include_black_swans else 0
    for t in range(steps):
        sampled=data[rng.integers(0,len(data),size=n_sims)].copy()
        if include_black_swans:
            mask=rng.random(n_sims)<prob; count=int(mask.sum())
            if count:
                ids=rng.integers(0,len(events),size=count)
                sampled[mask]=np.clip(shocks[ids]+rng.normal(0,.015,size=(count,len(weights))),-.65,.35)
        holdings*=1+sampled; value=holdings.sum(axis=1)
        if daily_cost:
            value*=1-daily_cost; totals=np.maximum(holdings.sum(axis=1),1e-12); holdings*=(value/totals)[:,None]
        peak=np.maximum(peak,value); dd=value/np.maximum(peak,1e-12)-1; mdd=np.minimum(mdd,dd)
        under=dd<-1e-10; uwcur=np.where(under,uwcur+1,0); uwlong=np.maximum(uwlong,uwcur)
        if include_put_hedge:
            pmask=(dd*100<=-abs(float(put_trigger_drawdown_pct))) & ~paid
            if np.any(pmask):
                value[pmask]+=.03*float(initial_value); totals=np.maximum(holdings[pmask].sum(axis=1),1e-12)
                holdings[pmask]*=(value[pmask]/totals)[:,None]; paid[pmask]=True
        if (t+1)%252==0: paid[:]=False
        if (t+1)%reb==0: holdings=value[:,None]*weights[None,:]
        paths[:,t+1]=value.astype(np.float32)
    days=np.arange(steps+1,dtype=float); real_paths=(paths.astype(float)/((1+float(inflation_rate))**(days/252))[None,:]).astype(np.float32)
    terminal=paths[:,-1].astype(float); real_terminal=real_paths[:,-1].astype(float); tr=terminal/float(initial_value)-1
    q=1-float(confidence); cutoff=float(np.quantile(tr,q)); tail=tr[tr<=cutoff]
    return {"paths":paths,"real_paths":real_paths,"terminal":terminal,"real_terminal":real_terminal,"terminal_returns":tr,
     "var":max(0,-cutoff),"cvar":max(0,-float(tail.mean())) if tail.size else 0.0,
     "median_terminal":float(np.median(terminal)),"mean_terminal":float(np.mean(terminal)),
     "p5_terminal":float(np.quantile(terminal,.05)),"p95_terminal":float(np.quantile(terminal,.95)),
     "median_real_terminal":float(np.median(real_terminal)),
     "median_return_annualized":float((np.median(terminal)/float(initial_value))**(1/years)-1),
     "max_drawdown_median":float(np.median(mdd)),"max_drawdown_p95":float(np.quantile(mdd,.05)),
     "underwater_days_median":float(np.median(uwlong)),"underwater_days_p95":float(np.quantile(uwlong,.95)),
     "returns":returns,"weights":weights,"assets":assets}
