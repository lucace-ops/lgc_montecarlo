"""Advanced Monte Carlo engine for L & G Consulting Portfolio Lab."""
from __future__ import annotations
from math import sqrt
from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import norm

BLACK_SWAN_EVENTS = {
    "Lunedì nero (1987)": {"date":"1987-10-19","equity":-0.22,"bonds":0.01,"gold":0.04,"other":-0.12},
    "Crisi asiatica / LTCM (1998)": {"date":"1998-08-31","equity":-0.12,"bonds":0.03,"gold":-0.02,"other":-0.08},
    "Bolla dot-com (2000-2002)": {"date":"2001-09-21","equity":-0.18,"bonds":0.04,"gold":0.03,"other":-0.12},
    "Crisi finanziaria globale (2008)": {"date":"2008-10-10","equity":-0.20,"bonds":0.02,"gold":-0.05,"other":-0.15},
    "Crisi Euro / downgrade USA (2011)": {"date":"2011-08-08","equity":-0.10,"bonds":0.03,"gold":0.05,"other":-0.08},
    "Sell-off COVID (2020)": {"date":"2020-03-16","equity":-0.12,"bonds":-0.02,"gold":-0.03,"other":-0.10},
    "Shock inflazione / tassi (2022)": {"date":"2022-06-13","equity":-0.05,"bonds":-0.04,"gold":-0.02,"other":-0.05},
}
ASSET_CLASS_HINTS={"equity":("equity","azion","world","sp500","s&p","nasdaq","msci","stock"),"bonds":("bond","obblig","treasury","govt","aggregate","agg","gilt"),"gold":("gold","oro","sgld","4gl")}
def classify_asset(asset:dict[str,Any])->str:
    h=(str(asset.get("asset_class",""))+" "+str(asset.get("ticker",""))).lower()
    for c,keys in ASSET_CLASS_HINTS.items():
        if any(k in h for k in keys): return c
    return "other"

def download_prices(tickers:list[str],period:str="10y")->pd.DataFrame:
    import yfinance as yf
    cleaned=list(dict.fromkeys(t.strip().upper() for t in tickers if t.strip()))
    if not cleaned: raise ValueError("Inserire almeno un ticker valido.")
    raw=yf.download(tickers=cleaned,period=period,auto_adjust=True,progress=False,group_by="ticker",threads=True)
    if raw.empty: raise ValueError("Yahoo Finance non ha restituito dati. Verifica ticker e connessione.")
    if isinstance(raw.columns,pd.MultiIndex):
        frames={}
        for ticker in cleaned:
            if ticker in raw.columns.get_level_values(0):
                b=raw[ticker]; frames[ticker]=b["Close" if "Close" in b.columns else b.columns[0]]
            elif ticker in raw.columns.get_level_values(-1):
                b=raw.xs(ticker,axis=1,level=-1); frames[ticker]=b["Close" if "Close" in b.columns else b.columns[0]]
        prices=pd.DataFrame(frames)
    else:
        c="Close" if "Close" in raw.columns else raw.columns[0]; prices=raw[[c]].rename(columns={c:cleaned[0]})
    prices=prices.sort_index().apply(pd.to_numeric,errors="coerce").dropna(axis=1,how="all").ffill().dropna(how="all")
    if prices.empty: raise ValueError("Non è stato possibile estrarre prezzi validi.")
    return prices

def historical_returns(prices:pd.DataFrame,years:int=10)->pd.DataFrame:
    cutoff=prices.index.max()-pd.Timedelta(days=int(years*365.25))
    r=prices.loc[prices.index>=cutoff].pct_change(fill_method=None).replace([np.inf,-np.inf],np.nan).dropna(how="all").dropna(axis=1,how="all")
    if len(r)<60: raise ValueError("Dati storici insufficienti.")
    return r

def _monthly_history(returns:pd.DataFrame)->pd.DataFrame:
    gross=(1+returns.fillna(0)).resample("ME").prod()-1
    return gross.dropna(how="any")

def _draws(hist:np.ndarray,n:int,months:int,method:str,distribution:str,df:int,block:int,rng)->np.ndarray:
    if method=="Bootstrap storico":
        block=max(1,min(int(block),months)); out=np.empty((n,months,hist.shape[1]))
        for s in range(0,months,block):
            w=min(block,months-s); starts=rng.integers(0,len(hist),size=n); ids=(starts[:,None]+np.arange(w)[None,:])%len(hist); out[:,s:s+w]=hist[ids]
        return out
    mu=hist.mean(0); cov=np.atleast_2d(np.cov(hist,rowvar=False,ddof=1)); cov=np.nan_to_num(cov)+np.eye(hist.shape[1])*1e-12
    z=rng.multivariate_normal(mu,cov,size=(n,months))
    if distribution=="Student-t multivariata":
        df=max(3,int(df)); chi=rng.chisquare(df,size=(n,months,1)); z=mu+(z-mu)*np.sqrt((df-2)/chi)
    return z

def _put_price(spot,strike,t,r,sigma):
    spot=np.maximum(np.asarray(spot,float),1e-12); t=max(float(t),1e-8); sigma=max(float(sigma),1e-8)
    d1=(np.log(spot/strike)+(r+0.5*sigma*sigma)*t)/(sigma*np.sqrt(t)); d2=d1-sigma*np.sqrt(t)
    return strike*np.exp(-r*t)*norm.cdf(-d2)-spot*norm.cdf(-d1)

def simulate_portfolio(returns,assets,weights_pct,initial_value=100000,years=10,n_sims=3000,confidence=.95,inflation_rate=.02,rebalance_frequency="Annuale",include_black_swans=False,include_put_hedge=False,put_annual_cost_pct=.75,put_trigger_drawdown_pct=15,seed=42,contribution=0.0,contribution_frequency="Mensile",method="Bootstrap storico",distribution="Student-t multivariata",degrees_of_freedom=6,block_length=3,risk_free_rate=.025,ruin_threshold=.5,put_moneyness=.20,put_maturity_months=12,put_implied_volatility=.25,put_hedge_fraction=1.0,custom_shock=None):
    tickers=[str(a["ticker"]).strip().upper() for a in assets]; wm={t:float(w) for t,w in zip(tickers,weights_pct)}
    usable=[t for t in tickers if t in returns.columns]
    if not usable: raise ValueError("Nessun ticker disponibile nei dati.")
    w=np.array([wm[t] for t in usable],float); w/=w.sum(); monthly=_monthly_history(returns[usable])
    if len(monthly)<12: raise ValueError("Servono almeno 12 mesi comuni di dati.")
    months=int(years)*12; n=max(1,int(n_sims)); rng=np.random.default_rng(int(seed)); draws=_draws(monthly.to_numpy(float),n,months,method,distribution,degrees_of_freedom,block_length,rng)
    pr=np.tensordot(draws,w,axes=([-1],[0])); np.clip(pr,-.999,5,out=pr)
    if custom_shock:
        start=max(0,int(custom_shock.get("start_month",1))-1); dur=max(1,int(custom_shock.get("duration",1))); end=min(months,start+dur); impact=float(custom_shock.get("impact",0)); vol=max(1,float(custom_shock.get("vol_multiplier",1))); seg=pr[:,start:end]; mean=float(monthly.to_numpy().dot(w).mean()); seg[:]=mean+(seg-mean)*vol; seg[:]=(1+seg)*((1+impact)**(1/max(1,end-start)))-1
    wealth=np.empty((n,months+1),dtype=np.float32); wealth[:,0]=initial_value; holdings=np.full(n,float(initial_value)); peaks=holdings.copy(); mdd=np.zeros(n); uw=np.zeros(n,int); uwmax=np.zeros(n,int); net=np.empty((n,months),float)
    premium_paid=0.0; payoff_received=0.0; put_value=np.zeros(n); strike=np.zeros(n); units=np.zeros(n); expiry=max(1,int(put_maturity_months)); iv=max(.01,float(put_implied_volatility)); hedge=max(0,min(1,float(put_hedge_fraction)))
    if include_put_hedge:
        spot=holdings.copy(); strike=spot*(1-float(put_moneyness)); units=holdings*hedge/np.maximum(spot,1e-12); put_value=units*_put_price(spot,strike,expiry/12,risk_free_rate,iv); holdings-=put_value; premium_paid+=float(np.mean(put_value))
    for m in range(1,months+1):
        prev=np.maximum(holdings+put_value,1e-12); cash=float(contribution) if contribution_frequency=="Mensile" or m%12==0 else 0.0; holdings+=cash; holdings*=1+pr[:,m-1]
        if include_put_hedge:
            spot=np.maximum(holdings,1e-12); remaining=expiry-((m-1)%expiry)
            if m%expiry==0:
                payoff=units*np.maximum(strike-spot,0); holdings+=payoff; payoff_received+=float(np.mean(payoff)); strike=spot*(1-float(put_moneyness)); units=holdings*hedge/spot; put_value=units*_put_price(spot,strike,expiry/12,risk_free_rate,iv); holdings-=put_value; premium_paid+=float(np.mean(put_value))
            else: put_value=units*_put_price(spot,strike,remaining/12,risk_free_rate,iv)
        total=np.maximum(holdings+put_value,0); wealth[:,m]=total; net[:,m-1]=(total-cash)/prev-1; peaks=np.maximum(peaks,total); dd=total/np.maximum(peaks,1e-12)-1; mdd=np.minimum(mdd,dd); under=dd<0; uw=np.where(under,uw+1,0); uwmax=np.maximum(uwmax,uw)
    real=wealth.astype(float)/((1+inflation_rate)**(np.arange(months+1)/12))[None,:]; terminal=wealth[:,-1].astype(float); realterminal=real[:,-1]; invested=float(initial_value)+float(contribution)*int(years)*(12 if contribution_frequency=="Mensile" else 1)
    flat=net.ravel(); q95,q99=np.quantile(flat,[.05,.01]); c95=flat[flat<=q95]; c99=flat[flat<=q99]; meanm=float(flat.mean()); stdm=float(flat.std(ddof=1)); ann=(1+meanm)**12-1; vol=stdm*sqrt(12); downside=np.minimum(flat,0); ddev=float(np.sqrt(np.mean(downside**2))*sqrt(12)); sharpe=(ann-risk_free_rate)/vol if vol>1e-12 else 0; sortino=(ann-risk_free_rate)/ddev if ddev>1e-12 else 0
    tr=terminal/invested-1; q=1-confidence; cut=float(np.quantile(tr,q)); tail=tr[tr<=cut]
    return {"paths":wealth,"real_paths":real.astype(np.float32),"terminal":terminal,"real_terminal":realterminal,"terminal_returns":tr,"var":max(0,-cut),"cvar":max(0,-float(tail.mean())) if len(tail) else 0,"median_terminal":float(np.median(terminal)),"mean_terminal":float(np.mean(terminal)),"p5_terminal":float(np.quantile(terminal,.05)),"p95_terminal":float(np.quantile(terminal,.95)),"median_real_terminal":float(np.median(realterminal)),"median_return_annualized":float((np.median(terminal)/max(invested,1e-12))**(1/years)-1),"max_drawdown_median":float(np.median(mdd)),"max_drawdown_p95":float(np.quantile(mdd,.05)),"underwater_days_median":float(np.median(uwmax)*21),"underwater_days_p95":float(np.quantile(uwmax,.95)*21),"probability_loss":float(np.mean(terminal<invested)),"probability_ruin":float(np.mean(terminal<invested*ruin_threshold)),"var95_monthly":max(0,-float(q95)),"var99_monthly":max(0,-float(q99)),"cvar95_monthly":max(0,-float(c95.mean())),"cvar99_monthly":max(0,-float(c99.mean())),"sharpe":float(sharpe),"sortino":float(sortino),"annualized_return":float(ann),"annualized_volatility":float(vol),"total_invested":invested,"hedging":{"premium_paid":premium_paid,"payoff_received":payoff_received},"returns":returns,"weights":w,"assets":[a for a in assets if str(a["ticker"]).strip().upper() in usable]}
