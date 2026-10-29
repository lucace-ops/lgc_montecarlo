"""Advanced monthly Monte Carlo engine for L & G Consulting Portfolio Lab."""
from __future__ import annotations
from math import sqrt
import numpy as np
import pandas as pd
from scipy.stats import norm

HISTORICAL_SHOCKS = {
    "Shock petrolifero / stagflazione": ("1973-10-01", "1974-03-31"),
    "Lunedì Nero": ("1987-10-01", "1987-12-31"),
    "Bolla Dot-Com": ("2000-03-01", "2002-10-31"),
    "Crisi Subprime / Lehman Brothers": ("2007-10-01", "2009-03-31"),
    "Crisi del debito sovrano Eurozona": ("2010-05-01", "2012-07-31"),
    "Crash COVID-19": ("2020-02-01", "2020-04-30"),
    "Shock inflazionistico e bond crash": ("2022-01-01", "2022-12-31"),
}
BLACK_SWAN_EVENTS = {name: {"date": start, "start": start, "end": end} for name, (start, end) in HISTORICAL_SHOCKS.items()}

def _draws(history, n_sims, months, method, distribution, degrees_of_freedom, block_length, rng):
    if history.shape[0] < 2: raise ValueError("Servono almeno due mesi di rendimenti storici.")
    if method == "Bootstrap storico":
        block=max(1,min(int(block_length),months)); out=np.empty((n_sims,months,history.shape[1]),float)
        for start in range(0,months,block):
            width=min(block,months-start); starts=rng.integers(0,history.shape[0],size=n_sims); ids=(starts[:,None]+np.arange(width)[None,:])%history.shape[0]; out[:,start:start+width,:]=history[ids]
        return out
    mu=np.mean(history,axis=0); cov=np.atleast_2d(np.cov(history,rowvar=False,ddof=1)); cov=np.nan_to_num(cov)+np.eye(history.shape[1])*1e-12
    z=rng.multivariate_normal(mu,cov,size=(n_sims,months),check_valid="warn")
    if distribution=="Student-t multivariata":
        df=max(3,int(degrees_of_freedom)); chi=rng.chisquare(df,size=(n_sims,months,1)); z=mu+(z-mu)*np.sqrt((df-2)/chi)
    return z

def compose_historical_event_returns(event_name, weights, asset_histories, proxy_mappings=None, proxy_histories=None):
    start,end=HISTORICAL_SHOCKS[event_name]; start_date,end_date=pd.Timestamp(start),pd.Timestamp(end); proxy_mappings,proxy_histories=proxy_mappings or {},proxy_histories or {}; components={}; used_proxies=[]; missing=[]
    for ticker,weight in weights.items():
        direct=asset_histories.get(ticker); direct_window=direct.loc[(direct.index>=start_date)&(direct.index<=end_date)].dropna() if direct is not None else pd.Series(dtype=float)
        if len(direct_window)>=2: components[ticker]=direct_window.astype(float)*float(weight); continue
        proxy=proxy_mappings.get(ticker); ps=proxy_histories.get(proxy) if proxy else None; proxy_window=ps.loc[(ps.index>=start_date)&(ps.index<=end_date)].dropna() if ps is not None else pd.Series(dtype=float)
        if len(proxy_window)>=2: components[ticker]=proxy_window.astype(float)*float(weight); used_proxies.append((ticker,proxy))
        else: missing.append(ticker)
    if missing or not components: return None,used_proxies,missing
    aligned=pd.concat(components,axis=1,join="inner").dropna(how="any")
    if len(aligned)<2: return None,used_proxies,list(weights)
    return aligned.sum(axis=1).astype(float),used_proxies,[]

def _apply_historical_events_to_portfolio(portfolio_returns, events, weights, asset_histories, proxy_mappings, proxy_histories):
    warnings,details=[],[]; horizon,next_start=portfolio_returns.shape[1],1
    for event in events or []:
        name=str(event["name"])
        if name not in HISTORICAL_SHOCKS: warnings.append(f"{name}: evento storico sconosciuto, ignorato."); continue
        start,end=HISTORICAL_SHOCKS[name]; full_months=(pd.Timestamp(end).year-pd.Timestamp(start).year)*12+pd.Timestamp(end).month-pd.Timestamp(start).month+1; start_month=int(event.get("start_month",next_start))
        event_returns,used,missing=compose_historical_event_returns(name,weights,asset_histories,proxy_mappings,proxy_histories)
        if event_returns is None: warnings.append(f"{name}: non applicato; storico/proxy insufficiente per {', '.join(missing) if missing else 'la finestra storica'}."); next_start=start_month+full_months; continue
        vals=event_returns.to_numpy(float); duration=min(len(vals),horizon-start_month+1)
        if duration<=0: warnings.append(f"{name}: fuori dall'orizzonte simulato."); continue
        left=start_month-1; portfolio_returns[:,left:left+duration]=vals[:duration][None,:]
        if used: warnings.append("PROXY_RAPPRESENTATIVO: "+name+": "+", ".join(f"{proxy} per {asset}" for asset,proxy in used)+".")
        warnings.append(f"{name}: sequenza storica applicata ai mesi {start_month}-{start_month+duration-1}."); details.append({"name":name,"start_month":start_month,"duration":duration,"used_proxies":used}); next_start=start_month+full_months
    return warnings,details

def _apply_custom_shock(portfolio_returns, shock, mean_return):
    if not shock: return []
    start=max(1,int(shock.get("start_month",1))); duration=max(1,int(shock.get("duration",1))); end=min(portfolio_returns.shape[1],start+duration-1)
    if start>portfolio_returns.shape[1]: return ["Shock personalizzato fuori dall'orizzonte."]
    width=end-start+1; impact=float(shock.get("impact",0)); vol=max(0,float(shock.get("vol_multiplier",1))); segment=portfolio_returns[:,start-1:end]; segment[:]=mean_return+(segment-mean_return)*vol; segment[:]=(1+segment)*((1+impact)**(1/width))-1; np.clip(segment,-.999,5,out=segment)
    return [f"Shock personalizzato applicato ai mesi {start}-{end}: impatto {impact:.1%}, volatilità x{vol:.2f}."]

def _put_price(spot,strike,t,r,sigma):
    spot=np.maximum(np.asarray(spot,float),1e-12); t,sigma=max(float(t),1e-8),max(float(sigma),1e-8); d1=(np.log(spot/strike)+(r+.5*sigma*sigma)*t)/(sigma*np.sqrt(t)); d2=d1-sigma*np.sqrt(t); return strike*np.exp(-r*t)*norm.cdf(-d2)-spot*norm.cdf(-d1)

def _monthly_contribution(base,month,frequency,annual_growth,stop_after_years,extraordinary_month,extraordinary_amount):
    year_index=(month-1)//12
    if stop_after_years and year_index>=int(stop_after_years): regular=0.
    else:
        regular=float(base)*((1+float(annual_growth))**year_index)
        if frequency=="Annuale" and month%12!=0: regular=0.
    return regular+(float(extraordinary_amount) if extraordinary_month and month==int(extraordinary_month) else 0.)

def simulate_portfolio(returns:pd.DataFrame,assets,weights_pct,initial_value=100000,years=10,n_sims=3000,confidence=.95,inflation_rate=.02,rebalance_frequency="Annuale",include_black_swans=False,include_put_hedge=False,put_annual_cost_pct=.75,put_trigger_drawdown_pct=15,seed=42,contribution=0.,contribution_frequency="Mensile",method="Bootstrap storico",distribution="Student-t multivariata",degrees_of_freedom=6,block_length=3,risk_free_rate=.025,ruin_threshold=.5,put_moneyness=.20,put_maturity_months=12,put_implied_volatility=.25,put_hedge_fraction=1.,custom_shock=None,historical_events=None,historical_asset_returns=None,historical_proxies=None,proxy_histories=None,rebalance_mode=None,rebalance_band=.05,contributions_first=True,contribution_growth_rate=0.,contribution_stop_after_years=None,extraordinary_month=None,extraordinary_amount=0.,taxation_enabled=False,default_tax_rate=.26,transaction_fee_fixed=0.,transaction_fee_pct=0.,annual_platform_fee_pct=0.):
    tickers=[str(a["ticker"]).strip().upper() for a in assets]; weight_map={t:float(w) for t,w in zip(tickers,weights_pct)}; usable=[t for t in tickers if t in returns.columns]
    if not usable: raise ValueError("Nessun ticker disponibile nei dati mensili in EUR.")
    w=np.array([weight_map[t] for t in usable],float)
    if np.any(w<0) or w.sum()<=0: raise ValueError("I pesi devono essere non negativi e con somma positiva.")
    w/=w.sum(); monthly=returns[usable].dropna(how="any")
    if len(monthly)<12: raise ValueError("Servono almeno 12 mesi comuni di rendimenti in EUR.")
    asset_map={str(a["ticker"]).strip().upper():a for a in assets}; ters=np.array([max(0.,float(asset_map[t].get("ter",0.) or 0.))/100 for t in usable]); tax_rates=np.array([max(0.,min(1.,float(asset_map[t].get("tax_rate",default_tax_rate) or default_tax_rate))) for t in usable]); monthly_ter=(1+ters)**(1/12)-1
    months,n=max(1,int(years))*12,max(1,int(n_sims)); rng=np.random.default_rng(int(seed)); asset_draws=_draws(monthly.to_numpy(float),n,months,method,distribution,degrees_of_freedom,block_length,rng); np.clip(asset_draws,-.999,5,out=asset_draws); portfolio_returns=np.tensordot(asset_draws,w,axes=([-1],[0])); warnings=[]; stress_details=[]
    if historical_events:
        hw={t:float(wi) for t,wi in zip(usable,w)}; ww,stress_details=_apply_historical_events_to_portfolio(portfolio_returns,historical_events,hw,historical_asset_returns or {},historical_proxies or {},proxy_histories or {}); warnings.extend(ww)
    warnings.extend(_apply_custom_shock(portfolio_returns,custom_shock,float(monthly.to_numpy().dot(w).mean()))); baseline_pr=np.tensordot(asset_draws,w,axes=([-1],[0])); ratio=np.divide(1+portfolio_returns,np.maximum(1+baseline_pr,1e-9)); stressed_asset_draws=(1+asset_draws)*ratio[:,:,None]-1; np.clip(stressed_asset_draws,-.999,5,out=stressed_asset_draws)
    holdings=np.broadcast_to(float(initial_value)*w,(n,len(w))).copy(); basis=holdings.copy(); gross_holdings=holdings.copy(); wealth=np.empty((n,months+1),np.float32); wealth[:,0]=initial_value; gross_wealth=np.empty((n,months+1),np.float32); gross_wealth[:,0]=initial_value; peaks=np.full(n,float(initial_value)); mdd=np.zeros(n); uw=np.zeros(n,int); uwmax=np.zeros(n,int); net_returns=np.empty((n,months),float); tax_paid=np.zeros(n); fees_paid=np.zeros(n); total_contributed=np.full(n,float(initial_value))
    premium_paid=0.; payoff_received=0.; put_value=np.zeros(n); strike=np.zeros(n); units=np.zeros(n); expiry=max(1,int(put_maturity_months)); iv=max(.01,float(put_implied_volatility)); hedge=np.clip(float(put_hedge_fraction),0,1)
    if include_put_hedge:
        spot=holdings.sum(1); strike=spot*(1-float(put_moneyness)); units=hedge; premium=units*_put_price(spot,strike,expiry/12,risk_free_rate,iv); scale=np.maximum((spot-premium)/np.maximum(spot,1e-12),0); holdings*=scale[:,None]; basis*=scale[:,None]; put_value=premium; premium_paid+=float(np.mean(premium))
    mode=rebalance_mode or rebalance_frequency; schedule={"Mensile":1,"Trimestrale":3,"Semestrale":6,"Annuale":12}.get(mode); fixed_fee=max(0.,float(transaction_fee_fixed)); pct_fee=max(0.,float(transaction_fee_pct))/100; platform_monthly=(1+max(0.,float(annual_platform_fee_pct))/100)**(1/12)-1
    for m in range(1,months+1):
        prev_total=np.maximum(holdings.sum(1)+put_value,1e-12); cash=_monthly_contribution(contribution,m,contribution_frequency,contribution_growth_rate,contribution_stop_after_years,extraordinary_month,extraordinary_amount)
        if cash>0:
            total_contributed+=cash
            if contributions_first:
                current=holdings.sum(1); target=current[:,None]*w[None,:]; gap=np.maximum(target-holdings,0); gap_sum=gap.sum(1); alloc=np.broadcast_to(w,holdings.shape).copy(); np.divide(gap,gap_sum[:,None],out=alloc,where=gap_sum[:,None]>1e-12)
            else: alloc=np.broadcast_to(w,holdings.shape)
            buy=cash*alloc; fee=np.where(buy>0,fixed_fee+buy*pct_fee,0).sum(1); investable=np.maximum(cash-fee,0); buy=investable[:,None]*alloc; holdings+=buy; basis+=buy; gross_holdings+=cash*alloc; fees_paid+=fee
        holdings*=1+stressed_asset_draws[:,m-1,:]; gross_holdings*=1+stressed_asset_draws[:,m-1,:]
        if np.any(monthly_ter>0): holdings*=1-monthly_ter[None,:]
        if platform_monthly>0: fee=holdings.sum(1)*platform_monthly; holdings*=max(1-platform_monthly,0); fees_paid+=fee
        total=holdings.sum(1); weights_now=np.divide(holdings,np.maximum(total[:,None],1e-12)); do_rebalance=np.zeros(n,dtype=bool)
        if mode=="Bande": do_rebalance=np.max(np.abs(weights_now-w[None,:]),axis=1)>=float(rebalance_band)
        elif schedule and m%schedule==0: do_rebalance[:]=True
        if np.any(do_rebalance):
            idx=np.where(do_rebalance)[0]; target_values=total[idx,None]*w[None,:]; diff=target_values-holdings[idx]; sells=np.maximum(-diff,0); buys=np.maximum(diff,0)
            if taxation_enabled: gain_fraction=np.maximum(holdings[idx]-basis[idx],0)/np.maximum(holdings[idx],1e-12); taxes=(sells*gain_fraction*tax_rates[None,:]).sum(1); tax_paid[idx]+=taxes
            else: taxes=np.zeros(len(idx))
            trade_fee=np.where(sells>0,fixed_fee+sells*pct_fee,0).sum(1)+np.where(buys>0,fixed_fee+buys*pct_fee,0).sum(1); fees_paid[idx]+=trade_fee; post_total=np.maximum(total[idx]-taxes-trade_fee,0); holdings[idx]=post_total[:,None]*w[None,:]; basis[idx]=holdings[idx]; gross_total=gross_holdings[idx].sum(1); gross_holdings[idx]=gross_total[:,None]*w[None,:]
        if include_put_hedge:
            spot=np.maximum(holdings.sum(1),1e-12); remaining=expiry-((m-1)%expiry)
            if m%expiry==0:
                payoff=units*np.maximum(strike-spot,0); payoff_received+=float(np.mean(payoff)); holdings+=payoff[:,None]*w[None,:]; basis+=payoff[:,None]*w[None,:]; spot=holdings.sum(1); strike=spot*(1-float(put_moneyness)); premium=units*_put_price(spot,strike,expiry/12,risk_free_rate,iv); factor=np.maximum((spot-premium)/np.maximum(spot,1e-12),0); holdings*=factor[:,None]; basis*=factor[:,None]; put_value=premium; premium_paid+=float(np.mean(premium))
            else: put_value=units*_put_price(spot,strike,remaining/12,risk_free_rate,iv)
        total=np.maximum(holdings.sum(1)+put_value,0); gross_total=np.maximum(gross_holdings.sum(1),0); wealth[:,m]=total; gross_wealth[:,m]=gross_total; net_returns[:,m-1]=(total-cash)/prev_total-1; peaks=np.maximum(peaks,total); dd=total/np.maximum(peaks,1e-12)-1; mdd=np.minimum(mdd,dd); under=dd<-1e-10; uw=np.where(under,uw+1,0); uwmax=np.maximum(uwmax,uw)
    terminal_pre_liquidation=wealth[:,-1].astype(float); terminal_gross=gross_wealth[:,-1].astype(float); final_tax=np.zeros(n)
    if taxation_enabled: gains=np.maximum(holdings-basis,0); final_tax=(gains*tax_rates[None,:]).sum(1)
    terminal_net=np.maximum(terminal_pre_liquidation-final_tax,0); tax_paid+=final_tax; real_terminal_net=terminal_net/((1+float(inflation_rate))**int(years)); real_paths=wealth.astype(float)/((1+float(inflation_rate))**(np.arange(months+1)/12))[None,:]
    invested=total_contributed; flat=net_returns.ravel(); q95,q99=np.quantile(flat,[.05,.01]); c95=flat[flat<=q95]; c99=flat[flat<=q99]; meanm=float(flat.mean()); stdm=float(flat.std(ddof=1)); ann=(1+meanm)**12-1; vol=stdm*sqrt(12); downside=np.minimum(flat,0); ddev=float(np.sqrt(np.mean(downside**2))*sqrt(12)); sharpe=(ann-risk_free_rate)/vol if vol>1e-12 else 0.; sortino=(ann-risk_free_rate)/ddev if ddev>1e-12 else 0.; tr=terminal_net/np.maximum(invested,1e-12)-1; cut=float(np.quantile(tr,1-float(confidence))); tail=tr[tr<=cut]; loss10=np.mean(terminal_net<invested*.90); loss20=np.mean(terminal_net<invested*.80); tail5=terminal_net[terminal_net<=np.quantile(terminal_net,.05)]
    return {"paths":wealth,"gross_paths":gross_wealth,"real_paths":real_paths.astype(np.float32),"simulated_returns":net_returns.astype(np.float32),"terminal":terminal_net,"terminal_net":terminal_net,"terminal_pre_liquidation":terminal_pre_liquidation,"terminal_gross":terminal_gross,"real_terminal":real_terminal_net,"median_terminal":float(np.median(terminal_net)),"median_terminal_gross":float(np.median(terminal_gross)),"median_terminal_pre_liquidation":float(np.median(terminal_pre_liquidation)),"mean_terminal":float(np.mean(terminal_net)),"p5_terminal":float(np.quantile(terminal_net,.05)),"p95_terminal":float(np.quantile(terminal_net,.95)),"median_real_terminal":float(np.median(real_terminal_net)),"terminal_returns":tr,"var":max(0,-cut),"cvar":max(0,-float(tail.mean())) if tail.size else 0.,"max_drawdown_median":float(np.median(mdd)),"max_drawdown_p95":float(np.quantile(mdd,.05)),"underwater_days_median":float(np.median(uwmax)*21),"underwater_days_p95":float(np.quantile(uwmax,.95)*21),"probability_loss":float(np.mean(terminal_net<invested)),"probability_loss_10":float(loss10),"probability_loss_20":float(loss20),"probability_ruin":float(np.mean(terminal_net<invested*float(ruin_threshold))),"tail5_terminal_mean":float(np.mean(tail5)) if tail5.size else float(np.min(terminal_net)),"var95_monthly":max(0,-float(q95)),"var99_monthly":max(0,-float(q99)),"cvar95_monthly":max(0,-float(c95.mean())) if c95.size else 0.,"cvar99_monthly":max(0,-float(c99.mean())) if c99.size else 0.,"sharpe":float(sharpe),"sortino":float(sortino),"annualized_return":float(ann),"annualized_volatility":float(vol),"total_invested":float(np.mean(invested)),"taxes_paid_mean":float(np.mean(tax_paid)),"fees_paid_mean":float(np.mean(fees_paid)),"hedging":{"premium_paid":premium_paid,"payoff_received":payoff_received,"moneyness":put_moneyness,"maturity_months":put_maturity_months,"implied_volatility":put_implied_volatility,"hedge_fraction":put_hedge_fraction},"warnings":warnings,"stress_details":stress_details,"weights":w,"assets":[asset_map[t] for t in usable]}

def probability_of_goal(result,target_value,real=False):
    values=np.asarray(result["real_terminal"] if real else result["terminal_net"],float); return float(np.mean(values>=float(target_value)))

def simulate_decumulation(monthly_returns,initial_capital,annual_withdrawal,years,inflation_rate=.02,withdrawal_growth_with_inflation=True):
    r=np.asarray(monthly_returns,float); months=min(int(years)*12,r.shape[1]); wealth=np.full(r.shape[0],float(initial_capital)); paths=np.empty((r.shape[0],months+1),np.float32); paths[:,0]=wealth; exhausted=np.zeros(r.shape[0],bool); exhaustion_month=np.full(r.shape[0],np.nan); base_monthly=float(annual_withdrawal)/12
    for m in range(1,months+1):
        growth=(1+float(inflation_rate))**((m-1)/12) if withdrawal_growth_with_inflation else 1.; withdrawal=base_monthly*growth; wealth=np.maximum(wealth-withdrawal,0); newly=(~exhausted)&(wealth<=0); exhaustion_month[newly]=m; exhausted|=newly; wealth*=1+r[:,m-1]; wealth=np.maximum(wealth,0); paths[:,m]=wealth
    terminal=wealth
    return {"paths":paths,"terminal":terminal,"survival_probability":float(np.mean(terminal>0)),"depletion_probability":float(np.mean(terminal<=0)),"median_terminal":float(np.median(terminal)),"p5_terminal":float(np.quantile(terminal,.05)),"median_exhaustion_year":float(np.nanmedian(exhaustion_month)/12) if np.any(~np.isnan(exhaustion_month)) else None}