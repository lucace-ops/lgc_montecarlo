"""Advanced monthly Monte Carlo engine with historical stresses and theoretical put overlay."""
from __future__ import annotations
from math import sqrt
from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import norm

HISTORICAL_SHOCKS: dict[str, tuple[str, str]] = {
    "Shock petrolifero / stagflazione": ("1973-10-01", "1974-03-31"),
    "Lunedì Nero": ("1987-10-01", "1987-12-31"),
    "Bolla Dot-Com": ("2000-03-01", "2002-10-31"),
    "Crisi Subprime / Lehman Brothers": ("2007-10-01", "2009-03-31"),
    "Crisi del debito sovrano Eurozona": ("2010-05-01", "2012-07-31"),
    "Crash COVID-19": ("2020-02-01", "2020-04-30"),
    "Shock inflazionistico e bond crash": ("2022-01-01", "2022-12-31"),
}

BLACK_SWAN_EVENTS = {name: {"date": start, "start": start, "end": end} for name, (start, end) in HISTORICAL_SHOCKS.items()}

def _draws(history: np.ndarray, n_sims: int, months: int, method: str, distribution: str, degrees_of_freedom: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    if history.shape[0] < 2:
        raise ValueError("Servono almeno due mesi di rendimenti storici.")
    if method == "Bootstrap storico":
        block = max(1, min(int(block_length), months))
        out = np.empty((n_sims, months, history.shape[1]), dtype=float)
        for start in range(0, months, block):
            width = min(block, months - start)
            starts = rng.integers(0, history.shape[0], size=n_sims)
            ids = (starts[:, None] + np.arange(width)[None, :]) % history.shape[0]
            out[:, start:start + width, :] = history[ids]
        return out
    mu = np.mean(history, axis=0)
    cov = np.atleast_2d(np.cov(history, rowvar=False, ddof=1))
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    cov += np.eye(cov.shape[0]) * 1e-12
    draws = rng.multivariate_normal(mu, cov, size=(n_sims, months), check_valid="warn")
    if distribution == "Student-t multivariata":
        df = max(3, int(degrees_of_freedom))
        chi = rng.chisquare(df, size=(n_sims, months, 1))
        draws = mu + (draws - mu) * np.sqrt((df - 2) / chi)
    return draws

def compose_historical_event_returns(event_name: str, weights: dict[str, float], asset_histories: dict[str, pd.Series], proxy_mappings: dict[str, str] | None = None, proxy_histories: dict[str, pd.Series] | None = None):
    start, end = HISTORICAL_SHOCKS[event_name]
    start_date, end_date = pd.Timestamp(start), pd.Timestamp(end)
    proxy_mappings = proxy_mappings or {}
    proxy_histories = proxy_histories or {}
    components = {}
    used_proxies=[]
    missing=[]
    for ticker, weight in weights.items():
        direct = asset_histories.get(ticker)
        direct_window = direct.loc[(direct.index >= start_date) & (direct.index <= end_date)].dropna() if direct is not None else pd.Series(dtype=float)
        if len(direct_window) >= 2:
            components[ticker] = direct_window.astype(float) * float(weight)
            continue
        proxy = proxy_mappings.get(ticker)
        ps = proxy_histories.get(proxy) if proxy else None
        proxy_window = ps.loc[(ps.index >= start_date) & (ps.index <= end_date)].dropna() if ps is not None else pd.Series(dtype=float)
        if len(proxy_window) >= 2:
            components[ticker] = proxy_window.astype(float) * float(weight)
            used_proxies.append((ticker, proxy))
        else:
            missing.append(ticker)
    if missing or not components:
        return None, used_proxies, missing
    aligned = pd.concat(components, axis=1, join="inner").dropna(how="any")
    if len(aligned) < 2:
        return None, used_proxies, list(weights)
    result = aligned.sum(axis=1).astype(float)
    result.name = "rendimento_portafoglio"
    return result, used_proxies, []

def _apply_historical_events(simulated: np.ndarray, events: list[dict[str, Any]], weights: dict[str, float], asset_histories: dict[str, pd.Series], proxy_mappings: dict[str, str], proxy_histories: dict[str, pd.Series]):
    warnings=[]
    details=[]
    horizon = simulated.shape[1]
    next_start = 1
    for event in events:
        name = str(event["name"])
        if name not in HISTORICAL_SHOCKS:
            warnings.append(f"{name}: evento storico sconosciuto, ignorato.")
            continue
        start, end = HISTORICAL_SHOCKS[name]
        full_months = (pd.Timestamp(end).year-pd.Timestamp(start).year)*12 + pd.Timestamp(end).month-pd.Timestamp(start).month + 1
        start_month = int(event.get("start_month", next_start))
        if start_month > horizon:
            warnings.append(f"{name}: fuori dall'orizzonte simulato.")
            next_start = start_month + full_months
            continue
        event_returns, used, missing = compose_historical_event_returns(name, weights, asset_histories, proxy_mappings, proxy_histories)
        if event_returns is None:
            if missing:
                warnings.append(f"{name}: non applicato; storico/proxy insufficiente per {', '.join(missing)}.")
            else:
                warnings.append(f"{name}: non applicato; mesi comuni insufficienti nella finestra storica.")
            next_start = start_month + full_months
            continue
        vals = event_returns.to_numpy(dtype=float)
        duration = min(len(vals), horizon-start_month+1)
        if duration <= 0:
            continue
        left = start_month-1
        simulated[:, left:left+duration] = vals[:duration][None, :]
        if used:
            proxy_txt = ", ".join(f"{proxy} per {asset}" for asset, proxy in used)
            warnings.append(f"PROXY_RAPPRESENTATIVO: {name}: {proxy_txt}.")
        warnings.append(f"{name}: sequenza storica applicata ai mesi {start_month}-{start_month+duration-1}.")
        details.append({"name":name,"start_month":start_month,"duration":duration,"used_proxies":used,"historical_start":str(event_returns.index.min().date()),"historical_end":str(event_returns.index.max().date())})
        next_start = start_month + full_months
    return warnings, details

def _apply_custom_shock(simulated: np.ndarray, shock: dict[str, Any] | None, mean_return: float):
    if not shock: return []
    start=max(1,int(shock.get("start_month",1))); duration=max(1,int(shock.get("duration",1)))
    end=min(simulated.shape[1],start+duration-1)
    if start>simulated.shape[1]: return ["Shock personalizzato fuori dall'orizzonte."]
    width=end-start+1; impact=float(shock.get("impact",0)); vol=max(0,float(shock.get("vol_multiplier",1)))
    segment=simulated[:,start-1:end]
    segment[:] = mean_return + (segment-mean_return)*vol
    segment[:] = (1+segment)*((1+impact)**(1/width))-1
    np.clip(segment,-0.999,5,out=segment)
    return [f"Shock personalizzato applicato ai mesi {start}-{end}: impatto {impact:.1%}, volatilità x{vol:.2f}."]

def _put_price(spot, strike, t, r, sigma):
    spot=np.maximum(np.asarray(spot,float),1e-12); t=max(float(t),1e-8); sigma=max(float(sigma),1e-8)
    d1=(np.log(spot/strike)+(r+0.5*sigma*sigma)*t)/(sigma*np.sqrt(t)); d2=d1-sigma*np.sqrt(t)
    return strike*np.exp(-r*t)*norm.cdf(-d2)-spot*norm.cdf(-d1)

def simulate_portfolio(returns: pd.DataFrame, assets, weights_pct, initial_value=100000, years=10, n_sims=3000, confidence=.95, inflation_rate=.02, rebalance_frequency="Annuale", include_black_swans=False, include_put_hedge=False, put_annual_cost_pct=.75, put_trigger_drawdown_pct=15, seed=42, contribution=0.0, contribution_frequency="Mensile", method="Bootstrap storico", distribution="Student-t multivariata", degrees_of_freedom=6, block_length=3, risk_free_rate=.025, ruin_threshold=.5, put_moneyness=.20, put_maturity_months=12, put_implied_volatility=.25, put_hedge_fraction=1.0, custom_shock=None, historical_events=None, historical_asset_returns=None, historical_proxies=None, proxy_histories=None):
    tickers=[str(a["ticker"]).strip().upper() for a in assets]
    weight_map={t:float(w) for t,w in zip(tickers,weights_pct)}
    usable=[t for t in tickers if t in returns.columns]
    if not usable: raise ValueError("Nessun ticker disponibile nei dati mensili in EUR.")
    w=np.array([weight_map[t] for t in usable],float)
    if np.any(w<0) or w.sum()<=0: raise ValueError("I pesi devono essere non negativi e con somma positiva.")
    w/=w.sum()
    monthly=returns[usable].dropna(how="any")
    if len(monthly)<12: raise ValueError("Servono almeno 12 mesi comuni di rendimenti in EUR.")
    months=max(1,int(years))*12; n=max(1,int(n_sims)); rng=np.random.default_rng(int(seed))
    draws=_draws(monthly.to_numpy(float),n,months,method,distribution,degrees_of_freedom,block_length,rng)
    pr=np.tensordot(draws,w,axes=([-1],[0])); np.clip(pr,-.999,5,out=pr)
    warnings=[]; stress_details=[]
    historical_events=historical_events or []
    if historical_events:
        hw={t:float(wi) for t,wi in zip(usable,w)}
        ww, stress_details = _apply_historical_events(pr,historical_events,hw,historical_asset_returns or {},historical_proxies or {},proxy_histories or {})
        warnings.extend(ww)
    warnings.extend(_apply_custom_shock(pr,custom_shock,float(monthly.to_numpy().dot(w).mean())))

    wealth=np.empty((n,months+1),dtype=np.float32); wealth[:,0]=initial_value
    holdings=np.full(n,float(initial_value)); peaks=holdings.copy(); mdd=np.zeros(n); uw=np.zeros(n,int); uwmax=np.zeros(n,int); net=np.empty((n,months),float)
    premium_paid=0.0; payoff_received=0.0; put_value=np.zeros(n); strike=np.zeros(n); units=np.zeros(n)
    expiry=max(1,int(put_maturity_months)); iv=max(.01,float(put_implied_volatility)); hedge=max(0,min(1,float(put_hedge_fraction)))
    if include_put_hedge:
        spot=holdings.copy(); strike=spot*(1-float(put_moneyness)); units=holdings*hedge/np.maximum(spot,1e-12)
        put_value=units*_put_price(spot,strike,expiry/12,risk_free_rate,iv); holdings-=put_value; premium_paid+=float(np.mean(put_value))
    for m in range(1,months+1):
        prev=np.maximum(holdings+put_value,1e-12)
        cash=float(contribution) if contribution_frequency=="Mensile" or m%12==0 else 0.0
        holdings+=cash; holdings*=1+pr[:,m-1]
        if include_put_hedge:
            spot=np.maximum(holdings,1e-12); remaining=expiry-((m-1)%expiry)
            if m%expiry==0:
                payoff=units*np.maximum(strike-spot,0); holdings+=payoff; payoff_received+=float(np.mean(payoff))
                strike=spot*(1-float(put_moneyness)); units=holdings*hedge/spot
                put_value=units*_put_price(spot,strike,expiry/12,risk_free_rate,iv); holdings-=put_value; premium_paid+=float(np.mean(put_value))
            else:
                put_value=units*_put_price(spot,strike,remaining/12,risk_free_rate,iv)
        total=np.maximum(holdings+put_value,0); wealth[:,m]=total
        net[:,m-1]=(total-cash)/prev-1
        peaks=np.maximum(peaks,total); dd=total/np.maximum(peaks,1e-12)-1; mdd=np.minimum(mdd,dd)
        under=dd<-1e-10; uw=np.where(under,uw+1,0); uwmax=np.maximum(uwmax,uw)
    real=wealth.astype(float)/((1+float(inflation_rate))**(np.arange(months+1)/12))[None,:]
    terminal=wealth[:,-1].astype(float); real_terminal=real[:,-1]
    annual_contrib = float(contribution)*(12 if contribution_frequency=="Mensile" else 1)
    invested=float(initial_value)+annual_contrib*int(years)
    flat=net.ravel(); q95,q99=np.quantile(flat,[.05,.01]); c95=flat[flat<=q95]; c99=flat[flat<=q99]
    meanm=float(flat.mean()); stdm=float(flat.std(ddof=1)); ann=(1+meanm)**12-1; vol=stdm*sqrt(12)
    downside=np.minimum(flat,0); ddev=float(np.sqrt(np.mean(downside**2))*sqrt(12))
    sharpe=(ann-risk_free_rate)/vol if vol>1e-12 else 0.0; sortino=(ann-risk_free_rate)/ddev if ddev>1e-12 else 0.0
    tr=terminal/max(invested,1e-12)-1; q=1-float(confidence); cut=float(np.quantile(tr,q)); tail=tr[tr<=cut]
    return {
        "paths":wealth,"real_paths":real.astype(np.float32),"terminal":terminal,"real_terminal":real_terminal,"terminal_returns":tr,
        "var":max(0,-cut),"cvar":max(0,-float(tail.mean())) if tail.size else 0.0,
        "median_terminal":float(np.median(terminal)),"mean_terminal":float(np.mean(terminal)),"p5_terminal":float(np.quantile(terminal,.05)),"p95_terminal":float(np.quantile(terminal,.95)),
        "median_real_terminal":float(np.median(real_terminal)),"max_drawdown_median":float(np.median(mdd)),"max_drawdown_p95":float(np.quantile(mdd,.05)),
        "underwater_days_median":float(np.median(uwmax)*21),"underwater_days_p95":float(np.quantile(uwmax,.95)*21),
        "probability_loss":float(np.mean(terminal<invested)),"probability_ruin":float(np.mean(terminal<invested*float(ruin_threshold))),
        "var95_monthly":max(0,-float(q95)),"var99_monthly":max(0,-float(q99)),"cvar95_monthly":max(0,-float(c95.mean())) if c95.size else 0.0,"cvar99_monthly":max(0,-float(c99.mean())) if c99.size else 0.0,
        "sharpe":float(sharpe),"sortino":float(sortino),"annualized_return":float(ann),"annualized_volatility":float(vol),"total_invested":invested,
        "hedging":{"premium_paid":premium_paid,"payoff_received":payoff_received,"moneyness":put_moneyness,"maturity_months":put_maturity_months,"implied_volatility":put_implied_volatility,"hedge_fraction":put_hedge_fraction},
        "warnings":warnings,"stress_details":stress_details,"weights":w,"assets":[a for a in assets if str(a["ticker"]).strip().upper() in usable]
    }