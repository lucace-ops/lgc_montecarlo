"""Core calculations for the L & G Consulting Monte Carlo portfolio simulator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd

# Seven historically meaningful stress episodes. Shocks are illustrative portfolio-level
# approximations, not a claim that every ETF experienced these exact returns.
BLACK_SWAN_EVENTS = {
    "Lunedì nero (1987)": {"date": "1987-10-19", "equity": -0.22, "bonds": 0.01, "gold": 0.04, "other": -0.12},
    "Crisi asiatica / LTCM (1998)": {"date": "1998-08-31", "equity": -0.12, "bonds": 0.03, "gold": -0.02, "other": -0.08},
    "Bolla dot-com (2000-2002)": {"date": "2001-09-21", "equity": -0.18, "bonds": 0.04, "gold": 0.03, "other": -0.12},
    "Crisi finanziaria globale (2008)": {"date": "2008-10-10", "equity": -0.20, "bonds": 0.02, "gold": -0.05, "other": -0.15},
    "Crisi Euro / downgrade USA (2011)": {"date": "2011-08-08", "equity": -0.10, "bonds": 0.03, "gold": 0.05, "other": -0.08},
    "Sell-off COVID (2020)": {"date": "2020-03-16", "equity": -0.12, "bonds": -0.02, "gold": -0.03, "other": -0.10},
    "Shock inflazione / tassi (2022)": {"date": "2022-06-13", "equity": -0.05, "bonds": -0.04, "gold": -0.02, "other": -0.05},
}

ASSET_CLASS_HINTS = {
    "equity": ("equity", "azion", "world", "sp500", "s&p", "nasdaq", "msci", "stock"),
    "bonds": ("bond", "obblig", "treasury", "govt", "aggregate", "agg", "gilt"),
    "gold": ("gold", "oro", "sgld", "4gl"),
}

def classify_asset(asset: dict[str, Any]) -> str:
    explicit = str(asset.get("asset_class", "")).lower()
    ticker = str(asset.get("ticker", "")).lower()
    haystack = explicit + " " + ticker
    for cls, hints in ASSET_CLASS_HINTS.items():
        if any(h in haystack for h in hints):
            return cls
    return "other"

def download_prices(tickers: list[str], period: str = "10y") -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance via yfinance."""
    import yfinance as yf
    cleaned = list(dict.fromkeys(t.strip().upper() for t in tickers if t.strip()))
    if not cleaned:
        raise ValueError("Inserire almeno un ticker valido.")
    raw = yf.download(
        tickers=cleaned, period=period, auto_adjust=True, progress=False,
        group_by="ticker", threads=True
    )
    if raw.empty:
        raise ValueError("Yahoo Finance non ha restituito dati. Verifica ticker e connessione.")
    if isinstance(raw.columns, pd.MultiIndex):
        frames = {}
        for ticker in cleaned:
            if ticker in raw.columns.get_level_values(0):
                block = raw[ticker]
                col = "Close" if "Close" in block.columns else block.columns[0]
                frames[ticker] = block[col]
            elif ticker in raw.columns.get_level_values(-1):
                block = raw.xs(ticker, axis=1, level=-1)
                col = "Close" if "Close" in block.columns else block.columns[0]
                frames[ticker] = block[col]
        prices = pd.DataFrame(frames)
    else:
        col = "Close" if "Close" in raw.columns else raw.columns[0]
        prices = raw[[col]].rename(columns={col: cleaned[0]})
    prices = prices.sort_index().apply(pd.to_numeric, errors="coerce")
    prices = prices.dropna(axis=1, how="all").ffill().dropna(how="all")
    if prices.empty or prices.shape[1] == 0:
        raise ValueError("Non è stato possibile estrarre prezzi validi.")
    return prices

def historical_returns(prices: pd.DataFrame, years: int = 10) -> pd.DataFrame:
    if prices.empty:
        raise ValueError("La serie dei prezzi è vuota.")
    cutoff = prices.index.max() - pd.Timedelta(days=int(years * 365.25))
    selected = prices.loc[prices.index >= cutoff]
    returns = selected.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")
    returns = returns.dropna(axis=1, how="all")
    if len(returns) < 60:
        raise ValueError("Dati insufficienti: servono almeno circa 60 osservazioni giornaliere.")
    return returns

def _portfolio_daily_returns(returns: pd.DataFrame, assets: list[dict], weights: np.ndarray) -> pd.Series:
    tickers = [a["ticker"] for a in assets]
    usable = [t for t in tickers if t in returns.columns]
    if not usable:
        raise ValueError("Nessuno dei ticker del portafoglio è presente nei dati scaricati.")
    w_map = {a["ticker"]: float(w) for a, w in zip(assets, weights)}
    w = np.array([w_map[t] for t in usable], dtype=float)
    w = w / w.sum()
    return returns[usable].dropna().dot(w)

def max_drawdown(path: np.ndarray) -> float:
    peaks = np.maximum.accumulate(path, axis=0)
    dd = path / np.maximum(peaks, 1e-12) - 1.0
    return float(np.nanmin(dd))

def underwater_duration(path: np.ndarray) -> int:
    """Longest number of simulated daily observations below a previous high."""
    peaks = np.maximum.accumulate(path)
    underwater = path < peaks * (1 - 1e-10)
    longest = current = 0
    for flag in underwater:
        if flag:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)

def _apply_event_shocks(daily_returns: np.ndarray, assets: list[dict], weights: np.ndarray,
                        rng: np.random.Generator, enabled: bool) -> np.ndarray:
    """Inject rare cross-asset event days using asset-class shock assumptions."""
    out = daily_returns.copy()
    if not enabled:
        return out
    # Event-day probability is calibrated to produce occasional stress years, not to
    # reproduce a historical chronology. Events are sampled with replacement.
    probability = min(0.035, 7 / 252)
    event_indices = np.flatnonzero(rng.random(out.shape[0]) < probability)
    events = list(BLACK_SWAN_EVENTS.values())
    for idx in event_indices:
        event = events[int(rng.integers(0, len(events)))]
        for j, asset in enumerate(assets):
            cls = classify_asset(asset)
            shock = event.get(cls, event["other"])
            # Use a modest jitter so repeated sampled events do not become identical.
            out[idx, j] = np.clip(shock + rng.normal(0, 0.015), -0.65, 0.35)
    return out

def simulate_portfolio(
    returns: pd.DataFrame,
    assets: list[dict],
    weights_pct: list[float],
    initial_value: float = 100_000,
    years: int = 10,
    n_sims: int = 5000,
    confidence: float = 0.95,
    inflation_rate: float = 0.02,
    rebalance_frequency: str = "Annuale",
    include_black_swans: bool = True,
    include_put_hedge: bool = False,
    put_annual_cost_pct: float = 0.75,
    put_trigger_drawdown_pct: float = 15.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap daily returns with cross-asset dependence preserved by row sampling.

    Put overlay is a transparent stylized payoff model, not option-chain pricing:
    annual premium is charged daily; after a drawdown trigger, an illustrative payout
    is applied once per calendar year. It must not be interpreted as a tradable quote.
    """
    tickers = [a["ticker"] for a in assets]
    available = [t for t in tickers if t in returns.columns]
    if not available:
        raise ValueError("I ticker del portafoglio non sono disponibili.")
    assets = [a for a in assets if a["ticker"] in available]
    weights = np.asarray(weights_pct[:len(tickers)], dtype=float)
    # Re-align weights in case a ticker was omitted from downloaded data.
    original_map = {ticker: float(w) for ticker, w in zip(tickers, weights_pct)}
    weights = np.array([original_map[a["ticker"]] for a in assets], dtype=float)
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("I pesi devono essere non negativi e avere somma positiva.")
    weights /= weights.sum()
    data = returns[[a["ticker"] for a in assets]].dropna().to_numpy(dtype=float)
    if data.shape[0] < 30:
        raise ValueError("Dati storici comuni insufficienti per simulare.")
    rng = np.random.default_rng(seed)
    steps = max(1, int(years * 252))
    paths = np.empty((n_sims, steps + 1), dtype=np.float64)
    paths[:, 0] = initial_value
    base_daily_cost = max(0.0, put_annual_cost_pct / 100) / 252 if include_put_hedge else 0.0
    rebalance_every = {"Mensile": 21, "Trimestrale": 63, "Annuale": 252}.get(rebalance_frequency, steps + 1)
    for s in range(n_sims):
        sampled_idx = rng.integers(0, len(data), size=steps)
        sampled = data[sampled_idx].copy()
        sampled = _apply_event_shocks(sampled, assets, weights, rng, include_black_swans)
        value = float(initial_value)
        holdings = value * weights.copy()
        peak = value
        put_paid_this_year = False
        for t in range(steps):
            holdings *= (1.0 + sampled[t])
            value = float(np.sum(holdings))
            value *= (1.0 - base_daily_cost)
            peak = max(peak, value)
            if include_put_hedge and not put_paid_this_year:
                drawdown = (value / peak - 1.0) * 100
                if drawdown <= -abs(put_trigger_drawdown_pct):
                    # Illustrative payout sized as 3% of initial capital. Not option valuation.
                    payout = 0.03 * initial_value
                    value += payout
                    holdings = holdings / max(np.sum(holdings), 1e-12) * value
                    put_paid_this_year = True
            if (t + 1) % 252 == 0:
                put_paid_this_year = False
            if (t + 1) % rebalance_every == 0:
                holdings = value * weights
            paths[s, t + 1] = value
    # Inflation-adjusted paths use deterministic CPI assumption.
    day_index = np.arange(steps + 1)
    real_paths = paths / ((1 + inflation_rate) ** (day_index / 252))
    terminal = paths[:, -1]
    real_terminal = real_paths[:, -1]
    terminal_returns = terminal / initial_value - 1
    q = 1 - confidence
    var_loss = max(0.0, -float(np.quantile(terminal_returns, q)))
    cvars = max(0.0, -float(terminal_returns[terminal_returns <= np.quantile(terminal_returns, q)].mean())) if np.any(terminal_returns <= np.quantile(terminal_returns, q)) else 0.0
    mdds = np.array([max_drawdown(p) for p in paths])
    uwds = np.array([underwater_duration(p) for p in paths])
    return {
        "paths": paths,
        "real_paths": real_paths,
        "terminal": terminal,
        "real_terminal": real_terminal,
        "terminal_returns": terminal_returns,
        "var": var_loss,
        "cvar": cvars,
        "median_terminal": float(np.median(terminal)),
        "mean_terminal": float(np.mean(terminal)),
        "p5_terminal": float(np.quantile(terminal, 0.05)),
        "p95_terminal": float(np.quantile(terminal, 0.95)),
        "median_real_terminal": float(np.median(real_terminal)),
        "median_return_annualized": float((np.median(terminal) / initial_value) ** (1 / years) - 1),
        "max_drawdown_median": float(np.median(mdds)),
        "max_drawdown_p95": float(np.quantile(mdds, 0.95)),
        "underwater_days_median": float(np.median(uwds)),
        "underwater_days_p95": float(np.quantile(uwds, 0.95)),
        "returns": returns,
        "weights": weights,
        "assets": assets,
    }
