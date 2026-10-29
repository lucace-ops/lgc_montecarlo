"""Yahoo Finance retrieval with EUR-normalized monthly total-return series."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import pandas as pd
import yfinance as yf

@dataclass
class MarketHistory:
    asset_returns: pd.DataFrame
    currencies: dict[str, str]
    warnings: list[str]

class MarketDataError(ValueError):
    pass

def _currency_for(ticker: yf.Ticker, symbol: str) -> str:
    currency: Any = None
    try: currency = ticker.get_info().get("currency")
    except Exception: pass
    if not currency:
        try: currency = ticker.fast_info.get("currency")
        except Exception: pass
    if not currency:
        raise MarketDataError(f"Yahoo Finance non ha restituito la valuta per {symbol}.")
    return str(currency)

def _history(symbol: str, start: str = "1970-01-01", adjusted_required: bool = True) -> pd.Series:
    try:
        frame = yf.Ticker(symbol).history(start=start, auto_adjust=False, actions=False, raise_errors=True)
    except Exception as exc:
        raise MarketDataError(f"Errore nel download di {symbol}: {exc}") from exc
    if frame.empty: raise MarketDataError(f"Nessun dato storico disponibile per {symbol}.")
    col = "Adj Close" if adjusted_required else "Close"
    if col not in frame.columns: raise MarketDataError(f"Yahoo Finance non fornisce {col} per {symbol}.")
    s = frame[col].dropna()
    if s.empty: raise MarketDataError(f"{col} non contiene valori validi per {symbol}.")
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s.sort_index().astype(float)

def _fx_history(currency: str) -> tuple[pd.Series, str, bool]:
    currency = currency.upper()
    if currency == "EUR": return pd.Series(dtype=float), "EUR", False
    errors=[]
    for pair, inverse in ((f"{currency}EUR=X", False), (f"EUR{currency}=X", True)):
        try: return _history(pair, adjusted_required=False), pair, inverse
        except Exception as exc: errors.append(str(exc))
    raise MarketDataError(f"Impossibile convertire {currency} in EUR. {'; '.join(errors)}")

def _eur_monthly_returns(symbol: str, start: str = "1970-01-01") -> tuple[pd.Series, str, list[str]]:
    ticker = yf.Ticker(symbol)
    raw = _history(symbol, start=start)
    currency = _currency_for(ticker, symbol)
    warnings=[]
    scale = 0.01 if currency.strip() in {"GBp", "GBX"} else 1.0
    fx_currency = "GBP" if scale == 0.01 else currency.strip().upper()
    if scale == 0.01: warnings.append(f"{symbol}: quotazione in pence convertita in GBP.")
    fx, pair, inverse = _fx_history(fx_currency)
    dates = pd.DatetimeIndex(raw.index)
    if fx_currency == "EUR": rates = pd.Series(1.0, index=dates)
    else:
        rates = fx.reindex(fx.index.union(dates)).sort_index().ffill().reindex(dates)
        if inverse: rates = 1.0 / rates
        if rates.isna().all(): raise MarketDataError(f"Cambio {fx_currency}/EUR non disponibile per {symbol}.")
        if rates.isna().any():
            first_valid=rates.first_valid_index()
            warnings.append(f"{symbol}: conversione EUR disponibile da {first_valid:%Y-%m}; mesi precedenti esclusi.")
        warnings.append(f"{symbol}: cambio {fx_currency}/EUR tramite {pair}.")
    eur_prices = raw * scale * rates
    monthly = eur_prices.resample("ME").last().pct_change(fill_method=None).dropna()
    monthly.name = symbol.upper()
    return monthly.astype(float), currency, warnings

def load_monthly_eur_return_series(symbol: str, start: str = "1970-01-01") -> pd.Series:
    series, _, _ = _eur_monthly_returns(symbol.strip().upper(), start=start)
    return series

def load_monthly_eur_returns(symbols: list[str], start: str = "1970-01-01") -> MarketHistory:
    normalized = list(dict.fromkeys(s.strip().upper() for s in symbols if s.strip()))
    if not normalized: raise MarketDataError("Inserisci almeno un ticker.")
    series_map={}; currencies={}; warnings=[]
    for symbol in normalized:
        s,c,w = _eur_monthly_returns(symbol,start=start)
        series_map[symbol]=s; currencies[symbol]=c; warnings.extend(w)
    frame = pd.concat(series_map,axis=1,join="inner").dropna(how="any")
    if len(frame)<12: raise MarketDataError("Dopo allineamento e conversione EUR restano meno di 12 mesi comuni.")
    frame.index.name="Mese"
    return MarketHistory(frame.astype(float), currencies, list(dict.fromkeys(warnings)))