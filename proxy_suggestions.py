"""Conservative proxy suggestions for historical stress reconstruction.

Suggestions are never applied silently: the UI asks the user to choose/confirm one.
"""
from __future__ import annotations
import re

_SUGGESTIONS = {
    "VWCE.DE": (("ACWI", "Azionario globale"), ("VT", "Azionario globale"), ("^GSPC", "Azionario USA - proxy di ultima istanza")),
    "VWCE": (("ACWI", "Azionario globale"), ("VT", "Azionario globale"), ("^GSPC", "Azionario USA - proxy di ultima istanza")),
    "IWDA.AS": (("ACWI", "Azionario globale"), ("VEA", "Sviluppati"), ("^GSPC", "Azionario USA - proxy di ultima istanza")),
    "SPY": (("^GSPC", "S&P 500 index"), ("VOO", "S&P 500"), ("IVV", "S&P 500")),
    "VOO": (("^GSPC", "S&P 500 index"), ("SPY", "S&P 500"), ("IVV", "S&P 500")),
    "IVV": (("^GSPC", "S&P 500 index"), ("SPY", "S&P 500")),
    "QQQ": (("^IXIC", "Nasdaq Composite"), ("QQQM", "Nasdaq-100")),
    "VTI": (("^GSPC", "Azionario USA - proxy ampio"), ("ITOT", "Azionario USA totale")),
    "EEM": (("VWO", "Mercati emergenti"), ("IEMG", "Mercati emergenti")),
    "VWO": (("EEM", "Mercati emergenti"), ("IEMG", "Mercati emergenti")),
    "IEMG": (("EEM", "Mercati emergenti"), ("VWO", "Mercati emergenti")),
    "AGGH.L": (("AGG", "Obbligazionario aggregate USA"), ("BND", "Obbligazionario aggregate USA")),
    "AGGH.DE": (("AGG", "Obbligazionario aggregate USA"), ("BND", "Obbligazionario aggregate USA")),
    "AGG": (("BND", "Obbligazionario aggregate USA"),),
    "BND": (("AGG", "Obbligazionario aggregate USA"),),
    "BNDX": (("IAGG", "Obbligazionario globale ex-USA"), ("BNDW", "Obbligazionario globale")),
    "TLT": (("IEF", "Treasury USA"), ("EDV", "Treasury USA lunga duration")),
    "IEF": (("TLT", "Treasury USA lunga duration"), ("VGIT", "Treasury USA intermedio")),
    "TIP": (("SCHP", "TIPS USA"),),
    "SGLD.MI": (("GLD", "Oro"), ("IAU", "Oro")),
    "SGLD.L": (("GLD", "Oro"), ("IAU", "Oro")),
    "GLD": (("IAU", "Oro"),),
    "IAU": (("GLD", "Oro"),),
    "XEON.DE": (("BIL", "Treasury bills USA - proxy liquidità"), ("SHV", "Treasury USA breve")),
}

_CLASS_FALLBACKS = {
    "Azionario globale": (("ACWI", "Azionario globale"), ("VT", "Azionario globale"), ("^GSPC", "Proxy azionario lungo")),
    "Azionario USA": (("^GSPC", "S&P 500 index"), ("SPY", "S&P 500")),
    "Azionario sviluppati": (("VEA", "Sviluppati ex-USA"), ("ACWI", "Azionario globale")),
    "Emergenti": (("EEM", "Mercati emergenti"), ("VWO", "Mercati emergenti")),
    "Obbligazionario globale": (("AGG", "Aggregate USA"), ("BND", "Aggregate USA")),
    "Obbligazionario governativo": (("IEF", "Treasury USA intermedio"), ("TLT", "Treasury USA lungo")),
    "Obbligazionario inflation-linked": (("TIP", "TIPS USA"), ("SCHP", "TIPS USA")),
    "Oro": (("GLD", "Oro"), ("IAU", "Oro")),
    "Liquidità": (("BIL", "Treasury bills USA"), ("SHV", "Treasury USA breve")),
}

def suggested_proxies(ticker: str, asset_class: str | None = None):
    normalized = ticker.strip().upper()
    combined = list(_SUGGESTIONS.get(normalized, ()))
    combined.extend(_CLASS_FALLBACKS.get(str(asset_class or ""), ()))
    seen = set()
    out = []
    for symbol, reason in combined:
        if symbol != normalized and symbol not in seen:
            seen.add(symbol); out.append((symbol, reason))
    return tuple(out)

def normalize_proxy_ticker(value: str) -> str:
    normalized = value.strip().upper()
    if not re.fullmatch(r"[A-Z0-9^=._-]{1,24}", normalized):
        raise ValueError("Inserisci un ticker Yahoo Finance valido.")
    return normalized