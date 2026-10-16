"""Core performance and risk metrics.

Every function takes periodic *simple* returns (pandas Series or array-like).
`ppy` is periods per year (252 for daily).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _clean(r) -> pd.Series:
    return pd.Series(r).dropna().astype(float)


def equity_curve(r) -> pd.Series:
    return (1 + _clean(r)).cumprod()


def drawdown_series(r) -> pd.Series:
    """Drawdown from the running peak. The starting equity of 1.0 counts as a peak."""
    eq = equity_curve(r)
    peak = eq.cummax().clip(lower=1.0)
    return eq / peak - 1


def max_drawdown(r) -> float:
    dd = drawdown_series(r)
    return float(dd.min()) if len(dd) else float("nan")


def longest_drawdown(r) -> int:
    """Longest run of periods spent below a prior equity high (recovery time)."""
    longest = cur = 0
    for x in drawdown_series(r).to_numpy():
        cur = cur + 1 if x < 0 else 0
        longest = max(longest, cur)
    return longest


def ann_return(r, ppy: int = 252) -> float:
    r = _clean(r)
    if len(r) == 0:
        return float("nan")
    return float((1 + r).prod() ** (ppy / len(r)) - 1)


def ann_vol(r, ppy: int = 252) -> float:
    return float(_clean(r).std(ddof=1) * np.sqrt(ppy))


def sharpe(r, ppy: int = 252, rf: float = 0.0) -> float:
    r = _clean(r) - rf / ppy
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(ppy)) if sd and sd > 0 else float("nan")


def sortino(r, ppy: int = 252, rf: float = 0.0) -> float:
    r = _clean(r) - rf / ppy
    downside = float(np.sqrt(np.mean(np.minimum(r, 0) ** 2)))
    return float(r.mean() / downside * np.sqrt(ppy)) if downside > 0 else float("nan")


def calmar(r, ppy: int = 252) -> float:
    mdd = abs(max_drawdown(r))
    return float(ann_return(r, ppy) / mdd) if mdd > 0 else float("nan")


def summary(r, ppy: int = 252) -> dict:
    return {
        "ann_return": ann_return(r, ppy),
        "ann_vol": ann_vol(r, ppy),
        "sharpe": sharpe(r, ppy),
        "sortino": sortino(r, ppy),
        "max_drawdown": max_drawdown(r),
        "longest_drawdown_periods": longest_drawdown(r),
        "calmar": calmar(r, ppy),
        "n": int(_clean(r).shape[0]),
    }
