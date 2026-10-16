"""Simple, transparent strategies plus an honest way to choose their settings.

No look-ahead: a signal computed from today's close is only held starting
TOMORROW (`shift(1)`). Settings are chosen on the first `split` of history only;
the remainder is kept as untouched data to judge the choice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import execution as ex
from . import metrics as m


def sma_cross(p: pd.Series, fast: int, slow: int) -> pd.Series:
    return (p.rolling(fast).mean() > p.rolling(slow).mean()).astype(float)


def momentum(p: pd.Series, lookback: int) -> pd.Series:
    return (p.pct_change(lookback) > 0).astype(float)


def mean_reversion(p: pd.Series, window: int, z: float) -> pd.Series:
    zs = (p - p.rolling(window).mean()) / p.rolling(window).std()
    return (zs < -z).astype(float)


STRATEGIES = {
    "sma": {
        "label": "Trend following (moving-average crossover)",
        "blurb": "Holds the asset while its short-term average price is above its long-term average; otherwise sits in cash.",
        "fn": sma_cross, "warmup": lambda p: p["slow"],
        "grid": [{"fast": f, "slow": s} for f in (10, 20, 50) for s in (100, 150, 200)],
    },
    "momentum": {
        "label": "Momentum",
        "blurb": "Holds the asset while it has risen over the past N days; otherwise sits in cash.",
        "fn": momentum, "warmup": lambda p: p["lookback"],
        "grid": [{"lookback": n} for n in (20, 60, 120, 252)],
    },
    "meanrev": {
        "label": "Mean reversion (buy dips)",
        "blurb": "Buys when the price falls unusually far below its recent average and exits otherwise.",
        "fn": mean_reversion, "warmup": lambda p: p["window"],
        "grid": [{"window": w, "z": z} for w in (10, 20, 50) for z in (1.0, 1.5, 2.0)],
    },
}


def backtest(prices: pd.Series, name: str, params: dict, cost_bps: float = 2.0):
    """Full-length (gross, net, position) series. Position is what was HELD that day."""
    sig = STRATEGIES[name]["fn"](prices, **params)
    pos = sig.shift(1).fillna(0.0)
    gross = (pos * prices.pct_change()).fillna(0.0)
    net = ex.apply_costs(gross, pos, cost_bps)
    return gross, net, pos


def run_strategy(prices: pd.Series, name: str, cost_bps: float = 2.0, split: float = 0.7) -> dict:
    """Try every setting in the grid, pick the best on the first `split` of history only."""
    spec = STRATEGIES[name]
    warm = max(spec["warmup"](p) for p in spec["grid"])
    if len(prices) < warm + 300:
        raise ValueError(f"Need at least {warm + 300} price points; got {len(prices)}.")
    trials, runs = {}, {}
    for params in spec["grid"]:
        g, n, pos = backtest(prices, name, params, cost_bps)
        key = ", ".join(f"{k}={v}" for k, v in params.items())
        trials[key] = n.iloc[warm:]
        runs[key] = (params, g.iloc[warm:], n.iloc[warm:], pos.iloc[warm:])
    T = pd.DataFrame(trials)
    cut = int(len(T) * split)
    best = max(trials, key=lambda k: m.sharpe(trials[k].iloc[:cut]))
    params, gross, net, pos = runs[best]
    return {"name": name, "best_key": best, "best_params": params, "gross": gross, "net": net, "pos": pos,
            "trials": T, "cut": cut, "n_trials": len(trials)}
