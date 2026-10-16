"""Project 2: Live vs. backtest decay analyzer.

Question answered: "Is this strategy's live performance still consistent with
its backtest, or is it just a bad patch / real decay?"

Two complementary tests:
1. Block-bootstrap: resample live-length windows from the backtest and see
   what percentile the observed live Sharpe falls in.
2. CUSUM: a sequential detector that flags the first time live returns have
   drifted persistently below the backtest mean.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics as m


def bootstrap_sharpe_dist(backtest, length: int, ppy: int = 252, n_boot: int = 2000,
                          block: int = 10, seed: int = 0) -> np.ndarray:
    """Distribution of annualized Sharpe over `length`-period windows, drawn from the backtest."""
    x = pd.Series(backtest).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    block = max(1, min(block, len(x)))
    n_blocks = int(np.ceil(length / block))
    starts = rng.integers(0, len(x) - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :length]
    samples = x[idx]
    sd = samples.std(axis=1, ddof=1)
    return np.where(sd > 0, samples.mean(axis=1) / sd * np.sqrt(ppy), np.nan)


def cusum_alarm(live, bt_mean: float, bt_std: float, k: float = 0.5, h: float = 5.0):
    """One-sided (downward) CUSUM on standardized live returns.

    Returns the index position of the first alarm, or None.
    """
    z = (pd.Series(live).dropna().to_numpy() - bt_mean) / bt_std
    s = 0.0
    for i, zi in enumerate(z):
        s = min(0.0, s + zi + k)
        if s < -h:
            return i
    return None


def compare(backtest, live, ppy: int = 252, n_boot: int = 2000, block: int = 10, seed: int = 0) -> dict:
    bt = pd.Series(backtest).dropna()
    lv = pd.Series(live).dropna()
    live_sr = m.sharpe(lv, ppy)
    dist = bootstrap_sharpe_dist(bt, len(lv), ppy, n_boot, block, seed)
    pct = float(np.nanmean(dist <= live_sr))
    alarm = cusum_alarm(lv, bt.mean(), bt.std(ddof=1))
    if pct < 0.05 or alarm is not None:
        verdict = "decayed"
    elif pct < 0.15:
        verdict = "watch"
    else:
        verdict = "consistent"
    return {
        "backtest_sharpe": m.sharpe(bt, ppy), "live_sharpe": live_sr,
        "live_percentile_vs_backtest": pct,
        "cusum_alarm_index": alarm,
        "cusum_alarm_date": (lv.index[alarm] if alarm is not None and hasattr(lv.index, "__getitem__") else None),
        "verdict": verdict,
        "caveat": "Short live windows are noisy; treat 'watch' as a prompt to look, not to act.",
    }
