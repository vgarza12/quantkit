"""Synthetic data generators (so everything runs with no data subscription) and CSV loaders."""
from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_returns(n=1000, ann_mu=0.10, ann_vol=0.15, ppy=252, seed=0, start="2020-01-01") -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    return pd.Series(rng.normal(ann_mu / ppy, ann_vol / np.sqrt(ppy), n), index=idx, name="strategy")


def synthetic_universe(n_strats=20, n=1000, n_real=3, ppy=252, seed=0) -> pd.DataFrame:
    """Strategy return streams sharing a market factor.

    Only the first `n_real` have a true edge; the rest are pure noise. A good
    evaluator should not be fooled by the lucky noise ones.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    factor = rng.normal(0, 0.008, n)
    cols = {}
    for i in range(n_strats):
        edge = 0.12 / ppy if i < n_real else 0.0
        noise = rng.normal(0, 0.007, n)
        cols[f"strat_{i:02d}"] = edge + 0.6 * factor + noise
    return pd.DataFrame(cols, index=idx)


def synthetic_fills(n=500, seed=0, avg_slip_bps=1.5, miss_rate=0.02) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-02 09:30", periods=n, freq="15min")
    symbols = rng.choice(["ES", "NQ", "EURUSD", "CL"], n)
    side = rng.choice(["BUY", "SELL"], n)
    expected = rng.uniform(50, 5000, n)
    sign = np.where(side == "BUY", 1, -1)
    slip = rng.normal(avg_slip_bps, 1.0, n) / 1e4
    fill = expected * (1 + sign * slip)
    fill[rng.random(n) < miss_rate] = np.nan
    latency = rng.gamma(2.0, 0.25, n)
    return pd.DataFrame({
        "ts": ts, "symbol": symbols, "side": side, "qty": rng.integers(1, 10, n),
        "expected_price": expected, "fill_price": fill,
        "signal_ts": ts, "fill_ts": ts + pd.to_timedelta(latency, unit="s"),
    })


def load_returns_csv(path: str, col: str | None = None) -> pd.Series:
    """CSV with a date column first and one or more return columns."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df[col] if col else df.iloc[:, 0]


def synthetic_prices(n_assets: int = 3, n: int = 2500, seed: int = 0) -> pd.DataFrame:
    """Offline demo prices: assets drift up/down in persistent regimes; the last one is pure noise."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    cols = {}
    for i in range(n_assets):
        noise_only = i == n_assets - 1
        regime, rets = 1, []
        for _ in range(n):
            if rng.random() > 0.99:
                regime = -regime
            mu = 0.0 if noise_only else (0.0007 if regime > 0 else -0.0005)
            rets.append(rng.normal(mu, 0.01))
        cols[f"DEMO_{'NOISE' if noise_only else chr(65 + i)}"] = 100 * np.cumprod(1 + np.array(rets))
    return pd.DataFrame(cols, index=idx)
