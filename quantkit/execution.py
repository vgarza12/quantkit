"""Project 4: Execution and slippage monitor.

Input: a fills log with columns
  ts, symbol, side (BUY/SELL), qty, expected_price, fill_price (NaN = missed),
  signal_ts, fill_ts
Slippage is signed so that POSITIVE = worse than expected (a cost).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics as m

REQUIRED = {"ts", "symbol", "side", "qty", "expected_price", "fill_price", "signal_ts", "fill_ts"}


def analyze_fills(fills: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED - set(fills.columns)
    if missing:
        raise ValueError(f"fills missing columns: {sorted(missing)}")
    df = fills.copy()
    sign = np.where(df["side"].str.upper() == "BUY", 1, -1)
    df["filled"] = df["fill_price"].notna()
    diff = sign * (df["fill_price"] - df["expected_price"])
    df["slip_bps"] = diff / df["expected_price"] * 1e4
    df["slip_cost"] = diff * df["qty"]
    df["latency_s"] = (pd.to_datetime(df["fill_ts"]) - pd.to_datetime(df["signal_ts"])).dt.total_seconds()
    return df


def summarize(df: pd.DataFrame) -> dict:
    f = df[df["filled"]]
    by_symbol = f.groupby("symbol").agg(
        fills=("slip_bps", "size"), mean_bps=("slip_bps", "mean"),
        p95_bps=("slip_bps", lambda x: np.percentile(x, 95)), latency_s=("latency_s", "mean"),
    )
    return {
        "orders": int(len(df)),
        "fill_rate": float(df["filled"].mean()),
        "mean_slip_bps": float(f["slip_bps"].mean()),
        "p95_slip_bps": float(np.percentile(f["slip_bps"], 95)),
        "mean_latency_s": float(f["latency_s"].mean()),
        "total_slip_cost": float(f["slip_cost"].sum()),
        "by_symbol": by_symbol,
    }


def apply_costs(returns: pd.Series, positions: pd.Series, bps: float) -> pd.Series:
    """Subtract trading cost: |change in position| x bps. `positions` = fraction of capital held."""
    turnover = positions.diff().abs()
    turnover.iloc[0] = abs(positions.iloc[0])
    return returns - turnover * bps / 1e4


def gap_attribution(returns: pd.Series, positions: pd.Series, bps: float, ppy: int = 252) -> dict:
    """How much of a backtest's edge does execution cost alone remove?"""
    net = apply_costs(returns, positions, bps)
    return {
        "gross_sharpe": m.sharpe(returns, ppy), "net_sharpe": m.sharpe(net, ppy),
        "gross_ann_return": m.ann_return(returns, ppy), "net_ann_return": m.ann_return(net, ppy),
        "annual_cost_drag": m.ann_return(returns, ppy) - m.ann_return(net, ppy),
    }
