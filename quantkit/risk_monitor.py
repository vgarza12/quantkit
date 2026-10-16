"""Project 5: Risk monitoring and alerting.

Rule-based checks on an equity/returns stream and current portfolio weights.
`check` evaluates the latest state; `first_breach` replays history to show
when each rule would first have fired.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import metrics as m


@dataclass
class RiskLimits:
    max_drawdown: float = 0.10      # fraction, e.g. 0.10 = -10%
    daily_loss: float = 0.03        # single-period loss limit
    max_loss_streak: int = 6        # consecutive losing periods
    max_weight: float = 0.40        # largest single position / strategy weight


@dataclass
class Alert:
    level: str   # "WARN" or "CRITICAL"
    rule: str
    message: str


def loss_streak(returns: pd.Series) -> int:
    n = 0
    for x in reversed(list(pd.Series(returns).dropna())):
        if x < 0:
            n += 1
        else:
            break
    return n


def check(returns: pd.Series, weights: dict | None = None, limits: RiskLimits | None = None) -> list[Alert]:
    lim = limits or RiskLimits()
    r = pd.Series(returns).dropna()
    alerts: list[Alert] = []
    dd = m.max_drawdown(r)
    cur_dd = float(m.drawdown_series(r).iloc[-1])
    if cur_dd <= -lim.max_drawdown:
        alerts.append(Alert("CRITICAL", "drawdown", f"Current drawdown {cur_dd:.1%} breaches limit {lim.max_drawdown:.0%}"))
    elif cur_dd <= -0.7 * lim.max_drawdown:
        alerts.append(Alert("WARN", "drawdown", f"Current drawdown {cur_dd:.1%} is within 30% of the limit"))
    if r.iloc[-1] <= -lim.daily_loss:
        alerts.append(Alert("CRITICAL", "daily_loss", f"Last period return {r.iloc[-1]:.1%} breaches {lim.daily_loss:.0%} limit"))
    streak = loss_streak(r)
    if streak >= lim.max_loss_streak:
        alerts.append(Alert("WARN", "loss_streak", f"{streak} consecutive losing periods (limit {lim.max_loss_streak})"))
    if weights:
        total = sum(abs(v) for v in weights.values()) or 1.0
        for k, v in weights.items():
            if abs(v) / total > lim.max_weight:
                alerts.append(Alert("WARN", "concentration", f"{k} is {abs(v) / total:.0%} of exposure (limit {lim.max_weight:.0%})"))
    return alerts


def first_breach(returns: pd.Series, limits: RiskLimits | None = None) -> dict:
    """Replay history: when would each rule first have fired? (None = never)"""
    lim = limits or RiskLimits()
    r = pd.Series(returns).dropna()
    dd = m.drawdown_series(r)
    out = {
        "drawdown": next((i for i, x in dd.items() if x <= -lim.max_drawdown), None),
        "daily_loss": next((i for i, x in r.items() if x <= -lim.daily_loss), None),
        "loss_streak": None,
    }
    run = 0
    for i, x in r.items():
        run = run + 1 if x < 0 else 0
        if run >= lim.max_loss_streak:
            out["loss_streak"] = i
            break
    return out
