"""Project 1: Strategy evaluation and ranking, with overfitting checks.

Core idea: a high in-sample Sharpe means little when many strategies were
tried. We use the Probabilistic Sharpe Ratio (PSR) and a Deflated Sharpe Ratio
(DSR) that raises the bar according to how many strategies were screened
(Bailey & López de Prado). The final score is a heuristic, not a law.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import metrics as m

EULER_GAMMA = 0.5772156649


def prob_sharpe(r, sr_benchmark: float = 0.0) -> float:
    """P(true per-period Sharpe > benchmark), adjusting for skew and kurtosis."""
    r = pd.Series(r).dropna()
    n = len(r)
    if n < 3 or r.std(ddof=1) == 0:
        return float("nan")
    sr = r.mean() / r.std(ddof=1)
    g3 = stats.skew(r)
    g4 = stats.kurtosis(r, fisher=False)
    denom = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr**2, 1e-12))
    return float(stats.norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / denom))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """Per-period Sharpe you'd expect the BEST of n_trials pure-noise strategies to reach."""
    if n_trials < 2 or var_sr <= 0:
        return 0.0
    z1 = stats.norm.ppf(1 - 1 / n_trials)
    z2 = stats.norm.ppf(1 - 1 / (n_trials * np.e))
    return float(np.sqrt(var_sr) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def smoothness(r) -> float:
    """R^2 of log-equity vs time. Real strategies are bumpy; ~1.0 is suspicious."""
    eq = np.log(m.equity_curve(r).to_numpy())
    t = np.arange(len(eq))
    if len(eq) < 3 or np.ptp(eq) == 0:
        return float("nan")
    return float(np.corrcoef(t, eq)[0, 1] ** 2)


def evaluate(r, ppy: int = 252, oos_frac: float = 0.5, sr_benchmark: float = 0.0) -> dict:
    r = pd.Series(r).dropna()
    cut = int(len(r) * (1 - oos_frac))
    is_r, oos_r = r.iloc[:cut], r.iloc[cut:]
    out = m.summary(r, ppy)
    out.update({
        "is_sharpe": m.sharpe(is_r, ppy),
        "oos_sharpe": m.sharpe(oos_r, ppy),
        "psr": prob_sharpe(r, sr_benchmark),
        "smooth_r2": smoothness(r),
    })
    flags = []
    if len(r) < 250:
        flags.append("few_observations")
    if len(r) >= 60 and out["smooth_r2"] > 0.98:
        flags.append("too_smooth")
    if out["is_sharpe"] > 0 and not out["oos_sharpe"] > 0.5 * out["is_sharpe"]:
        flags.append("oos_decay")
    out["flags"] = flags
    return out


def rank_strategies(strats: dict | pd.DataFrame, ppy: int = 252, oos_frac: float = 0.5) -> pd.DataFrame:
    """Rank many strategies, deflating Sharpe for the number of strategies screened."""
    if isinstance(strats, pd.DataFrame):
        strats = {c: strats[c] for c in strats.columns}
    per_period_sr = {k: pd.Series(v).dropna().mean() / pd.Series(v).dropna().std(ddof=1) for k, v in strats.items()}
    bench = expected_max_sharpe(len(strats), float(np.var(list(per_period_sr.values()), ddof=1)) if len(strats) > 1 else 0.0)
    rows = []
    for name, r in strats.items():
        ev = evaluate(r, ppy, oos_frac)
        dsr = prob_sharpe(r, bench)
        score = float(np.clip(dsr - 0.15 * len(ev["flags"]), 0, 1))
        rows.append({
            "strategy": name, "sharpe": ev["sharpe"], "oos_sharpe": ev["oos_sharpe"],
            "max_dd": ev["max_drawdown"], "psr": ev["psr"], "deflated_sr_prob": dsr,
            "flags": ",".join(ev["flags"]), "score": score,
        })
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
