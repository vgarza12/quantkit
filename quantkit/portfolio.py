"""Project 3: Portfolio construction across strategies, tested out-of-sample.

Optimizers are long-only and fully invested. Inputs are estimated from a
lookback window only, then applied to the *next* window (walk-forward), so
results are not contaminated by hindsight.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import metrics as m


def _shrunk_cov(R: pd.DataFrame, delta: float = 0.2) -> np.ndarray:
    """Shrink covariance toward its diagonal to reduce estimation error."""
    S = R.cov().to_numpy()
    return (1 - delta) * S + delta * np.diag(np.diag(S))


def _solve(obj, n: int) -> np.ndarray:
    x0 = np.full(n, 1 / n)
    res = minimize(obj, x0, method="SLSQP", bounds=[(0, 1)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    w = res.x if res.success else x0
    w = np.clip(w, 0, None)
    return w / w.sum()


def equal_weight(R: pd.DataFrame) -> np.ndarray:
    return np.full(R.shape[1], 1 / R.shape[1])


def inverse_vol(R: pd.DataFrame) -> np.ndarray:
    iv = 1 / R.std(ddof=1).to_numpy()
    return iv / iv.sum()


def min_variance(R: pd.DataFrame) -> np.ndarray:
    S = _shrunk_cov(R)
    return _solve(lambda w: w @ S @ w, R.shape[1])


def risk_parity(R: pd.DataFrame) -> np.ndarray:
    S = _shrunk_cov(R)

    def obj(w):
        rc = w * (S @ w)
        return float(((rc - rc.mean()) ** 2).sum()) * 1e8

    return _solve(obj, R.shape[1])


def max_sharpe(R: pd.DataFrame) -> np.ndarray:
    """Max-Sharpe with expected returns shrunk halfway to the cross-sectional mean."""
    S = _shrunk_cov(R)
    mu = R.mean().to_numpy()
    mu = 0.5 * mu + 0.5 * mu.mean()
    return _solve(lambda w: -(w @ mu) / np.sqrt(w @ S @ w), R.shape[1])


METHODS = {
    "equal_weight": equal_weight, "inverse_vol": inverse_vol, "min_variance": min_variance,
    "risk_parity": risk_parity, "max_sharpe": max_sharpe,
}


def walk_forward(R: pd.DataFrame, method: str, lookback: int = 252, rebal: int = 21) -> pd.Series:
    """Out-of-sample returns: weights from the past `lookback` rows, held for `rebal` rows."""
    fn = METHODS[method]
    pieces = []
    for t in range(lookback, len(R), rebal):
        w = fn(R.iloc[t - lookback:t])
        pieces.append(R.iloc[t:t + rebal] @ w)
    return pd.concat(pieces)


def compare(R: pd.DataFrame, lookback: int = 252, rebal: int = 21, ppy: int = 252) -> pd.DataFrame:
    rows = {}
    for name in METHODS:
        s = m.summary(walk_forward(R, name, lookback, rebal), ppy)
        rows[name] = {k: s[k] for k in ("ann_return", "ann_vol", "sharpe", "max_drawdown")}
    return pd.DataFrame(rows).T.sort_values("sharpe", ascending=False)


def diversification(R: pd.DataFrame) -> dict:
    """How much real diversification is there?

    effective_bets = (sum of eigenvalues)^2 / sum of squared eigenvalues of the correlation
    matrix. If it's far below the strategy count, the strategies are largely the same bet.
    """
    C = R.corr().to_numpy()
    n = C.shape[0]
    eig = np.clip(np.linalg.eigvalsh(C), 0, None)
    off = C[~np.eye(n, dtype=bool)]
    w = equal_weight(R)
    vol = R.std(ddof=1).to_numpy()
    port_vol = np.sqrt(w @ R.cov().to_numpy() @ w)
    return {
        "n_strategies": n,
        "avg_pairwise_corr": float(off.mean()),
        "effective_bets": float(eig.sum() ** 2 / (eig ** 2).sum()),
        "diversification_ratio": float((w @ vol) / port_vol),
    }
