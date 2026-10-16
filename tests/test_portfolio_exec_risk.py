import numpy as np
import pandas as pd
import pytest

from quantkit import data, execution as ex, portfolio as pf, risk_monitor as rm


def test_weights_valid_for_all_methods():
    R = data.synthetic_universe(6, 600, seed=2)
    for name, fn in pf.METHODS.items():
        w = fn(R)
        assert w.sum() == pytest.approx(1.0), name
        assert (w >= -1e-9).all(), name


def test_min_variance_beats_equal_weight_in_sample_variance():
    R = data.synthetic_universe(6, 600, seed=2)
    var = lambda w: w @ R.cov().to_numpy() @ w
    assert var(pf.min_variance(R)) <= var(pf.equal_weight(R)) + 1e-12


def test_walk_forward_has_no_lookahead_rows():
    R = data.synthetic_universe(4, 600, seed=2)
    out = pf.walk_forward(R, "equal_weight", lookback=252, rebal=21)
    assert out.index[0] == R.index[252]


def test_correlated_strategies_have_few_effective_bets():
    d = pf.diversification(data.synthetic_universe(10, 800, n_real=0, seed=1))
    assert d["effective_bets"] < d["n_strategies"]


def test_slippage_sign_and_cost():
    f = data.synthetic_fills(300, seed=1, avg_slip_bps=2.0)
    s = ex.summarize(ex.analyze_fills(f))
    assert 1.0 < s["mean_slip_bps"] < 3.0       # positive = cost
    assert 0.9 < s["fill_rate"] <= 1.0


def test_sell_slippage_sign():
    f = pd.DataFrame({"ts": [0], "symbol": ["ES"], "side": ["SELL"], "qty": [1],
                      "expected_price": [100.0], "fill_price": [99.9],
                      "signal_ts": ["2024-01-01 10:00:00"], "fill_ts": ["2024-01-01 10:00:01"]})
    assert ex.analyze_fills(f)["slip_bps"].iloc[0] == pytest.approx(10.0)  # sold lower = cost


def test_costs_reduce_sharpe():
    idx = pd.bdate_range("2022-01-03", periods=500)
    rng = np.random.default_rng(0)
    pos = pd.Series(rng.choice([-1, 1], 500), index=idx)
    ret = pd.Series(rng.normal(0.0004, 0.01, 500), index=idx)
    g = ex.gap_attribution(ret, pos, bps=5)
    assert g["net_sharpe"] < g["gross_sharpe"] and g["annual_cost_drag"] > 0


def test_risk_alerts_fire():
    r = pd.Series([0.0] * 5 + [-0.04] * 4)
    rules = {a.rule for a in rm.check(r, {"A": 8, "B": 2})}
    assert {"drawdown", "daily_loss", "concentration"} <= rules


def test_first_breach_dates():
    r = pd.Series([0.01, -0.05, 0.0], index=pd.bdate_range("2024-01-01", periods=3))
    b = rm.first_breach(r, rm.RiskLimits(max_drawdown=0.04, daily_loss=0.03))
    assert b["daily_loss"] == r.index[1] and b["drawdown"] == r.index[1]
