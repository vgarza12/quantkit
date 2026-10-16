import numpy as np
import pandas as pd
import pytest

from quantkit import data, decay, metrics as m, strategy_eval as se


def test_max_drawdown_known_value():
    r = pd.Series([0.10, -0.20, 0.05])
    # equity: 1.1, 0.88, 0.924 -> worst dd = 0.88/1.1 - 1 = -0.2
    assert m.max_drawdown(r) == pytest.approx(-0.2)


def test_drawdown_counts_initial_loss():
    assert m.max_drawdown(pd.Series([-0.1, 0.0])) == pytest.approx(-0.1)


def test_sharpe_scales_with_sqrt_ppy():
    r = data.synthetic_returns(500, seed=1)
    assert m.sharpe(r, 252) == pytest.approx(m.sharpe(r, 63) * 2, rel=1e-9)


def test_zero_edge_noise_gets_low_psr():
    r = data.synthetic_returns(1000, ann_mu=0.0, seed=3)
    assert se.prob_sharpe(r) < 0.95


@pytest.mark.filterwarnings("ignore:Precision loss")
def test_perfectly_smooth_curve_is_flagged():
    r = pd.Series(np.full(300, 0.001))
    assert "too_smooth" in se.evaluate(r)["flags"]


def test_deflation_penalizes_many_trials():
    U = data.synthetic_universe(n_strats=30, n=750, n_real=0, seed=5)
    ranked = se.rank_strategies(U)
    # With no real edge anywhere, even the "best" strategy shouldn't score confidently high.
    assert ranked["score"].max() < 0.95


def test_expected_max_sharpe_grows_with_trials():
    assert se.expected_max_sharpe(100, 0.0004) > se.expected_max_sharpe(5, 0.0004) > 0


def test_decay_detects_collapse():
    bt = data.synthetic_returns(1500, ann_mu=0.15, ann_vol=0.10, seed=1)
    live = data.synthetic_returns(150, ann_mu=-0.20, ann_vol=0.10, seed=2)
    assert decay.compare(bt, live)["verdict"] == "decayed"


def test_decay_accepts_consistent_live():
    bt = data.synthetic_returns(1500, ann_mu=0.10, ann_vol=0.10, seed=1)
    live = data.synthetic_returns(250, ann_mu=0.10, ann_vol=0.10, seed=9)
    assert decay.compare(bt, live)["verdict"] == "consistent"


def test_cusum_none_when_on_track():
    x = np.random.default_rng(0).normal(0.001, 0.01, 200)
    assert decay.cusum_alarm(x, 0.001, 0.01) is None
