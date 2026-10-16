import numpy as np
import pandas as pd
import pytest

from quantkit import cli, data, marketdata, report, strategies as st


def test_signal_is_held_next_day_not_same_day():
    p = data.synthetic_prices(1, 800, seed=1).iloc[:, 0]
    sig = st.STRATEGIES["momentum"]["fn"](p, 20)
    _, _, pos = st.backtest(p, "momentum", {"lookback": 20})
    assert pos.iloc[0] == 0
    assert (pos.iloc[1:].to_numpy() == sig.iloc[:-1].to_numpy()).all()    # no look-ahead


def test_future_price_change_cannot_alter_past_returns():
    p = data.synthetic_prices(1, 800, seed=2).iloc[:, 0]
    _, n1, _ = st.backtest(p, "sma", {"fast": 10, "slow": 100})
    p2 = p.copy(); p2.iloc[-1] *= 1.5                      # change only the final price
    _, n2, _ = st.backtest(p2, "sma", {"fast": 10, "slow": 100})
    assert np.allclose(n1.iloc[:-1], n2.iloc[:-1])


def test_costs_reduce_net_returns():
    p = data.synthetic_prices(1, 800, seed=3).iloc[:, 0]
    _, net0, _ = st.backtest(p, "sma", {"fast": 10, "slow": 100}, cost_bps=0)
    _, net5, _ = st.backtest(p, "sma", {"fast": 10, "slow": 100}, cost_bps=5)
    assert net5.sum() < net0.sum()


def test_selection_uses_only_first_split():
    p = data.synthetic_prices(1, 1200, seed=4).iloc[:, 0]
    r = st.run_strategy(p, "sma", split=0.7)
    from quantkit import metrics as m
    cut = r["cut"]
    best_is = m.sharpe(r["net"].iloc[:cut])
    assert all(m.sharpe(r["trials"][k].iloc[:cut]) <= best_is + 1e-12 for k in r["trials"])


def test_too_little_data_raises_clear_error():
    p = data.synthetic_prices(1, 200).iloc[:, 0]
    with pytest.raises(ValueError, match="Need at least"):
        st.run_strategy(p, "sma")


def test_verdict_rules():
    ok = lambda k, s: {"key": k, "status": s}
    keys = ["money", "beats", "luck", "held_up", "costs", "risk", "smooth"]
    assert report.verdict([ok(k, "pass") for k in keys])[0] == "green"
    assert report.verdict([ok(k, "pass") if k != "risk" else ok(k, "warn") for k in keys])[0] == "yellow"
    assert report.verdict([ok(k, "pass") if k != "luck" else ok(k, "fail") for k in keys])[0] == "red"
    # a failing NON-core check alone should not make it red
    assert report.verdict([ok(k, "pass") if k != "costs" else ok(k, "fail") for k in keys])[0] == "yellow"


def test_report_builds_valid_html_with_all_sections():
    page, results = report.build_report(data.synthetic_prices(seed=0))
    assert page.startswith("<!doctype html>") and page.count("<svg") >= 6
    for needle in ["The seven checks", "Combining them", "glossary", "cannot tell you", "Summary"]:
        assert needle in page
    assert "nan" not in page.lower().replace("financial", "")
    assert len(results) == 3 and all(len(r["checks"]) == 7 for r in results)


def test_html_escapes_ticker_names():
    P = data.synthetic_prices(1, 1500, seed=0)
    P.columns = ["<b>X</b>"]
    page, _ = report.build_report(P)
    assert "<b>X</b>" not in page and "&lt;b&gt;X&lt;/b&gt;" in page


def test_stricter_bar_when_screening_many_tickers():
    p = data.synthetic_prices(1, 2000, seed=0).iloc[:, 0]
    one = report.analyze_ticker(p, "T", n_tickers=1)
    many = report.analyze_ticker(p, "T", n_tickers=20)
    rank = {"pass": 2, "warn": 1, "fail": 0}
    luck = lambda r: next(c for c in r["checks"] if c["key"] == "luck")["status"]
    assert rank[luck(many)] <= rank[luck(one)]


def test_cli_demo_writes_file(tmp_path, capsys):
    out = tmp_path / "r.html"
    assert cli.main(["demo", "--out", str(out)]) == 0
    assert out.exists() and "Strategy report card" in out.read_text()
    assert "Report written" in capsys.readouterr().out


def test_cli_csv_roundtrip(tmp_path):
    P = data.synthetic_prices(2, 1500, seed=1)
    csv = tmp_path / "p.csv"; P.to_csv(csv)
    out = tmp_path / "r.html"
    assert cli.main(["report", "--csv", str(csv), "--strategy", "momentum", "--out", str(out)]) == 0
    assert "DEMO_A" in out.read_text()
    assert list(marketdata.load_prices_csv(str(csv)).columns) == ["DEMO_A", "DEMO_NOISE"]


def test_cli_requires_input(capsys):
    with pytest.raises(SystemExit):
        cli.main(["report"])


def test_never_trading_strategy_does_not_break_report():
    P = data.synthetic_prices(2, 1500, seed=1)
    P["FLAT"] = 100.0 + np.arange(len(P)) * 1e-6          # tiny steady rise; momentum stays fully invested
    P["DOWN"] = np.linspace(100, 50, len(P))              # falls steadily; momentum never buys
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")                    # any numeric warning fails the test
        page, results = report.build_report(P, strategy="momentum")
    assert "DOWN" in page and len(results) == 4


def test_held_up_message_matches_reason():
    base = {"verdict": "watch", "live_percentile_vs_backtest": 0.9, "cusum_alarm_index": 12, "cusum_alarm_date": "2023-03-14"}
    st_, msg = report.held_up_status(base, is_sharpe=0.4)            # better overall, but a sudden drop
    assert st_ == "warn" and "sudden sharp drop" in msg and "2023-03-14" in msg and "weak side" not in msg
    weak = {"verdict": "watch", "live_percentile_vs_backtest": 0.08, "cusum_alarm_index": None, "cusum_alarm_date": None}
    assert "weak side" in report.held_up_status(weak, 0.4)[1]
    assert report.held_up_status({"verdict": "consistent", "live_percentile_vs_backtest": 0.5}, 0.4)[0] == "pass"
    assert report.held_up_status({"verdict": "decayed", "live_percentile_vs_backtest": 0.01}, 0.4)[0] == "fail"


def test_beats_holding_branches():
    f = report.beats_holding_status
    assert f(1.2, 0.8, 0.10, 0.25, "X")[0] == "pass"
    assert f(0.8, 0.85, 0.10, 0.25, "X")[0] == "warn"               # roughly matched
    assert f(0.5, 0.9, 0.10, 0.30, "X")[0] == "warn"               # less Sharpe but much smoother
    assert f(0.5, 0.9, 0.28, 0.30, "X")[0] == "fail"               # worse on both counts
    assert f(float("nan"), 0.9, 0.1, 0.2, "X")[0] == "fail"


def test_smoothness_needs_high_sharpe_too():
    f = report.smoothness_status
    assert f(0.97, 1.1)[0] == "pass"                                # trending market, ordinary Sharpe
    assert f(0.99, 1.1)[0] == "pass"                                # straight-ish but not too good
    assert f(0.99, 3.5)[0] == "fail"                                # straight AND implausibly good
    assert f(0.99, float("nan"))[0] == "fail"                       # zero variance


def test_buy_and_hold_comparison_appears_in_report():
    page, results = report.build_report(data.synthetic_prices(seed=0))
    assert "Beats just holding" in page
    assert all(any(c["key"] == "beats" for c in r["checks"]) for r in results)
