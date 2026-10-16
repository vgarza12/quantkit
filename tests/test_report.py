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
    keys = ["money", "luck", "held_up", "costs", "risk", "smooth"]
    assert report.verdict([ok(k, "pass") for k in keys])[0] == "green"
    assert report.verdict([ok(k, "pass") if k != "risk" else ok(k, "warn") for k in keys])[0] == "yellow"
    assert report.verdict([ok(k, "pass") if k != "luck" else ok(k, "fail") for k in keys])[0] == "red"
    # a failing NON-core check alone should not make it red
    assert report.verdict([ok(k, "pass") if k != "costs" else ok(k, "fail") for k in keys])[0] == "yellow"


def test_report_builds_valid_html_with_all_sections():
    page, results = report.build_report(data.synthetic_prices(seed=0))
    assert page.startswith("<!doctype html>") and page.count("<svg") >= 6
    for needle in ["The six checks", "Combining them", "glossary", "cannot tell you", "Summary"]:
        assert needle in page
    assert "nan" not in page.lower().replace("financial", "")
    assert len(results) == 3 and all(len(r["checks"]) == 6 for r in results)


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
