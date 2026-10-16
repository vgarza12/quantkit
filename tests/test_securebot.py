import json

import pytest

from quantkit.securebot import (AuditLog, KillSwitch, MissingSecret, PaperBroker, RiskGuard, TradingBot,
                                load_secrets, scan_for_secrets)


def make_bot(tmp_path, **broker_kw):
    broker = PaperBroker(**broker_kw)
    guard = RiskGuard(KillSwitch(str(tmp_path / "STOP")), max_order_qty=5, max_position=10)
    audit = AuditLog(str(tmp_path / "audit.jsonl"))
    return TradingBot(broker, guard, audit, sleep=lambda s: None), broker, audit


def test_secrets_never_in_repr(monkeypatch):
    s = load_secrets({"QK_API_KEY": "abc123supersecret", "QK_API_SECRET": "zzz999topsecret"})
    assert "supersecret" not in repr(s) and "topsecret" not in str(s)


def test_missing_secret_fails_loudly():
    with pytest.raises(MissingSecret):
        load_secrets({})


def test_scanner_finds_hardcoded_key(tmp_path):
    (tmp_path / "bad.py").write_text('api_key = "' + "X" * 20 + '"\n')
    (tmp_path / "ok.py").write_text('api_key = os.environ["QK_API_KEY"]\n')
    hits = scan_for_secrets(str(tmp_path))
    assert len(hits) == 1 and hits[0][0].endswith("bad.py")


def test_lost_response_retry_does_not_double_trade(tmp_path):
    bot, broker, _ = make_bot(tmp_path, drop_response_n=2)   # order arrives, reply lost, twice
    fill = bot.place("ES", "BUY", 1, 5000.0)
    assert fill and fill["status"] == "FILLED"
    assert len(broker.orders) == 1                            # exactly one order, not three


def test_retries_exhausted_trips_kill_switch(tmp_path):
    bot, _, _ = make_bot(tmp_path, fail_first_n=99)
    assert bot.place("ES", "BUY", 1, 5000.0) is None
    assert bot.guard.ks.tripped
    assert bot.place("ES", "BUY", 1, 5000.0) is None          # stays halted


def test_guard_rejections(tmp_path):
    bot, broker, _ = make_bot(tmp_path)
    assert bot.place("ES", "BUY", 99, 5000.0) is None                    # too big
    assert bot.place("ES", "BUY", 5, 5000.0, position=8) is None         # position limit
    assert bot.place("ES", "BUY", 1, 5000.0, daily_pnl_frac=-0.05) is None  # daily loss
    assert broker.orders == {} and bot.guard.ks.tripped


def test_manual_kill_file_halts_trading(tmp_path):
    bot, broker, _ = make_bot(tmp_path)
    (tmp_path / "STOP").write_text("")
    assert bot.place("ES", "BUY", 1, 5000.0) is None and broker.orders == {}


def test_audit_chain_detects_tampering(tmp_path):
    bot, _, audit = make_bot(tmp_path)
    for _ in range(3):
        bot.place("ES", "BUY", 1, 5000.0)
    path = str(tmp_path / "audit.jsonl")
    assert AuditLog.verify(path) == (True, None)
    lines = (tmp_path / "audit.jsonl").read_text().splitlines()
    rec = json.loads(lines[1]); rec["event"]["qty"] = 999
    lines[1] = json.dumps(rec, sort_keys=True)
    (tmp_path / "audit.jsonl").write_text("\n".join(lines) + "\n")
    ok, bad_line = AuditLog.verify(path)
    assert not ok and bad_line == 2


def test_audit_redacts_secrets(tmp_path):
    a = AuditLog(str(tmp_path / "a.jsonl"))
    a.append({"type": "x", "api_key": "SHOULD_NOT_APPEAR", "nested": {"token": "ALSO_NO"}})
    assert "SHOULD_NOT_APPEAR" not in (tmp_path / "a.jsonl").read_text()
    assert "ALSO_NO" not in (tmp_path / "a.jsonl").read_text()
