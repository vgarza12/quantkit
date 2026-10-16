"""End-to-end demo on synthetic data:  python examples/run_demo.py"""
import sys, tempfile, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from quantkit import data, decay, execution, metrics, portfolio, risk_monitor, strategy_eval
from quantkit.securebot import AuditLog, KillSwitch, PaperBroker, RiskGuard, TradingBot

pd.set_option("display.width", 140)
hr = lambda t: print(f"\n{'=' * 8} {t} {'=' * 8}")

U = data.synthetic_universe(n_strats=20, n=1000, n_real=3, seed=0)

hr("1. Strategy ranking (20 strategies, only 3 have a real edge)")
print(strategy_eval.rank_strategies(U).head(6).round(3).to_string(index=False))

hr("2. Live vs backtest decay (strategy that stops working)")
bt = data.synthetic_returns(1500, ann_mu=0.15, ann_vol=0.10, seed=1)
live = data.synthetic_returns(150, ann_mu=-0.20, ann_vol=0.10, seed=2, start="2026-01-01")
d = decay.compare(bt, live)
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items() if k != "caveat"})

hr("3. Portfolio construction (walk-forward, out-of-sample)")
print(portfolio.compare(U.iloc[:, :8]).round(3))
print("diversification:", {k: round(v, 2) for k, v in portfolio.diversification(U.iloc[:, :8]).items()})

hr("4. Execution quality")
s = execution.summarize(execution.analyze_fills(data.synthetic_fills(800, avg_slip_bps=2.0)))
print({k: round(v, 3) for k, v in s.items() if k != "by_symbol"})
pos = pd.Series(1.0, index=bt.index).where(bt.rolling(5).mean().shift(1) > 0, -1.0)
print("backtest vs after costs:", {k: round(v, 3) for k, v in execution.gap_attribution(bt, pos, bps=3).items()})

hr("5. Risk monitor")
bad = pd.concat([bt.iloc[-100:], pd.Series([-0.02] * 7, index=pd.bdate_range("2030-01-01", periods=7))])
for a in risk_monitor.check(bad, {"ES": 9, "NQ": 1}):
    print(f"[{a.level}] {a.rule}: {a.message}")

hr("6. Secure bot (response lost twice; retries must not double-trade)")
with tempfile.TemporaryDirectory() as tmp:
    broker = PaperBroker(drop_response_n=2)
    bot = TradingBot(broker, RiskGuard(KillSwitch(os.path.join(tmp, "STOP"))), AuditLog(os.path.join(tmp, "audit.jsonl")),
                     sleep=lambda s: None)
    fill = bot.place("ES", "BUY", 1, 5000.0)
    print("status:", fill["status"], "| orders at broker:", len(broker.orders), "| audit intact:",
          AuditLog.verify(os.path.join(tmp, "audit.jsonl")))
