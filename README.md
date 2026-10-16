# quantkit

Six small, tested tools that answer one question a systematic trading team keeps asking:
**"Is this automated strategy real, is it still working, how should it be combined, and can it be run safely?"**

```
pip install -r requirements.txt
python examples/run_demo.py     # all six on synthetic data, no data subscription needed
python -m pytest                # 28 tests
```

| # | Module | What it does | Why it matters |
|---|--------|--------------|----------------|
| 1 | `strategy_eval.py` | Scores and ranks strategies; Probabilistic and **Deflated** Sharpe (penalizes screening many strategies), IS/OOS split, "too smooth" check | Picking the best of many strategies is exactly how backtest overfitting happens |
| 2 | `decay.py` | Compares live to backtest using a block bootstrap and a CUSUM detector | Separates "bad patch" from "stopped working" |
| 3 | `portfolio.py` | Equal-weight, inverse-vol, min-variance, risk parity, max-Sharpe, all **walk-forward**; effective-bets diversification measure | Correlated strategies are often one bet in disguise |
| 4 | `execution.py` | Slippage (bps), latency, fill rate; shows how much of a backtest's edge costs remove | Execution quality is part of the strategy's return |
| 5 | `risk_monitor.py` | Drawdown, daily-loss, loss-streak and concentration alerts, plus historical replay of when each would have fired | Risk rules should be explicit and testable |
| 6 | `securebot/` | Paper bot with env-only secrets, idempotent retry-safe orders, kill switch, pre-trade limits, **hash-chained audit log**, hardcoded-secret scanner | In automated trading, one bug or leaked key costs real money |

## Design choices worth knowing
- **No look-ahead:** portfolio weights are estimated only from the lookback window, then applied to the following window.
- **Retries cannot double-trade:** one `client_order_id` per order across all attempts. A test simulates the broker receiving an order but the reply being lost.
- **Tamper evidence:** every audit record contains the previous record's hash; editing any line is detected by `AuditLog.verify`.
- **Secrets:** read only from environment variables, masked in `repr`, redacted from logs.

## Honest limitations
- Demo data is **synthetic**. Results on it prove the code works, not that any strategy makes money. Plug in your own return series with `data.load_returns_csv`.
- The ranking `score` is a heuristic (deflated PSR minus a penalty per flag), not a theorem. Thresholds (0.98 smoothness, 5% bootstrap tail, CUSUM k=0.5, h=5) are defensible defaults, not tuned values.
- The deflated Sharpe assumes roughly independent trials and treats the cross-sectional Sharpe variance as the trial variance; real strategy searches are messier.
- Bootstrap decay tests have low power on short live windows. "watch" means look closer, not act.
- Slippage modeling is descriptive (from logged fills), not a market-impact model.
- `PaperBroker` simulates failures; it is **not** connected to a real broker. A real adapter would need its own testing.
- Not investment advice.

## Extending it
Real broker adapter (subclass `Broker`), futures/FX cost models in `execution.apply_costs`, regime-conditional reporting, and a tear-sheet plot on top of `metrics.summary`.
