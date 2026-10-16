# quantkit

Six small, tested tools that answer one question a systematic trading team keeps asking:
**"Is this automated strategy real, is it still working, how should it be combined, and can it be run safely?"**

```
pip install -r requirements.txt
python examples/run_demo.py     # all six on synthetic data, no data subscription needed
python -m pytest                # 41 tests
```

## The report card (start here)

One command turns a price history into a single plain-English web page: a green / yellow / red verdict,
seven checks explained in everyday language, and two charts. No statistics background needed to read it.

```
python -m quantkit demo --open                              # offline, synthetic prices
python -m quantkit report --tickers SPY QQQ --open          # real prices (pip install yfinance first)
python -m quantkit report --csv prices.csv --strategy momentum --cost-bps 3 --max-dd 0.15
```
Options: `--strategy sma | momentum | meanrev`, `--cost-bps`, `--split` (share of history used to choose settings), `--max-dd`, `--out`.

| Check | Question it answers |
|---|---|
| Makes money after costs | Is there a profit on data the strategy never saw? |
| Beats just holding | Did it add value over simply holding the asset, in risk-adjusted terms or by a much smoother ride? |
| Distinguishable from luck | How confident can we be the edge is real? (stricter when many tickers are screened) |
| Held up on unseen data | Did the edge survive after the settings were chosen? |
| Costs don't eat the edge | How much of the edge is left after trading costs? |
| Losses stay tolerable | Is the worst fall within your limit? |
| Not suspiciously smooth | An almost straight curve AND an unusually high Sharpe is a red flag (a rising market alone is not) |

How it stays honest: every strategy's settings are chosen on the first 70% of history only, and the verdict is judged on the last 30%.
A green verdict means "worth testing further", never "safe to trade".

## The six building blocks

| # | Module | What it does | Why it matters |
|---|--------|--------------|----------------|
| 1 | `strategy_eval.py` | Scores and ranks strategies; Probabilistic and **Deflated** Sharpe (penalizes screening many strategies), IS/OOS split, "too smooth" check | Picking the best of many strategies is exactly how backtest overfitting happens |
| 2 | `decay.py` | Compares live to backtest using a block bootstrap (verdict) plus a CUSUM detector for sudden breakdowns | Separates "bad patch" from "stopped working" |
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
- Built-in strategies are deliberately simple (long/flat, one asset at a time) so the verdicts are easy to interpret.
- The Yahoo Finance loader depends on the third-party `yfinance` package; if it fails, export prices to CSV and use `--csv`.
- Demo data is **synthetic**. Results on it prove the code works, not that any strategy makes money. Plug in your own return series with `data.load_returns_csv`.
- The ranking `score` is a heuristic (deflated PSR minus a penalty per flag), not a theorem. Thresholds (0.98 smoothness, 5% bootstrap tail, CUSUM k=0.5, h=5) are defensible defaults, not tuned values.
- The deflated Sharpe assumes roughly independent trials and treats the cross-sectional Sharpe variance as the trial variance; real strategy searches are messier.
- Bootstrap decay tests have low power on short live windows. "watch" means look closer, not act.
- Slippage modeling is descriptive (from logged fills), not a market-impact model.
- `PaperBroker` simulates failures; it is **not** connected to a real broker. A real adapter would need its own testing.
- Not investment advice.

## Extending it
Real broker adapter (subclass `Broker`), futures/FX cost models in `execution.apply_costs`, regime-conditional reporting, and a tear-sheet plot on top of `metrics.summary`.
