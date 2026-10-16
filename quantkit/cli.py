"""Command line:  python -m quantkit demo | report"""
from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from . import data, marketdata
from .report import ICON, build_report
from .strategies import STRATEGIES


def _common(p):
    p.add_argument("--strategy", choices=list(STRATEGIES), default="sma", help="sma | momentum | meanrev")
    p.add_argument("--cost-bps", type=float, default=2.0, help="trading cost per unit traded, in basis points")
    p.add_argument("--split", type=float, default=0.7, help="fraction of history used to choose settings")
    p.add_argument("--max-dd", type=float, default=0.20, help="largest acceptable fall from a peak, e.g. 0.20")
    p.add_argument("--out", default="report.html")
    p.add_argument("--open", action="store_true", help="open the report in your browser")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="quantkit", description="Plain-English strategy report cards.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("report", help="build a report from real prices")
    rp.add_argument("--tickers", nargs="+", help="e.g. SPY QQQ (needs: pip install yfinance)")
    rp.add_argument("--csv", help="CSV of prices: dates in column 1, one column per ticker")
    rp.add_argument("--start", default="2015-01-01")
    rp.add_argument("--end")
    _common(rp)
    dp = sub.add_parser("demo", help="build a report on synthetic prices (works offline)")
    dp.add_argument("--seed", type=int, default=0)
    _common(dp)
    a = ap.parse_args(argv)

    if a.cmd == "demo":
        prices, src = data.synthetic_prices(seed=a.seed), "· SYNTHETIC DEMO DATA"
    elif a.csv:
        prices, src = marketdata.load_prices_csv(a.csv), f"· from {Path(a.csv).name}"
    elif a.tickers:
        prices, src = marketdata.fetch_prices(a.tickers, a.start, a.end), "· Yahoo Finance"
    else:
        ap.error("report needs --tickers or --csv")

    page, results = build_report(prices, a.strategy, a.cost_bps, a.split, a.max_dd, src)
    Path(a.out).write_text(page, encoding="utf-8")
    for r in results:
        print(f"\n{r['ticker']}: {r['verdict'][1]}")
        for c in r["checks"]:
            print(f"  {ICON[c['status']]} {c['title']:<28} {c['value']}")
    out = Path(a.out).resolve()
    print(f"\nReport written to {out}")
    if a.open:
        webbrowser.open(out.as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
