"""Plain-English strategy report card: one self-contained HTML file.

The heavy statistics (deflation, bootstrap, CUSUM, walk-forward optimisation)
run underneath; the page shows a verdict, six checks in everyday language,
and two charts.
"""
from __future__ import annotations

import datetime as dt
import html

import numpy as np
import pandas as pd

from . import decay, metrics as m, portfolio as pf, strategy_eval as se
from .strategies import STRATEGIES, run_strategy

PASS, WARN, FAIL = "pass", "warn", "fail"
ICON = {PASS: "✓", WARN: "!", FAIL: "✗"}


def _pct(x, d=1):
    return "n/a" if x is None or not np.isfinite(x) else f"{x * 100:.{d}f}%"


def _f(x, d=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def _duration(periods: int) -> str:
    months = periods / 21
    return f"about {months:.0f} months" if months >= 2 else f"{periods} trading days"


# ----------------------------------------------------------------------------- analysis
def analyze_ticker(prices: pd.Series, ticker: str, strategy: str = "sma", cost_bps: float = 2.0,
                   split: float = 0.7, max_dd: float = 0.20, n_tickers: int = 1, ppy: int = 252) -> dict:
    res = run_strategy(prices, strategy, cost_bps, split)
    net, gross, cut = res["net"], res["gross"], res["cut"]
    is_net, oos_net, oos_gross = net.iloc[:cut], net.iloc[cut:], gross.iloc[cut:]
    bh = prices.pct_change().reindex(net.index).fillna(0.0)

    s_is, s_oos = m.summary(is_net, ppy), m.summary(oos_net, ppy)
    s_full, bh_s = m.summary(net, ppy), m.summary(bh.iloc[cut:], ppy)
    n_oos = len(oos_net)
    psr = se.prob_sharpe(oos_net)
    dec = decay.compare(is_net, oos_net, ppy)
    gross_sr, net_sr = m.sharpe(oos_gross, ppy), s_oos["sharpe"]
    smooth = se.smoothness(net)

    # Deflated Sharpe on the whole sample, shown for information (accounts for the settings tried).
    per_sr = (res["trials"].mean() / res["trials"].std(ddof=1)).to_numpy()
    bench = se.expected_max_sharpe(res["n_trials"], float(np.var(per_sr, ddof=1)))
    dsr_full = se.prob_sharpe(net, bench)

    # Stricter bar when several tickers were screened (simple Bonferroni-style adjustment).
    bar_hi, bar_lo = 1 - 0.05 / n_tickers, 1 - 0.20 / n_tickers
    checks = []

    st = PASS if net_sr > 0.5 else WARN if net_sr > 0 else FAIL
    checks.append(dict(key="money", title="Makes money after costs", status=st, value=f"Sharpe {_f(net_sr)}",
        explain=f"On history the strategy never saw, it earned {_pct(s_oos['ann_return'])} a year after trading costs, "
                f"with a Sharpe ratio of {_f(net_sr)} (return per unit of risk; above 0.5 is decent, above 1 is strong)."))

    st, msg = beats_holding_status(net_sr, bh_s["sharpe"], abs(s_oos["max_drawdown"]), abs(bh_s["max_drawdown"]), ticker)
    checks.append(dict(key="beats", title="Beats just holding", status=st, value=f"Sharpe {_f(net_sr)} vs {_f(bh_s['sharpe'])}", explain=msg))

    st = PASS if psr >= bar_hi else WARN if psr >= bar_lo else FAIL
    if st == PASS and n_oos < 250:
        st = WARN
    extra = f" Because {n_tickers} tickers were screened, the bar is raised to {bar_hi * 100:.1f}%." if n_tickers > 1 else ""
    short = " The unseen period is short, so treat this with caution." if n_oos < 250 else ""
    checks.append(dict(key="luck", title="Distinguishable from luck", status=st, value=f"{psr * 100:.0f}% confident",
        explain=f"Judging only by unseen data ({n_oos} days), there is a {psr * 100:.0f}% chance the strategy's true edge is positive. "
                f"The usual bar is 95%.{extra}{short}"))

    st, msg = held_up_status(dec, s_is["sharpe"])
    checks.append(dict(key="held_up", title="Held up on unseen data", status=st,
        value=f"{_f(s_is['sharpe'])} → {_f(net_sr)}",
        explain=f"Sharpe was {_f(s_is['sharpe'])} when settings were chosen and {_f(net_sr)} afterwards. {msg}"))

    if gross_sr > 0:
        ratio = net_sr / gross_sr
        st = PASS if ratio >= 0.7 else WARN if ratio >= 0.4 else FAIL
        cexp = (f"Trading costs ({_f(cost_bps, 1)} basis points per trade) cut the Sharpe from {_f(gross_sr)} to {_f(net_sr)}, "
                f"removing {_pct(max(0.0, 1 - ratio), 0)} of the edge.")
        cval = f"{_pct(max(0.0, ratio), 0)} of edge kept"
    else:
        st, cexp, cval = FAIL, "Before costs the strategy already had no edge on unseen data, so costs can only make it worse.", "no edge"
    checks.append(dict(key="costs", title="Costs don't eat the edge", status=st, value=cval, explain=cexp))

    mdd = abs(s_oos["max_drawdown"])
    st = PASS if mdd <= max_dd else WARN if mdd <= 1.5 * max_dd else FAIL
    checks.append(dict(key="risk", title="Losses stay tolerable", status=st, value=f"worst loss {_pct(mdd)}",
        explain=f"The worst fall from a peak on unseen data was {_pct(mdd)} (limit set: {_pct(max_dd, 0)}). "
                f"The longest stretch without a new high was {_duration(s_oos['longest_drawdown_periods'])}."))

    st, msg = smoothness_status(smooth, s_full["sharpe"])
    checks.append(dict(key="smooth", title="Not suspiciously smooth", status=st,
                       value=f"smoothness {_f(smooth)}", explain=msg))

    return dict(ticker=ticker, strategy=strategy, res=res, net=net, gross=gross, bh=bh, cut=cut, checks=checks,
                s_is=s_is, s_oos=s_oos, s_full=m.summary(net, ppy), bh_oos=m.summary(bh.iloc[cut:], ppy),
                dsr_full=dsr_full, verdict=verdict(checks), oos_net=oos_net, cost_bps=cost_bps, split=split)


def held_up_status(dec: dict, is_sharpe: float) -> tuple[str, str]:
    """Status and plain-English message for the 'held up on unseen data' check."""
    pct = dec["live_percentile_vs_backtest"]
    if dec["verdict"] == "no_data":
        return FAIL, "The strategy made no trades (or its returns never varied) on unseen data, so there is nothing to test."
    if dec["verdict"] == "decayed":
        return FAIL, "Performance on unseen data is much worse than when the settings were chosen: a sign the earlier result was partly luck or the edge faded."
    if is_sharpe <= 0:
        return WARN, "Even the best-looking settings showed no edge in the earlier period. A good result only on unseen data is usually luck."
    if dec["verdict"] == "watch":
        if np.isfinite(pct) and pct >= 0.15 and dec.get("cusum_alarm_index") is not None:
            when = str(dec.get("cusum_alarm_date"))[:10]
            return WARN, (f"Overall it did at least as well as before, but returns suffered a sudden sharp drop around {when}. "
                          "Worth checking what happened then and whether it could happen again.")
        return WARN, "Unseen-data performance is on the weak side of what the earlier results would suggest."
    return PASS, "Unseen-data performance is in line with what the earlier results would suggest."


def beats_holding_status(net_sr: float, bh_sr: float, mdd: float, bh_mdd: float, ticker: str) -> tuple[str, str]:
    """Does the strategy add value over simply holding the asset? (unseen data, after costs)"""
    if not (np.isfinite(net_sr) and np.isfinite(bh_sr)):
        return FAIL, "There was not enough variation in returns to compare the strategy with simply holding."
    nums = (f"Sharpe {_f(net_sr)} vs {_f(bh_sr)} for holding {ticker}; "
            f"worst fall {_pct(mdd)} vs {_pct(bh_mdd)}.")
    diff = net_sr - bh_sr
    if diff >= 0.1:
        return PASS, f"It earned more per unit of risk than simply holding. {nums}"
    if diff >= -0.1:
        return WARN, f"It roughly matched simply holding, so the extra effort adds little. {nums}"
    if bh_mdd > 0 and mdd <= 0.7 * bh_mdd:
        return WARN, f"It earned less per unit of risk than holding, but gave a much smoother ride. {nums}"
    return FAIL, f"Simply holding {ticker} did as well or better with no effort, so the strategy added no clear value. {nums}"


def smoothness_status(smooth: float, sharpe: float) -> tuple[str, str]:
    """Flag 'too good to be true': an almost straight equity curve AND an unusually high Sharpe.

    A steadily rising market alone produces a fairly straight curve, so smoothness by itself is not suspicious.
    """
    too_smooth = np.isfinite(smooth) and smooth > 0.98
    too_good = (not np.isfinite(sharpe)) or sharpe > 2.5
    if too_smooth and too_good:
        return FAIL, ("The return curve is almost a straight line with an unusually high Sharpe, which real trading essentially never produces. "
                      "Check for hindsight bias or data errors.")
    return PASS, ("The return curve has the ups and downs real trading has. "
                  "An almost perfectly straight line combined with a Sharpe above about 2.5 would be a red flag.")


def verdict(checks: list[dict]) -> tuple[str, str, str]:
    st = {c["key"]: c["status"] for c in checks}
    if FAIL in (st["money"], st["luck"], st["held_up"]):
        return ("red", "Not convincing",
                "The evidence does not support a real edge. At least one core test (profit after costs, statistical confidence, or holding up on unseen data) failed.")
    if all(s == PASS for s in st.values()):
        return ("green", "Promising: worth deeper testing",
                "It passed every check. That is a reason to test further (more data, other markets, live paper trading), not a reason to trade it.")
    return ("yellow", "Mixed: needs more evidence",
            "No core test failed outright, but one or more checks raised a warning. Read the checks below to see which.")


def portfolio_section(results: list[dict]) -> dict | None:
    if len(results) < 2:
        return None
    R = pd.concat({r["ticker"]: r["oos_net"] for r in results}, axis=1).dropna()
    R = R.loc[:, R.std(ddof=1) > 1e-12]          # drop strategies that never traded (zero variance)
    if R.shape[1] < 2 or len(R) < 60:
        return None
    div = pf.diversification(R)
    ew = R.mean(axis=1)
    out = {"div": div, "ew": m.summary(ew), "singles": {c: m.sharpe(R[c]) for c in R}, "n": len(R)}
    look = min(252, len(R) // 3)
    if len(R) >= look + 63:
        out["compare"] = pf.compare(R, lookback=look, rebal=21)
    return out


# ----------------------------------------------------------------------------- charts
def svg_chart(series: dict[str, pd.Series], split_at: int | None = None, width=720, height=250, fmt=_f) -> str:
    names = list(series)
    n = len(series[names[0]])
    allv = np.concatenate([s.to_numpy() for s in series.values()])
    lo, hi = float(np.nanmin(allv)), float(np.nanmax(allv))
    pad = (hi - lo) * 0.08 or 1.0
    lo, hi = lo - pad, hi + pad
    L, R, T, B = 54, 10, 10, 46
    X = lambda i: L + (width - L - R) * i / max(n - 1, 1)
    Y = lambda v: T + (height - T - B) * (1 - (v - lo) / (hi - lo))
    step = max(1, n // 400)
    idx = sorted(set(range(0, n, step)) | {n - 1})
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" xmlns="http://www.w3.org/2000/svg">']
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        parts.append(f'<line class="grid" x1="{L}" x2="{width - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>'
                     f'<text class="ax" x="{L - 6}" y="{Y(v) + 4:.1f}" text-anchor="end">{fmt(v)}</text>')
    if split_at is not None:
        x = X(split_at)
        parts.append(f'<line class="split" x1="{x:.1f}" x2="{x:.1f}" y1="{T}" y2="{height - B}"/>'
                     f'<text class="ax" x="{x + 5:.1f}" y="{T + 11}">unseen data →</text>')
    for j, name in enumerate(names):
        vals = series[name].to_numpy()
        pts = " ".join(f"{X(i):.1f},{Y(vals[i]):.1f}" for i in idx)
        parts.append(f'<polyline class="l{j}" fill="none" points="{pts}"/>')
    ix = series[names[0]].index
    parts.append(f'<text class="ax" x="{L}" y="{height - 28}">{html.escape(str(ix[0])[:10])}</text>'
                 f'<text class="ax" x="{width - R}" y="{height - 28}" text-anchor="end">{html.escape(str(ix[-1])[:10])}</text>')
    if len(names) > 1:                                   # legend on its own row, clear of the dates
        for j, nm in enumerate(names):
            parts.append(f'<text class="lg{j}" x="{L + j * 210}" y="{height - 6}" font-size="12">■ {html.escape(nm)}</text>')
    parts.append("</svg>")
    return "".join(parts)


# ----------------------------------------------------------------------------- html
CSS = """
:root{--bg:#fff;--fg:#111827;--muted:#6b7280;--card:#f9fafb;--line:#e5e7eb;--g:#15803d;--gb:#dcfce7;--y:#a16207;--yb:#fef9c3;--r:#b91c1c;--rb:#fee2e2;--l0:#2563eb;--l1:#9ca3af}
@media (prefers-color-scheme:dark){:root{--bg:#0b0f17;--fg:#e5e7eb;--muted:#9ca3af;--card:#111827;--line:#1f2937;--g:#4ade80;--gb:#052e16;--y:#facc15;--yb:#422006;--r:#f87171;--rb:#450a0a;--l0:#60a5fa;--l1:#6b7280}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 -apple-system,Segoe UI,Helvetica,Arial,sans-serif}
main{max-width:860px;margin:0 auto;padding:28px 18px 60px}h1{font-size:26px;margin:0 0 4px}h2{font-size:20px;margin:34px 0 10px}
.sub{color:var(--muted);margin:0 0 20px}.banner{border-radius:12px;padding:16px 18px;margin:14px 0 8px;border:1px solid var(--line)}
.banner b{font-size:20px;display:block}.green{background:var(--gb);color:var(--g)}.yellow{background:var(--yb);color:var(--y)}.red{background:var(--rb);color:var(--r)}
.banner span{color:var(--fg)}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:14px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}.card small{color:var(--muted);display:block}.card b{font-size:20px}
.check{display:grid;grid-template-columns:30px 1fr;gap:10px;padding:12px 0;border-bottom:1px solid var(--line)}
.ic{width:26px;height:26px;border-radius:50%;text-align:center;font-weight:700;line-height:26px}.ic.pass{background:var(--gb);color:var(--g)}.ic.warn{background:var(--yb);color:var(--y)}.ic.fail{background:var(--rb);color:var(--r)}
.check b{display:block}.check em{color:var(--muted);font-style:normal;font-size:14px}.check p{margin:2px 0 0}
table{border-collapse:collapse;width:100%;font-size:15px}td,th{padding:7px 8px;border-bottom:1px solid var(--line);text-align:left}th{color:var(--muted);font-weight:600}
svg{width:100%;height:auto}.grid{stroke:var(--line)}.ax{fill:var(--muted);font-size:11px}.split{stroke:var(--muted);stroke-dasharray:4 4}
.l0{stroke:var(--l0);stroke-width:1.8}.l1{stroke:var(--l1);stroke-width:1.4}.lg0{fill:var(--l0)}.lg1{fill:var(--l1)}
details{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px;margin:10px 0}summary{cursor:pointer;font-weight:600}
.note{color:var(--muted);font-size:14px}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
.dot.green{background:var(--g)}.dot.yellow{background:var(--y)}.dot.red{background:var(--r)}
"""

GLOSSARY = [
    ("Sharpe ratio", "Return earned per unit of risk taken. Around 0.5 is decent, 1 is strong, above 2 on real trading is rare and deserves suspicion."),
    ("Drawdown", "How far the account falls from its previous high before recovering. It tells you how bad the pain gets."),
    ("Unseen data", "The most recent part of history, kept aside and never used when choosing the strategy's settings. It is the fairest test available."),
    ("Basis point (bp)", "One hundredth of one percent. 2 bp of cost means paying 0.02% every time the position changes."),
    ("Overfitting", "Tuning a strategy until it fits past noise. It looks great on history and then fails on new data."),
]


def _card(label, value):
    return f'<div class="card"><small>{html.escape(label)}</small><b>{html.escape(value)}</b></div>'


def _ticker_section(r: dict, split_pct: int) -> str:
    cls, head, sub = r["verdict"]
    s, bh, spec = r["s_oos"], r["bh_oos"], STRATEGIES[r["strategy"]]
    eq = (1 + r["net"]).cumprod()
    bheq = (1 + r["bh"]).cumprod()
    dd = m.drawdown_series(r["net"])
    checks = "".join(
        f'<div class="check"><div class="ic {c["status"]}">{ICON[c["status"]]}</div>'
        f'<div><b>{html.escape(c["title"])} <em>· {html.escape(c["value"])}</em></b><p>{html.escape(c["explain"])}</p></div></div>'
        for c in r["checks"])
    cards = "".join([
        _card("Yearly return (unseen, after costs)", _pct(s["ann_return"])),
        _card("Same period, just holding", _pct(bh["ann_return"])),
        _card("Sharpe ratio (unseen)", _f(s["sharpe"])),
        _card("Worst fall (unseen)", _pct(abs(s["max_drawdown"]))),
        _card("Longest without a new high", _duration(s["longest_drawdown_periods"])),
    ])
    return f"""
<section><h2>{html.escape(r['ticker'])}</h2>
<div class="banner {cls}"><b>{html.escape(head)}</b><span>{html.escape(sub)}</span></div>
<p class="note">Strategy: {html.escape(spec['label'])}. {html.escape(spec['blurb'])}
Settings chosen from {r['res']['n_trials']} variations using only the first {split_pct}% of history ({html.escape(r['res']['best_key'])}).
Because the best of several tries always looks flattering, the verdict is judged on the last {100 - split_pct}%, which played no part in choosing.</p>
<div class="cards">{cards}</div>
<h3>The seven checks</h3>{checks}
<h3>Growth of $1</h3>{svg_chart({'Strategy (after costs)': eq, 'Just holding': bheq}, split_at=r['cut'])}
<h3>Losses from the previous high</h3>{svg_chart({'Strategy': dd}, split_at=r['cut'], fmt=lambda v: _pct(v, 0))}
</section>"""


def build_report(prices: pd.DataFrame, strategy: str = "sma", cost_bps: float = 2.0, split: float = 0.7,
                 max_dd: float = 0.20, source: str = "") -> tuple[str, list[dict]]:
    results = [analyze_ticker(prices[c].dropna(), c, strategy, cost_bps, split, max_dd, n_tickers=prices.shape[1])
               for c in prices.columns]
    split_pct = int(round(split * 100))
    summary = ""
    if len(results) > 1:
        rows = "".join(
            f'<tr><td><span class="dot {r["verdict"][0]}"></span>{html.escape(r["ticker"])}</td><td>{html.escape(r["verdict"][1])}</td>'
            f'<td>{_pct(r["s_oos"]["ann_return"])}</td><td>{_f(r["s_oos"]["sharpe"])}</td><td>{_pct(abs(r["s_oos"]["max_drawdown"]))}</td></tr>'
            for r in results)
        summary = (f'<h2>Summary</h2><table><tr><th>Ticker</th><th>Verdict</th><th>Yearly return</th><th>Sharpe</th><th>Worst fall</th></tr>{rows}</table>'
                   f'<p class="note">All figures are from unseen data, after costs.</p>')
    port = portfolio_section(results)
    port_html = ""
    if port:
        d = port["div"]
        cmp_html = ""
        if "compare" in port:
            rows = "".join(f"<tr><td>{html.escape(k.replace('_', ' '))}</td><td>{_pct(v.ann_return)}</td><td>{_f(v.sharpe)}</td><td>{_pct(abs(v.max_drawdown))}</td></tr>"
                           for k, v in port["compare"].iterrows())
            cmp_html = (f'<p>Ways of weighting them, each tested without peeking ahead:</p>'
                        f'<table><tr><th>Method</th><th>Yearly return</th><th>Sharpe</th><th>Worst fall</th></tr>{rows}</table>')
        port_html = (f'<h2>Combining them</h2><p>You have {d["n_strategies"]} strategies, but because they tend to move together they behave like '
                     f'about <b>{d["effective_bets"]:.1f}</b> independent bets (average correlation {d["avg_pairwise_corr"]:.2f}). '
                     f'Holding them equally weighted would have returned {_pct(port["ew"]["ann_return"])} a year with a Sharpe of {_f(port["ew"]["sharpe"])}, '
                     f'on unseen data.</p>{cmp_html}')
    gloss = "".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>" for k, v in GLOSSARY)
    limits = ("This report tests simple rules on past prices. It cannot account for changing markets, liquidity, taxes, or how you would behave in a real drawdown. "
              "Results on a few years of data can be luck either way. A green verdict means \"worth testing further\", never \"safe to trade\". Not investment advice.")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Strategy report card</title><style>{CSS}</style></head><body><main>
<h1>Strategy report card</h1>
<p class="sub">{html.escape(', '.join(prices.columns))} · {str(prices.index[0])[:10]} to {str(prices.index[-1])[:10]} · generated {dt.date.today()} {html.escape(source)}</p>
{summary}{''.join(_ticker_section(r, split_pct) for r in results)}{port_html}
<h2>Plain-English glossary</h2><table>{gloss}</table>
<h2>What this report cannot tell you</h2><p class="note">{html.escape(limits)}</p>
</main></body></html>"""
    return page, results
