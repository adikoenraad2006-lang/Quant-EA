#!/usr/bin/env python3
"""Invariant checks for the strategy tester.

Runs on synthetic prices -- these test the machinery, not any edge.

    python tests/test_pipeline.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from qea.backtest import (annual_return, max_drawdown, net_returns, run_backtest,
                          sharpe, trade_count, walk_forward)
from qea.strategies import REGISTRY, build_configs
from qea.synthetic import synthetic_ohlcv, synthetic_universe

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append(name)


# ------------------------------------------------------------------ metrics
def test_metrics() -> None:
    print("\nmetrics")
    r = pd.Series([0.01] * 252)
    check("sharpe of a constant series is undefined", np.isnan(sharpe(r)))

    rng = np.random.default_rng(1)
    x = rng.normal(0.0005, 0.01, 5000)
    expected = x.mean() / x.std(ddof=1) * np.sqrt(252)
    check("sharpe matches the definition", abs(sharpe(x) - expected) < 1e-12)

    check("max drawdown of a rising series is 0", max_drawdown([0.01, 0.01, 0.01]) == 0.0)
    dd = max_drawdown([0.5, -0.5])          # 1.5 -> 0.75
    check("max drawdown is measured from the peak", abs(dd - (-0.5)) < 1e-12, f"{dd}")

    check("annual return compounds", abs(annual_return([0.0] * 252) - 0.0) < 1e-12)

    check("trade count counts position changes",
          trade_count([0, 1, 1, -1, -1, 0]) == 3, str(trade_count([0, 1, 1, -1, -1, 0])))
    check("entering from flat counts as a trade", trade_count([1, 1, 1]) == 1)


# ------------------------------------------------------------------ costs
def test_costs() -> None:
    print("\ncosts")
    df = synthetic_ohlcv(0, n=500)
    flat = pd.Series(1.0, index=df.index)
    free = net_returns(df, flat, 0.0)
    charged = net_returns(df, flat, 10.0)
    check("buy and hold pays cost once", abs((free - charged).sum() - 10.0 / 1e4) < 1e-12)

    flip = pd.Series(np.where(np.arange(len(df)) % 2 == 0, 1.0, -1.0), index=df.index)
    paid = (net_returns(df, flip, 10.0) - net_returns(df, flip, 0.0)).sum()
    expected = -(len(df) - 1) * 2 * 10.0 / 1e4 - 10.0 / 1e4
    check("a flip pays twice the one-way cost", abs(paid - expected) < 1e-10,
          f"{paid} vs {expected}")

    check("costs never help", (net_returns(df, flip, 10.0) <= net_returns(df, flip, 0.0)).all())


# ------------------------------------------------------------------ strategies
def test_strategy_contract() -> None:
    print("\nstrategy contract (all configs)")
    df = synthetic_ohlcv(3, n=1500)
    configs = build_configs()

    bad_values, non_causal, fired = [], [], set()
    cut = len(df) - 100
    truncated_df = df.iloc[:cut]
    for cfg in configs:
        pos = cfg.function(df, **cfg.params)
        vals = set(np.unique(pos.to_numpy()))
        if not vals.issubset({-1.0, 0.0, 1.0}):
            bad_values.append(cfg.name)
        if len(pos) != len(df) or not pos.index.equals(df.index):
            bad_values.append(cfg.name + " (index)")
        if pos.abs().sum() > 0:
            fired.add(cfg.family)
        trunc = cfg.function(truncated_df, **cfg.params)
        if not np.allclose(pos.iloc[:cut].to_numpy(), trunc.to_numpy(), equal_nan=True):
            non_causal.append(cfg.name)

    check(f"{len(configs)} configs return positions in {{-1,0,1}}", not bad_values,
          str(bad_values[:5]))
    check("no config sees the future (truncation invariance)", not non_causal,
          str(non_causal[:5]))
    # a single rare parameterisation may sit flat on one series (filter 5 drops it),
    # but a family that never fires anywhere is a bug
    silent = sorted(set(REGISTRY) - fired)
    check("every family fires at least once", not silent, str(silent))

    # an explicit one-bar-lag check: a signal built on bar t trades from bar t+1
    idx = df.index
    fake = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1.0},
                        index=idx)
    from qea.strategies import strategy

    @strategy("__lagprobe", "trend")
    def _probe(d):
        return pd.Series(1.0, index=d.index)

    lagged = _probe(fake)
    check("the execution lag is applied", lagged.iloc[0] == 0.0 and lagged.iloc[1] == 1.0)
    REGISTRY.pop("__lagprobe", None)


def test_families() -> None:
    print("\nlibrary coverage")
    configs = build_configs()
    check("47 families registered", len(REGISTRY) == 47, str(len(REGISTRY)))
    check("config count is in the hundreds", 200 <= len(configs) <= 999, str(len(configs)))
    check("config names are unique", len({c.name for c in configs}) == len(configs))
    cats = {m["category"] for m in REGISTRY.values()}
    check("all six categories are used", cats == {"trend", "meanrev", "volume",
                                                  "volatility", "pattern", "composite"}, str(cats))
    check("every config carries its params", all(isinstance(c.params, dict) for c in configs))


# ------------------------------------------------------------------ walk-forward
def test_walk_forward() -> None:
    print("\nwalk-forward")
    df = synthetic_ohlcv(4, n=2000)
    pos = pd.Series(1.0, index=df.index)
    rets = net_returns(df, pos, 0.0)
    wf = walk_forward(rets, pos, train_bars=1000, test_bars=250)
    check("out-of-sample starts after the training block",
          pd.Timestamp(wf["oos_start"]) == df.index[1000])
    check("out-of-sample windows tile the remainder", wf["oos_bars"] == 1000, str(wf["oos_bars"]))
    check("window count is right", wf["oos_windows"] == 4, str(wf["oos_windows"]))
    check("in-sample and out-of-sample do not overlap",
          wf["oos_returns"].index[0] > rets.index[999])
    check("too little history returns nothing",
          walk_forward(rets.iloc[:100], pos.iloc[:100], 1000, 250) == {})


def test_filters() -> None:
    print("\nfilters")
    from layer2_sweep import apply_filters
    rows = pd.DataFrame([
        {"oos_maxdd": -0.10, "oos_sharpe": 1.0, "is_sharpe": 1.0, "trades": 100},   # survives
        {"oos_maxdd": -0.50, "oos_sharpe": 1.0, "is_sharpe": 1.0, "trades": 100},   # deep dd
        {"oos_maxdd": -0.10, "oos_sharpe": 0.2, "is_sharpe": 1.0, "trades": 100},   # weak
        {"oos_maxdd": -0.10, "oos_sharpe": 3.0, "is_sharpe": 3.0, "trades": 100},   # too good
        {"oos_maxdd": -0.10, "oos_sharpe": 1.0, "is_sharpe": 0.2, "trades": 100},   # oos >> is
        {"oos_maxdd": -0.10, "oos_sharpe": 1.0, "is_sharpe": 1.0, "trades": 5},     # too few
        {"oos_maxdd": -0.10, "oos_sharpe": 1.0, "is_sharpe": -0.5, "trades": 100},  # is negative
        {"oos_maxdd": np.nan, "oos_sharpe": np.nan, "is_sharpe": np.nan, "trades": 0},
    ])
    f = apply_filters(rows)
    check("only the clean row survives", list(f["survived"]) == [True] + [False] * 7,
          str(list(f["survived"])))
    check("NaNs never survive", not bool(f["survived"].iloc[-1]))


# ------------------------------------------------------------------ end to end
def test_end_to_end() -> None:
    print("\nend to end")
    prices = synthetic_universe(["SPY", "QQQ", "BTC-USD"], n=2000)
    configs = build_configs()[:40]
    rows = [run_backtest(df, c, t) for t, df in prices.items() for c in configs]
    rows = [r for r in rows if r]
    check("every backtest produced a row", len(rows) == 3 * len(configs), str(len(rows)))
    check("no backtest errored", not any("error" in r for r in rows))
    d = pd.DataFrame(rows)
    check("crypto is charged more than SPY",
          d[d.asset == "BTC-USD"].cost_bps.iloc[0] > d[d.asset == "SPY"].cost_bps.iloc[0])
    check("drawdowns are non-positive", bool((d["oos_maxdd"] <= 0).all()))
    check("exposure is a fraction", bool(((d["exposure"] >= 0) & (d["exposure"] <= 1)).all()))

    from layer3_robustness import bootstrap
    rng = np.random.default_rng(2)
    r = rng.normal(0.0004, 0.01, 2000)
    b = bootstrap(r, n=200, method="resample")
    check("bootstrap percentiles are ordered",
          b["boot_sharpe_p05"] < b["boot_sharpe_p50"] < b["boot_sharpe_p95"])
    check("worst bootstrap drawdown is the worst",
          b["boot_worst_dd"] <= b["boot_dd_p05"] <= b["boot_dd_median"])
    shuf = bootstrap(r, n=50, method="shuffle")
    check("permutation leaves the Sharpe fixed",
          abs(shuf["boot_sharpe_p05"] - shuf["boot_sharpe_p95"]) < 1e-9)


def test_cross_sectional() -> None:
    print("\ncross-sectional (layer 4)")
    from layer4_cross_sectional import build_panel, cross_sectional_portfolio, momentum
    tickers = [f"A{i}" for i in range(12)]
    prices = synthetic_universe(tickers, n=1500, drift=0.0)
    closes, returns = build_panel(prices)
    check("panel is aligned", closes.shape == (1500, 12), str(closes.shape))

    m = momentum(closes, 252, 21)
    manual = closes.iloc[-22] / closes.iloc[-274] - 1.0
    check("12-1 momentum skips the last month",
          np.allclose(m.iloc[-1].to_numpy(), manual.to_numpy()))

    port = cross_sectional_portfolio(closes, returns, 126, 0, rebalance=21)
    check("portfolio is dollar neutral when fully invested",
          abs(port["gross"].iloc[-1] - 2.0) < 0.5, str(port["gross"].iloc[-1]))
    check("turnover only on rebalances",
          bool((port["turnover"][~port["rebalances"]] == 0).all()))
    # on driftless random walks the ranking has nothing to find
    s = sharpe(port["returns"])
    check("no look-ahead leak on random data (|Sharpe| < 0.6)", abs(s) < 0.6, f"sharpe={s:.2f}")


if __name__ == "__main__":
    test_metrics()
    test_costs()
    test_strategy_contract()
    test_families()
    test_walk_forward()
    test_filters()
    test_end_to_end()
    test_cross_sectional()
    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        sys.exit(1)
    print("all checks passed")
