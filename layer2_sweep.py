#!/usr/bin/env python3
"""LAYER 2 -- the sweep, the six filters, and the funnel report.

Runs every (config x asset) combination, records in-sample Sharpe, walk-forward
out-of-sample Sharpe, out-of-sample max drawdown and trade count, writes
results/sweep_results.csv, then prints the attrition funnel.

    python layer2_sweep.py
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

from qea.config import (END, FILTERS, MIN_BARS, RESULTS_DIR, START, TEST_BARS,
                        TRAIN_BARS, UNIVERSE)
from qea.data import load_dataset
from qea.strategies import CATEGORIES, build_configs
from qea.backtest import buy_and_hold, run_backtest

SWEEP_CSV = os.path.join(RESULTS_DIR, "sweep_results.csv")
SURVIVORS_CSV = os.path.join(RESULTS_DIR, "survivors.csv")


# ------------------------------------------------------------------ sweep
def run_sweep(prices, configs, train_bars=TRAIN_BARS, test_bars=TEST_BARS) -> pd.DataFrame:
    rows, t0 = [], time.time()
    for i, (ticker, df) in enumerate(prices.items(), 1):
        n_before = len(rows)
        for cfg in configs:
            row = run_backtest(df, cfg, ticker, train_bars, test_bars)
            if row is not None:
                rows.append(row)
        print(f"  [{i:2d}/{len(prices)}] {ticker:<9} {len(rows) - n_before:4d} backtests "
              f"({time.time() - t0:6.1f}s elapsed)")
    df = pd.DataFrame(rows)
    if "error" in df.columns:
        broken = df[df["error"].notna()]
        if len(broken):
            print(f"  {len(broken)} backtests errored: "
                  f"{sorted(broken['config'].unique())[:5]}")
        df = df[df["error"].isna()].drop(columns=["error"])
    return df.reset_index(drop=True)


# ------------------------------------------------------------------ filters
def apply_filters(df: pd.DataFrame, thresholds: dict = FILTERS) -> pd.DataFrame:
    """The six survival tests, all applied to the out-of-sample numbers."""
    f = pd.DataFrame(index=df.index)
    f["f1_drawdown"] = df["oos_maxdd"] > thresholds["max_drawdown"]
    f["f2_sharpe_floor"] = df["oos_sharpe"] > thresholds["min_sharpe"]
    f["f3_sharpe_ceiling"] = df["oos_sharpe"] < thresholds["max_sharpe"]
    # OOS may not run away from IS; a big positive gap is the overfit signature.
    limit = df["is_sharpe"].abs() * (1.0 + thresholds["max_oos_over_is"])
    f["f4_oos_vs_is"] = df["oos_sharpe"] <= np.where(df["is_sharpe"] > 0, limit, np.inf)
    f["f5_trades"] = df["trades"] >= thresholds["min_trades"]
    f["f6_is_positive"] = df["is_sharpe"] > 0 if thresholds["is_sharpe_positive"] else True
    f = f.fillna(False)
    f["survived"] = f.all(axis=1)
    return f


# ------------------------------------------------------------------ report
def funnel_report(df: pd.DataFrame, flags: pd.DataFrame, bench: pd.DataFrame) -> None:
    total = len(df)
    pos_oos = int((df["oos_sharpe"] > 0).sum())
    cleared = int((df["oos_sharpe"] > FILTERS["min_sharpe"]).sum())
    survived = int(flags["survived"].sum())

    def pct(x):
        return f"{100.0 * x / total:5.1f}%" if total else "  n/a"

    print("\n" + "=" * 78)
    print("THE FUNNEL")
    print("=" * 78)
    print(f"  total backtests run           {total:7d}   {pct(total)}")
    print(f"  positive out-of-sample Sharpe {pos_oos:7d}   {pct(pos_oos)}")
    print(f"  out-of-sample Sharpe > {FILTERS['min_sharpe']:<4}   {cleared:7d}   {pct(cleared)}")
    print(f"  survived all six filters      {survived:7d}   {pct(survived)}")

    print("\n  attrition, filter by filter (each on its own, over all backtests)")
    labels = {
        "f1_drawdown": f"OOS max drawdown better than {FILTERS['max_drawdown']:.0%}",
        "f2_sharpe_floor": f"OOS Sharpe above {FILTERS['min_sharpe']}",
        "f3_sharpe_ceiling": f"OOS Sharpe below {FILTERS['max_sharpe']}",
        "f4_oos_vs_is": f"OOS Sharpe <= IS +{FILTERS['max_oos_over_is']:.0%}",
        "f5_trades": f"at least {FILTERS['min_trades']} trades",
        "f6_is_positive": "in-sample Sharpe positive",
    }
    running = pd.Series(True, index=flags.index)
    for key, label in labels.items():
        running &= flags[key]
        print(f"    {label:<42} passes {int(flags[key].sum()):6d}   "
              f"cumulative {int(running.sum()):6d}")

    print("\n" + "-" * 78)
    print("SURVIVAL BY CATEGORY")
    print("-" * 78)
    print(f"  {'category':<12} {'tested':>7} {'survived':>9} {'rate':>7} "
          f"{'mean OOS Sharpe':>17} {'survivor mean':>14}")
    work = df.assign(survived=flags["survived"].to_numpy())
    for cat in CATEGORIES:
        sub = work[work["category"] == cat]
        if not len(sub):
            continue
        surv = sub[sub["survived"]]
        print(f"  {cat:<12} {len(sub):>7} {len(surv):>9} {len(surv) / len(sub):>6.1%} "
              f"{sub['oos_sharpe'].mean():>17.3f} "
              f"{(surv['oos_sharpe'].mean() if len(surv) else float('nan')):>14.3f}")

    print("\n" + "-" * 78)
    print("SURVIVAL BY FAMILY")
    print("-" * 78)
    fam = work.groupby("family").agg(
        category=("category", "first"),
        tested=("oos_sharpe", "size"),
        survived=("survived", "sum"),
        mean_oos_sharpe=("oos_sharpe", "mean"),
    )
    fam["rate"] = fam["survived"] / fam["tested"]
    fam = fam.sort_values(["survived", "mean_oos_sharpe"], ascending=False)
    print(f"  {'family':<22} {'cat':<11} {'tested':>7} {'surv':>5} {'rate':>7} {'mean OOS Sh':>12}")
    for name, r in fam.iterrows():
        print(f"  {name:<22} {r['category']:<11} {int(r['tested']):>7} "
              f"{int(r['survived']):>5} {r['rate']:>6.1%} {r['mean_oos_sharpe']:>12.3f}")

    print("\n" + "-" * 78)
    print("TOP SURVIVORS")
    print("-" * 78)
    surv = work[work["survived"]].sort_values("oos_sharpe", ascending=False)
    if not len(surv):
        print("  none — nothing cleared all six filters.")
    else:
        bench_map = bench.set_index("asset")["oos_sharpe"].to_dict() if len(bench) else {}
        print(f"  {'asset':<9} {'config':<44} {'IS Sh':>7} {'OOS Sh':>7} "
              f"{'OOS DD':>8} {'trades':>7} {'B&H Sh':>7}")
        for _, r in surv.head(25).iterrows():
            print(f"  {r['asset']:<9} {r['config'][:44]:<44} {r['is_sharpe']:>7.2f} "
                  f"{r['oos_sharpe']:>7.2f} {r['oos_maxdd']:>7.1%} {int(r['trades']):>7} "
                  f"{bench_map.get(r['asset'], float('nan')):>7.2f}")
        print(f"\n  survivors span {surv['family'].nunique()} families "
              f"and {surv['asset'].nunique()} assets")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    ap.add_argument("--min-bars", type=int, default=MIN_BARS)
    ap.add_argument("--train-bars", type=int, default=TRAIN_BARS)
    ap.add_argument("--test-bars", type=int, default=TEST_BARS)
    ap.add_argument("--assets", nargs="*", default=UNIVERSE)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 78)
    print("LAYER 2 -- SWEEP")
    print("=" * 78)
    prices = load_dataset(args.synthetic, args.assets, args.start, args.end, args.min_bars)
    if not prices:
        raise SystemExit("no price data — cannot sweep")

    configs = build_configs()
    print(f"\n{len(configs)} configs x {len(prices)} assets = "
          f"{len(configs) * len(prices)} backtests")
    print(f"in-sample: first {args.train_bars} bars | walk-forward OOS windows of "
          f"{args.test_bars} bars\n")

    df = run_sweep(prices, configs, args.train_bars, args.test_bars)
    flags = apply_filters(df)
    out = pd.concat([df, flags], axis=1)
    out.to_csv(SWEEP_CSV, index=False)
    out[out["survived"]].to_csv(SURVIVORS_CSV, index=False)

    bench = pd.DataFrame([b for b in (buy_and_hold(d, t, args.train_bars, args.test_bars)
                                      for t, d in prices.items()) if b])
    print(f"\nTotal backtests: {len(df)}  ->  {SWEEP_CSV}")
    funnel_report(df, flags, bench)
    print(f"\nWrote {SWEEP_CSV} and {SURVIVORS_CSV}")


if __name__ == "__main__":
    main()
