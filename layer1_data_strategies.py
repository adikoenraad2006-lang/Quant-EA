#!/usr/bin/env python3
"""LAYER 1 -- data and the strategy library.

Downloads daily OHLCV for the asset universe and builds the full config grid,
then prints what it got.  Run it on its own to check the foundation before the
sweep; layers 2-4 import the same modules.

    python layer1_data_strategies.py
"""

from __future__ import annotations

import argparse
import collections

import numpy as np
import pandas as pd

from qea.config import END, MIN_BARS, START, UNIVERSE
from qea.data import load_dataset
from qea.strategies import CATEGORIES, REGISTRY, build_configs


def sanity_check(prices: dict[str, pd.DataFrame], configs, n: int = 25) -> None:
    """Positions must be in {-1,0,1} and must not use the same bar's data."""
    if not prices:
        print("\nSanity check skipped: no price data loaded.")
        return
    ticker, df = next(iter(prices.items()))
    rng = np.random.default_rng(0)
    picks = [configs[i] for i in rng.choice(len(configs), size=min(n, len(configs)), replace=False)]

    bad_values, look_ahead = [], []
    cut = len(df) - 50
    for cfg in picks:
        pos = cfg.function(df, **cfg.params)
        if not set(np.unique(pos.to_numpy())).issubset({-1.0, 0.0, 1.0}):
            bad_values.append(cfg.name)
        # a position must not change when a *later* bar changes
        truncated = cfg.function(df.iloc[:cut], **cfg.params)
        overlap = pos.iloc[:cut - 1]
        if not np.allclose(overlap.to_numpy(), truncated.iloc[:cut - 1].to_numpy(), equal_nan=True):
            look_ahead.append(cfg.name)

    print(f"\nSanity check on {ticker} ({len(picks)} random configs)")
    print(f"  positions in {{-1,0,1}} : {'PASS' if not bad_values else 'FAIL ' + str(bad_values)}")
    print(f"  no look-ahead          : {'PASS' if not look_ahead else 'FAIL ' + str(look_ahead)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    ap.add_argument("--min-bars", type=int, default=MIN_BARS)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--synthetic", action="store_true",
                    help="random-walk prices, for smoke-testing the pipeline offline")
    args = ap.parse_args()

    print("=" * 78)
    print("LAYER 1 -- DATA")
    print("=" * 78)
    prices = load_dataset(args.synthetic, UNIVERSE, args.start, args.end,
                          args.min_bars, use_cache=not args.no_cache)

    print("\n" + "=" * 78)
    print("LAYER 1 -- STRATEGY LIBRARY")
    print("=" * 78)
    configs = build_configs()
    by_cat = collections.Counter(c.category for c in configs)
    fam_by_cat = collections.Counter(m["category"] for m in REGISTRY.values())
    print(f"{'category':<12} {'families':>9} {'configs':>9}")
    for cat in CATEGORIES:
        print(f"{cat:<12} {fam_by_cat[cat]:>9} {by_cat[cat]:>9}")
    print(f"{'TOTAL':<12} {len(REGISTRY):>9} {len(configs):>9}")
    print(f"\nTotal configs: {len(configs)}")
    print(f"Assets loaded: {len(prices)}")
    print(f"Backtests the sweep will run: {len(configs) * len(prices)}")

    sanity_check(prices, configs)


if __name__ == "__main__":
    main()
