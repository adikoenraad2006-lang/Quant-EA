#!/usr/bin/env python3
"""LAYER 3 -- robustness: parameter sensitivity and a bootstrap stress test.

Two checks that catch survivors which only worked by luck or on one magic
setting.  Reads results/sweep_results.csv from layer 2.

    python layer3_robustness.py
"""

from __future__ import annotations

import argparse
import ast
import os

import numpy as np
import pandas as pd

from qea.cli import add_data_args, load_from_args
from qea.config import (BOOTSTRAP_N, FRAGILE_DD, RESULTS_DIR, TEST_BARS, TRAIN_BARS)
from qea.backtest import max_drawdown, run_backtest, sharpe
from qea.strategies import Config, REGISTRY

SWEEP_CSV = os.path.join(RESULTS_DIR, "sweep_results.csv")
SENSITIVITY_CSV = os.path.join(RESULTS_DIR, "param_sensitivity.csv")
BOOTSTRAP_CSV = os.path.join(RESULTS_DIR, "bootstrap_results.csv")


# ------------------------------------------------------ parameter sensitivity
def parameter_sensitivity(sweep: pd.DataFrame) -> pd.DataFrame:
    """Per family: how much does the out-of-sample Sharpe depend on the setting?

    Each config is first collapsed to its mean OOS Sharpe across assets, so the
    spread that gets reported is spread *across parameter settings*, which is
    the thing that tells curve-fitting from a real edge.
    """
    per_config = (sweep.groupby(["family", "category", "config"], as_index=False)
                       .agg(config_oos_sharpe=("oos_sharpe", "mean"),
                            assets=("oos_sharpe", "size")))
    out = (per_config.groupby(["family", "category"], as_index=False)
                     .agg(n_configs=("config", "size"),
                          mean_oos_sharpe=("config_oos_sharpe", "mean"),
                          std_oos_sharpe=("config_oos_sharpe", "std"),
                          best_config_sharpe=("config_oos_sharpe", "max"),
                          worst_config_sharpe=("config_oos_sharpe", "min"),
                          frac_configs_positive=("config_oos_sharpe",
                                                 lambda s: float((s > 0).mean()))))
    row_level = (sweep.groupby("family")
                      .agg(frac_backtests_positive=("oos_sharpe", lambda s: float((s > 0).mean())),
                           n_backtests=("oos_sharpe", "size")))
    out = out.merge(row_level, on="family")
    out["spread_ratio"] = out["std_oos_sharpe"] / out["mean_oos_sharpe"].abs().replace(0.0, np.nan)
    return out.sort_values("mean_oos_sharpe", ascending=False).reset_index(drop=True)


def print_sensitivity(sens: pd.DataFrame) -> None:
    print("\n" + "=" * 96)
    print("PARAMETER SENSITIVITY -- one row per strategy family, across its parameter grid")
    print("=" * 96)
    print(f"  {'family':<22} {'cat':<11} {'cfgs':>5} {'mean OOS':>9} {'std':>7} "
          f"{'+frac':>7} {'best':>7} {'worst':>7}  verdict")
    for _, r in sens.iterrows():
        if r["mean_oos_sharpe"] <= 0:
            verdict = "no edge"
        elif r["frac_configs_positive"] >= 0.8 and r["std_oos_sharpe"] < 0.5 * abs(r["mean_oos_sharpe"]):
            verdict = "stable across settings"
        elif r["frac_configs_positive"] < 0.5:
            verdict = "one magic setting"
        else:
            verdict = "mixed"
        print(f"  {r['family']:<22} {r['category']:<11} {int(r['n_configs']):>5} "
              f"{r['mean_oos_sharpe']:>9.3f} {r['std_oos_sharpe']:>7.3f} "
              f"{r['frac_configs_positive']:>6.0%} {r['best_config_sharpe']:>7.3f} "
              f"{r['worst_config_sharpe']:>7.3f}  {verdict}")
    print("\n  A tight std with a high positive fraction means the edge does not rest on one")
    print("  magic number.  A high best with a low positive fraction is a curve-fit warning.")


# ------------------------------------------------------------- bootstrap
def bootstrap(returns: np.ndarray, n: int = BOOTSTRAP_N, method: str = "resample",
              seed: int = 7) -> dict:
    """Reshuffle the out-of-sample daily returns n times and score each path.

    method="resample" draws with replacement (the usual bootstrap): both the
    Sharpe and the drawdown vary.  method="shuffle" permutes the actual returns,
    which leaves the Sharpe numerically unchanged by construction -- only the
    equity path, and so the drawdown, moves.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 30:
        return {}
    rng = np.random.default_rng(seed)
    idx = (rng.integers(0, r.size, size=(n, r.size)) if method == "resample"
           else np.argsort(rng.random((n, r.size)), axis=1))
    paths = r[idx]

    sds = paths.std(axis=1, ddof=1)
    sharpes = np.where(sds > 0, paths.mean(axis=1) / np.where(sds > 0, sds, 1.0) * np.sqrt(252), np.nan)
    equity = np.cumprod(1.0 + paths, axis=1)
    dds = (equity / np.maximum.accumulate(equity, axis=1) - 1.0).min(axis=1)

    return {
        "boot_sharpe_p05": float(np.nanpercentile(sharpes, 5)),
        "boot_sharpe_p50": float(np.nanpercentile(sharpes, 50)),
        "boot_sharpe_p95": float(np.nanpercentile(sharpes, 95)),
        "boot_dd_p05": float(np.percentile(dds, 5)),
        "boot_dd_median": float(np.percentile(dds, 50)),
        "boot_worst_dd": float(dds.min()),
        "boot_frac_sharpe_positive": float(np.nanmean(sharpes > 0)),
        "n_paths": int(n),
    }


def stress_survivors(survivors: pd.DataFrame, prices: dict, n_boot: int,
                     method: str, train_bars: int, test_bars: int) -> pd.DataFrame:
    rows = []
    for _, r in survivors.iterrows():
        df = prices.get(r["asset"])
        if df is None:
            continue
        meta = REGISTRY[r["family"]]
        cfg = Config(r["config"], meta["function"], ast.literal_eval(r["params"]),
                     meta["category"], r["family"])
        res = run_backtest(df, cfg, r["asset"], train_bars, test_bars, keep_returns=True)
        if not res or "oos_returns" not in res:
            continue
        oos = res["oos_returns"].to_numpy()
        boot = bootstrap(oos, n_boot, method)
        if not boot:
            continue
        row = {"asset": r["asset"], "config": r["config"], "family": r["family"],
               "category": r["category"], "actual_oos_sharpe": res["oos_sharpe"],
               "actual_oos_maxdd": res["oos_maxdd"], "trades": res["trades"], **boot}
        row["flag"] = "solid" if row["boot_worst_dd"] > FRAGILE_DD else "fragile"
        rows.append(row)
    return pd.DataFrame(rows)


def print_bootstrap(boot: pd.DataFrame, method: str) -> None:
    print("\n" + "=" * 104)
    print(f"BOOTSTRAP STRESS TEST -- {int(boot['n_paths'].iloc[0]) if len(boot) else 0} "
          f"reshuffles per survivor (method={method})")
    print("=" * 104)
    if not len(boot):
        print("  no survivors to stress.")
        return
    print(f"  {'asset':<9} {'config':<40} {'actual':>7} {'Sh p05':>7} {'Sh p50':>7} "
          f"{'Sh p95':>7} {'med DD':>8} {'worst DD':>9}  flag")
    for _, r in boot.iterrows():
        print(f"  {r['asset']:<9} {r['config'][:40]:<40} {r['actual_oos_sharpe']:>7.2f} "
              f"{r['boot_sharpe_p05']:>7.2f} {r['boot_sharpe_p50']:>7.2f} "
              f"{r['boot_sharpe_p95']:>7.2f} {r['boot_dd_median']:>7.1%} "
              f"{r['boot_worst_dd']:>8.1%}  {r['flag']}")
    n_solid = int((boot["flag"] == "solid").sum())
    print(f"\n  solid {n_solid} / {len(boot)}   fragile {len(boot) - n_solid} / {len(boot)}"
          f"   (fragile = worst-case bootstrap drawdown worse than {FRAGILE_DD:.0%})")
    if method == "shuffle":
        print("  Note: permuting returns cannot change the Sharpe ratio, so the Sharpe")
        print("  percentiles above are identical by construction; only the drawdowns move.")


def main() -> None:
    ap = add_data_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--sweep", default=SWEEP_CSV)
    ap.add_argument("--top", type=int, default=25, help="survivors to stress test")
    ap.add_argument("--n-boot", type=int, default=BOOTSTRAP_N)
    ap.add_argument("--method", choices=["resample", "shuffle"], default="resample")
    ap.add_argument("--train-bars", type=int, default=TRAIN_BARS)
    ap.add_argument("--test-bars", type=int, default=TEST_BARS)
    args = ap.parse_args()

    if not os.path.exists(args.sweep):
        raise SystemExit(f"{args.sweep} not found — run layer2_sweep.py first")
    sweep = pd.read_csv(args.sweep)

    print("=" * 78)
    print("LAYER 3 -- ROBUSTNESS")
    print("=" * 78)
    print(f"Loaded {len(sweep)} sweep rows from {args.sweep}")

    sens = parameter_sensitivity(sweep)
    sens.to_csv(SENSITIVITY_CSV, index=False)
    print_sensitivity(sens)

    survivors = sweep[sweep["survived"]].sort_values("oos_sharpe", ascending=False).head(args.top)
    prices = load_from_args(args, tickers=sorted(survivors["asset"].unique()) or None,
                            verbose=False)
    boot = stress_survivors(survivors, prices, args.n_boot, args.method,
                            args.train_bars, args.test_bars)
    boot.to_csv(BOOTSTRAP_CSV, index=False)
    print_bootstrap(boot, args.method)
    print(f"\nWrote {SENSITIVITY_CSV} and {BOOTSTRAP_CSV}")


if __name__ == "__main__":
    main()
