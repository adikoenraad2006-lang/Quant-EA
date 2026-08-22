#!/usr/bin/env python3
"""LAYER 4 -- cross-sectional momentum, a standalone check.

In the main sweep momentum was tested on each asset by itself.  The strongest
documented form of momentum is cross-sectional: rank the assets against each
other, buy the winners, sell the losers.  This builds that and scores it exactly
the way the main tester scores everything else, so the numbers are comparable.

    python layer4_cross_sectional.py
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from qea.config import (END, MIN_BARS, RESULTS_DIR, START, TEST_BARS, TRAIN_BARS,
                        UNIVERSE, XS_LOOKBACKS, XS_REBALANCE, cost_bps)
from qea.backtest import annual_return, max_drawdown, sharpe, walk_forward
from qea.data import load_dataset

SWEEP_CSV = os.path.join(RESULTS_DIR, "sweep_results.csv")
XS_CSV = os.path.join(RESULTS_DIR, "cross_sectional_momentum.csv")
XS_WINDOWS_CSV = os.path.join(RESULTS_DIR, "cross_sectional_windows.csv")


# ------------------------------------------------------------------ panel
def build_panel(prices: dict[str, pd.DataFrame], coverage: float = 0.5):
    """Aligned close panel, restricted to dates most of the universe trades."""
    closes = pd.DataFrame({t: df["Close"] for t, df in prices.items()}).sort_index()
    keep = closes.notna().sum(axis=1) >= max(3, int(coverage * closes.shape[1]))
    closes = closes[keep]
    returns = closes.pct_change()
    # a gap in one name must not become a fake return for that name
    returns = returns.where(closes.notna() & closes.shift(1).notna())
    return closes, returns


def momentum(closes: pd.DataFrame, lookback: int, skip: int) -> pd.DataFrame:
    """Trailing return over `lookback` bars, ending `skip` bars ago."""
    end = closes.shift(skip)
    start = closes.shift(skip + lookback)
    return end / start - 1.0


# ------------------------------------------------------------------ portfolio
def cross_sectional_portfolio(closes, returns, lookback, skip,
                              rebalance=XS_REBALANCE, min_assets=6):
    """Long the top third, short the bottom third, equal weight, held to the
    next rebalance.  Weights drift with the assets in between, so turnover --
    and therefore cost -- is only charged when the book is actually rebalanced.
    """
    mom = momentum(closes, lookback, skip)
    tickers = list(closes.columns)
    bps = np.array([cost_bps(t) for t in tickers]) / 1e4
    ret_mat = returns[tickers].to_numpy(dtype=float)
    ret_mat = np.nan_to_num(ret_mat, nan=0.0)
    mom_mat = mom[tickers].to_numpy(dtype=float)
    tradable = closes[tickers].notna().to_numpy()

    n_dates = len(closes)
    valid = (np.isfinite(mom_mat) & tradable).sum(axis=1)
    starts = np.flatnonzero(valid >= min_assets)
    if starts.size == 0:
        return None
    first = int(starts[0])

    # signal dates: every `rebalance` bars from the first date with enough history
    signal_dates = set(range(first, n_dates, rebalance))

    weights = np.zeros(len(tickers))
    port_ret = np.zeros(n_dates)
    gross = np.zeros(n_dates)
    turnover = np.zeros(n_dates)
    rebalances = np.zeros(n_dates, dtype=bool)
    target = np.zeros(len(tickers))
    have_target = False

    for t in range(first + 1, n_dates):
        if (t - 1) in signal_dates:
            m = mom_mat[t - 1]
            ok = np.isfinite(m) & tradable[t - 1] & tradable[t]
            idx = np.flatnonzero(ok)
            third = len(idx) // 3
            if third >= 2:
                order = idx[np.argsort(m[idx])]
                new = np.zeros(len(tickers))
                new[order[-third:]] = 1.0 / third      # winners, +100% gross
                new[order[:third]] = -1.0 / third      # losers,  -100% gross
                target, have_target = new, True
                rebalances[t] = True
        if not have_target:
            continue
        held = target if rebalances[t] else weights
        turnover[t] = float(np.abs(held - weights).sum())
        cost = float((np.abs(held - weights) * bps).sum())
        port_ret[t] = float((held * ret_mat[t]).sum()) - cost
        gross[t] = float(np.abs(held).sum())
        weights = held * (1.0 + ret_mat[t])            # drift into the next bar

    idx_dates = closes.index
    return {
        "returns": pd.Series(port_ret, index=idx_dates).iloc[first + 1:],
        "gross": pd.Series(gross, index=idx_dates).iloc[first + 1:],
        "turnover": pd.Series(turnover, index=idx_dates).iloc[first + 1:],
        "rebalances": pd.Series(rebalances, index=idx_dates).iloc[first + 1:],
    }


# ------------------------------------------------------------------ scoring
def score(port: dict, train_bars: int, test_bars: int) -> dict | None:
    rets, gross = port["returns"], port["gross"]
    wf = walk_forward(rets, gross, train_bars, test_bars)
    if not wf:
        return None
    oos = wf["oos_returns"]
    rebal_oos = int(port["rebalances"].loc[oos.index].sum())
    turn_oos = float(port["turnover"].loc[oos.index].sum())

    # regime check: how much of the result is one good stretch?
    win_sharpes = np.array(wf["oos_window_sharpes"], dtype=float)
    by_year = oos.groupby(oos.index.year).apply(lambda s: sharpe(s))
    best_year = by_year.idxmax() if len(by_year) else None
    ex_best = oos[oos.index.year != best_year] if best_year is not None else oos

    return {
        "is_sharpe": wf["is_sharpe"],
        "oos_sharpe": wf["oos_sharpe"],
        "oos_maxdd": wf["oos_maxdd"],
        "oos_ann_return": wf["oos_return"],
        "oos_bars": wf["oos_bars"],
        "oos_windows": wf["oos_windows"],
        "rebalances_oos": rebal_oos,
        "avg_turnover_per_rebalance": turn_oos / max(rebal_oos, 1),
        "best_window_sharpe": float(np.nanmax(win_sharpes)),
        "worst_window_sharpe": float(np.nanmin(win_sharpes)),
        "frac_windows_positive": float(np.nanmean(win_sharpes > 0)),
        "best_year": int(best_year) if best_year is not None else None,
        "oos_sharpe_ex_best_year": sharpe(ex_best),
        "oos_start": wf["oos_start"],
        "oos_end": wf["oos_end"],
        "_windows": list(zip(wf["oos_window_labels"], wf["oos_window_sharpes"])),
        "_by_year": by_year,
    }


def single_asset_momentum(sweep_path: str) -> pd.DataFrame:
    if not os.path.exists(sweep_path):
        return pd.DataFrame()
    sweep = pd.read_csv(sweep_path)
    fams = ["ts_momentum", "roc_momentum", "dual_momentum"]
    return sweep[sweep["family"].isin(fams)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    ap.add_argument("--min-bars", type=int, default=MIN_BARS)
    ap.add_argument("--train-bars", type=int, default=TRAIN_BARS)
    ap.add_argument("--test-bars", type=int, default=TEST_BARS)
    ap.add_argument("--rebalance", type=int, default=XS_REBALANCE)
    ap.add_argument("--sweep", default=SWEEP_CSV)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 78)
    print("LAYER 4 -- CROSS-SECTIONAL MOMENTUM")
    print("=" * 78)
    prices = load_dataset(args.synthetic, UNIVERSE, args.start, args.end, args.min_bars,
                          verbose=False)
    if not prices:
        raise SystemExit("no price data — cannot run the cross-sectional test")

    closes, returns = build_panel(prices)
    print(f"universe: {closes.shape[1]} assets, {closes.shape[0]} common dates "
          f"({closes.index[0].date()} -> {closes.index[-1].date()})")
    print(f"rebalance every {args.rebalance} trading days, long top third / short bottom "
          f"third, equal weight,\nper-asset transaction costs charged on turnover\n")

    rows, window_rows = [], []
    for label, (lookback, skip) in XS_LOOKBACKS.items():
        port = cross_sectional_portfolio(closes, returns, lookback, skip, args.rebalance)
        if port is None:
            print(f"  {label}: not enough history")
            continue
        res = score(port, args.train_bars, args.test_bars)
        if res is None:
            print(f"  {label}: not enough out-of-sample data")
            continue
        windows, by_year = res.pop("_windows"), res.pop("_by_year")
        rows.append({"lookback": label, "lookback_bars": lookback, "skip_bars": skip, **res})
        for lab, sh in windows:
            window_rows.append({"lookback": label, "window": lab, "sharpe": sh})
        print(f"  {label:<8} lookback {lookback:>3}d skip {skip:>2}d  "
              f"IS Sharpe {res['is_sharpe']:>6.2f}  OOS Sharpe {res['oos_sharpe']:>6.2f}  "
              f"OOS maxDD {res['oos_maxdd']:>7.1%}  ann {res['oos_ann_return']:>7.2%}")

    if not rows:
        raise SystemExit("no cross-sectional results")

    xs = pd.DataFrame(rows)
    xs.to_csv(XS_CSV, index=False)
    pd.DataFrame(window_rows).to_csv(XS_WINDOWS_CSV, index=False)

    # -------------------------------------------------- side by side
    single = single_asset_momentum(args.sweep)
    print("\n" + "=" * 78)
    print("REPORT -- cross-sectional vs single-asset momentum (out-of-sample)")
    print("=" * 78)
    print(f"  {'lookback':<10} {'OOS Sharpe':>11} {'OOS maxDD':>11} {'ann return':>11} "
          f"{'windows +':>10}")
    for _, r in xs.iterrows():
        print(f"  {r['lookback']:<10} {r['oos_sharpe']:>11.2f} {r['oos_maxdd']:>10.1%} "
              f"{r['oos_ann_return']:>10.2%} {r['frac_windows_positive']:>9.0%}")

    best = xs.loc[xs["oos_sharpe"].idxmax()]
    if len(single):
        s_mean = single["oos_sharpe"].mean()
        s_med = single["oos_sharpe"].median()
        s_best = single["oos_sharpe"].max()
        s_pos = float((single["oos_sharpe"] > 0).mean())
        print(f"\n  single-asset momentum from the sweep ({len(single)} backtests, "
              f"families ts_momentum / roc_momentum / dual_momentum)")
        print(f"    mean OOS Sharpe   {s_mean:>6.2f}")
        print(f"    median OOS Sharpe {s_med:>6.2f}")
        print(f"    best OOS Sharpe   {s_best:>6.2f}   (best single backtest, "
              f"cherry-picked out of {len(single)})")
        print(f"    positive fraction {s_pos:>6.0%}")
        verdict = ("BEAT" if best["oos_sharpe"] > s_mean else "DID NOT BEAT")
        print(f"\n  Plain statement: ranking assets against each other {verdict} trading "
              f"momentum on each\n  one alone. Best cross-sectional OOS Sharpe "
              f"{best['oos_sharpe']:.2f} ({best['lookback']}) vs a mean of "
              f"{s_mean:.2f}\n  across single-asset momentum backtests "
              f"(median {s_med:.2f}).")
    else:
        print(f"\n  {args.sweep} not found — run layer2_sweep.py for the side-by-side.")

    print("\n  Drawdowns, as they came out:")
    for _, r in xs.iterrows():
        print(f"    {r['lookback']:<8} OOS max drawdown {r['oos_maxdd']:>7.1%}  "
              f"(worst walk-forward window Sharpe {r['worst_window_sharpe']:>5.2f})")

    print("\n  Regime check:")
    for _, r in xs.iterrows():
        drop = r["oos_sharpe"] - r["oos_sharpe_ex_best_year"]
        note = ("leans on one year" if drop > 0.3 else "spread across years")
        print(f"    {r['lookback']:<8} best year {r['best_year']}  OOS Sharpe "
              f"{r['oos_sharpe']:>5.2f} -> {r['oos_sharpe_ex_best_year']:>5.2f} without it "
              f"({note})")

    print(f"\nWrote {XS_CSV} and {XS_WINDOWS_CSV}")


if __name__ == "__main__":
    main()
