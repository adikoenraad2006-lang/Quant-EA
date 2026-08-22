"""Layer 2 engine: costs, metrics and the walk-forward split.

Cost model
    A position of +1 means fully long the asset, -1 fully short.  Every change
    in position pays |pos_t - pos_{t-1}| * cost_bps / 10_000 of notional, so a
    long -> short flip pays twice the one-way cost.  Costs are per asset
    (config.COST_BPS): 2bp for SPY, 20bp for crypto, and so on.

Walk-forward
    The first TRAIN_BARS of an asset are the in-sample block.  From there the
    tester steps forward in TEST_BARS windows; each window's returns are
    appended to one stitched out-of-sample series.  All the reported OOS numbers
    (Sharpe, max drawdown, trade count) come from that stitched series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ANNUALIZATION, TEST_BARS, TRAIN_BARS, cost_bps


# ------------------------------------------------------------------ metrics
def sharpe(returns: pd.Series | np.ndarray, periods: int = ANNUALIZATION) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return np.nan
    sd = r.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return np.nan
    return float(r.mean() / sd * np.sqrt(periods))


def max_drawdown(returns: pd.Series | np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    r = np.nan_to_num(r, nan=0.0)
    if r.size == 0:
        return np.nan
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def annual_return(returns: pd.Series | np.ndarray, periods: int = ANNUALIZATION) -> float:
    r = np.nan_to_num(np.asarray(returns, dtype=float), nan=0.0)
    if r.size == 0:
        return np.nan
    total = float(np.prod(1.0 + r))
    if total <= 0:
        return -1.0
    return total ** (periods / r.size) - 1.0


def trade_count(positions: pd.Series | np.ndarray) -> int:
    p = np.nan_to_num(np.asarray(positions, dtype=float), nan=0.0)
    if p.size == 0:
        return 0
    return int((np.diff(p, prepend=0.0) != 0).sum())


def _finite(values) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    return v[np.isfinite(v)]


def _nanmax(values) -> float:
    v = _finite(values)
    return float(v.max()) if v.size else np.nan


def _nanmin(values) -> float:
    v = _finite(values)
    return float(v.min()) if v.size else np.nan


def _pos_frac(values) -> float:
    v = _finite(values)
    return float((v > 0).mean()) if v.size else np.nan


# ------------------------------------------------------------------ returns
def net_returns(df: pd.DataFrame, positions: pd.Series, bps: float) -> pd.Series:
    """Daily strategy returns after per-asset transaction costs."""
    asset_ret = df["Close"].pct_change().fillna(0.0)
    pos = positions.reindex(df.index).fillna(0.0)
    turnover = pos.diff().abs().fillna(pos.abs())
    return pos * asset_ret - turnover * (bps / 1e4)


# ------------------------------------------------------------------ walk-forward
def walk_forward(returns: pd.Series, positions: pd.Series,
                 train_bars: int = TRAIN_BARS, test_bars: int = TEST_BARS) -> dict:
    """In-sample block + stitched out-of-sample windows."""
    n = len(returns)
    if n < train_bars + test_bars:
        return {}

    is_ret = returns.iloc[:train_bars]
    is_pos = positions.iloc[:train_bars]

    oos_slices, window_sharpes, window_labels = [], [], []
    start = train_bars
    while start < n:
        stop = min(start + test_bars, n)
        if stop - start < test_bars // 2:      # drop a stub final window
            break
        seg = returns.iloc[start:stop]
        oos_slices.append(seg)
        window_sharpes.append(sharpe(seg))
        window_labels.append(f"{seg.index[0].date()}:{seg.index[-1].date()}")
        start = stop

    if not oos_slices:
        return {}

    oos_ret = pd.concat(oos_slices)
    oos_pos = positions.loc[oos_ret.index]

    return {
        "is_sharpe": sharpe(is_ret),
        "is_maxdd": max_drawdown(is_ret),
        "is_trades": trade_count(is_pos),
        "oos_sharpe": sharpe(oos_ret),
        "oos_maxdd": max_drawdown(oos_ret),
        "oos_return": annual_return(oos_ret),
        "oos_trades": trade_count(oos_pos),
        "oos_bars": len(oos_ret),
        "oos_windows": len(oos_slices),
        "oos_window_sharpes": window_sharpes,
        "oos_window_labels": window_labels,
        "oos_returns": oos_ret,
        "is_start": str(is_ret.index[0].date()),
        "oos_start": str(oos_ret.index[0].date()),
        "oos_end": str(oos_ret.index[-1].date()),
    }


def run_backtest(df: pd.DataFrame, config, ticker: str,
                 train_bars: int = TRAIN_BARS, test_bars: int = TEST_BARS,
                 keep_returns: bool = False) -> dict | None:
    """One (config x asset) backtest -> a single sweep row."""
    try:
        pos = config.function(df, **config.params)
    except Exception as exc:                       # a broken config must not kill the sweep
        return {"asset": ticker, "config": config.name, "family": config.family,
                "category": config.category, "error": repr(exc)}

    bps = cost_bps(ticker)
    rets = net_returns(df, pos, bps)
    wf = walk_forward(rets, pos, train_bars, test_bars)
    if not wf:
        return None

    row = {
        "asset": ticker,
        "config": config.name,
        "family": config.family,
        "category": config.category,
        "cost_bps": bps,
        "params": str(config.params),
        "is_sharpe": wf["is_sharpe"],
        "is_maxdd": wf["is_maxdd"],
        "oos_sharpe": wf["oos_sharpe"],
        "oos_maxdd": wf["oos_maxdd"],
        "oos_ann_return": wf["oos_return"],
        "trades": wf["oos_trades"],
        "oos_bars": wf["oos_bars"],
        "oos_windows": wf["oos_windows"],
        "best_window_sharpe": _nanmax(wf["oos_window_sharpes"]),
        "worst_window_sharpe": _nanmin(wf["oos_window_sharpes"]),
        "pos_window_frac": _pos_frac(wf["oos_window_sharpes"]),
        "exposure": float(pos.loc[wf["oos_returns"].index].abs().mean()),
        "is_start": wf["is_start"],
        "oos_start": wf["oos_start"],
        "oos_end": wf["oos_end"],
    }
    if keep_returns:
        row["oos_returns"] = wf["oos_returns"]
        row["oos_window_sharpes"] = wf["oos_window_sharpes"]
        row["oos_window_labels"] = wf["oos_window_labels"]
    return row


def buy_and_hold(df: pd.DataFrame, ticker: str,
                 train_bars: int = TRAIN_BARS, test_bars: int = TEST_BARS) -> dict:
    """Reference row: long the asset the whole way through, same cost model."""
    pos = pd.Series(1.0, index=df.index)
    rets = net_returns(df, pos, cost_bps(ticker))
    wf = walk_forward(rets, pos, train_bars, test_bars)
    if not wf:
        return {}
    return {"asset": ticker, "oos_sharpe": wf["oos_sharpe"], "oos_maxdd": wf["oos_maxdd"],
            "oos_ann_return": wf["oos_return"], "is_sharpe": wf["is_sharpe"]}
