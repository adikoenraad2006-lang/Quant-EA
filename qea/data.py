"""Layer 1 data: daily OHLCV from yfinance, with an on-disk cache."""

from __future__ import annotations

import os

import pandas as pd

from .config import CACHE_DIR, END, MIN_BARS, START, UNIVERSE

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def _cache_path(ticker: str, start: str, end: str) -> str:
    safe = ticker.replace("/", "-")
    return os.path.join(CACHE_DIR, f"{safe}_{start}_{end}.csv")


def _normalise(raw: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    if raw is None or len(raw) == 0:
        return None
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance returns (field, ticker); drop whichever level holds the ticker
        levels = [lvl for lvl in range(df.columns.nlevels)
                  if ticker in df.columns.get_level_values(lvl)]
        df = df.xs(ticker, axis=1, level=levels[0]) if levels else df.droplevel(1, axis=1)
    df.columns = [str(c).split(" ")[0].title() for c in df.columns]
    if not set(OHLCV).issubset(df.columns):
        return None
    df = df[OHLCV].astype(float)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[df["Close"] > 0]
    df["Volume"] = df["Volume"].fillna(0.0)
    return df


def load_prices(tickers=None, start=START, end=END, min_bars=MIN_BARS,
                use_cache=True, verbose=True) -> dict[str, pd.DataFrame]:
    """Daily OHLCV per ticker, auto_adjust=True, assets under `min_bars` skipped."""
    tickers = list(tickers or UNIVERSE)
    os.makedirs(CACHE_DIR, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}
    skipped: list[tuple[str, str]] = []

    for ticker in tickers:
        path = _cache_path(ticker, start, end)
        df = None
        if use_cache and os.path.exists(path):
            df = _normalise(pd.read_csv(path, index_col=0, parse_dates=True), ticker)
        if df is None:
            import yfinance as yf
            raw = yf.download(ticker, start=start, end=end, auto_adjust=True,
                              progress=False, actions=False)
            df = _normalise(raw, ticker)
            if df is not None and use_cache:
                df.to_csv(path)
        if df is None:
            skipped.append((ticker, "no data"))
            continue
        if len(df) < min_bars:
            skipped.append((ticker, f"only {len(df)} bars"))
            continue
        out[ticker] = df

    if verbose:
        print(f"Loaded {len(out)} assets ({start} -> {end})")
        for ticker, df in out.items():
            print(f"  {ticker:9s} {len(df):5d} bars  "
                  f"{df.index[0].date()} -> {df.index[-1].date()}")
        for ticker, why in skipped:
            print(f"  SKIPPED {ticker}: {why}")
        if not out:
            print("  (no data — see the network note in README.md if downloads are blocked)")
    return out


def load_dataset(synthetic: bool = False, tickers=None, start=START, end=END,
                 min_bars=MIN_BARS, use_cache=True, verbose=True) -> dict[str, pd.DataFrame]:
    """Real prices, or -- with `synthetic` -- random walks for a pipeline smoke test."""
    if synthetic:
        from .synthetic import synthetic_universe
        tickers = list(tickers or UNIVERSE)
        print("!" * 78)
        print("!! SYNTHETIC RANDOM-WALK DATA -- pipeline smoke test only.")
        print("!! Numbers produced from this are NOT research results.")
        print("!" * 78)
        return synthetic_universe(tickers)
    return load_prices(tickers, start, end, min_bars, use_cache, verbose)
