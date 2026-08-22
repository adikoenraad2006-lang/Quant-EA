"""Synthetic OHLCV generator.

For pipeline smoke tests ONLY.  These are random-walk prices with no economic
content -- never report performance measured on them as a research result.
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd


def synthetic_ohlcv(seed: int = 0, n: int = 3800, start: str = "2010-01-04",
                    drift: float = 0.0003, vol: float = 0.012) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    ret = rng.normal(drift, vol, n)
    close = 100.0 * np.exp(np.cumsum(ret))
    open_ = close * (1.0 + rng.normal(0.0, vol / 3.0, n))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, vol / 2.0, n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, vol / 2.0, n)))
    volume = rng.lognormal(15.0, 0.4, n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close, "Volume": volume}, index=idx)


def _ticker_seed(ticker: str) -> int:
    """Stable per-ticker seed, so a subset of the universe gets the same series."""
    return zlib.crc32(ticker.encode()) % (2 ** 31)


def synthetic_universe(tickers, **kw) -> dict[str, pd.DataFrame]:
    return {t: synthetic_ohlcv(seed=_ticker_seed(t), **kw) for t in tickers}
