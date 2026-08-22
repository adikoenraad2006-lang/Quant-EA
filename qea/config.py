"""Shared configuration for the four-layer strategy tester."""

from __future__ import annotations

import os

# ---------------------------------------------------------------- data
START = "2010-01-01"
END = "2025-01-01"
MIN_BARS = 500

INDEX_ETFS = ["SPY", "QQQ", "IWM", "DIA"]
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLU", "XLY", "XLP"]
MACRO_ETFS = ["GLD", "USO", "TLT", "HYG", "EFA", "EEM", "EWZ"]
CRYPTO = ["BTC-USD", "ETH-USD"]
LARGE_CAPS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META", "JPM"]

UNIVERSE = INDEX_ETFS + SECTOR_ETFS + MACRO_ETFS + CRYPTO + LARGE_CAPS

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# ---------------------------------------------------------------- costs
# Round-trip cost is charged on turnover: cost_t = |pos_t - pos_{t-1}| * COST_BPS/1e4.
# A flat long -> short flip therefore pays twice the one-way cost, which is the
# realistic per-asset commission + spread + slippage estimate for a retail account.
DEFAULT_COST_BPS = 5.0
COST_BPS = {
    **{t: 2.0 for t in INDEX_ETFS},        # penny-wide, deepest books
    **{t: 3.0 for t in SECTOR_ETFS},
    **{t: 4.0 for t in MACRO_ETFS},
    **{t: 3.0 for t in LARGE_CAPS},
    "USO": 6.0,
    "EWZ": 7.0,
    "HYG": 5.0,
    **{t: 20.0 for t in CRYPTO},           # wide spreads + exchange fees
}

def cost_bps(ticker: str) -> float:
    return COST_BPS.get(ticker, DEFAULT_COST_BPS)

# ---------------------------------------------------------------- walk-forward
# The first TRAIN_BARS of every asset are the in-sample block.  Everything after
# it is covered by rolling walk-forward windows of TEST_BARS, whose out-of-sample
# segments are stitched end to end into one continuous OOS return series.
TRAIN_BARS = 1260   # ~5 years
TEST_BARS = 252     # ~1 year, also the walk-forward step
ANNUALIZATION = 252

# ---------------------------------------------------------------- six filters
FILTERS = {
    "max_drawdown": -0.35,   # OOS max drawdown must be better than this
    "min_sharpe": 0.5,       # OOS Sharpe above this
    "max_sharpe": 2.5,       # OOS Sharpe below this (too good = the asset did the work)
    "max_oos_over_is": 0.30, # OOS Sharpe no more than +30% above IS Sharpe
    "min_trades": 30,        # enough trades to mean anything
    "is_sharpe_positive": True,
}

# ---------------------------------------------------------------- layer 3
BOOTSTRAP_N = 200
FRAGILE_DD = -0.35   # worst-case bootstrap drawdown worse than this => fragile

# ---------------------------------------------------------------- layer 4
XS_REBALANCE = 21
XS_LOOKBACKS = {"3m": (63, 0), "6m": (126, 0), "12m_1m": (252, 21)}
