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

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(_ROOT, "data")
RESULTS_DIR = os.path.join(_ROOT, "results")
TICKSTORY_DIR = os.path.join(_ROOT, "tickstory")

# ---------------------------------------------------------------- tickstory
# Tickstory exports are timestamped in whatever offset you chose at export time
# (Dukascopy source data is UTC).  Daily bars are cut at this hour of that clock:
# 22 == 17:00 New York, the standard FX daily close.  0 gives plain UTC days.
SESSION_CLOSE_HOUR = 22

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

# Spread-based one-way estimates for the instruments Tickstory carries, as a
# fraction of notional.  A 1-pip EURUSD spread on a 1.10 price is ~0.9bp.
FX_MAJOR_BPS = 1.0     # EURUSD, USDJPY, GBPUSD, ...
FX_CROSS_BPS = 2.0     # EURGBP, GBPJPY, AUDNZD, ...
METAL_BPS = 2.5        # XAUUSD, XAGUSD
INDEX_CFD_BPS = 2.0    # US30, GER40, NAS100, ...

CURRENCIES = {"USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD", "SEK",
              "NOK", "DKK", "SGD", "HKD", "MXN", "ZAR", "TRY", "PLN", "HUF",
              "CZK", "CNH", "RUB"}
METALS = {"XAU", "XAG", "XPT", "XPD"}

_COST_OVERRIDES: dict[str, float] = {}


def set_cost_overrides(mapping: dict) -> None:
    """Point the cost model at your own per-symbol numbers (bps, one way)."""
    _COST_OVERRIDES.update({str(k).upper(): float(v) for k, v in mapping.items()})


def load_cost_overrides(path: str) -> dict:
    """Read a two-column `symbol,bps` csv and install it."""
    import csv
    out = {}
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 2 or not row[0].strip():
                continue
            try:
                out[row[0].strip().upper()] = float(row[1])
            except ValueError:
                continue        # header line
    set_cost_overrides(out)
    return out


def _guess_fx_cost(symbol: str) -> float | None:
    """Infer a cost from the shape of an FX / metal / index symbol."""
    s = symbol.upper()
    if len(s) == 6:
        base, quote = s[:3], s[3:]
        if base in METALS and quote in CURRENCIES:
            return METAL_BPS
        if base in CURRENCIES and quote in CURRENCIES:
            return FX_MAJOR_BPS if "USD" in (base, quote) else FX_CROSS_BPS
    if s in {"US30", "US500", "USTEC", "NAS100", "SPX500", "GER40", "GER30",
             "UK100", "JP225", "FRA40", "AUS200", "EU50", "HK50"}:
        return INDEX_CFD_BPS
    return None


def cost_bps(ticker: str) -> float:
    """One-way transaction cost in basis points of notional."""
    key = ticker.upper()
    if key in _COST_OVERRIDES:
        return _COST_OVERRIDES[key]
    if ticker in COST_BPS:
        return COST_BPS[ticker]
    guess = _guess_fx_cost(key)
    return guess if guess is not None else DEFAULT_COST_BPS

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
