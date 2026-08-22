"""Layer 1 strategy library: the popular-retail spectrum.

Every strategy is `f(df, **params) -> pd.Series` of daily positions in
{-1, 0, 1}: long / flat / short.  The @strategy decorator applies the one-bar
execution lag centrally, so today's position is built only from data up to
yesterday's close -- no look-ahead anywhere in the library.

`build_configs()` expands each family over a small parameter grid and returns
(name, function, params, category, family) tuples.
"""

from __future__ import annotations

import itertools
from functools import wraps
from typing import Callable, NamedTuple

import numpy as np
import pandas as pd

from . import indicators as ind

CATEGORIES = ("trend", "meanrev", "volume", "volatility", "pattern", "composite")

REGISTRY: dict[str, dict] = {}


class Config(NamedTuple):
    name: str
    function: Callable
    params: dict
    category: str
    family: str


def strategy(family: str, category: str):
    """Register a signal function and wrap it with the one-bar execution lag."""
    if category not in CATEGORIES:
        raise ValueError(f"unknown category {category!r}")

    def deco(fn):
        @wraps(fn)
        def wrapper(df: pd.DataFrame, **params) -> pd.Series:
            raw = fn(df, **params)
            pos = pd.Series(raw, index=df.index).astype(float)
            pos = pos.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            # the signal computed from bar t is only tradeable from bar t+1
            return pos.shift(1).fillna(0.0).clip(-1.0, 1.0)

        wrapper.family = family
        wrapper.category = category
        REGISTRY[family] = {"function": wrapper, "category": category}
        return wrapper

    return deco


# --------------------------------------------------------------- helpers
def _flip(long: pd.Series, short: pd.Series) -> pd.Series:
    """Stateless: long where `long`, short where `short`, flat otherwise."""
    pos = pd.Series(0.0, index=long.index)
    pos[short.fillna(False)] = -1.0
    pos[long.fillna(False)] = 1.0
    return pos


def _hold(index, long_entry, short_entry, exit_cond=None) -> pd.Series:
    """Stateful: enter on a signal and hold until an exit or an opposite entry."""
    raw = pd.Series(np.nan, index=index)
    if exit_cond is not None:
        raw[exit_cond.fillna(False)] = 0.0
    raw[short_entry.fillna(False)] = -1.0
    raw[long_entry.fillna(False)] = 1.0
    return raw.ffill().fillna(0.0)


def _hold_bars(signal: pd.Series, bars: int) -> pd.Series:
    """Take the signal and stay in it for `bars` days (a new signal overrides)."""
    s = signal.replace(0.0, np.nan)
    if bars > 1:
        s = s.ffill(limit=bars - 1)
    return s.fillna(0.0)


def _cross_up(a: pd.Series, b) -> pd.Series:
    b = b if isinstance(b, pd.Series) else pd.Series(b, index=a.index)
    return (a > b) & (a.shift(1) <= b.shift(1))


def _cross_dn(a: pd.Series, b) -> pd.Series:
    b = b if isinstance(b, pd.Series) else pd.Series(b, index=a.index)
    return (a < b) & (a.shift(1) >= b.shift(1))


# =========================================================== TREND
@strategy("ma_crossover", "trend")
def ma_crossover(df, fast=20, slow=100):
    f, s = ind.sma(df["Close"], fast), ind.sma(df["Close"], slow)
    return _flip(f > s, f < s)


@strategy("ts_momentum", "trend")
def ts_momentum(df, lookback=126, long_only=False):
    r = df["Close"].pct_change(lookback)
    pos = _flip(r > 0, r < 0)
    return pos.clip(lower=0.0) if long_only else pos


@strategy("roc_momentum", "trend")
def roc_momentum(df, n=20, threshold=0.0):
    r = ind.roc(df["Close"], n)
    return _flip(r > threshold, r < -threshold)


@strategy("macd_trend", "trend")
def macd_trend(df, fast=12, slow=26, signal=9):
    _, _, hist = ind.macd(df["Close"], fast, slow, signal)
    return _flip(hist > 0, hist < 0)


@strategy("donchian_breakout", "trend")
def donchian_breakout(df, n=20, exit_n=10):
    lo, hi = ind.donchian(df, n)
    xlo, xhi = ind.donchian(df, exit_n)
    c = df["Close"]
    return _hold(df.index, c > hi, c < lo, exit_cond=(c < xlo) | (c > xhi))


@strategy("bollinger_breakout", "trend")
def bollinger_breakout(df, n=20, k=2.0):
    lower, mid, upper = ind.bbands(df["Close"], n, k)
    c = df["Close"]
    return _hold(df.index, c > upper, c < lower, exit_cond=_cross_dn(c, mid) | _cross_up(c, mid))


@strategy("supertrend", "trend")
def supertrend_strategy(df, n=10, mult=3.0):
    t = ind.supertrend(df, n, mult)
    return _flip(t > 0, t < 0)


@strategy("parabolic_sar", "trend")
def parabolic_sar(df, af_step=0.02, af_max=0.2):
    t = ind.psar(df, af_step, af_max)
    return _flip(t > 0, t < 0)


@strategy("adx_trend", "trend")
def adx_trend(df, n=14, threshold=25.0):
    a, plus, minus = ind.adx(df, n)
    strong = a > threshold
    return _flip(strong & (plus > minus), strong & (minus > plus))


@strategy("ichimoku", "trend")
def ichimoku_strategy(df, tenkan=9, kijun=26, senkou=52):
    conv, base, span_a, span_b = ind.ichimoku(df, tenkan, kijun, senkou)
    cloud_hi = pd.concat([span_a, span_b], axis=1).max(axis=1)
    cloud_lo = pd.concat([span_a, span_b], axis=1).min(axis=1)
    c = df["Close"]
    return _flip((c > cloud_hi) & (conv > base), (c < cloud_lo) & (conv < base))


@strategy("linreg_slope", "trend")
def linreg_slope_strategy(df, n=60, threshold=0.0):
    sl = ind.linreg_slope(df["Close"], n)
    return _flip(sl > threshold, sl < -threshold)


@strategy("aroon", "trend")
def aroon_strategy(df, n=25, threshold=70.0):
    up, dn = ind.aroon(df, n)
    return _flip((up > threshold) & (up > dn), (dn > threshold) & (dn > up))


@strategy("vortex", "trend")
def vortex_strategy(df, n=14):
    vp, vm = ind.vortex(df, n)
    return _flip(vp > vm, vm > vp)


@strategy("trix", "trend")
def trix_strategy(df, n=15, signal=9):
    t = ind.trix(df["Close"], n)
    sig = ind.ema(t, signal)
    return _flip(t > sig, t < sig)


@strategy("hull_ma", "trend")
def hull_ma(df, n=36):
    h = ind.hma(df["Close"], n)
    slope = h.diff()
    return _flip(slope > 0, slope < 0)


@strategy("kama", "trend")
def kama_strategy(df, n=10, fast=2, slow=30):
    k = ind.kama(df["Close"], n, fast, slow)
    slope = k.diff()
    c = df["Close"]
    return _flip((c > k) & (slope > 0), (c < k) & (slope < 0))


@strategy("turtle", "trend")
def turtle(df, entry_n=55, exit_n=20):
    lo_e, hi_e = ind.donchian(df, entry_n)
    lo_x, hi_x = ind.donchian(df, exit_n)
    c = df["Close"]
    return _hold(df.index, c > hi_e, c < lo_e, exit_cond=(c < lo_x) | (c > hi_x))


@strategy("dual_momentum", "trend")
def dual_momentum(df, lookback=252, trend_n=200):
    r = df["Close"].pct_change(lookback)
    ma = ind.sma(df["Close"], trend_n)
    c = df["Close"]
    return _flip((r > 0) & (c > ma), (r < 0) & (c < ma))


@strategy("elder_ray", "trend")
def elder_ray_strategy(df, n=13):
    bull, bear = ind.elder_ray(df, n)
    base_slope = ind.ema(df["Close"], n).diff()
    return _flip((base_slope > 0) & (bear < 0), (base_slope < 0) & (bull > 0))


# =========================================================== MEAN REVERSION
@strategy("rsi_revert", "meanrev")
def rsi_revert(df, n=14, lo=30.0, hi=70.0):
    r = ind.rsi(df["Close"], n)
    return _hold(df.index, r < lo, r > hi, exit_cond=_cross_up(r, 50.0) | _cross_dn(r, 50.0))


@strategy("bollinger_revert", "meanrev")
def bollinger_revert(df, n=20, k=2.0):
    lower, mid, upper = ind.bbands(df["Close"], n, k)
    c = df["Close"]
    return _hold(df.index, c < lower, c > upper, exit_cond=_cross_up(c, mid) | _cross_dn(c, mid))


@strategy("zscore_revert", "meanrev")
def zscore_revert(df, n=20, z=2.0, exit_z=0.5):
    zs = ind.zscore(df["Close"], n)
    return _hold(df.index, zs < -z, zs > z, exit_cond=zs.abs() < exit_z)


@strategy("stochastic_revert", "meanrev")
def stochastic_revert(df, k=14, d=3, lo=20.0, hi=80.0):
    pct_k, pct_d = ind.stochastic(df, k, d)
    return _hold(df.index, pct_d < lo, pct_d > hi,
                 exit_cond=_cross_up(pct_d, 50.0) | _cross_dn(pct_d, 50.0))


@strategy("cci_revert", "meanrev")
def cci_revert(df, n=20, threshold=100.0):
    c = ind.cci(df, n)
    return _hold(df.index, c < -threshold, c > threshold,
                 exit_cond=_cross_up(c, 0.0) | _cross_dn(c, 0.0))


@strategy("williams_revert", "meanrev")
def williams_revert(df, n=14, lo=-80.0, hi=-20.0):
    w = ind.williams_r(df, n)
    return _hold(df.index, w < lo, w > hi, exit_cond=_cross_up(w, -50.0) | _cross_dn(w, -50.0))


@strategy("keltner_revert", "meanrev")
def keltner_revert(df, n=20, k=2.0):
    lower, mid, upper = ind.keltner(df, n, k)
    c = df["Close"]
    return _hold(df.index, c < lower, c > upper, exit_cond=_cross_up(c, mid) | _cross_dn(c, mid))


@strategy("vwap_revert", "meanrev")
def vwap_revert(df, n=20, threshold=0.02):
    v = ind.rolling_vwap(df, n)
    dev = df["Close"] / v - 1.0
    return _hold(df.index, dev < -threshold, dev > threshold,
                 exit_cond=_cross_up(dev, 0.0) | _cross_dn(dev, 0.0))


@strategy("percent_b", "meanrev")
def percent_b_revert(df, n=20, lo=0.05, hi=0.95):
    pb = ind.percent_b(df["Close"], n, 2.0)
    return _hold(df.index, pb < lo, pb > hi, exit_cond=_cross_up(pb, 0.5) | _cross_dn(pb, 0.5))


@strategy("connors_rsi", "meanrev")
def connors_rsi_revert(df, rsi_n=3, lo=20.0, hi=80.0):
    c = ind.connors_rsi(df["Close"], rsi_n=rsi_n)
    return _hold(df.index, c < lo, c > hi, exit_cond=_cross_up(c, 50.0) | _cross_dn(c, 50.0))


@strategy("ultimate_oscillator", "meanrev")
def ultimate_revert(df, a=7, b=14, c=28, lo=30.0, hi=70.0):
    u = ind.ultimate_oscillator(df, a, b, c)
    return _hold(df.index, u < lo, u > hi, exit_cond=_cross_up(u, 50.0) | _cross_dn(u, 50.0))


@strategy("gap_fade", "meanrev")
def gap_fade(df, gap=0.01, hold=1):
    g = df["Open"] / df["Close"].shift(1) - 1.0
    sig = _flip(g < -gap, g > gap)
    return _hold_bars(sig, hold)


# =========================================================== VOLUME
@strategy("obv_trend", "volume")
def obv_trend(df, n=20):
    o = ind.obv(df)
    return _flip(o > ind.sma(o, n), o < ind.sma(o, n))


@strategy("chaikin_money_flow", "volume")
def chaikin_money_flow(df, n=20, threshold=0.0):
    c = ind.cmf(df, n)
    return _flip(c > threshold, c < -threshold)


@strategy("money_flow_index", "volume")
def money_flow_index(df, n=14, lo=20.0, hi=80.0):
    m = ind.mfi(df, n)
    return _hold(df.index, m < lo, m > hi, exit_cond=_cross_up(m, 50.0) | _cross_dn(m, 50.0))


@strategy("volume_surge", "volume")
def volume_surge(df, n=20, mult=2.0, hold=3):
    surge = df["Volume"] > mult * ind.sma(df["Volume"], n)
    up = df["Close"] > df["Open"]
    sig = _flip(surge & up, surge & ~up)
    return _hold_bars(sig, hold)


@strategy("force_index", "volume")
def force_index_trend(df, n=13):
    f = ind.force_index(df, n)
    return _flip(f > 0, f < 0)


@strategy("chaikin_oscillator", "volume")
def chaikin_oscillator_trend(df, fast=3, slow=10):
    o = ind.chaikin_oscillator(df, fast, slow)
    return _flip(o > 0, o < 0)


# =========================================================== VOLATILITY
@strategy("atr_breakout", "volatility")
def atr_breakout(df, n=14, mult=2.0):
    a = ind.atr(df, n)
    ref = df["Close"].shift(1)
    c = df["Close"]
    return _hold(df.index, c > ref + mult * a, c < ref - mult * a,
                 exit_cond=(c - ref).abs() > 3.0 * mult * a)


@strategy("volatility_breakout", "volatility")
def volatility_breakout(df, n=20, k=1.5, hold=5):
    ret = df["Close"].pct_change()
    sd = ret.rolling(n, min_periods=n).std(ddof=0)
    sig = _flip(ret > k * sd, ret < -k * sd)
    return _hold_bars(sig, hold)


@strategy("squeeze_breakout", "volatility")
def squeeze_breakout(df, n=20, bb_k=2.0, kc_k=1.5):
    bl, bm, bu = ind.bbands(df["Close"], n, bb_k)
    kl, km, ku = ind.keltner(df, n, kc_k)
    squeeze = (bl > kl) & (bu < ku)
    released = (~squeeze) & squeeze.shift(1).fillna(False)
    mom = df["Close"] - ind.sma(df["Close"], n)
    return _hold(df.index, released & (mom > 0), released & (mom < 0), exit_cond=squeeze)


# =========================================================== PATTERN
@strategy("engulfing", "pattern")
def engulfing(df, hold=3, trend_n=0):
    o, c = df["Open"], df["Close"]
    po, pc = o.shift(1), c.shift(1)
    bull = (c > o) & (pc < po) & (c > po) & (o < pc)
    bear = (c < o) & (pc > po) & (c < po) & (o > pc)
    if trend_n:
        ma = ind.sma(c, trend_n)
        bull &= c > ma
        bear &= c < ma
    return _hold_bars(_flip(bull, bear), hold)


@strategy("three_bar_reversal", "pattern")
def three_bar_reversal(df, hold=3):
    lo, hi, c, o = df["Low"], df["High"], df["Close"], df["Open"]
    down3 = (lo < lo.shift(1)) & (lo.shift(1) < lo.shift(2))
    up3 = (hi > hi.shift(1)) & (hi.shift(1) > hi.shift(2))
    bull = down3 & (c > o) & (c > hi.shift(1))
    bear = up3 & (c < o) & (c < lo.shift(1))
    return _hold_bars(_flip(bull, bear), hold)


@strategy("higher_highs_lows", "pattern")
def higher_highs_lows(df, n=20):
    hh = df["High"].rolling(n, min_periods=n).max()
    ll = df["Low"].rolling(n, min_periods=n).min()
    rising = (hh > hh.shift(n)) & (ll > ll.shift(n))
    falling = (hh < hh.shift(n)) & (ll < ll.shift(n))
    return _flip(rising, falling)


@strategy("pivot_bounce", "pattern")
def pivot_bounce(df, n=10, hold=5):
    lo = df["Low"].rolling(n, min_periods=n).min()
    hi = df["High"].rolling(n, min_periods=n).max()
    bull = (df["Low"] <= lo) & (df["Close"] > df["Open"])
    bear = (df["High"] >= hi) & (df["Close"] < df["Open"])
    return _hold_bars(_flip(bull, bear), hold)


# =========================================================== COMPOSITE
@strategy("macd_rsi_confirm", "composite")
def macd_rsi_confirm(df, fast=12, slow=26, signal=9, rsi_n=14, rsi_lvl=50.0):
    _, _, hist = ind.macd(df["Close"], fast, slow, signal)
    r = ind.rsi(df["Close"], rsi_n)
    return _flip((hist > 0) & (r > rsi_lvl), (hist < 0) & (r < 100.0 - rsi_lvl))


@strategy("triple_screen", "composite")
def triple_screen(df, tide_n=100, wave_n=13, osc_n=14):
    tide = ind.ema(df["Close"], tide_n).diff()
    force = ind.force_index(df, wave_n)
    pct_k, _ = ind.stochastic(df, osc_n, 3)
    long = (tide > 0) & (force < 0) & (pct_k < 40.0)
    short = (tide < 0) & (force > 0) & (pct_k > 60.0)
    return _hold(df.index, long, short, exit_cond=(tide > 0) & (pct_k > 80.0) | (tide < 0) & (pct_k < 20.0))


@strategy("chandelier", "composite")
def chandelier(df, n=22, mult=3.0):
    a = ind.atr(df, n)
    long_stop = df["High"].rolling(n, min_periods=n).max() - mult * a
    short_stop = df["Low"].rolling(n, min_periods=n).min() + mult * a
    c = df["Close"]
    return _hold(df.index, c > short_stop.shift(1), c < long_stop.shift(1))


# =========================================================== config grids
def _grid(family: str, **axes) -> list[Config]:
    """Cartesian product of the parameter axes for one family."""
    meta = REGISTRY[family]
    keys = list(axes)
    out = []
    for combo in itertools.product(*(axes[k] for k in keys)):
        params = dict(zip(keys, combo))
        tag = "_".join(f"{k}{_fmt(v)}" for k, v in params.items())
        out.append(Config(f"{family}__{tag}", meta["function"], params,
                          meta["category"], family))
    return out


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "T" if v else "F"
    if isinstance(v, float):
        return f"{v:g}".replace("-", "m").replace(".", "p")
    return str(v)


def build_configs() -> list[Config]:
    """Every (name, function, params, category, family) the sweep will run."""
    cfgs: list[Config] = []

    # ---- trend
    cfgs += [c for c in _grid("ma_crossover", fast=[5, 10, 20, 50], slow=[50, 100, 150, 200])
             if c.params["fast"] < c.params["slow"]]
    cfgs += _grid("ts_momentum", lookback=[21, 42, 63, 126, 252], long_only=[False, True])
    cfgs += _grid("roc_momentum", n=[5, 10, 20, 60, 120], threshold=[0.0, 0.02])
    cfgs += _grid("macd_trend", fast=[8, 12, 19], slow=[17, 26, 39], signal=[9])
    cfgs += _grid("donchian_breakout", n=[10, 20, 55, 100], exit_n=[10, 20])
    cfgs += _grid("bollinger_breakout", n=[20, 50, 100], k=[1.5, 2.0, 2.5])
    cfgs += _grid("supertrend", n=[7, 10, 14], mult=[1.5, 2.0, 3.0])
    cfgs += _grid("parabolic_sar", af_step=[0.01, 0.02, 0.04], af_max=[0.1, 0.2])
    cfgs += _grid("adx_trend", n=[14, 20], threshold=[15.0, 20.0, 25.0])
    cfgs += _grid("ichimoku", tenkan=[7, 9, 12], kijun=[22, 26], senkou=[52])
    cfgs += _grid("linreg_slope", n=[20, 60, 120], threshold=[0.0, 0.0005])
    cfgs += _grid("aroon", n=[14, 25, 50], threshold=[50.0, 70.0])
    cfgs += _grid("vortex", n=[14, 21, 30, 45])
    cfgs += _grid("trix", n=[9, 15, 30], signal=[9])
    cfgs += _grid("hull_ma", n=[16, 36, 64, 100])
    cfgs += _grid("kama", n=[10, 20], fast=[2], slow=[20, 30])
    cfgs += _grid("turtle", entry_n=[20, 55, 100], exit_n=[10, 20])
    cfgs += _grid("dual_momentum", lookback=[126, 252], trend_n=[100, 200])
    cfgs += _grid("elder_ray", n=[8, 13, 21, 34])

    # ---- mean reversion
    cfgs += _grid("rsi_revert", n=[2, 7, 14], lo=[10.0, 20.0, 30.0])
    cfgs += _grid("bollinger_revert", n=[10, 20, 50], k=[1.5, 2.0, 2.5])
    cfgs += _grid("zscore_revert", n=[10, 20, 60], z=[1.0, 1.5, 2.0])
    cfgs += _grid("stochastic_revert", k=[14, 21], d=[3], lo=[10.0, 20.0], hi=[80.0, 90.0])
    cfgs += _grid("cci_revert", n=[14, 20, 50], threshold=[100.0, 200.0])
    cfgs += _grid("williams_revert", n=[14, 28], lo=[-90.0, -80.0], hi=[-20.0, -10.0])
    cfgs += _grid("keltner_revert", n=[20, 50], k=[1.5, 2.0, 2.5])
    cfgs += _grid("vwap_revert", n=[20, 50], threshold=[0.01, 0.02, 0.05])
    cfgs += _grid("percent_b", n=[20, 50], lo=[0.0, 0.05, 0.2], hi=[1.0, 0.95, 0.8])
    cfgs += _grid("connors_rsi", rsi_n=[3], lo=[10.0, 20.0], hi=[80.0, 90.0])
    cfgs += _grid("ultimate_oscillator", a=[5, 7], b=[14], c=[28], lo=[30.0, 40.0], hi=[70.0, 60.0])
    cfgs += _grid("gap_fade", gap=[0.005, 0.01, 0.02, 0.03], hold=[1, 3, 5])

    # ---- volume
    cfgs += _grid("obv_trend", n=[20, 50, 100, 200])
    cfgs += _grid("chaikin_money_flow", n=[20, 50], threshold=[0.0, 0.05, 0.1])
    cfgs += _grid("money_flow_index", n=[14, 21], lo=[20.0, 30.0], hi=[80.0, 70.0])
    cfgs += _grid("volume_surge", n=[20, 50], mult=[1.5, 2.0, 3.0], hold=[3, 5])
    cfgs += _grid("force_index", n=[2, 13, 26, 50])
    cfgs += _grid("chaikin_oscillator", fast=[3, 5], slow=[10, 20, 30])

    # ---- volatility
    cfgs += _grid("atr_breakout", n=[14, 20], mult=[0.5, 1.0, 2.0])
    cfgs += _grid("volatility_breakout", n=[20, 50], k=[1.0, 1.5, 2.0], hold=[3, 5])
    cfgs += _grid("squeeze_breakout", n=[20, 50], bb_k=[2.0], kc_k=[1.5, 2.0, 2.5])

    # ---- pattern
    cfgs += _grid("engulfing", hold=[1, 3, 5, 10], trend_n=[0, 50, 200])
    cfgs += _grid("three_bar_reversal", hold=[1, 3, 5, 10])
    cfgs += _grid("higher_highs_lows", n=[10, 20, 50, 100])
    cfgs += _grid("pivot_bounce", n=[5, 10, 20], hold=[3, 5, 10])

    # ---- composite
    cfgs += _grid("macd_rsi_confirm", fast=[8, 12], slow=[26, 39], signal=[9],
                  rsi_n=[14], rsi_lvl=[50.0, 55.0])
    cfgs += _grid("triple_screen", tide_n=[50, 100, 200], wave_n=[13], osc_n=[14, 21])
    cfgs += _grid("chandelier", n=[14, 22, 50], mult=[2.0, 3.0, 4.0])

    return cfgs
