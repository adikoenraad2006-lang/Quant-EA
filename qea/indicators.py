"""Vectorised technical indicators.

Every function takes a price frame (columns Open/High/Low/Close/Volume) or a
single Close series and returns values aligned to the input index.  Nothing in
here peeks forward: value at bar t only ever uses bars <= t.  The one-bar
execution lag is applied once, centrally, by the @strategy decorator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ averages
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def wilder(s: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing (used by RSI / ATR / ADX)."""
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def stdev(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).std(ddof=0)


def _wma(s: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1, dtype=float)
    w /= w.sum()
    vals = s.to_numpy(dtype=float)
    if len(vals) < n:
        return pd.Series(np.nan, index=s.index)
    out = np.full(len(vals), np.nan)
    out[n - 1:] = np.convolve(vals, w[::-1], mode="valid")
    return pd.Series(out, index=s.index)


def hma(s: pd.Series, n: int) -> pd.Series:
    """Hull moving average."""
    half = max(1, int(n / 2))
    root = max(1, int(np.sqrt(n)))
    return _wma(2.0 * _wma(s, half) - _wma(s, n), root)


def kama(s: pd.Series, n: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """Kaufman adaptive moving average (recursive, so an explicit loop)."""
    v = s.to_numpy(dtype=float)
    change = np.abs(v - np.roll(v, n))
    change[:n] = np.nan
    vol = pd.Series(np.abs(np.diff(v, prepend=np.nan))).rolling(n, min_periods=n).sum().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        er = np.where(vol > 0, change / vol, 0.0)
    sc = (er * (2.0 / (fast + 1) - 2.0 / (slow + 1)) + 2.0 / (slow + 1)) ** 2
    out = np.full(len(v), np.nan)
    prev = np.nan
    for i in range(len(v)):
        if np.isnan(sc[i]):
            continue
        if np.isnan(prev):
            prev = v[i]
        else:
            prev = prev + sc[i] * (v[i] - prev)
        out[i] = prev
    return pd.Series(out, index=s.index)


def linreg_slope(s: pd.Series, n: int) -> pd.Series:
    """Slope of an n-bar OLS fit, expressed per bar and normalised by price."""
    x = np.arange(n, dtype=float)
    xc = x - x.mean()
    denom = float((xc ** 2).sum())
    v = s.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    if len(v) >= n:
        num = np.convolve(v, xc[::-1], mode="valid")
        out[n - 1:] = num / denom
    slope = pd.Series(out, index=s.index)
    return slope / s.replace(0.0, np.nan)


# ------------------------------------------------------------------ range
def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    a = df["High"] - df["Low"]
    b = (df["High"] - prev_close).abs()
    c = (df["Low"] - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return wilder(true_range(df), n)


def adx(df: pd.DataFrame, n: int = 14):
    """Returns (adx, +DI, -DI)."""
    up = df["High"].diff()
    down = -df["Low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr_n = wilder(true_range(df), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100.0 * wilder(plus_dm, n) / tr_n
        minus_di = 100.0 * wilder(minus_dm, n) / tr_n
        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return wilder(dx, n), plus_di, minus_di


# ------------------------------------------------------------------ oscillators
def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = wilder(gain, n)
    avg_loss = wilder(loss, n)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.where(avg_loss != 0.0, 100.0).where(avg_gain.notna(), np.nan)


def stochastic(df: pd.DataFrame, k: int = 14, d: int = 3):
    lo = df["Low"].rolling(k, min_periods=k).min()
    hi = df["High"].rolling(k, min_periods=k).max()
    pct_k = 100.0 * (df["Close"] - lo) / (hi - lo).replace(0.0, np.nan)
    return pct_k, pct_k.rolling(d, min_periods=d).mean()


def cci(df: pd.DataFrame, n: int = 20) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    ma = tp.rolling(n, min_periods=n).mean()
    md = (tp - ma).abs().rolling(n, min_periods=n).mean()
    return (tp - ma) / (0.015 * md.replace(0.0, np.nan))


def williams_r(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hi = df["High"].rolling(n, min_periods=n).max()
    lo = df["Low"].rolling(n, min_periods=n).min()
    return -100.0 * (hi - df["Close"]) / (hi - lo).replace(0.0, np.nan)


def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(s, fast) - ema(s, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


def trix(s: pd.Series, n: int = 15) -> pd.Series:
    e3 = ema(ema(ema(s, n), n), n)
    return 100.0 * e3.pct_change()


def ultimate_oscillator(df: pd.DataFrame, a: int = 7, b: int = 14, c: int = 28) -> pd.Series:
    prev_close = df["Close"].shift(1)
    bp = df["Close"] - pd.concat([df["Low"], prev_close], axis=1).min(axis=1)
    tr = true_range(df)

    def avg(n):
        return bp.rolling(n, min_periods=n).sum() / tr.rolling(n, min_periods=n).sum().replace(0.0, np.nan)

    return 100.0 * (4 * avg(a) + 2 * avg(b) + avg(c)) / 7.0


def connors_rsi(s: pd.Series, rsi_n: int = 3, streak_n: int = 2, pct_n: int = 100) -> pd.Series:
    chg = s.diff()
    sign = np.sign(chg.fillna(0.0).to_numpy())
    streak = np.zeros(len(sign))
    run = 0.0
    for i in range(len(sign)):
        if sign[i] > 0:
            run = run + 1 if run > 0 else 1.0
        elif sign[i] < 0:
            run = run - 1 if run < 0 else -1.0
        else:
            run = 0.0
        streak[i] = run
    streak = pd.Series(streak, index=s.index)
    ret = s.pct_change()
    pct_rank = ret.rolling(pct_n, min_periods=pct_n).rank(pct=True) * 100.0
    return (rsi(s, rsi_n) + rsi(streak, streak_n) + pct_rank) / 3.0


def elder_ray(df: pd.DataFrame, n: int = 13):
    base = ema(df["Close"], n)
    return df["High"] - base, df["Low"] - base


# ------------------------------------------------------------------ bands / channels
def bbands(s: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(s, n)
    sd = stdev(s, n)
    return mid - k * sd, mid, mid + k * sd


def percent_b(s: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    lower, _, upper = bbands(s, n, k)
    return (s - lower) / (upper - lower).replace(0.0, np.nan)


def keltner(df: pd.DataFrame, n: int = 20, k: float = 2.0):
    mid = ema(df["Close"], n)
    rng = k * atr(df, n)
    return mid - rng, mid, mid + rng


def donchian(df: pd.DataFrame, n: int = 20):
    """Channel over the n bars ending at t-1 so a touch at t is a real break."""
    return (df["Low"].rolling(n, min_periods=n).min().shift(1),
            df["High"].rolling(n, min_periods=n).max().shift(1))


def supertrend(df: pd.DataFrame, n: int = 10, mult: float = 3.0) -> pd.Series:
    """Returns +1 while in the up-trend, -1 while in the down-trend."""
    hl2 = (df["High"] + df["Low"]) / 2.0
    a = atr(df, n)
    upper = (hl2 + mult * a).to_numpy()
    lower = (hl2 - mult * a).to_numpy()
    close = df["Close"].to_numpy(dtype=float)
    n_obs = len(close)
    trend = np.full(n_obs, np.nan)
    fu, fl = np.nan, np.nan
    direction = 1
    for i in range(n_obs):
        if np.isnan(upper[i]):
            continue
        if np.isnan(fu):
            fu, fl = upper[i], lower[i]
            trend[i] = direction
            continue
        fu = min(upper[i], fu) if close[i - 1] <= fu else upper[i]
        fl = max(lower[i], fl) if close[i - 1] >= fl else lower[i]
        if close[i] > fu:
            direction = 1
        elif close[i] < fl:
            direction = -1
        trend[i] = direction
    return pd.Series(trend, index=df.index)


def psar(df: pd.DataFrame, af_step: float = 0.02, af_max: float = 0.2) -> pd.Series:
    """Returns +1 when the parabolic SAR sits below price, -1 when above."""
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    n = len(high)
    out = np.full(n, np.nan)
    if n < 2:
        return pd.Series(out, index=df.index)
    bull = True
    sar = low[0]
    ep = high[0]
    af = af_step
    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if bull:
            sar = min(sar, low[i - 1], low[max(0, i - 2)])
            if low[i] < sar:
                bull, sar, ep, af = False, ep, low[i], af_step
            elif high[i] > ep:
                ep, af = high[i], min(af + af_step, af_max)
        else:
            sar = max(sar, high[i - 1], high[max(0, i - 2)])
            if high[i] > sar:
                bull, sar, ep, af = True, ep, high[i], af_step
            elif low[i] < ep:
                ep, af = low[i], min(af + af_step, af_max)
        out[i] = 1.0 if bull else -1.0
    return pd.Series(out, index=df.index)


def ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou: int = 52):
    def mid(n):
        return (df["High"].rolling(n, min_periods=n).max() + df["Low"].rolling(n, min_periods=n).min()) / 2.0

    conv, base = mid(tenkan), mid(kijun)
    span_a = ((conv + base) / 2.0).shift(kijun)   # shifted forward = known in advance
    span_b = mid(senkou).shift(kijun)
    return conv, base, span_a, span_b


def aroon(df: pd.DataFrame, n: int = 25):
    up = df["High"].rolling(n + 1, min_periods=n + 1).apply(np.argmax, raw=True) / n * 100.0
    dn = df["Low"].rolling(n + 1, min_periods=n + 1).apply(np.argmin, raw=True) / n * 100.0
    return up, dn


def vortex(df: pd.DataFrame, n: int = 14):
    tr_n = true_range(df).rolling(n, min_periods=n).sum()
    vm_p = (df["High"] - df["Low"].shift(1)).abs().rolling(n, min_periods=n).sum()
    vm_m = (df["Low"] - df["High"].shift(1)).abs().rolling(n, min_periods=n).sum()
    tr_n = tr_n.replace(0.0, np.nan)
    return vm_p / tr_n, vm_m / tr_n


# ------------------------------------------------------------------ volume
def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["Close"].diff().fillna(0.0))
    return (direction * df["Volume"].fillna(0.0)).cumsum()


def cmf(df: pd.DataFrame, n: int = 20) -> pd.Series:
    rng = (df["High"] - df["Low"]).replace(0.0, np.nan)
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / rng
    mfv = mfm * df["Volume"]
    return mfv.rolling(n, min_periods=n).sum() / df["Volume"].rolling(n, min_periods=n).sum().replace(0.0, np.nan)


def mfi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    raw = tp * df["Volume"]
    up = raw.where(tp.diff() > 0, 0.0).rolling(n, min_periods=n).sum()
    dn = raw.where(tp.diff() < 0, 0.0).rolling(n, min_periods=n).sum()
    return 100.0 - 100.0 / (1.0 + up / dn.replace(0.0, np.nan))


def force_index(df: pd.DataFrame, n: int = 13) -> pd.Series:
    return ema(df["Close"].diff() * df["Volume"], n)


def chaikin_oscillator(df: pd.DataFrame, fast: int = 3, slow: int = 10) -> pd.Series:
    rng = (df["High"] - df["Low"]).replace(0.0, np.nan)
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / rng
    adl = (mfm * df["Volume"]).fillna(0.0).cumsum()
    return ema(adl, fast) - ema(adl, slow)


def rolling_vwap(df: pd.DataFrame, n: int = 20) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vol = df["Volume"].replace(0.0, np.nan)
    return (tp * vol).rolling(n, min_periods=n).sum() / vol.rolling(n, min_periods=n).sum()


# ------------------------------------------------------------------ misc
def zscore(s: pd.Series, n: int) -> pd.Series:
    return (s - sma(s, n)) / stdev(s, n).replace(0.0, np.nan)


def roc(s: pd.Series, n: int) -> pd.Series:
    return s.pct_change(n)
