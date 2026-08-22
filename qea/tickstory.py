"""Load Tickstory exports and roll them up into the daily OHLCV the tester wants.

Tickstory writes several shapes and the exact one depends on the template you
picked at export time, so the reader sniffs the file instead of assuming:

    tick csv    2015.01.02 00:00:00.123,1.20998,1.21024,0.75,1.50
    m1 csv      2015.01.02,00:00,1.21000,1.21010,1.20990,1.21005,42
    mt5 csv     <DATE>\\t<TIME>\\t<OPEN>\\t<HIGH>\\t<LOW>\\t<CLOSE>\\t<TICKVOL>...
    generic     2015-01-02 00:00:00,1.21000,1.21010,1.20990,1.21005,42
    mt4 hst     binary, version 400 or 401

.gz and .zip are read in place.  Big tick files are streamed in chunks and
aggregated as they go, so memory stays flat whatever the file size.

Two things about this data that the yfinance path does not have to think about:

* Timestamps carry whatever offset you chose at export (Dukascopy source data is
  UTC).  A daily bar has to be cut somewhere, and for FX that is conventionally
  17:00 New York = 22:00 UTC, not UTC midnight -- cutting at midnight splits the
  Sunday-evening open into a stub bar of its own.  `session_close_hour` sets it.
* Volume is *tick* volume, not traded size.  It is a usable activity proxy in FX
  and the volume strategies will run on it, but it is not the same quantity the
  equity side of the universe reports.
"""

from __future__ import annotations

import gzip
import os
import re
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CACHE_DIR, MIN_BARS, SESSION_CLOSE_HOUR

OHLCV = ["Open", "High", "Low", "Close", "Volume"]
CHUNK_ROWS = 4_000_000
DATA_SUFFIXES = (".csv", ".txt", ".tsv", ".csv.gz", ".txt.gz", ".zip", ".hst")

_DELIMITERS = [",", ";", "\t", "|"]

# fixed-width timestamp shapes, so day bucketing can slice instead of parse
_TS_FORMS = [
    # (regex, date slice, hour slice, strptime format)
    (re.compile(r"^\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}"), (0, 10), (11, 13), "%Y.%m.%d %H:%M:%S"),
    (re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}"), (0, 10), (11, 13), "%Y-%m-%d %H:%M:%S"),
    (re.compile(r"^\d{4}/\d{2}/\d{2} \d{2}:\d{2}"), (0, 10), (11, 13), "%Y/%m/%d %H:%M:%S"),
    (re.compile(r"^\d{8} \d{6}"), (0, 8), (9, 11), "%Y%m%d %H%M%S"),
    (re.compile(r"^\d{8}T?\d{6}$"), (0, 8), (8, 10), "%Y%m%d%H%M%S"),
    (re.compile(r"^\d{4}\.\d{2}\.\d{2}$"), (0, 10), None, "%Y.%m.%d"),
    (re.compile(r"^\d{4}-\d{2}-\d{2}$"), (0, 10), None, "%Y-%m-%d"),
    (re.compile(r"^\d{8}$"), (0, 8), None, "%Y%m%d"),
]


@dataclass
class TickstoryFormat:
    kind: str                 # "tick" | "bars"
    delimiter: str
    skiprows: int
    n_datetime_cols: int      # 1 = one timestamp column, 2 = date + time
    columns: list[str]        # names for the value columns after the timestamp
    date_slice: tuple | None
    hour_slice: tuple | None
    parse_format: str | None
    decimal: str = "."

    def describe(self) -> str:
        d = {",": "comma", ";": "semicolon", "\t": "tab", "|": "pipe"}[self.delimiter]
        return (f"{self.kind} data, {d}-separated, "
                f"{'date+time' if self.n_datetime_cols == 2 else 'single timestamp'} "
                f"column(s), values {self.columns}")


# ------------------------------------------------------------------ opening
def _open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", newline="")
    if path.endswith(".zip"):
        zf = zipfile.ZipFile(path)
        inner = [n for n in zf.namelist() if not n.endswith("/")]
        if not inner:
            raise ValueError(f"{path} is an empty archive")
        import io
        return io.TextIOWrapper(zf.open(inner[0]), newline="")
    return open(path, "rt", newline="")


def _sample_lines(path: str, n: int = 40) -> list[str]:
    out = []
    with _open_text(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(line)
            if len(out) >= n:
                break
    return out


# ------------------------------------------------------------------ sniffing
def _is_number(tok: str) -> bool:
    try:
        float(tok.replace(",", ".") if tok.count(",") == 1 and "." not in tok else tok)
        return True
    except ValueError:
        return False


def _match_ts(tok: str):
    for rx, dsl, hsl, fmt in _TS_FORMS:
        if rx.match(tok):
            return dsl, hsl, fmt
    return None


def _pick_delimiter(lines: list[str]) -> str:
    best, best_score = ",", -1
    for d in _DELIMITERS:
        counts = [ln.count(d) for ln in lines]
        if not counts or counts[0] == 0:
            continue
        # a real delimiter appears the same number of times on every row
        score = counts[0] if len(set(counts)) == 1 else 0
        if score > best_score:
            best, best_score = d, score
    return best


def _looks_ohlc(rows: list[list[float]], i: int) -> bool:
    """Do four consecutive numeric fields behave like open/high/low/close?"""
    ok = 0
    for r in rows:
        if len(r) < i + 4 or any(not np.isfinite(v) for v in r[i:i + 4]):
            return False
        o, h, l, c = r[i:i + 4]
        if h >= max(o, c) and l <= min(o, c) and l > 0 and h / max(l, 1e-12) < 10:
            ok += 1
    return ok == len(rows)


def sniff_format(path: str, kind: str = "auto") -> TickstoryFormat:
    """Work out how a Tickstory export is laid out."""
    lines = _sample_lines(path)
    if not lines:
        raise ValueError(f"{path} is empty")

    delim = _pick_delimiter(lines)
    first = [t.strip() for t in lines[0].split(delim)]
    skip = 0 if _match_ts(first[0]) else 1          # header row
    body = [[t.strip() for t in ln.split(delim)] for ln in lines[skip:skip + 12]]
    body = [r for r in body if len(r) == len(body[0])]
    if not body:
        raise ValueError(f"{path}: could not read a consistent row")

    ts0 = _match_ts(body[0][0])
    if ts0 is None:
        raise ValueError(f"{path}: first field {body[0][0]!r} is not a timestamp "
                         f"this reader recognises")
    date_slice, hour_slice, fmt = ts0

    # a bare date in field 0 followed by a time in field 1
    n_dt = 1
    if hour_slice is None and len(body[0]) > 1 and re.match(r"^\d{2}:?\d{2}", body[0][1]):
        n_dt = 2

    values = [r[n_dt:] for r in body]
    numeric = [[float(t) if _is_number(t) else float("nan") for t in r] for r in values]
    n_val = len(numeric[0])

    if kind == "auto":
        if n_val >= 5:
            kind = "bars"
        elif n_val == 4:
            kind = "bars" if _looks_ohlc(numeric, 0) else "tick"
        else:
            kind = "tick"

    if kind == "bars":
        cols = OHLCV[:min(n_val, 5)]
        if n_val < 4:
            raise ValueError(f"{path}: {n_val} value columns is too few for OHLC bars")
        if n_val == 4:
            cols = ["Open", "High", "Low", "Close"]
    else:
        cols = ["Bid", "Ask"] + (["BidVolume", "AskVolume"][:max(0, n_val - 2)])

    return TickstoryFormat(kind=kind, delimiter=delim, skiprows=skip, n_datetime_cols=n_dt,
                           columns=cols, date_slice=date_slice, hour_slice=hour_slice,
                           parse_format=fmt)


# ------------------------------------------------------------------ day keys
def _day_keys(stamps: pd.Series, times: pd.Series | None, fmt: TickstoryFormat,
              session_close_hour: int) -> pd.Series:
    """Map each row to the trading day it belongs to, as a YYYY-MM-DD string.

    Sliced rather than parsed: these files run to hundreds of millions of rows
    and `to_datetime` on every one of them is the whole cost of the job.
    """
    s = stamps.astype(str)
    a, b = fmt.date_slice
    date = s.str.slice(a, b)

    if session_close_hour % 24 == 0:
        return date

    if fmt.hour_slice is not None:
        ha, hb = fmt.hour_slice
        hour = pd.to_numeric(s.str.slice(ha, hb), errors="coerce")
    elif times is not None:
        hour = pd.to_numeric(times.astype(str).str.slice(0, 2), errors="coerce")
    else:
        return date

    day = pd.to_datetime(date, format="%Y.%m.%d" if "." in fmt.parse_format else
                         ("%Y%m%d" if b == 8 else "%Y-%m-%d"), errors="coerce")
    day = day + pd.to_timedelta((hour >= session_close_hour).astype("int8"), unit="D")
    return day.dt.strftime("%Y-%m-%d")


# ------------------------------------------------------------------ aggregation
class _DailyAccumulator:
    """Fold chunks of intraday rows into one daily OHLCV frame."""

    def __init__(self) -> None:
        self.parts: list[pd.DataFrame] = []

    def add_bars(self, day: pd.Series, frame: pd.DataFrame) -> None:
        agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
        if "Volume" in frame.columns:
            agg["Volume"] = "sum"
        g = frame.groupby(day, sort=True).agg(agg)
        g["rows"] = frame.groupby(day, sort=True).size()
        self.parts.append(g)

    def add_ticks(self, day: pd.Series, mid: pd.Series, size: pd.Series | None) -> None:
        g = mid.groupby(day, sort=True).agg(Open="first", High="max", Low="min", Close="last")
        g["Volume"] = (size.groupby(day, sort=True).sum() if size is not None
                       else mid.groupby(day, sort=True).size().astype(float))
        g["rows"] = mid.groupby(day, sort=True).size()
        self.parts.append(g)

    def result(self) -> pd.DataFrame:
        if not self.parts:
            return pd.DataFrame(columns=OHLCV + ["rows"])
        allp = pd.concat(self.parts)
        # chunks arrive in time order, so first/last across them stay correct
        out = allp.groupby(level=0, sort=True).agg(
            Open=("Open", "first"), High=("High", "max"), Low=("Low", "min"),
            Close=("Close", "last"),
            Volume=("Volume", "sum") if "Volume" in allp.columns else ("Open", "size"),
            rows=("rows", "sum"))
        out.index = pd.to_datetime(out.index)
        return out.sort_index()


def _read_hst(path: str) -> pd.DataFrame:
    """MetaTrader 4 .hst history, version 400 or 401."""
    with open(path, "rb") as fh:
        header = fh.read(148)
        if len(header) < 148:
            raise ValueError(f"{path}: truncated hst header")
        version = int(np.frombuffer(header, dtype="<i4", count=1)[0])
        payload = fh.read()
    if version == 400:
        dt = np.dtype([("ctm", "<i4"), ("Open", "<f8"), ("Low", "<f8"),
                       ("High", "<f8"), ("Close", "<f8"), ("Volume", "<f8")])
    elif version == 401:
        dt = np.dtype([("ctm", "<i8"), ("Open", "<f8"), ("High", "<f8"), ("Low", "<f8"),
                       ("Close", "<f8"), ("Volume", "<i8"), ("spread", "<i4"),
                       ("real_volume", "<i8")])
    else:
        raise ValueError(f"{path}: unsupported hst version {version}")
    n = len(payload) // dt.itemsize
    rec = np.frombuffer(payload, dtype=dt, count=n)
    df = pd.DataFrame({c: rec[c].astype(float) for c in OHLCV})
    df.index = pd.to_datetime(rec["ctm"].astype("int64"), unit="s")
    return df


# ------------------------------------------------------------------ one symbol
def read_symbol(path: str, session_close_hour: int = SESSION_CLOSE_HOUR,
                kind: str = "auto", keep_weekends: bool = False,
                min_rows_frac: float = 0.1, verbose: bool = True) -> pd.DataFrame:
    """Read one Tickstory export and return daily OHLCV."""
    if path.endswith(".hst"):
        bars = _read_hst(path)
        shift = (24 - session_close_hour % 24) % 24
        day = (bars.index + pd.Timedelta(hours=shift)).floor("D")
        acc = _DailyAccumulator()
        acc.add_bars(pd.Series(day, index=bars.index), bars)
        daily = acc.result()
        fmt_note = f"mt4 hst, {len(bars)} bars"
    else:
        fmt = sniff_format(path, kind)
        fmt_note = fmt.describe()
        acc = _DailyAccumulator()
        n_cols = fmt.n_datetime_cols + len(fmt.columns)
        names = (["_d", "_t"] if fmt.n_datetime_cols == 2 else ["_d"]) + fmt.columns
        reader = pd.read_csv(_open_text(path), sep=fmt.delimiter, header=None,
                             names=names, usecols=range(n_cols), skiprows=fmt.skiprows,
                             chunksize=CHUNK_ROWS, dtype={"_d": str, "_t": str},
                             on_bad_lines="skip", engine="c")
        for chunk in reader:
            times = chunk["_t"] if fmt.n_datetime_cols == 2 else None
            day = _day_keys(chunk["_d"], times, fmt, session_close_hour)
            good = day.notna()
            if fmt.kind == "bars":
                cols = [c for c in OHLCV if c in chunk.columns]
                frame = chunk.loc[good, cols].apply(pd.to_numeric, errors="coerce")
                frame = frame.dropna(subset=["Open", "High", "Low", "Close"])
                acc.add_bars(day[frame.index], frame)
            else:
                bid = pd.to_numeric(chunk.loc[good, "Bid"], errors="coerce")
                ask = pd.to_numeric(chunk.loc[good, "Ask"], errors="coerce")
                mid = ((bid + ask) / 2.0).where(ask.notna() & (ask > 0), bid).dropna()
                size = None
                if "BidVolume" in chunk.columns:
                    vol = (pd.to_numeric(chunk.loc[mid.index, "BidVolume"], errors="coerce")
                           .fillna(0.0))
                    if "AskVolume" in chunk.columns:
                        vol = vol + pd.to_numeric(chunk.loc[mid.index, "AskVolume"],
                                                  errors="coerce").fillna(0.0)
                    size = vol if float(vol.abs().sum()) > 0 else None
                acc.add_ticks(day[mid.index], mid, size)
        daily = acc.result()

    if daily.empty:
        raise ValueError(f"{path}: no usable rows")

    rows = daily.pop("rows")
    dropped_stub = 0
    if rows.median() > 5:                     # only meaningful for intraday input
        floor = max(1.0, min_rows_frac * float(rows.median()))
        keep = rows >= floor
        dropped_stub = int((~keep).sum())
        daily = daily[keep]

    dropped_wknd = 0
    if not keep_weekends:
        wknd = daily.index.dayofweek >= 5
        dropped_wknd = int(wknd.sum())
        daily = daily[~wknd]

    if "Volume" not in daily.columns:
        daily["Volume"] = 0.0
    daily = daily[OHLCV].astype(float)
    daily = daily[(daily["Close"] > 0) & daily["Close"].notna()]

    if verbose:
        note = f"  {os.path.basename(path):<34} {fmt_note}"
        print(note)
        print(f"  {'':34} -> {len(daily)} daily bars "
              f"{daily.index[0].date()} -> {daily.index[-1].date()}"
              + (f", dropped {dropped_stub} stub sessions" if dropped_stub else "")
              + (f", {dropped_wknd} weekend bars" if dropped_wknd else ""))
        if float(daily["Volume"].abs().sum()) == 0:
            print(f"  {'':34} !! volume is all zero — the volume strategies "
                  f"cannot work on this export")
    return daily


# ------------------------------------------------------------------ directory
_SUFFIX_NOISE = re.compile(r"[_\-. ]?(m1|m5|m15|m30|h1|h4|d1|ticks?|bars?|"
                           r"\d{4}(-\d{2})?(-\d{2})?|utc|gmt[+-]?\d*|pro|micro|ecn|cash)$",
                           re.IGNORECASE)


def symbol_from_filename(path: str) -> str:
    stem = os.path.basename(path)
    for suf in (".csv.gz", ".txt.gz", ".csv", ".txt", ".tsv", ".zip", ".hst"):
        if stem.lower().endswith(suf):
            stem = stem[: -len(suf)]
            break
    prev = None
    while prev != stem:                        # strip EURUSD_M1_2015-2025 -> EURUSD
        prev = stem
        stem = _SUFFIX_NOISE.sub("", stem).rstrip("_-. ")
    return (stem.split("_")[0].split("-")[0] or os.path.basename(path)).upper()


def discover(data_dir: str) -> dict[str, str]:
    """Map symbol -> file path for every export in a directory."""
    found: dict[str, str] = {}
    for root, _, files in os.walk(data_dir):
        for name in sorted(files):
            low = name.lower()
            if not low.endswith(DATA_SUFFIXES):
                continue
            path = os.path.join(root, name)
            sym = symbol_from_filename(path)
            found.setdefault(sym, path)
    return found


def load_tickstory(data_dir: str, symbols=None, start: str | None = None,
                   end: str | None = None, min_bars: int = MIN_BARS,
                   session_close_hour: int = SESSION_CLOSE_HOUR, kind: str = "auto",
                   keep_weekends: bool = False, use_cache: bool = True,
                   verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Every Tickstory export in `data_dir`, rolled up to daily OHLCV.

    Returns the same {symbol: dataframe} contract as the yfinance loader, so the
    four layers do not care which one produced it.
    """
    if not os.path.isdir(data_dir):
        raise SystemExit(f"tickstory directory not found: {data_dir}")
    files = discover(data_dir)
    if symbols:
        wanted = {s.upper() for s in symbols}
        files = {k: v for k, v in files.items() if k in wanted}
    if not files:
        raise SystemExit(f"no Tickstory exports found in {data_dir} "
                         f"(looked for {', '.join(DATA_SUFFIXES)})")

    cache_dir = os.path.join(CACHE_DIR, "tickstory")
    os.makedirs(cache_dir, exist_ok=True)
    if verbose:
        print(f"Reading Tickstory exports from {data_dir}")
        print(f"  daily bars cut at {session_close_hour:02d}:00 of the file's own clock"
              + (" (17:00 New York, the FX convention)" if session_close_hour == 22 else ""))

    out, skipped = {}, []
    for sym, path in sorted(files.items()):
        cache = os.path.join(cache_dir, f"{sym}_daily_h{session_close_hour}.csv")
        daily = None
        if use_cache and os.path.exists(cache) and \
                os.path.getmtime(cache) >= os.path.getmtime(path):
            daily = pd.read_csv(cache, index_col=0, parse_dates=True)
            if verbose:
                print(f"  {os.path.basename(path):<34} cached -> {len(daily)} daily bars")
        if daily is None:
            try:
                daily = read_symbol(path, session_close_hour, kind, keep_weekends,
                                    verbose=verbose)
            except Exception as exc:
                skipped.append((sym, f"{type(exc).__name__}: {exc}"))
                continue
            daily.to_csv(cache)

        if start:
            daily = daily[daily.index >= pd.Timestamp(start)]
        if end:
            daily = daily[daily.index < pd.Timestamp(end)]
        if len(daily) < min_bars:
            skipped.append((sym, f"only {len(daily)} bars in range"))
            continue
        out[sym] = daily

    if verbose:
        print(f"\nLoaded {len(out)} symbols from Tickstory")
        for sym, why in skipped:
            print(f"  SKIPPED {sym}: {why}")
    return out
