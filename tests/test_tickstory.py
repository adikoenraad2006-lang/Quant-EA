#!/usr/bin/env python3
"""Checks for the Tickstory reader: every export shape must land on the same bars.

    python tests/test_tickstory.py
"""

from __future__ import annotations

import os
import struct
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from qea.tickstory import (discover, load_tickstory, read_symbol, sniff_format,
                           symbol_from_filename)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append(name)


# ------------------------------------------------------------------ fixtures
def minute_bars(days: int = 700, seed: int = 0) -> pd.DataFrame:
    """M1 bars over `days` sessions, 24 hourly stamps a day (00:00 .. 23:00)."""
    rng = np.random.default_rng(seed)
    stamps, rows = [], []
    day = pd.Timestamp("2015-01-05")
    price = 1.2
    sessions = 0
    while sessions < days:                     # always emit whole sessions
        if day.dayofweek < 5:
            sessions += 1
            # the FX week ends 17:00 New York on Friday, so no late Friday bars
            last_hour = 22 if day.dayofweek == 4 else 24
            for hour in range(last_hour):
                o = price
                c = o * float(np.exp(rng.normal(0, 0.0008)))
                h = max(o, c) * (1 + abs(float(rng.normal(0, 0.0003))))
                l = min(o, c) * (1 - abs(float(rng.normal(0, 0.0003))))
                stamps.append(day + pd.Timedelta(hours=hour))
                rows.append((o, h, l, c, float(rng.integers(10, 500))))
                price = c
        day += pd.Timedelta(days=1)
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"],
                        index=pd.DatetimeIndex(stamps))


def expected_daily(bars: pd.DataFrame, session_close_hour: int) -> pd.DataFrame:
    shift = (24 - session_close_hour % 24) % 24
    day = (bars.index + pd.Timedelta(hours=shift)).floor("D")
    out = bars.groupby(day).agg(Open=("Open", "first"), High=("High", "max"),
                                Low=("Low", "min"), Close=("Close", "last"),
                                Volume=("Volume", "sum"))
    return out[out.index.dayofweek < 5]


# ------------------------------------------------------------------ writers
def write_mt4_csv(bars, path):
    with open(path, "w") as fh:
        for ts, r in bars.iterrows():
            fh.write(f"{ts:%Y.%m.%d},{ts:%H:%M},{r.Open:.5f},{r.High:.5f},"
                     f"{r.Low:.5f},{r.Close:.5f},{int(r.Volume)}\n")


def write_mt5_tsv(bars, path):
    with open(path, "w") as fh:
        fh.write("<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n")
        for ts, r in bars.iterrows():
            fh.write(f"{ts:%Y.%m.%d}\t{ts:%H:%M:%S}\t{r.Open:.5f}\t{r.High:.5f}\t"
                     f"{r.Low:.5f}\t{r.Close:.5f}\t{int(r.Volume)}\t0\t2\n")


def write_generic_iso(bars, path):
    with open(path, "w") as fh:
        fh.write("Timestamp,Open,High,Low,Close,Volume\n")
        for ts, r in bars.iterrows():
            fh.write(f"{ts:%Y-%m-%d %H:%M:%S},{r.Open:.5f},{r.High:.5f},"
                     f"{r.Low:.5f},{r.Close:.5f},{int(r.Volume)}\n")


def write_ninja(bars, path):
    with open(path, "w") as fh:
        for ts, r in bars.iterrows():
            fh.write(f"{ts:%Y%m%d %H%M%S};{r.Open:.5f};{r.High:.5f};"
                     f"{r.Low:.5f};{r.Close:.5f};{int(r.Volume)}\n")


def write_ticks(bars, path, spread=0.00010, with_volume=True):
    """One bar -> four ticks (open, high, low, close) so the daily roll-up matches."""
    with open(path, "w") as fh:
        for ts, r in bars.iterrows():
            for i, mid in enumerate([r.Open, r.High, r.Low, r.Close]):
                stamp = ts + pd.Timedelta(milliseconds=i * 100)
                bid, ask = mid - spread / 2, mid + spread / 2
                tail = f",{r.Volume / 4:.2f},0" if with_volume else ""
                fh.write(f"{stamp:%Y.%m.%d %H:%M:%S}.{stamp.microsecond // 1000:03d},"
                         f"{bid:.5f},{ask:.5f}{tail}\n")


def write_hst(bars, path, version=401):
    with open(path, "wb") as fh:
        header = bytearray(148)
        struct.pack_into("<i", header, 0, version)
        fh.write(header)
        for ts, r in bars.iterrows():
            ctm = int(ts.timestamp())
            if version == 400:
                fh.write(struct.pack("<iddddd", ctm, r.Open, r.Low, r.High,
                                     r.Close, r.Volume))
            else:
                fh.write(struct.pack("<qddddqiq", ctm, r.Open, r.High, r.Low,
                                     r.Close, int(r.Volume), 2, 0))


# ------------------------------------------------------------------ tests
def test_formats(tmp: str) -> None:
    print("\nformat detection and roll-up (session close 22:00)")
    bars = minute_bars()
    want = expected_daily(bars, 22)

    writers = {
        "EURUSD_M1.csv": (write_mt4_csv, "bars"),
        "GBPUSD.tsv": (write_mt5_tsv, "bars"),
        "USDJPY_2015-2025.csv": (write_generic_iso, "bars"),
        "AUDUSD_M1_UTC.txt": (write_ninja, "bars"),
        "XAUUSD_ticks.csv": (write_ticks, "tick"),
    }
    for name, (writer, kind) in writers.items():
        path = os.path.join(tmp, name)
        writer(bars, path)
        fmt = sniff_format(path)
        check(f"{name}: detected as {kind}", fmt.kind == kind, f"got {fmt.kind}")
        got = read_symbol(path, session_close_hour=22, verbose=False)
        aligned = got.reindex(want.index)
        # exports carry 5 decimals, so that is the precision to compare at
        same = np.allclose(aligned[["Open", "High", "Low", "Close"]].to_numpy(),
                           want[["Open", "High", "Low", "Close"]].to_numpy(),
                           rtol=0, atol=2e-5, equal_nan=False)
        check(f"{name}: daily OHLC matches", bool(same))
        check(f"{name}: bar count matches", len(got) == len(want),
              f"{len(got)} vs {len(want)}")

    # ticks with no volume column at all
    path = os.path.join(tmp, "EURGBP_tick_novol.csv")
    write_ticks(bars, path, with_volume=False)
    got = read_symbol(path, session_close_hour=22, verbose=False)
    check("tick file without volume: counts ticks instead",
          float(got["Volume"].sum()) == 4 * len(bars), str(got["Volume"].sum()))


def test_hst(tmp: str) -> None:
    print("\nmt4 .hst binary")
    bars = minute_bars(days=600)
    want = expected_daily(bars, 22)
    for version in (400, 401):
        path = os.path.join(tmp, f"EURCHF{version}.hst")
        write_hst(bars, path, version)
        got = read_symbol(path, session_close_hour=22, verbose=False)
        aligned = got.reindex(want.index)
        check(f"hst {version}: daily OHLC matches",
              bool(np.allclose(aligned[["Open", "High", "Low", "Close"]].to_numpy(),
                               want[["Open", "High", "Low", "Close"]].to_numpy(),
                               rtol=0, atol=1e-12)))   # binary, so exact


def test_sessions(tmp: str) -> None:
    print("\nsession boundary")
    bars = minute_bars(days=520)
    path = os.path.join(tmp, "SESSION.csv")
    write_mt4_csv(bars, path)

    utc = read_symbol(path, session_close_hour=0, verbose=False)
    ny = read_symbol(path, session_close_hour=22, verbose=False)
    check("cutting at 22:00 matches a UTC-midnight cut on count within a bar",
          abs(len(utc) - len(ny)) <= 1, f"{len(utc)} vs {len(ny)}")

    want_utc = expected_daily(bars, 0)
    check("UTC-day close is the 23:00 bar close",
          np.allclose(utc.reindex(want_utc.index)["Close"].to_numpy(),
                      want_utc["Close"].to_numpy(), rtol=0, atol=2e-5))
    want_ny = expected_daily(bars, 22)
    check("22:00 close rolls the last two hours into the next session",
          np.allclose(ny.reindex(want_ny.index)["Close"].to_numpy(),
                      want_ny["Close"].to_numpy(), rtol=0, atol=2e-5))
    check("the two cuts really differ", not np.allclose(
        utc["Close"].to_numpy()[:100], ny["Close"].to_numpy()[:100]))


def test_compressed_and_names(tmp: str) -> None:
    print("\ncompression and symbol naming")
    import gzip
    import zipfile
    bars = minute_bars(days=520)
    plain = os.path.join(tmp, "NZDUSD.csv")
    write_mt4_csv(bars, plain)
    ref = read_symbol(plain, verbose=False)

    gz = os.path.join(tmp, "NZDCAD.csv.gz")
    with open(plain, "rb") as src, gzip.open(gz, "wb") as dst:
        dst.write(src.read())
    check("gzip round-trips", read_symbol(gz, verbose=False).equals(ref))

    zp = os.path.join(tmp, "NZDCHF.zip")
    with zipfile.ZipFile(zp, "w") as zf:
        zf.write(plain, "inner.csv")
    check("zip round-trips", read_symbol(zp, verbose=False).equals(ref))

    cases = {
        "EURUSD_M1_2015-2025.csv": "EURUSD",
        "GBPJPY-M15.txt": "GBPJPY",
        "XAUUSD_ticks.csv.gz": "XAUUSD",
        "USDCAD.hst": "USDCAD",
        "US30_H1_UTC.csv": "US30",
        "EURUSD_2020.csv": "EURUSD",
    }
    bad = {k: symbol_from_filename(k) for k, v in cases.items()
           if symbol_from_filename(k) != v}
    check("symbols come out of filenames", not bad, str(bad))


def test_directory(tmp: str) -> None:
    print("\ndirectory load")
    d = os.path.join(tmp, "dir")
    os.makedirs(d, exist_ok=True)
    bars = minute_bars(days=700)
    for sym in ["EURUSD", "GBPUSD", "USDJPY"]:
        write_mt4_csv(bars, os.path.join(d, f"{sym}_M1.csv"))
    write_mt4_csv(minute_bars(days=100), os.path.join(d, "SHORTY_M1.csv"))

    check("discover finds every export", set(discover(d)) ==
          {"EURUSD", "GBPUSD", "USDJPY", "SHORTY"}, str(set(discover(d))))

    prices = load_tickstory(d, min_bars=500, verbose=False)
    check("short history is skipped", set(prices) == {"EURUSD", "GBPUSD", "USDJPY"},
          str(set(prices)))
    check("frames carry OHLCV", list(prices["EURUSD"].columns) ==
          ["Open", "High", "Low", "Close", "Volume"])
    check("index is a sorted DatetimeIndex",
          prices["EURUSD"].index.is_monotonic_increasing and
          isinstance(prices["EURUSD"].index, pd.DatetimeIndex))

    sub = load_tickstory(d, symbols=["eurusd"], min_bars=500, verbose=False)
    check("symbol filter is case-insensitive", set(sub) == {"EURUSD"}, str(set(sub)))

    cached = load_tickstory(d, min_bars=500, verbose=False)
    check("cached read matches the first read", cached["EURUSD"].equals(prices["EURUSD"]))


def test_runs_the_tester(tmp: str) -> None:
    print("\nthe tester runs on it")
    from qea.backtest import run_backtest
    from qea.config import cost_bps
    from qea.strategies import build_configs
    d = os.path.join(tmp, "dir")
    prices = load_tickstory(d, min_bars=500, verbose=False)
    configs = build_configs()
    df = prices["EURUSD"]
    rows = [run_backtest(df, c, "EURUSD", train_bars=300, test_bars=100) for c in configs]
    rows = [r for r in rows if r]
    check("every config ran on Tickstory bars", len(rows) == len(configs),
          f"{len(rows)}/{len(configs)}")
    check("none errored", not any("error" in r for r in rows),
          str([r["config"] for r in rows if "error" in r][:3]))
    check("FX gets an FX cost, not the equity default", cost_bps("EURUSD") == 1.0,
          str(cost_bps("EURUSD")))


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        test_formats(tmp)
        test_hst(tmp)
        test_sessions(tmp)
        test_compressed_and_names(tmp)
        test_directory(tmp)
        test_runs_the_tester(tmp)
    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        sys.exit(1)
    print("all checks passed")
