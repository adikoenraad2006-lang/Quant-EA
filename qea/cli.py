"""Shared data-source arguments, so all four layers load prices the same way."""

from __future__ import annotations

import argparse

from .config import (END, MIN_BARS, SESSION_CLOSE_HOUR, START, TICKSTORY_DIR,
                     UNIVERSE, load_cost_overrides)


def add_data_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = ap.add_argument_group("data source")
    g.add_argument("--source", choices=["yahoo", "tickstory", "synthetic"], default="yahoo",
                   help="where prices come from (default: yahoo)")
    g.add_argument("--data-dir", default=TICKSTORY_DIR,
                   help="directory of Tickstory exports (--source tickstory)")
    g.add_argument("--session-close-utc", type=int, default=SESSION_CLOSE_HOUR,
                   help="hour of the export's own clock to cut daily bars at; "
                        "22 = 17:00 New York (default), 0 = plain UTC days")
    g.add_argument("--tick-format", choices=["auto", "tick", "bars"], default="auto",
                   help="override Tickstory format detection")
    g.add_argument("--keep-weekends", action="store_true",
                   help="keep Saturday/Sunday sessions (for instruments that trade then)")
    g.add_argument("--costs", default=None,
                   help="csv of `symbol,bps` one-way costs, overriding the built-in table")
    g.add_argument("--assets", nargs="*", default=None,
                   help="restrict to these symbols (default: the whole universe / "
                        "every export found)")
    g.add_argument("--start", default=None)
    g.add_argument("--end", default=None)
    g.add_argument("--min-bars", type=int, default=MIN_BARS)
    g.add_argument("--no-cache", action="store_true")
    g.add_argument("--synthetic", action="store_true",
                   help="alias for --source synthetic")
    return ap


def load_from_args(args, tickers=None, verbose: bool = True) -> dict:
    """Prices for whichever source the arguments select."""
    source = "synthetic" if getattr(args, "synthetic", False) else args.source
    if args.costs:
        loaded = load_cost_overrides(args.costs)
        if verbose:
            print(f"Cost overrides: {len(loaded)} symbols from {args.costs}")

    wanted = tickers if tickers is not None else args.assets

    if source == "tickstory":
        from .tickstory import load_tickstory
        return load_tickstory(args.data_dir, symbols=wanted, start=args.start,
                              end=args.end, min_bars=args.min_bars,
                              session_close_hour=args.session_close_utc,
                              kind=args.tick_format, keep_weekends=args.keep_weekends,
                              use_cache=not args.no_cache, verbose=verbose)

    from .data import load_dataset
    return load_dataset(source == "synthetic", wanted or UNIVERSE,
                        args.start or START, args.end or END, args.min_bars,
                        use_cache=not args.no_cache, verbose=verbose)
