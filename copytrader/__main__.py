"""Command line.

    python -m copytrader check   -c config.toml   # log in, list accounts + positions, test the socket
    python -m copytrader run     -c config.toml   # the copier itself (what systemd runs)
    python -m copytrader flatten -c config.toml   # emergency: close every follower position
    python -m copytrader alert   -c config.toml   # send a test Telegram message
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import fcntl
import logging
import os
import signal
import sys

import aiohttp

from . import config as config_mod
from .engine import Copier, Follower
from .notify import Notifier
from .tradovate import Connection, TradovateError

log = logging.getLogger("copytrader")


def _connections(cfg: dict, session: aiohttp.ClientSession, on_change=lambda conn: None) -> dict:
    return {name: Connection(name, login, cfg["environment"], session, on_change)
            for name, login in cfg["logins"].items()}


# ---------------------------------------------------------------- check

async def cmd_check(cfg: dict) -> int:
    ok = True
    wanted = {(cfg["leader"]["login"], cfg["leader"]["account"]): "leader"}
    for f in cfg["followers"]:
        wanted[(f["login"], f["account"])] = f"follower x{f['multiplier']} max {f['max_position']}"

    print(f"environment: {cfg['environment']}   dry_run: {cfg['dry_run']}")
    visible: dict[str, set] = {}
    async with aiohttp.ClientSession() as session:
        conns = _connections(cfg, session)
        for name, conn in conns.items():
            print(f"\n[{name}] logging in as {conn.cfg['username']} ...")
            try:
                await conn.login()
                accounts = await conn.list_accounts()
                positions = await conn.list_positions()
            except (TradovateError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
                print(f"  FAIL  {exc}")
                ok = False
                continue
            print(f"  OK    logged in, user id {conn.user_id}")
            visible[name] = {a["name"] for a in accounts} | {a.get("nickname") for a in accounts}
            for acc in accounts:
                role = wanted.get((name, acc["name"])) or wanted.get((name, acc.get("nickname")))
                nets = {}
                for p in positions:
                    if p.get("accountId") == acc["id"] and p.get("netPos"):
                        nets[p["contractId"]] = nets.get(p["contractId"], 0) + p["netPos"]
                held = ", ".join([f"{await conn.contract_name(c)} {n:+d}" for c, n in nets.items()]) or "flat"
                print(f"        account {acc['name']:<20} id {acc['id']:<10} {held:<24}"
                      f"{'<- ' + role if role else ''}")

            # Socket: connect, authorize, sync, then disconnect.
            stop = asyncio.Event()
            task = asyncio.create_task(conn.run(stop))
            try:
                await asyncio.wait_for(conn.ready.wait(), 20)
                print("  OK    websocket authorized and synced")
            except asyncio.TimeoutError:
                print("  FAIL  websocket did not sync within 20s (see log lines above)")
                ok = False
            stop.set()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(task, 10)

    for (login, account), role in wanted.items():
        if login in visible and account not in visible[login]:
            print(f"\nFAIL  {role} account {account!r} is not visible on login {login!r}")
            ok = False

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


# ---------------------------------------------------------------- run

def _single_instance(cfg: dict):
    """Hold a lock next to the pause file so two copiers can never run at once
    (e.g. a foreground test while the service is also running): both would order."""
    state_dir = os.path.dirname(cfg["pause_file"])
    if not os.path.isdir(state_dir):
        return None
    fh = open(os.path.join(state_dir, "copytrader.lock"), "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        raise SystemExit("another copier is already running (is the service up? "
                         "`sudo systemctl stop copytrader` first)")
    return fh


async def cmd_run(cfg: dict) -> int:
    lock = _single_instance(cfg)  # noqa: F841  (held for the life of the process)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)

    async with aiohttp.ClientSession() as session:
        tg = cfg["telegram"]
        notify = Notifier(session, tg.get("bot_token", ""), tg.get("chat_id", ""))
        copier: Copier | None = None
        conns = _connections(cfg, session, on_change=lambda conn: copier and copier.poke(conn))
        followers = [Follower(conns[f["login"]], f["account"], float(f["multiplier"]),
                              int(f["max_position"])) for f in cfg["followers"]]
        copier = Copier(conns[cfg["leader"]["login"]], cfg["leader"]["account"], followers,
                        dry_run=cfg["dry_run"], adopt_on_start=cfg["adopt_on_start"],
                        reconcile_interval=float(cfg["reconcile_interval"]),
                        order_timeout=float(cfg["order_timeout"]),
                        max_orders_per_minute=int(cfg["max_orders_per_minute"]),
                        pause_file=cfg["pause_file"], notify=notify)

        mode = "DRY RUN" if cfg["dry_run"] else "LIVE ORDERS"
        summary = (f"started ({cfg['environment']}, {mode}): leader {cfg['leader']['login']}/"
                   f"{cfg['leader']['account']} -> " + ", ".join(f.label for f in followers))
        log.info(summary)
        notify(summary)

        for conn in conns.values():
            try:
                await conn.login()
            except TradovateError as exc:
                log.error("%s", exc)
                notify(f"login failed, not starting: {exc}")
                return 1

        tasks = [asyncio.create_task(conn.run(stop)) for conn in conns.values()]
        tasks.append(asyncio.create_task(copier.run(stop)))
        await stop.wait()
        log.info("stopping")
        await notify.send("stopped")
        for task in tasks:
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await asyncio.wait_for(task, 10)
    return 0


# ---------------------------------------------------------------- flatten

async def cmd_flatten(cfg: dict, yes: bool) -> int:
    async with aiohttp.ClientSession() as session:
        conns = _connections(cfg, session)
        todo = []
        for f in cfg["followers"]:
            conn = conns[f["login"]]
            if conn.token is None:
                await conn.login()
            acc = next((a for a in await conn.list_accounts()
                        if f["account"] in (a["name"], a.get("nickname"))), None)
            if acc is None:
                print(f"account {f['account']} not found on {f['login']}")
                continue
            nets: dict[int, int] = {}
            for p in await conn.list_positions():
                if p.get("accountId") == acc["id"]:
                    nets[p["contractId"]] = nets.get(p["contractId"], 0) + p.get("netPos", 0)
            for cid, net in nets.items():
                if net:
                    todo.append((conn, acc, cid, net))
                    print(f"  {f['login']}/{acc['name']}: {await conn.contract_name(cid)} {net:+d}")
        if not todo:
            print("every follower is already flat")
            return 0
        if not yes and input("\nClose all of these at market? type YES: ").strip() != "YES":
            print("aborted")
            return 1
        for conn, acc, cid, _ in todo:
            try:
                await conn.liquidate(acc["id"], cid)
                print(f"  liquidation sent for {acc['name']} {await conn.contract_name(cid)}")
            except TradovateError as exc:
                print(f"  FAILED {acc['name']}: {exc}")
    print("\nIf the copier is running and the leader is still in a position, it will re-open the "
          "followers. Create the pause file first (see SETUP.md).")
    return 0


async def cmd_alert(cfg: dict) -> int:
    async with aiohttp.ClientSession() as session:
        tg = cfg["telegram"]
        notifier = Notifier(session, tg.get("bot_token", ""), tg.get("chat_id", ""))
        if not notifier.enabled:
            print("no [telegram] bot_token/chat_id in the config")
            return 1
        ok = await notifier.send("test alert: if you can read this, alerts work")
        print("sent" if ok else "failed (see log line above)")
        return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="copytrader", description="Tradovate copy trader")
    parser.add_argument("command", choices=["check", "run", "flatten", "alert"])
    parser.add_argument("-c", "--config", default="/etc/copytrader/config.toml")
    parser.add_argument("--yes", action="store_true", help="flatten without asking")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    try:
        cfg = config_mod.load(args.config)
    except config_mod.ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.command == "check":
        return asyncio.run(cmd_check(cfg))
    if args.command == "run":
        return asyncio.run(cmd_run(cfg))
    if args.command == "flatten":
        return asyncio.run(cmd_flatten(cfg, args.yes))
    return asyncio.run(cmd_alert(cfg))


if __name__ == "__main__":
    sys.exit(main())
