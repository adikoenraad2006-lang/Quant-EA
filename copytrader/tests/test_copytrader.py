#!/usr/bin/env python3
"""Checks for the copy trader. No Tradovate account or network needed: the last
group runs the real client against a fake Tradovate server on localhost.

    python copytrader/tests/test_copytrader.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile

sys.path.insert(0, __file__.rsplit("/", 3)[0])

import aiohttp
from aiohttp import web

from copytrader import config as config_mod
from copytrader.engine import Copier, Follower
from copytrader.tradovate import Connection, build_request, parse_frame

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------------- framing

def test_framing() -> None:
    print("framing")
    check("open frame", parse_frame("o") == ("o", []))
    check("heartbeat frame", parse_frame("h") == ("h", []))
    kind, items = parse_frame('a[{"s":200,"i":3,"d":{}}]')
    check("data frame", kind == "a" and items == [{"s": 200, "i": 3, "d": {}}])
    check("close frame", parse_frame('c[1000,"bye"]') == ("c", [1000, "bye"]))
    check("authorize request", build_request("authorize", 0, "TOKEN") == "authorize\n0\n\nTOKEN")
    check("json request", build_request("user/syncrequest", 7, {"users": [1]})
          == 'user/syncrequest\n7\n\n{"users": [1]}')
    check("empty-body request", build_request("account/list", 2) == "account/list\n2\n\n")


# ---------------------------------------------------------------- config

def _write(text: str) -> str:
    fh = tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False)
    fh.write(text)
    fh.close()
    return fh.name


GOOD = """
environment = "demo"
[logins.main]
username = "u"
password = "env:CT_TEST_PW"
cid = 1
sec = "s"
[leader]
login = "main"
account = "L"
[[followers]]
login = "main"
account = "F"
"""


def test_config() -> None:
    print("config")
    os.environ["CT_TEST_PW"] = "secret"
    cfg = config_mod.load(_write(GOOD))
    check("env: values resolved", cfg["logins"]["main"]["password"] == "secret")
    check("dry_run defaults on", cfg["dry_run"] is True)
    check("follower defaults", cfg["followers"][0]["multiplier"] == 1.0
          and cfg["followers"][0]["max_position"] == 1)

    def rejects(text: str) -> bool:
        try:
            config_mod.load(_write(text))
        except config_mod.ConfigError:
            return True
        return False

    check("rejects follower == leader", rejects(GOOD.replace('account = "F"', 'account = "L"')))
    check("rejects unknown login", rejects(GOOD.replace('login = "main"\naccount = "F"',
                                                        'login = "nope"\naccount = "F"')))
    check("rejects bad environment", rejects(GOOD.replace('"demo"', '"paper"')))
    check("rejects missing env var", rejects(GOOD.replace("CT_TEST_PW", "CT_TEST_UNSET_XYZ")))
    check("rejects zero multiplier", rejects(GOOD + "multiplier = 0\n"))


# ---------------------------------------------------------------- engine (fake connections)

class FakeConn:
    def __init__(self, name: str, accounts: dict[str, int]):
        self.name = name
        self.ready = asyncio.Event()
        self.ready.set()
        self.accounts = {i: {"id": i, "name": n} for n, i in accounts.items()}
        self.net: dict[int, dict[int, int]] = {i: {} for i in accounts.values()}
        self.orders: list[tuple[str, str, str, int]] = []
        self.fail_next = False
        self.fill_orders = True

    def account_by_name(self, name):
        return next((a for a in self.accounts.values() if a["name"] == name), None)

    def net_positions(self, account_id):
        return dict(self.net[account_id])

    async def contract_name(self, cid):
        return f"C{cid}"

    async def place_market_order(self, account, symbol, action, qty):
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("rejected")
        self.orders.append((account["name"], symbol, action, qty))
        if self.fill_orders:
            cid = int(symbol[1:])
            book = self.net[account["id"]]
            book[cid] = book.get(cid, 0) + (qty if action == "Buy" else -qty)
        return len(self.orders)


async def _setup(leader_start=None, adopt=False, mult=1.0, cap=5, dry=False, pause_file="", rate=10):
    """Leader + follower, after the first pass has recorded the start-up baseline."""
    lead = FakeConn("lead", {"L": 1})
    fol = FakeConn("fol", {"F": 2})
    lead.net[1] = dict(leader_start or {})
    f = Follower(fol, "F", mult, cap)
    cop = Copier(lead, "L", [f], dry_run=dry, adopt_on_start=adopt, order_timeout=0.2,
                 max_orders_per_minute=rate, pause_file=pause_file)
    await cop.reconcile()
    return lead, fol, f, cop


async def _engine() -> None:
    print("engine")
    lead, fol, _, cop = await _setup()
    await cop.reconcile()
    check("flat leader -> no orders", fol.orders == [])
    lead.net[1][100] = 2
    await cop.reconcile()
    check("leader opens 2 -> follower buys 2", fol.orders == [("F", "C100", "Buy", 2)])
    await cop.reconcile()
    check("no duplicate order on next pass", len(fol.orders) == 1)
    lead.net[1][100] = -1
    await cop.reconcile()
    check("reversal sells 3", fol.orders[-1] == ("F", "C100", "Sell", 3) and fol.net[2][100] == -1)
    lead.net[1][100] = 0
    await cop.reconcile()
    check("leader flat -> follower flat", fol.net[2][100] == 0)

    # Existing leader position at start-up is left alone unless adopt_on_start.
    lead, fol, _, cop = await _setup(leader_start={100: 3})
    await cop.reconcile()
    check("pre-existing leader position ignored", fol.orders == [])
    lead.net[1][100] = 4
    await cop.reconcile()
    check("...until the leader changes it, then mirrored", fol.net[2].get(100) == 4)
    lead, fol, _, cop = await _setup(leader_start={100: 3}, adopt=True)
    await cop.reconcile()
    check("adopt_on_start mirrors immediately", fol.net[2].get(100) == 3)

    # Multiplier truncates toward zero; cap clamps.
    lead, fol, _, cop = await _setup(mult=0.5, cap=2)
    lead.net[1][7] = 3
    await cop.reconcile()
    check("0.5 x 3 -> 1 (truncated)", fol.net[2].get(7) == 1)
    lead.net[1][7] = -9
    await cop.reconcile()
    check("0.5 x -9 -> -2 (capped)", fol.net[2].get(7) == -2)

    # In-flight order: not re-sent while unfilled, retried after timeout.
    lead, fol, _, cop = await _setup()
    fol.fill_orders = False
    lead.net[1][5] = 1
    await cop.reconcile()
    await cop.reconcile()
    check("unfilled order not duplicated", len(fol.orders) == 1)
    await asyncio.sleep(0.25)
    await cop.reconcile()
    check("re-sent after order_timeout", len(fol.orders) == 2)

    # Failed order is retried after the timeout.
    lead, fol, _, cop = await _setup()
    fol.fail_next = True
    lead.net[1][5] = 1
    await cop.reconcile()
    check("failed order leaves follower flat", fol.net[2].get(5, 0) == 0)
    await asyncio.sleep(0.25)
    await cop.reconcile()
    check("failed order retried", fol.net[2].get(5) == 1)

    # Dry run sends nothing.
    lead, fol, _, cop = await _setup(dry=True)
    lead.net[1][5] = 2
    await cop.reconcile()
    check("dry run sends nothing", fol.orders == [])

    # Pause file.
    pause = os.path.join(tempfile.mkdtemp(), "PAUSE")
    lead, fol, _, cop = await _setup(pause_file=pause)
    open(pause, "w").close()
    lead.net[1][5] = 1
    await cop.reconcile()
    check("pause file blocks orders", fol.orders == [])
    os.remove(pause)
    await cop.reconcile()
    check("removing pause file resumes", fol.net[2].get(5) == 1)

    # Rate breaker.
    lead, fol, f, cop = await _setup(rate=3)
    for i in range(6):
        lead.net[1][5] = 1 if i % 2 == 0 else -1
        await cop.reconcile()
    check("rate breaker halts a looping follower", f.halted and len(fol.orders) == 3)

    # Follower or leader offline -> nothing.
    lead, fol, _, cop = await _setup()
    lead.net[1][5] = 1
    fol.ready.clear()
    await cop.reconcile()
    check("offline follower skipped", fol.orders == [])
    fol.ready.set()
    lead.ready.clear()
    await cop.reconcile()
    check("offline leader -> nothing sent", fol.orders == [])
    lead.ready.set()
    await cop.reconcile()
    check("catches up when both are back", fol.net[2].get(5) == 1)


# ---------------------------------------------------------------- fake Tradovate server

class FakeTradovate:
    """Speaks just enough Tradovate to drive Connection + Copier for real."""

    def __init__(self):
        self.positions = {1: {"id": 1, "accountId": 10, "contractId": 500, "netPos": 0}}
        self.orders: list[dict] = []
        self.sockets: list[web.WebSocketResponse] = []
        self.logins = 0

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/v1/auth/accesstokenrequest", self.login)
        app.router.add_get("/v1/auth/renewaccesstoken", self.login)
        app.router.add_get("/v1/contract/item", self.contract)
        app.router.add_post("/v1/order/placeorder", self.place)
        app.router.add_get("/v1/websocket", self.socket)
        return app

    async def login(self, request):
        self.logins += 1
        if request.method == "POST":
            body = await request.json()
            if body.get("password") != "pw":
                return web.json_response({"errorText": "Incorrect username or password"})
        return web.json_response({"accessToken": "TOK", "userId": 42,
                                  "expirationTime": "2099-01-01T00:00:00.000Z"})

    async def contract(self, request):
        return web.json_response({"id": int(request.query["id"]), "name": "MESZ6"})

    async def place(self, request):
        assert request.headers.get("Authorization") == "Bearer TOK"
        body = await request.json()
        self.orders.append(body)
        # Fill instantly: update the follower's position and push it.
        pos = self.positions.setdefault(2, {"id": 2, "accountId": 20, "contractId": 500, "netPos": 0})
        pos["netPos"] += body["orderQty"] if body["action"] == "Buy" else -body["orderQty"]
        await self.push(pos)
        return web.json_response({"orderId": len(self.orders)})

    async def push(self, pos):
        frame = "a" + json.dumps([{"e": "props", "d": {"entityType": "position",
                                                        "eventType": "Updated", "entity": pos}}])
        for ws in list(self.sockets):
            if not ws.closed:
                await ws.send_str(frame)

    async def socket(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.sockets.append(ws)
        await ws.send_str("o")
        async for msg in ws:
            if msg.data == "[]":
                continue
            endpoint, rid, _, body = msg.data.split("\n", 3)
            if endpoint == "authorize":
                ok = body == "TOK"
                await ws.send_str("a" + json.dumps([{"s": 200 if ok else 401, "i": int(rid)}]))
            elif endpoint == "user/syncrequest":
                d = {"accounts": [{"id": 10, "name": "LEAD"}, {"id": 20, "name": "FOLLOW"}],
                     "positions": list(self.positions.values())}
                await ws.send_str("a" + json.dumps([{"s": 200, "i": int(rid), "d": d}]))
        return ws


async def _wait_for(pred, timeout=5.0) -> bool:
    end = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < end:
        if pred():
            return True
        await asyncio.sleep(0.02)
    return False


async def _integration() -> None:
    print("against a fake Tradovate server")
    fake = FakeTradovate()
    runner = web.AppRunner(fake.app())
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    urls = (f"http://127.0.0.1:{port}/v1", f"ws://127.0.0.1:{port}/v1/websocket")

    async with aiohttp.ClientSession() as session:
        bad = Connection("bad", {"username": "u", "password": "wrong", "cid": 1, "sec": "s"},
                         "demo", session, urls=urls)
        try:
            await bad.login()
            check("wrong password is reported", False)
        except Exception as exc:
            check("wrong password is reported", "Incorrect" in str(exc))

        copier: Copier | None = None
        conn = Connection("main", {"username": "u", "password": "pw", "cid": 1, "sec": "s"},
                          "demo", session, on_change=lambda c: copier and copier.poke(c), urls=urls)
        copier = Copier(conn, "LEAD", [Follower(conn, "FOLLOW", 1.0, 3)], dry_run=False,
                        reconcile_interval=0.2)
        await conn.login()
        stop = asyncio.Event()
        tasks = [asyncio.create_task(conn.run(stop)), asyncio.create_task(copier.run(stop))]

        check("socket authorizes and syncs", await _wait_for(conn.ready.is_set))
        check("accounts loaded from sync", conn.account_by_name("FOLLOW") is not None)
        check("start-up baseline recorded", await _wait_for(lambda: copier.baseline == {500: 0}))

        fake.positions[1]["netPos"] = 2
        await fake.push(fake.positions[1])
        check("leader fill copied to follower", await _wait_for(lambda: len(fake.orders) == 1))
        if fake.orders:
            o = fake.orders[0]
            check("order fields", (o["action"], o["orderQty"], o["symbol"], o["accountSpec"],
                                   o["orderType"], o["isAutomated"])
                  == ("Buy", 2, "MESZ6", "FOLLOW", "Market", True), str(o))
        check("follower position updated by push",
              await _wait_for(lambda: conn.net_positions(20).get(500) == 2))

        # Drop the socket: the client must reconnect, re-sync, and catch up.
        for ws in fake.sockets:
            await ws.close()
        check("notices the dropped socket", await _wait_for(lambda: not conn.ready.is_set(), 3))
        fake.positions[1]["netPos"] = 0           # leader closes while we are disconnected
        check("reconnects and re-syncs", await _wait_for(conn.ready.is_set, 6))
        check("catches up the missed close",
              await _wait_for(lambda: conn.net_positions(20).get(500) == 0, 3)
              and fake.orders[-1]["action"] == "Sell")

        stop.set()
        for task in tasks:
            await asyncio.wait_for(task, 5)
        check("stops cleanly", all(t.done() for t in tasks))
    await runner.cleanup()


def main() -> int:
    test_framing()
    test_config()
    asyncio.run(_engine())
    asyncio.run(_integration())
    print(f"\n{'ALL PASSED' if not FAILURES else f'{len(FAILURES)} FAILED: {FAILURES}'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
