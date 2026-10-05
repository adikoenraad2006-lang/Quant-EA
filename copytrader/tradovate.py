"""The slice of the Tradovate API the copier needs: login, one live socket, orders.

REST does the request/response work (login, token renewal, placing orders).
The WebSocket is used only for `user/syncrequest`, which pushes every change to
the login's accounts and positions as it happens.

Socket protocol (Tradovate's SockJS-style framing):
    server -> client   'o' open, 'h' heartbeat, 'a[...]' JSON messages, 'c[...]' close
    client -> server   "endpoint\\nrequestId\\nquery\\nbody", plus "[]" as a heartbeat
Responses carry {"i": requestId, "s": status, "d": data}; pushed changes arrive
as {"e": "props", "d": {"entityType", "eventType", "entity"}}.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import itertools
import json
import logging
import time
from datetime import datetime
from typing import Any, Callable

import aiohttp

ENVIRONMENTS = {
    "demo": ("https://demo.tradovateapi.com/v1", "wss://demo.tradovateapi.com/v1/websocket"),
    "live": ("https://live.tradovateapi.com/v1", "wss://live.tradovateapi.com/v1/websocket"),
}

HEARTBEAT_EVERY = 2.5      # seconds; Tradovate drops sockets that go quiet
SILENCE_LIMIT = 20.0       # no frame from the server for this long -> reconnect
RENEW_BEFORE = 15 * 60     # renew the access token this long before it expires
REQUEST_TIMEOUT = 15.0

log = logging.getLogger("copytrader.tradovate")


class TradovateError(Exception):
    pass


# ---------------------------------------------------------------- framing

def parse_frame(raw: str) -> tuple[str, list]:
    """Split one socket frame into (kind, messages)."""
    if not raw:
        return "", []
    kind = raw[0]
    if kind in "ac" and len(raw) > 1:
        return kind, json.loads(raw[1:])
    return kind, []


def build_request(endpoint: str, request_id: int, body: Any = None, query: str = "") -> str:
    """One request frame. A str body is sent verbatim (the authorize token)."""
    if body is None:
        payload = ""
    elif isinstance(body, str):
        payload = body
    else:
        payload = json.dumps(body)
    return f"{endpoint}\n{request_id}\n{query}\n{payload}"


def _expiry(text: str | None) -> float:
    if not text:
        return time.time() + 60 * 60
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


# ---------------------------------------------------------------- one login

class Connection:
    """One Tradovate login: its token, its socket, and its live account/position state."""

    def __init__(self, name: str, cfg: dict, environment: str, session: aiohttp.ClientSession,
                 on_change: Callable[["Connection"], None] = lambda conn: None,
                 urls: tuple[str, str] | None = None):
        self.name = name
        self.cfg = cfg
        self.session = session
        self.on_change = on_change
        self.rest_url, self.ws_url = urls or ENVIRONMENTS[environment]

        self.token: str | None = None
        self.token_expiry = 0.0
        self.user_id: int | None = None

        self.accounts: dict[int, dict] = {}     # account id -> account
        self.positions: dict[int, dict] = {}    # position id -> position
        self.ready = asyncio.Event()            # set while the socket is synced

        self._ids = itertools.count(1)
        self._pending: dict[int, asyncio.Future] = {}
        self._contract_names: dict[int, str] = {}
        self._last_rx = 0.0
        self._renewing = False

    # ------------------------------------------------------------ REST

    async def _http(self, method: str, path: str, body: Any = None, params: dict | None = None,
                    auth: bool = True) -> Any:
        headers = {"Accept": "application/json"}
        if auth and self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        async with self.session.request(method, self.rest_url + path, json=body, params=params,
                                        headers=headers, timeout=timeout) as resp:
            text = await resp.text()
            if resp.status >= 400:
                raise TradovateError(f"{method} {path} -> HTTP {resp.status}: {text[:300]}")
            return json.loads(text) if text else None

    def _credentials(self) -> dict:
        cfg = self.cfg
        device = cfg.get("device_id") or hashlib.sha1(
            f"copytrader-{self.name}-{cfg['username']}".encode()).hexdigest()
        return {
            "name": cfg["username"],
            "password": cfg["password"],
            "appId": cfg.get("app_id", "copytrader"),
            "appVersion": cfg.get("app_version", "1.0"),
            "cid": cfg["cid"],
            "sec": cfg["sec"],
            "deviceId": device,
        }

    def _take_token(self, data: dict) -> None:
        if not data or "accessToken" not in data:
            raise TradovateError(f"[{self.name}] no access token in response: {data}")
        self.token = data["accessToken"]
        self.token_expiry = _expiry(data.get("expirationTime"))
        self.user_id = data.get("userId", self.user_id)

    async def login(self) -> None:
        body = self._credentials()
        for _ in range(5):
            data = await self._http("POST", "/auth/accesstokenrequest", body, auth=False)
            if data and "p-ticket" in data:
                # Rate-limit "penalty ticket": wait, then retry carrying the ticket.
                if data.get("p-captcha"):
                    raise TradovateError(
                        f"[{self.name}] Tradovate wants a captcha for this login. Log in once "
                        "through the Tradovate web app, wait an hour, then try again.")
                wait = float(data.get("p-time", 5))
                log.warning("[%s] login rate-limited, retrying in %.0fs", self.name, wait)
                body["p-ticket"] = data["p-ticket"]
                await asyncio.sleep(wait)
                continue
            if data and data.get("errorText"):
                raise TradovateError(f"[{self.name}] login refused: {data['errorText']}")
            self._take_token(data)
            log.info("[%s] logged in (user %s)", self.name, self.user_id)
            return
        raise TradovateError(f"[{self.name}] login kept getting rate-limited")

    async def renew(self) -> None:
        try:
            self._take_token(await self._http("GET", "/auth/renewaccesstoken"))
            log.info("[%s] access token renewed", self.name)
        except (TradovateError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
            log.warning("[%s] token renewal failed (%s); logging in again", self.name, exc)
            await self.login()

    async def ensure_token(self) -> None:
        if not self.token:
            await self.login()
        elif time.time() > self.token_expiry - RENEW_BEFORE:
            await self.renew()

    async def list_accounts(self) -> list[dict]:
        return await self._http("GET", "/account/list") or []

    async def list_positions(self) -> list[dict]:
        return await self._http("GET", "/position/list") or []

    async def contract_name(self, contract_id: int) -> str:
        if contract_id not in self._contract_names:
            item = await self._http("GET", "/contract/item", params={"id": contract_id})
            self._contract_names[contract_id] = item["name"]
        return self._contract_names[contract_id]

    async def place_market_order(self, account: dict, symbol: str, action: str, qty: int) -> int:
        result = await self._http("POST", "/order/placeorder", {
            "accountSpec": account["name"],
            "accountId": account["id"],
            "action": action,
            "symbol": symbol,
            "orderQty": qty,
            "orderType": "Market",
            "isAutomated": True,   # CME requires automated orders to be flagged
        })
        if not result or result.get("failureReason") or "orderId" not in result:
            raise TradovateError(f"order rejected: {result}")
        return result["orderId"]

    async def liquidate(self, account_id: int, contract_id: int) -> Any:
        return await self._http("POST", "/order/liquidateposition",
                                {"accountId": account_id, "contractId": contract_id, "admin": False})

    # ------------------------------------------------------------ state

    def account_by_name(self, name: str) -> dict | None:
        for acc in self.accounts.values():
            if acc.get("name") == name or acc.get("nickname") == name:
                return acc
        return None

    def net_positions(self, account_id: int) -> dict[int, int]:
        """contract id -> net quantity, summed over every position row for the account."""
        out: dict[int, int] = {}
        for pos in self.positions.values():
            if pos.get("accountId") == account_id:
                cid = pos["contractId"]
                out[cid] = out.get(cid, 0) + int(pos.get("netPos", 0))
        return out

    def _apply(self, entity_type: str, event_type: str, entity: dict) -> None:
        table = {"position": self.positions, "account": self.accounts}.get(entity_type)
        if table is None:
            if entity_type == "fill":
                log.info("[%s] fill: %s %s @ %s (order %s)", self.name, entity.get("action"),
                         entity.get("qty"), entity.get("price"), entity.get("orderId"))
            return
        if event_type == "Deleted":
            table.pop(entity["id"], None)
        else:
            table[entity["id"]] = entity
        self.on_change(self)

    # ------------------------------------------------------------ socket

    async def run(self, stop: asyncio.Event) -> None:
        """Keep one synced socket open until `stop` is set, reconnecting with backoff."""
        backoff = 1.0
        while not stop.is_set():
            started = time.monotonic()
            try:
                await self.ensure_token()
                await self._session(stop)
            except (aiohttp.ClientError, asyncio.TimeoutError, TradovateError, ConnectionError) as exc:
                log.warning("[%s] socket down: %s", self.name, exc)
            except Exception:
                log.exception("[%s] unexpected socket error", self.name)
            self._drop()
            if stop.is_set():
                break
            if time.monotonic() - started > 60:
                backoff = 1.0
            log.info("[%s] reconnecting in %.0fs", self.name, backoff)
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(stop.wait(), backoff)
            backoff = min(backoff * 2, 60.0)

    def _drop(self) -> None:
        was_ready = self.ready.is_set()
        self.ready.clear()
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(ConnectionError("socket closed"))
        self._pending.clear()
        if was_ready:
            self.on_change(self)

    async def _session(self, stop: asyncio.Event) -> None:
        async with self.session.ws_connect(self.ws_url, autoping=True) as ws:
            self._last_rx = time.monotonic()
            tasks = [asyncio.create_task(self._heartbeat(ws))]
            stopper = asyncio.create_task(stop.wait())
            stopper.add_done_callback(lambda _: asyncio.ensure_future(ws.close()))
            try:
                async for msg in ws:
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        if msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                            break
                        continue
                    self._last_rx = time.monotonic()
                    kind, items = parse_frame(msg.data)
                    if kind == "o":
                        tasks.append(asyncio.create_task(self._handshake(ws)))
                    elif kind == "a":
                        for item in items:
                            self._dispatch(item, ws)
                    elif kind == "c":
                        raise ConnectionError(f"server closed the socket: {items}")
            finally:
                stopper.cancel()
                for task in tasks:
                    task.cancel()
                for task in tasks:
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await task
            if not stop.is_set():
                raise ConnectionError("socket closed")

    async def _call(self, ws: aiohttp.ClientWebSocketResponse, endpoint: str, body: Any = None,
                    request_id: int | None = None) -> Any:
        rid = next(self._ids) if request_id is None else request_id
        fut = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        await ws.send_str(build_request(endpoint, rid, body))
        reply = await asyncio.wait_for(fut, REQUEST_TIMEOUT)
        if reply.get("s") != 200:
            raise TradovateError(f"[{self.name}] {endpoint} -> {reply.get('s')}: {reply.get('d')}")
        return reply.get("d")

    async def _handshake(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        try:
            await self._call(ws, "authorize", self.token, request_id=0)
            data = await self._call(ws, "user/syncrequest", {"users": [self.user_id]}) or {}
        except Exception as exc:
            log.warning("[%s] handshake failed: %s", self.name, exc)
            await ws.close()
            return
        self.accounts = {a["id"]: a for a in data.get("accounts", [])}
        self.positions = {p["id"]: p for p in data.get("positions", [])}
        self.ready.set()
        log.info("[%s] synced: %d account(s), %d position row(s)",
                 self.name, len(self.accounts), len(self.positions))
        self.on_change(self)

    def _dispatch(self, item: dict, ws: aiohttp.ClientWebSocketResponse) -> None:
        rid = item.get("i")
        if rid is not None and rid in self._pending:
            fut = self._pending.pop(rid)
            if not fut.done():
                fut.set_result(item)
            return
        event = item.get("e")
        if event == "props":
            d = item.get("d") or {}
            self._apply(d.get("entityType", ""), d.get("eventType", ""), d.get("entity") or {})
        elif event == "shutdown":
            log.warning("[%s] server announced shutdown: %s", self.name, item.get("d"))
            asyncio.ensure_future(ws.close())

    async def _heartbeat(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        while not ws.closed:
            await asyncio.sleep(HEARTBEAT_EVERY)
            if time.monotonic() - self._last_rx > SILENCE_LIMIT:
                log.warning("[%s] server silent for %.0fs, reconnecting", self.name, SILENCE_LIMIT)
                await ws.close()
                return
            await ws.send_str("[]")
            if time.time() > self.token_expiry - RENEW_BEFORE and not self._renewing:
                # Renew off to the side so a slow REST call never delays a heartbeat.
                self._renewing = True
                asyncio.create_task(self._renew_in_background())

    async def _renew_in_background(self) -> None:
        try:
            await self.renew()
        except Exception as exc:
            log.error("[%s] could not renew or re-login: %s", self.name, exc)
        finally:
            self._renewing = False
