"""The copy logic: keep every follower's position equal to the leader's, scaled.

This mirrors *positions*, not individual fills. On every change (and every few
seconds as a safety net) each follower's target is recomputed from the leader's
current net position, and a market order closes the gap. A missed event, a
dropped socket or a restart therefore cannot leave a follower drifting: the
next pass sees the gap and fixes it.

Guards:
  * a contract is only copied once the leader's position in it changes after
    start-up (unless adopt_on_start), so a restart never touches old trades
  * an order "in flight" is assumed filled until the follower's position shows
    it or `order_timeout` passes, so one gap is never ordered twice
  * targets are truncated toward zero and clamped to the follower's max_position
  * a per-follower order-rate breaker halts that follower if it starts looping
  * a pause file stops all new orders while it exists
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field

from .tradovate import Connection

log = logging.getLogger("copytrader.engine")


@dataclass
class Follower:
    conn: Connection
    account_name: str
    multiplier: float = 1.0
    max_position: int = 1
    halted: bool = False
    recent_orders: deque = field(default_factory=deque)

    @property
    def label(self) -> str:
        return f"{self.conn.name}/{self.account_name}"

    def target(self, leader_net: int) -> int:
        raw = math.trunc(leader_net * self.multiplier)
        return max(-self.max_position, min(self.max_position, raw))


@dataclass
class Flight:
    target: int
    deadline: float


class Copier:
    def __init__(self, leader: Connection, leader_account: str, followers: list[Follower], *,
                 dry_run: bool = True, adopt_on_start: bool = False, reconcile_interval: float = 5.0,
                 order_timeout: float = 15.0, max_orders_per_minute: int = 10,
                 pause_file: str = "", notify=lambda text: None):
        self.leader = leader
        self.leader_account = leader_account
        self.followers = followers
        self.dry_run = dry_run
        self.adopt_on_start = adopt_on_start
        self.reconcile_interval = reconcile_interval
        self.order_timeout = order_timeout
        self.max_orders_per_minute = max_orders_per_minute
        self.pause_file = pause_file
        self.notify = notify

        self.baseline: dict[int, int] | None = None   # leader positions at start-up
        self.tracked: set[int] = set()                # contracts being mirrored
        self.in_flight: dict[tuple[str, int], Flight] = {}
        self._dry_logged: dict[tuple[str, int], int] = {}
        self._dirty = asyncio.Event()
        self._was_paused = False
        self._warned: set[str] = set()

    # ------------------------------------------------------------ wiring

    def poke(self, conn: Connection | None = None) -> None:
        """Called by a connection whenever its state changes."""
        self._dirty.set()

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._dirty.wait(), self.reconcile_interval)
            await asyncio.sleep(0.05)   # let a burst of events land before acting
            self._dirty.clear()
            if stop.is_set():
                break
            try:
                await self.reconcile()
            except Exception:
                log.exception("reconcile pass failed")

    def _warn_once(self, key: str, text: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            log.warning(text)
            self.notify(text)

    # ------------------------------------------------------------ logic

    def paused(self) -> bool:
        return bool(self.pause_file) and os.path.exists(self.pause_file)

    def _leader_positions(self) -> dict[int, int] | None:
        if not self.leader.ready.is_set():
            return None
        account = self.leader.account_by_name(self.leader_account)
        if account is None:
            self._warn_once("leader-missing", f"leader account {self.leader_account!r} not found "
                                              f"on login {self.leader.name!r}")
            return None
        return self.leader.net_positions(account["id"])

    def _update_tracked(self, leader_pos: dict[int, int]) -> None:
        if self.baseline is None:
            self.baseline = dict(leader_pos)
            if self.adopt_on_start:
                self.tracked.update(cid for cid, net in leader_pos.items() if net)
            elif any(leader_pos.values()):
                log.info("leader already holds %s at start-up; those are left alone until "
                         "the leader changes them", {c: n for c, n in leader_pos.items() if n})
        for cid in set(leader_pos) | set(self.baseline):
            if leader_pos.get(cid, 0) != self.baseline.get(cid, 0):
                self.tracked.add(cid)

    async def reconcile(self) -> None:
        leader_pos = self._leader_positions()
        if leader_pos is None:
            return
        self._update_tracked(leader_pos)

        paused = self.paused()
        if paused != self._was_paused:
            self._was_paused = paused
            text = "PAUSED: pause file present, no orders will be sent" if paused else "resumed"
            log.warning(text)
            self.notify(text)

        for follower in self.followers:
            if follower.halted or not follower.conn.ready.is_set():
                continue
            account = follower.conn.account_by_name(follower.account_name)
            if account is None:
                self._warn_once(f"missing-{follower.label}",
                                f"follower account {follower.label!r} not found")
                continue
            current = follower.conn.net_positions(account["id"])
            for cid in sorted(self.tracked):
                await self._reconcile_one(follower, account, cid, leader_pos.get(cid, 0),
                                          current.get(cid, 0), paused)

    async def _reconcile_one(self, follower: Follower, account: dict, cid: int,
                             leader_net: int, current: int, paused: bool) -> None:
        key = (follower.label, cid)
        target = follower.target(leader_net)
        now = time.monotonic()

        flight = self.in_flight.get(key)
        if flight is not None:
            if current == flight.target:
                del self.in_flight[key]
                flight = None
            elif now > flight.deadline:
                log.warning("%s: order toward %+d not reflected after %.0fs (position %+d); "
                            "re-checking", follower.label, flight.target, self.order_timeout, current)
                del self.in_flight[key]
                flight = None
        assumed = flight.target if flight else current

        diff = target - assumed
        if diff == 0 or paused:
            return

        action = "Buy" if diff > 0 else "Sell"
        qty = abs(diff)
        symbol = await self.leader.contract_name(cid)
        what = (f"{follower.label}: {action} {qty} {symbol}  "
                f"(leader {leader_net:+d} -> target {target:+d}, follower {assumed:+d})")

        if self.dry_run:
            if self._dry_logged.get(key) != target:
                self._dry_logged[key] = target
                log.info("DRY RUN would send %s", what)
                self.notify(f"[dry run] {what}")
            return

        if not self._rate_ok(follower):
            return
        try:
            order_id = await follower.conn.place_market_order(account, symbol, action, qty)
        except Exception as exc:   # TradovateError, network errors, ...
            log.error("%s FAILED: %s", what, exc)
            self.notify(f"ORDER FAILED {what}: {exc}")
            # Treat the gap as in flight so it is retried after order_timeout, not hammered.
            self.in_flight[key] = Flight(target=target, deadline=now + self.order_timeout)
            return
        log.info("sent %s  order %s", what, order_id)
        self.notify(f"sent {what}")
        self.in_flight[key] = Flight(target=target, deadline=now + self.order_timeout)

    def _rate_ok(self, follower: Follower) -> bool:
        now = time.monotonic()
        q = follower.recent_orders
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= self.max_orders_per_minute:
            follower.halted = True
            text = (f"HALTED {follower.label}: {len(q)} orders in the last minute. Check the "
                    f"accounts, then restart the service to resume.")
            log.error(text)
            self.notify(text)
            return False
        q.append(now)
        return True

