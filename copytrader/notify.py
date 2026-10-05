"""Optional Telegram alerts. With no bot token configured, alerts are only logged."""

from __future__ import annotations

import asyncio
import logging

import aiohttp

log = logging.getLogger("copytrader.notify")


class Notifier:
    def __init__(self, session: aiohttp.ClientSession, bot_token: str = "", chat_id: str = "",
                 prefix: str = "copytrader"):
        self.session = session
        self.bot_token = bot_token
        self.chat_id = str(chat_id)
        self.prefix = prefix
        self._tasks: set[asyncio.Task] = set()

    @property
    def enabled(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def __call__(self, text: str) -> None:
        """Fire and forget: alerting must never block or break order flow."""
        if not self.enabled:
            return
        task = asyncio.ensure_future(self.send(text))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def send(self, text: str) -> bool:
        if not self.enabled:
            return False
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        try:
            async with self.session.post(url, json={"chat_id": self.chat_id,
                                                    "text": f"[{self.prefix}] {text}"},
                                         timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    log.warning("telegram returned HTTP %s: %s", resp.status, (await resp.text())[:200])
                    return False
                return True
        except Exception as exc:
            log.warning("telegram alert failed: %s", exc)
            return False
