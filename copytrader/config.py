"""Load and validate config.toml. Any string value written as "env:NAME" is read
from the environment variable NAME, so secrets can live outside the file."""

from __future__ import annotations

import os
import tomllib

from .tradovate import ENVIRONMENTS


class ConfigError(Exception):
    pass


def _resolve(value):
    if isinstance(value, str) and value.startswith("env:"):
        name = value[4:]
        if name not in os.environ:
            raise ConfigError(f"environment variable {name} is not set")
        return os.environ[name]
    if isinstance(value, dict):
        return {k: _resolve(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v) for v in value]
    return value


def load(path: str) -> dict:
    try:
        with open(path, "rb") as fh:
            cfg = _resolve(tomllib.load(fh))
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: {exc}") from None

    cfg.setdefault("environment", "demo")
    cfg.setdefault("dry_run", True)
    cfg.setdefault("adopt_on_start", False)
    cfg.setdefault("reconcile_interval", 5.0)
    cfg.setdefault("order_timeout", 15.0)
    cfg.setdefault("max_orders_per_minute", 10)
    cfg.setdefault("pause_file", "/var/lib/copytrader/PAUSE")
    cfg.setdefault("telegram", {})

    if cfg["environment"] not in ENVIRONMENTS:
        raise ConfigError(f"environment must be one of {sorted(ENVIRONMENTS)}")

    logins = cfg.get("logins") or {}
    if not logins:
        raise ConfigError("add at least one [logins.<name>] section")
    for name, login in logins.items():
        for key in ("username", "password", "cid", "sec"):
            if login.get(key) in (None, "", 0):
                raise ConfigError(f"[logins.{name}] is missing {key}")

    leader = cfg.get("leader") or {}
    if leader.get("login") not in logins or not leader.get("account"):
        raise ConfigError("[leader] needs login (one of the [logins.*] names) and account")

    followers = cfg.get("followers") or []
    if not followers:
        raise ConfigError("add at least one [[followers]] entry")
    seen = {(leader["login"], leader["account"])}
    for i, f in enumerate(followers, 1):
        if f.get("login") not in logins or not f.get("account"):
            raise ConfigError(f"follower #{i} needs login (one of the [logins.*] names) and account")
        if (f["login"], f["account"]) in seen:
            raise ConfigError(f"follower #{i} ({f['account']}) is the leader or listed twice")
        seen.add((f["login"], f["account"]))
        f.setdefault("multiplier", 1.0)
        f.setdefault("max_position", 1)
        if f["multiplier"] <= 0 or int(f["max_position"]) < 1:
            raise ConfigError(f"follower #{i}: multiplier must be > 0 and max_position >= 1")
    return cfg
