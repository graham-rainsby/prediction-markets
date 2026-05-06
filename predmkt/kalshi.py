"""Kalshi public API client (historical + live read endpoints, no auth).

Base: https://api.elections.kalshi.com/trade-api/v2
Historical cutoff (queryable): /historical/cutoff
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator

import requests

BASE = "https://api.elections.kalshi.com/trade-api/v2"

# Kalshi documents 10 req/s as a safe ceiling for unauthenticated historical endpoints.
_MIN_INTERVAL_S = 0.12


@dataclass
class _Throttle:
    last: float = 0.0

    def wait(self) -> None:
        delta = time.monotonic() - self.last
        if delta < _MIN_INTERVAL_S:
            time.sleep(_MIN_INTERVAL_S - delta)
        self.last = time.monotonic()


class KalshiClient:
    def __init__(self, base: str = BASE, timeout: float = 30.0):
        self.base = base.rstrip("/")
        self.timeout = timeout
        self._s = requests.Session()
        self._s.headers["accept"] = "application/json"
        self._throttle = _Throttle()

    def _get(self, path: str, params: dict | None = None) -> dict:
        self._throttle.wait()
        url = f"{self.base}{path}"
        for attempt in range(5):
            r = self._s.get(url, params=params, timeout=self.timeout)
            if r.status_code == 429 or 500 <= r.status_code < 600:
                time.sleep(0.5 * (2**attempt))
                continue
            r.raise_for_status()
            return r.json()
        r.raise_for_status()
        return r.json()

    def cutoff(self) -> dict:
        return self._get("/historical/cutoff")

    def historical_markets(
        self,
        series_ticker: str | None = None,
        event_ticker: str | None = None,
        tickers: str | None = None,
        limit: int = 1000,
    ) -> Iterator[dict]:
        cursor = ""
        while True:
            params: dict = {"limit": limit}
            if series_ticker:
                params["series_ticker"] = series_ticker
            if event_ticker:
                params["event_ticker"] = event_ticker
            if tickers:
                params["tickers"] = tickers
            if cursor:
                params["cursor"] = cursor
            data = self._get("/historical/markets", params=params)
            for m in data.get("markets", []):
                yield m
            cursor = data.get("cursor") or ""
            if not cursor:
                return

    def historical_market(self, ticker: str) -> dict:
        data = self._get(f"/historical/markets/{ticker}")
        return data.get("market", data)

    def historical_candlesticks(
        self,
        ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int,
    ) -> list[dict]:
        if period_interval not in (1, 60, 1440):
            raise ValueError("period_interval must be 1, 60, or 1440")
        data = self._get(
            f"/historical/markets/{ticker}/candlesticks",
            params={
                "start_ts": start_ts,
                "end_ts": end_ts,
                "period_interval": period_interval,
            },
        )
        return data.get("candlesticks", [])

    def historical_trades(
        self,
        ticker: str | None = None,
        min_ts: int | None = None,
        max_ts: int | None = None,
        limit: int = 1000,
    ) -> Iterator[dict]:
        cursor = ""
        while True:
            params: dict = {"limit": limit}
            if ticker:
                params["ticker"] = ticker
            if min_ts is not None:
                params["min_ts"] = min_ts
            if max_ts is not None:
                params["max_ts"] = max_ts
            if cursor:
                params["cursor"] = cursor
            data = self._get("/historical/trades", params=params)
            for t in data.get("trades", []):
                yield t
            cursor = data.get("cursor") or ""
            if not cursor:
                return


def to_unix(dt_or_str: datetime | str) -> int:
    if isinstance(dt_or_str, str):
        dt = datetime.fromisoformat(dt_or_str.replace("Z", "+00:00"))
    else:
        dt = dt_or_str
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())
