"""Polymarket public API client (Gamma + CLOB read endpoints, no auth).

Bases:
    https://gamma-api.polymarket.com   - market metadata, search, listings
    https://clob.polymarket.com        - orderbook, last trade, price history

Rate limit: ~60 req/min on the public US Retail API (verified, not the
20 req/sec figure that's been floating around). We throttle at ~1 req/s.

Polymarket binary markets have two outcome tokens (YES/NO). The CLOB endpoints
take a `token_id` (one of `clobTokenIds[0]` for YES, `clobTokenIds[1]` for NO).
The CLOB `market` parameter for prices-history actually expects a token_id.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

import requests

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"

_MIN_INTERVAL_S = 1.0  # ~60/min, conservative


@dataclass
class _Throttle:
    last: float = 0.0

    def wait(self) -> None:
        delta = time.monotonic() - self.last
        if delta < _MIN_INTERVAL_S:
            time.sleep(_MIN_INTERVAL_S - delta)
        self.last = time.monotonic()


class PolymarketClient:
    def __init__(self, gamma: str = GAMMA, clob: str = CLOB, timeout: float = 30.0):
        self.gamma = gamma.rstrip("/")
        self.clob = clob.rstrip("/")
        self.timeout = timeout
        self._s = requests.Session()
        self._s.headers["accept"] = "application/json"
        self._throttle = _Throttle()

    def _get(self, base: str, path: str, params: dict | None = None) -> dict | list:
        self._throttle.wait()
        url = f"{base}{path}"
        for attempt in range(5):
            r = self._s.get(url, params=params, timeout=self.timeout)
            if r.status_code == 429 or 500 <= r.status_code < 600:
                time.sleep(0.5 * (2**attempt))
                continue
            r.raise_for_status()
            return r.json()
        r.raise_for_status()
        return r.json()

    # --- Gamma (metadata + search) ---

    def list_markets(
        self,
        active: bool | None = True,
        closed: bool | None = False,
        limit: int = 100,
        offset: int = 0,
        order: str | None = None,
        ascending: bool | None = None,
        **extra,
    ) -> list[dict]:
        params: dict = {"limit": limit, "offset": offset}
        if active is not None:
            params["active"] = str(active).lower()
        if closed is not None:
            params["closed"] = str(closed).lower()
        if order:
            params["order"] = order
        if ascending is not None:
            params["ascending"] = str(ascending).lower()
        params.update(extra)
        data = self._get(self.gamma, "/markets", params=params)
        return data if isinstance(data, list) else data.get("data", [])

    def iter_markets(
        self,
        active: bool | None = True,
        closed: bool | None = False,
        page_size: int = 100,
        max_pages: int = 50,
        **extra,
    ) -> Iterator[dict]:
        for i in range(max_pages):
            page = self.list_markets(
                active=active, closed=closed, limit=page_size, offset=i * page_size, **extra
            )
            if not page:
                return
            for m in page:
                yield m
            if len(page) < page_size:
                return

    def get_market(self, market_id: int | str) -> dict:
        data = self._get(self.gamma, f"/markets/{market_id}")
        return data if isinstance(data, dict) else data[0]

    def search(self, q: str, limit_per_type: int = 10) -> dict:
        return self._get(
            self.gamma,
            "/public-search",
            params={"q": q, "limit_per_type": limit_per_type},
        )

    # --- CLOB (orderbook + price data) ---

    def orderbook(self, token_id: str) -> dict:
        return self._get(self.clob, "/book", params={"token_id": token_id})

    def last_trade_price(self, token_id: str) -> dict:
        return self._get(self.clob, "/last-trade-price", params={"token_id": token_id})

    def price_history(
        self,
        token_id: str,
        interval: str = "1h",
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> list[dict]:
        params: dict = {"market": token_id, "interval": interval}
        if start_ts is not None:
            params["startTs"] = start_ts
        if end_ts is not None:
            params["endTs"] = end_ts
        data = self._get(self.clob, "/prices-history", params=params)
        return data.get("history", []) if isinstance(data, dict) else data

    # --- Convenience ---

    @staticmethod
    def yes_token(market: dict) -> str | None:
        """Return the YES outcome's CLOB token_id from a Gamma market record.

        Polymarket binary markets have outcomes ['Yes', 'No'] aligned with
        clobTokenIds[0]/[1]. Sometimes outcomes are stored as a JSON-encoded
        string rather than a list — handle both.
        """
        import json as _json

        outcomes = market.get("outcomes")
        if isinstance(outcomes, str):
            outcomes = _json.loads(outcomes)
        token_ids = market.get("clobTokenIds")
        if isinstance(token_ids, str):
            token_ids = _json.loads(token_ids)
        if not outcomes or not token_ids:
            return None
        for o, t in zip(outcomes, token_ids):
            if str(o).strip().lower() == "yes":
                return t
        return None

    @staticmethod
    def best_bid_ask(orderbook: dict) -> tuple[float | None, float | None]:
        """Top-of-book bid/ask from a CLOB /book response."""
        bids = orderbook.get("bids") or []
        asks = orderbook.get("asks") or []
        best_bid = max((float(b["price"]) for b in bids), default=None)
        best_ask = min((float(a["price"]) for a in asks), default=None)
        return best_bid, best_ask
