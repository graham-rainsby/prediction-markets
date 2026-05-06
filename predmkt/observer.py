"""Periodic snapshot of top-volume markets on Kalshi + Polymarket.

One row per market per snapshot. Daily parquet files in data/observations/.

Purpose: build a multi-week price/volume dataset. We don't pick strategies
upfront — we let the collected data suggest where edges live (spread widening,
correlated-market divergence, volume migrations, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from predmkt.polymarket import PolymarketClient

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "observations"
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"


SCHEMA = [
    "ts", "venue", "market_id", "title", "category", "event_id",
    "yes_bid", "yes_ask", "last_price", "spread",
    "volume_24h", "open_interest", "close_time",
]


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _to_float(x) -> float | None:
    if x is None or x == "":
        return None
    try:
        v = float(x)
        return v if v == v else None  # filter NaN
    except (TypeError, ValueError):
        return None


def snapshot_kalshi(
    top_n: int = 200,
    min_volume_24h_contracts: float = 100.0,
    horizon_days: int = 30,
) -> list[dict]:
    """Pull open Kalshi markets closing within `horizon_days`, keep top_n by 24h
    *dollar-estimated* volume above min contract threshold.

    Kalshi `volume_24h_fp` is contract count, not dollars. We estimate dollar
    volume as `contracts * yes_mid` (ignoring NO-side trades, ~equivalent).
    """
    import time as _time
    now = int(_time.time())
    end = now + horizon_days * 24 * 3600

    out: list[dict] = []
    cursor = ""
    pages = 0
    while pages < 15:
        params = {
            "status": "open", "limit": 1000,
            "min_close_ts": now, "max_close_ts": end,
        }
        if cursor:
            params["cursor"] = cursor
        r = requests.get(f"{KALSHI_BASE}/markets", params=params, timeout=30)
        r.raise_for_status()
        d = r.json()
        out.extend(d.get("markets", []))
        cursor = d.get("cursor") or ""
        pages += 1
        if not cursor:
            break

    ts = _now_utc().isoformat(timespec="seconds")
    rows = []
    for m in out:
        contracts = _to_float(m.get("volume_24h_fp"))
        if contracts is None or contracts < min_volume_24h_contracts:
            continue
        yes_bid = _to_float(m.get("yes_bid_dollars"))
        yes_ask = _to_float(m.get("yes_ask_dollars"))
        last = _to_float(m.get("last_price_dollars"))
        # Estimate per-contract price for $-volume: mid if available, else last
        price_est = None
        if yes_bid is not None and yes_ask is not None:
            price_est = (yes_bid + yes_ask) / 2.0
        elif last is not None and last > 0:
            price_est = last
        v24_dollars = (contracts * price_est) if price_est else 0.0
        spread = (yes_ask - yes_bid) if (yes_bid is not None and yes_ask is not None) else None
        rows.append({
            "ts": ts,
            "venue": "kalshi",
            "market_id": m.get("ticker"),
            "title": m.get("title"),
            "category": None,  # fetched separately at series level if needed
            "event_id": m.get("event_ticker"),
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "last_price": last,
            "spread": spread,
            "volume_24h": v24_dollars,
            "open_interest": _to_float(m.get("open_interest_fp")),
            "close_time": m.get("close_time"),
        })
    rows.sort(key=lambda r: r["volume_24h"] or 0, reverse=True)
    return rows[:top_n]


def snapshot_polymarket(top_n: int = 200, min_volume_24h: float = 100.0) -> list[dict]:
    """Pull active Polymarket markets, keep top_n by 24h volume above threshold."""
    poly = PolymarketClient()
    raw = poly.list_markets(
        active=True, closed=False, limit=top_n, order="volume24hr", ascending=False
    )
    ts = _now_utc().isoformat(timespec="seconds")
    rows = []
    for m in raw:
        v24 = _to_float(m.get("volume24hr"))
        if v24 is None or v24 < min_volume_24h:
            continue
        yes_bid = _to_float(m.get("bestBid"))
        yes_ask = _to_float(m.get("bestAsk"))
        spread = (yes_ask - yes_bid) if (yes_bid is not None and yes_ask is not None) else None
        category = None
        events = m.get("events") or []
        if events and isinstance(events, list):
            ev = events[0] if isinstance(events[0], dict) else {}
            category = ev.get("category") or ev.get("ticker")
        rows.append({
            "ts": ts,
            "venue": "polymarket",
            "market_id": str(m.get("id")),
            "title": m.get("question"),
            "category": category,
            "event_id": m.get("conditionId"),
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "last_price": _to_float(m.get("lastTradePrice")),
            "spread": spread,
            "volume_24h": v24,
            "open_interest": _to_float(m.get("openInterest")),
            "close_time": m.get("endDate"),
        })
    return rows


def snapshot_all(kalshi_top_n: int = 200, poly_top_n: int = 200) -> pd.DataFrame:
    rows = snapshot_kalshi(top_n=kalshi_top_n) + snapshot_polymarket(top_n=poly_top_n)
    df = pd.DataFrame(rows, columns=SCHEMA)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def append_snapshot(df: pd.DataFrame) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    day = df["ts"].iloc[0].strftime("%Y-%m-%d") if len(df) else _now_utc().strftime("%Y-%m-%d")
    path = DATA_DIR / f"{day}.parquet"
    if path.exists():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_parquet(path, index=False)
    return path


def load_observations(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Load all observation files, optionally filtered to a date range (YYYY-MM-DD)."""
    if not DATA_DIR.exists():
        return pd.DataFrame(columns=SCHEMA)
    files = sorted(DATA_DIR.glob("*.parquet"))
    if start:
        files = [f for f in files if f.stem >= start]
    if end:
        files = [f for f in files if f.stem <= end]
    if not files:
        return pd.DataFrame(columns=SCHEMA)
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
