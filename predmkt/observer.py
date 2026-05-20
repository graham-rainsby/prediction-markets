"""Periodic snapshot of top-volume markets on Kalshi + Polymarket.

One row per market per snapshot. Daily parquet files in data/observations/.

Two capture modes:
  1. top-N by 24h volume per venue (catches what's hot right now)
  2. watchlist (pinned markets — once seen, keep tracking until they close)

The watchlist fixes the cross-venue overlap problem: when volume migrates
between venues as an event approaches, top-N alone misses half the price
history. Watchlist ensures continuous per-market time series.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from predmkt.kalshi import KalshiClient
from predmkt.polymarket import PolymarketClient
from predmkt.watchlist import Watchlist

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "observations"

# Cap how many watchlist entries we re-fetch per snapshot, to keep CI under
# the 10-min budget. Eviction prefers oldest last_seen.
WATCHLIST_FETCH_CAP_PER_VENUE = 250

SCHEMA = [
    "ts", "venue", "market_id", "title", "category", "event_id",
    "yes_bid", "yes_ask", "last_price", "spread",
    "volume_24h", "open_interest", "close_time", "source",
]


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _to_float(x) -> float | None:
    if x is None or x == "":
        return None
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def _row_from_kalshi(m: dict, ts: str, source: str) -> dict:
    yes_bid = _to_float(m.get("yes_bid_dollars"))
    yes_ask = _to_float(m.get("yes_ask_dollars"))
    last = _to_float(m.get("last_price_dollars"))
    price_est = None
    if yes_bid is not None and yes_ask is not None:
        price_est = (yes_bid + yes_ask) / 2.0
    elif last is not None and last > 0:
        price_est = last
    contracts = _to_float(m.get("volume_24h_fp"))
    v24_dollars = (contracts * price_est) if (contracts and price_est) else 0.0
    spread = (yes_ask - yes_bid) if (yes_bid is not None and yes_ask is not None) else None
    return {
        "ts": ts,
        "venue": "kalshi",
        "market_id": m.get("ticker"),
        "title": m.get("title"),
        "category": None,
        "event_id": m.get("event_ticker"),
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "last_price": last,
        "spread": spread,
        "volume_24h": v24_dollars,
        "open_interest": _to_float(m.get("open_interest_fp")),
        "close_time": m.get("close_time"),
        "source": source,
    }


def _row_from_polymarket(m: dict, ts: str, source: str) -> dict:
    yes_bid = _to_float(m.get("bestBid"))
    yes_ask = _to_float(m.get("bestAsk"))
    spread = (yes_ask - yes_bid) if (yes_bid is not None and yes_ask is not None) else None
    category = None
    events = m.get("events") or []
    if events and isinstance(events, list):
        ev = events[0] if isinstance(events[0], dict) else {}
        category = ev.get("category") or ev.get("ticker")
    return {
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
        "volume_24h": _to_float(m.get("volume24hr")),
        "open_interest": _to_float(m.get("openInterest")),
        "close_time": m.get("endDate"),
        "source": source,
    }


def snapshot_kalshi(
    top_n: int = 200,
    min_volume_24h_contracts: float = 100.0,
    horizon_days: int = 30,
    client: KalshiClient | None = None,
) -> list[dict]:
    """Top-N open Kalshi markets closing within horizon, by $-volume."""
    import time as _time
    now = int(_time.time())
    end = now + horizon_days * 24 * 3600
    client = client or KalshiClient()
    raw = list(client.markets(
        status="open", min_close_ts=now, max_close_ts=end, limit=1000, max_pages=15,
    ))
    ts = _now_utc().isoformat(timespec="seconds")
    rows = []
    for m in raw:
        contracts = _to_float(m.get("volume_24h_fp"))
        if contracts is None or contracts < min_volume_24h_contracts:
            continue
        row = _row_from_kalshi(m, ts, source="topN")
        if (row["volume_24h"] or 0) <= 0:
            continue
        rows.append(row)
    rows.sort(key=lambda r: r["volume_24h"] or 0, reverse=True)
    return rows[:top_n]


def snapshot_polymarket(
    top_n: int = 200,
    min_volume_24h: float = 100.0,
    client: PolymarketClient | None = None,
) -> list[dict]:
    """Top-N active Polymarket markets by 24h $-volume."""
    client = client or PolymarketClient()
    raw = client.list_markets(
        active=True, closed=False, limit=top_n, order="volume24hr", ascending=False
    )
    ts = _now_utc().isoformat(timespec="seconds")
    rows = []
    for m in raw:
        v24 = _to_float(m.get("volume24hr"))
        if v24 is None or v24 < min_volume_24h:
            continue
        rows.append(_row_from_polymarket(m, ts, source="topN"))
    return rows


def _is_kalshi_closed(m: dict) -> bool:
    return m.get("status") in ("settled", "finalized", "closed")


def fetch_watchlist(
    watchlist: Watchlist,
    exclude_keys: set[tuple[str, str]],
    cap_per_venue: int = WATCHLIST_FETCH_CAP_PER_VENUE,
    kclient: KalshiClient | None = None,
    pclient: PolymarketClient | None = None,
) -> list[dict]:
    """Fetch current state for watchlist markets not already in `exclude_keys`.

    Drops markets from the watchlist after one final post-close snapshot.
    Mutates `watchlist` in place; caller is responsible for saving.
    """
    kclient = kclient or KalshiClient()
    pclient = pclient or PolymarketClient()
    ts = _now_utc().isoformat(timespec="seconds")
    rows: list[dict] = []

    # ---- Kalshi: batch by `tickers=A,B,...` ----
    k_to_fetch = [t for t in watchlist.tickers("kalshi") if ("kalshi", t) not in exclude_keys]
    # Prioritize oldest last_seen first (so we don't starve old entries)
    k_to_fetch.sort(key=lambda t: watchlist.entries["kalshi"][t].get("last_seen", ""))
    k_to_fetch = k_to_fetch[:cap_per_venue]
    # Chunk: Kalshi limit per call is 1000 but practical URL length safer at ~50
    CHUNK_K = 50
    for i in range(0, len(k_to_fetch), CHUNK_K):
        chunk = k_to_fetch[i:i+CHUNK_K]
        try:
            ms = list(kclient.markets(tickers=",".join(chunk)) if False else [])  # dead path; see below
        except TypeError:
            ms = []
        # KalshiClient.markets doesn't accept tickers; use a direct call via session.
        # Use the raw session for batch tickers (no series filter):
        try:
            kclient._throttle.wait()
            r = kclient._s.get(
                f"{kclient.base}/markets",
                params={"tickers": ",".join(chunk), "limit": 1000},
                timeout=kclient.timeout,
            )
            r.raise_for_status()
            ms = r.json().get("markets", [])
        except Exception as e:
            print(f"[watchlist] kalshi batch fetch failed: {e}")
            continue
        seen_tickers = set()
        for m in ms:
            tk = m.get("ticker")
            if not tk:
                continue
            seen_tickers.add(tk)
            row = _row_from_kalshi(m, ts, source="watch")
            rows.append(row)
            if _is_kalshi_closed(m):
                if watchlist.needs_one_more_snapshot("kalshi", tk):
                    watchlist.mark_closed("kalshi", tk)
                else:
                    watchlist.drop("kalshi", tk)
            else:
                # Refresh last_seen
                watchlist.add("kalshi", tk)
        # Tickers in the chunk that weren't returned by Kalshi → drop them
        for missing in set(chunk) - seen_tickers:
            watchlist.drop("kalshi", missing)

    # ---- Polymarket: single-fetch (Gamma's id= filter doesn't batch) ----
    p_to_fetch = [t for t in watchlist.tickers("polymarket") if ("polymarket", t) not in exclude_keys]
    p_to_fetch.sort(key=lambda t: watchlist.entries["polymarket"][t].get("last_seen", ""))
    p_to_fetch = p_to_fetch[:cap_per_venue]
    for mid in p_to_fetch:
        try:
            m = pclient.get_market(mid)
        except Exception as e:
            print(f"[watchlist] polymarket {mid} fetch failed: {e}")
            # Don't drop on transient errors
            continue
        if not m or not m.get("id"):
            watchlist.drop("polymarket", mid)
            continue
        row = _row_from_polymarket(m, ts, source="watch")
        rows.append(row)
        if m.get("closed"):
            if watchlist.needs_one_more_snapshot("polymarket", mid):
                watchlist.mark_closed("polymarket", mid)
            else:
                watchlist.drop("polymarket", mid)
        else:
            watchlist.add("polymarket", mid)

    return rows


def snapshot_all(
    kalshi_top_n: int = 200,
    poly_top_n: int = 200,
    use_watchlist: bool = True,
) -> pd.DataFrame:
    kclient = KalshiClient()
    pclient = PolymarketClient()

    top_rows = snapshot_kalshi(top_n=kalshi_top_n, client=kclient) \
             + snapshot_polymarket(top_n=poly_top_n, client=pclient)
    seen = {(r["venue"], r["market_id"]) for r in top_rows}

    all_rows = list(top_rows)
    if use_watchlist:
        wl = Watchlist.load()
        # Add/refresh anything from top-N
        for r in top_rows:
            wl.add(r["venue"], r["market_id"])
        # Fetch watchlist entries not already in top-N
        wl_rows = fetch_watchlist(wl, exclude_keys=seen, kclient=kclient, pclient=pclient)
        all_rows.extend(wl_rows)
        wl.evict_if_over_cap()
        wl.save()

    df = pd.DataFrame(all_rows, columns=SCHEMA)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def append_snapshot(df: pd.DataFrame) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    day = df["ts"].iloc[0].strftime("%Y-%m-%d") if len(df) else _now_utc().strftime("%Y-%m-%d")
    path = DATA_DIR / f"{day}.parquet"
    if path.exists():
        existing = pd.read_parquet(path)
        # Ensure schema compatibility (older files lack the 'source' column)
        for col in SCHEMA:
            if col not in existing.columns:
                existing[col] = None
        df = pd.concat([existing, df], ignore_index=True)
    df.to_parquet(path, index=False)
    return path


def load_observations(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    if not DATA_DIR.exists():
        return pd.DataFrame(columns=SCHEMA)
    files = sorted(DATA_DIR.glob("*.parquet"))
    if start:
        files = [f for f in files if f.stem >= start]
    if end:
        files = [f for f in files if f.stem <= end]
    if not files:
        return pd.DataFrame(columns=SCHEMA)
    frames = []
    for f in files:
        d = pd.read_parquet(f)
        for col in SCHEMA:
            if col not in d.columns:
                d[col] = None
        frames.append(d)
    return pd.concat(frames, ignore_index=True)
