"""Surface candidate Kalshi <-> Polymarket pairs for manual verification.

Kalshi side: pull all series in political categories, then enumerate open
markets across them. Polymarket side: top markets by 24h volume.

Match by token-Jaccard on titles. Print a diagnostic table for human review.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import requests
import pandas as pd

from predmkt.polymarket import PolymarketClient

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
POLITICAL_CATS = {"Politics", "Elections"}

STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "by", "at", "to", "for", "and", "or",
    "be", "is", "are", "was", "were", "will", "would", "could", "should",
    "this", "that", "these", "those", "it", "its", "with", "before", "after",
    "from", "into", "out", "over", "under", "than", "then", "do", "does",
    "did", "any", "all", "no", "yes",
}
_PUNCT_RE = re.compile(r"[^\w\s]")


def tokens(s: str) -> set[str]:
    s = (s or "").lower()
    s = _PUNCT_RE.sub(" ", s)
    return {t for t in s.split() if t and t not in STOPWORDS and len(t) > 2}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a and b) else 0.0


def fetch_kalshi_series(categories: set[str]) -> list[dict]:
    out: list[dict] = []
    cursor = ""
    while True:
        params = {"limit": 200}
        if cursor:
            params["cursor"] = cursor
        r = requests.get(f"{KALSHI_BASE}/series", params=params, timeout=30)
        r.raise_for_status()
        d = r.json()
        for s in d.get("series", []):
            if s.get("category") in categories:
                out.append(s)
        cursor = d.get("cursor") or ""
        if not cursor:
            break
    return out


def fetch_kalshi_open_markets(series_ticker: str) -> list[dict]:
    out: list[dict] = []
    cursor = ""
    while True:
        params = {"series_ticker": series_ticker, "status": "open", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        r = requests.get(f"{KALSHI_BASE}/markets", params=params, timeout=30)
        r.raise_for_status()
        d = r.json()
        out.extend(d.get("markets", []))
        cursor = d.get("cursor") or ""
        if not cursor:
            break
    return out


def main() -> None:
    pd.set_option("display.width", 180)
    pd.set_option("display.max_colwidth", 75)

    print("loading Kalshi series in political categories...")
    series_list = fetch_kalshi_series(POLITICAL_CATS)
    print(f"  political series found: {len(series_list)}")

    print("\nloading open markets across those series (sampling top 80 series)...")
    kalshi_markets: list[dict] = []
    for s in series_list[:80]:
        try:
            mkts = fetch_kalshi_open_markets(s["ticker"])
        except Exception:
            continue
        kalshi_markets.extend(mkts)
    print(f"  open Kalshi political markets: {len(kalshi_markets)}")

    # Sort by liquidity_dollars (or notional) descending; take top 60
    def _vol(m):
        return float(m.get("liquidity_dollars") or m.get("notional_value_dollars") or 0)
    kalshi_markets.sort(key=_vol, reverse=True)
    kalshi_markets = kalshi_markets[:60]

    print("\n--- Kalshi political markets sample (top 12 by liquidity) ---")
    for k in kalshi_markets[:12]:
        print(f"  liq=${_vol(k):>10,.0f}  {k.get('ticker'):<36}  {(k.get('title') or '')[:75]}")

    print("\nloading top-volume active Polymarket markets...")
    poly = PolymarketClient()
    poly_mkts = poly.list_markets(active=True, closed=False, limit=200, order="volume24hr", ascending=False)
    poly_with_tokens = [m for m in poly_mkts if PolymarketClient.yes_token(m)]
    print(f"  Polymarket markets with YES token: {len(poly_with_tokens)}")

    print("\n--- Polymarket top-volume markets (first 12) ---")
    for p in poly_with_tokens[:12]:
        print(f"  vol24h=${float(p.get('volume24hr') or 0):>12,.0f}  id={p.get('id'):>7}  {(p.get('question') or '')[:75]}")

    print("\n=== Candidate pairings (jaccard >= 0.20, top 3 per Kalshi market) ===")
    rows = []
    for k in kalshi_markets:
        k_title = k.get("title") or ""
        k_tokens = tokens(k_title)
        scored = [(jaccard(k_tokens, tokens(p.get("question") or "")), p) for p in poly_with_tokens]
        scored = [(s, p) for s, p in scored if s >= 0.20]
        scored.sort(key=lambda x: x[0], reverse=True)
        for sim, p in scored[:3]:
            rows.append({
                "sim": round(sim, 2),
                "k_ticker": k.get("ticker"),
                "k_title": k_title[:60],
                "k_liq": _vol(k),
                "p_id": p.get("id"),
                "p_question": (p.get("question") or "")[:60],
                "p_vol24h": float(p.get("volume24hr") or 0),
            })
    if not rows:
        print("  no candidate pairs above threshold")
        return
    df = pd.DataFrame(rows).sort_values("sim", ascending=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
