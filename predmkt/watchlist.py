"""Persistent watchlist of markets to track regardless of current volume rank.

Problem this solves: the top-N-by-volume observer captured each venue's PGA
markets at different times (Kalshi pre-tournament, Polymarket mid-tournament),
because volume migrated between venues as the event approached. Result: no
overlapping observations, no measurable cross-venue gap.

Fix: pin each market the first time it appears in a top-N snapshot. Keep
fetching it (regardless of volume rank) until it closes. After it closes,
take one final post-resolution snapshot, then drop it from the watchlist.

Watchlist file: data/watchlist.json
  {
    "kalshi":     {"KXPGATOUR-PGC26-SSCH": {"first_seen": "2026-05-05T...", "last_seen": "...", "closed_seen": false}},
    "polymarket": {"2234555":              {"first_seen": "2026-05-15T...", "last_seen": "...", "closed_seen": false}}
  }

Cap: watchlist size capped per venue (default 800) to prevent unbounded growth
during sport-season tickets that spawn many markets. Eviction by oldest
last_seen when over cap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "data" / "watchlist.json"
MAX_PER_VENUE = 800


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Watchlist:
    entries: dict[str, dict[str, dict]] = field(default_factory=lambda: {"kalshi": {}, "polymarket": {}})

    @classmethod
    def load(cls, path: Path = PATH) -> "Watchlist":
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text())
        return cls(entries={
            "kalshi": dict(raw.get("kalshi", {})),
            "polymarket": dict(raw.get("polymarket", {})),
        })

    def save(self, path: Path = PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.entries, indent=2, sort_keys=True))

    def add(self, venue: str, market_id: str) -> None:
        venue_map = self.entries.setdefault(venue, {})
        now = _now_iso()
        if market_id not in venue_map:
            venue_map[market_id] = {"first_seen": now, "last_seen": now, "closed_seen": False}
        else:
            venue_map[market_id]["last_seen"] = now

    def mark_closed(self, venue: str, market_id: str) -> None:
        venue_map = self.entries.get(venue, {})
        if market_id in venue_map:
            venue_map[market_id]["closed_seen"] = True
            venue_map[market_id]["last_seen"] = _now_iso()

    def drop(self, venue: str, market_id: str) -> None:
        self.entries.get(venue, {}).pop(market_id, None)

    def tickers(self, venue: str) -> list[str]:
        return list(self.entries.get(venue, {}).keys())

    def needs_one_more_snapshot(self, venue: str, market_id: str) -> bool:
        e = self.entries.get(venue, {}).get(market_id)
        return e is not None and not e.get("closed_seen", False)

    def evict_if_over_cap(self, max_per_venue: int = MAX_PER_VENUE) -> int:
        """Drop oldest-last-seen entries to stay under cap. Returns # dropped."""
        dropped = 0
        for venue, vmap in self.entries.items():
            if len(vmap) <= max_per_venue:
                continue
            # Sort by last_seen ascending; drop oldest
            sorted_items = sorted(vmap.items(), key=lambda kv: kv[1].get("last_seen", ""))
            excess = len(vmap) - max_per_venue
            for mid, _ in sorted_items[:excess]:
                vmap.pop(mid, None)
                dropped += 1
        return dropped
