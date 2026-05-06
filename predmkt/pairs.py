"""Cross-venue market pair watchlist.

A pair is a manually verified set of markets across Kalshi/Polymarket (and
later PredictIt) that resolve on the same underlying outcome. We keep this
manual rather than auto-matched: false matches in cross-venue arb are
expensive (you bet on different events thinking they're the same), so a
human-in-the-loop verification is worth the friction at v0.5 scale.

Watchlist is a JSON file at data/pairs.json. Each entry has:
    name              human-readable label, snake_case
    notes             one-line description for traceability
    kalshi            { ticker } — single market ticker on Kalshi
    polymarket        { market_id, yes_token_id } — Gamma id + CLOB token
    predictit         optional, { market_id }

Both legs should resolve on the same real-world event. The pair-tracker code
treats kalshi YES and polymarket YES as the *same* outcome — make sure the
naming is aligned when you add a pair.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

PAIRS_PATH = Path(__file__).resolve().parents[1] / "data" / "pairs.json"


@dataclass
class KalshiLeg:
    ticker: str


@dataclass
class PolymarketLeg:
    market_id: int
    yes_token_id: str


@dataclass
class PredictItLeg:
    market_id: int


@dataclass
class Pair:
    name: str
    notes: str = ""
    kalshi: KalshiLeg | None = None
    polymarket: PolymarketLeg | None = None
    predictit: PredictItLeg | None = None
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in list(d.items()):
            if v is None:
                del d[k]
        return d


def load_pairs(path: Path = PAIRS_PATH) -> list[Pair]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    out: list[Pair] = []
    for r in raw:
        out.append(
            Pair(
                name=r["name"],
                notes=r.get("notes", ""),
                tags=r.get("tags", []),
                kalshi=KalshiLeg(**r["kalshi"]) if r.get("kalshi") else None,
                polymarket=PolymarketLeg(**r["polymarket"]) if r.get("polymarket") else None,
                predictit=PredictItLeg(**r["predictit"]) if r.get("predictit") else None,
            )
        )
    return out


def save_pairs(pairs: list[Pair], path: Path = PAIRS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([p.to_dict() for p in pairs], indent=2))
