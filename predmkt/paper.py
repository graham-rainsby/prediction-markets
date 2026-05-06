"""Append-only paper-trading log.

One JSONL file per strategy in data/paper/<strategy>.jsonl. Each line is a
single event with a `type` discriminator:

    obs         - periodic observation of a tracked market pair
    bet         - a hypothetical entry; entry_price/side/size locked in
    resolution  - settlement value applied to a prior bet, with realized PnL

A `bet_id` (UUID) ties a resolution back to its bet. Designed for forward-only
paper trading (we never rewrite history). Reading/aggregation happens via
load_events() into a DataFrame.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "paper"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _path(strategy: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in strategy)
    return DATA_DIR / f"{safe}.jsonl"


def _append(strategy: str, record: dict[str, Any]) -> None:
    with _path(strategy).open("a") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")


@dataclass
class PaperLog:
    strategy: str

    def log_obs(self, **fields) -> None:
        rec = {"type": "obs", "ts": _now_iso(), "strategy": self.strategy, **fields}
        _append(self.strategy, rec)

    def log_bet(
        self,
        venue: str,
        market_id: str,
        side: str,                # "YES" or "NO"
        entry_price: float,       # in dollars (0..1)
        size_usd: float,
        fee: float = 0.0,
        metadata: dict | None = None,
    ) -> str:
        bet_id = uuid.uuid4().hex[:12]
        rec = {
            "type": "bet",
            "bet_id": bet_id,
            "ts": _now_iso(),
            "strategy": self.strategy,
            "venue": venue,
            "market_id": market_id,
            "side": side.upper(),
            "entry_price": float(entry_price),
            "size_usd": float(size_usd),
            "fee": float(fee),
            "metadata": metadata or {},
        }
        _append(self.strategy, rec)
        return bet_id

    def log_resolution(
        self,
        bet_id: str,
        settlement_price: float,  # 0.0 or 1.0 for binary
        notes: str | None = None,
    ) -> None:
        # PnL is computed on aggregation; just store raw inputs.
        rec = {
            "type": "resolution",
            "ts": _now_iso(),
            "strategy": self.strategy,
            "bet_id": bet_id,
            "settlement_price": float(settlement_price),
        }
        if notes:
            rec["notes"] = notes
        _append(self.strategy, rec)


def load_events(strategy: str) -> pd.DataFrame:
    p = _path(strategy)
    if not p.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def realized_bets(strategy: str) -> pd.DataFrame:
    """Join bets with their resolutions, compute PnL. Unresolved bets excluded."""
    df = load_events(strategy)
    if df.empty:
        return df
    bet_cols = ["bet_id", "ts", "venue", "market_id", "side", "entry_price", "size_usd", "fee", "metadata"]
    bets = df[df["type"] == "bet"][[c for c in bet_cols if c in df.columns]].copy()
    res = df[df["type"] == "resolution"].copy()
    if bets.empty or res.empty:
        return pd.DataFrame()
    res = res[["bet_id", "settlement_price", "ts"]].rename(columns={"ts": "resolved_ts"})
    j = bets.merge(res, on="bet_id", how="inner")

    def _pnl(r: pd.Series) -> float:
        # contracts purchased = size_usd / entry_price
        n = r["size_usd"] / r["entry_price"] if r["entry_price"] > 0 else 0
        if r["side"] == "YES":
            payoff = n * float(r["settlement_price"])
        else:
            payoff = n * (1.0 - float(r["settlement_price"]))
        cost = r["size_usd"]
        return payoff - cost - float(r.get("fee") or 0.0)

    j["pnl_usd"] = j.apply(_pnl, axis=1)
    return j


def open_bets(strategy: str) -> pd.DataFrame:
    df = load_events(strategy)
    if df.empty:
        return df
    bets = df[df["type"] == "bet"]
    res_ids = set(df.loc[df["type"] == "resolution", "bet_id"].dropna().tolist())
    return bets[~bets["bet_id"].isin(res_ids)].copy()
