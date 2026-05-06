"""Smoke test the paper-trading log: write a few events, read them back."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from predmkt.paper import PaperLog, load_events, open_bets, realized_bets


def main() -> None:
    log = PaperLog(strategy="smoke_test")

    log.log_obs(market_pair="kalshi:X / polymarket:Y", kalshi_mid=0.62, poly_mid=0.66, gap=0.04)

    bid_a = log.log_bet(
        venue="kalshi", market_id="X", side="YES", entry_price=0.62, size_usd=100,
        fee=1.65, metadata={"gap": 0.04, "counter_venue": "polymarket"},
    )
    bid_b = log.log_bet(
        venue="polymarket", market_id="Y", side="NO", entry_price=0.34, size_usd=100,
        fee=0.0, metadata={"gap": 0.04, "counter_venue": "kalshi"},
    )
    print(f"opened bets: {bid_a} (kalshi YES) and {bid_b} (polymarket NO)")

    # Pretend Kalshi market resolved YES, Polymarket market resolved YES (so NO settles 0).
    log.log_resolution(bid_a, settlement_price=1.0, notes="market resolved YES")
    log.log_resolution(bid_b, settlement_price=1.0, notes="market resolved YES")

    print("\n-- all events --")
    print(load_events("smoke_test")[["ts", "type", "bet_id", "venue", "side", "entry_price", "size_usd"]].to_string(index=False))

    print("\n-- realized bets with PnL --")
    rb = realized_bets("smoke_test")
    cols = ["bet_id", "venue", "side", "entry_price", "size_usd", "settlement_price", "pnl_usd"]
    print(rb[cols].to_string(index=False))

    print("\n-- open bets --")
    print(open_bets("smoke_test"))


if __name__ == "__main__":
    main()
