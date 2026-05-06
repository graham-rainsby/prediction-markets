"""Smoke test: pull a known FOMC market and its candlestick history.

The Dec 10 2025 FOMC delivered a 25bp cut, so KXFEDDECISION-25DEC-C25 should
have settled YES (final price = 1.00) and KXFEDDECISION-25DEC-H0 (hold)
should have settled NO (final price = 0.00).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datetime import datetime, timezone

from predmkt.kalshi import KalshiClient, to_unix


def main() -> None:
    c = KalshiClient()
    cutoff = c.cutoff()
    print("cutoff:", cutoff)

    print("\n-- FOMC markets discovered for series KXFEDDECISION --")
    fed_markets = list(c.historical_markets(series_ticker="KXFEDDECISION", limit=200))
    print(f"total markets: {len(fed_markets)}")

    settled = [m for m in fed_markets if m.get("status") == "finalized"]
    print(f"finalized:    {len(settled)}")
    yes_outcomes = [m for m in settled if m.get("result") == "yes"]
    print(f"yes-resolved: {len(yes_outcomes)}")

    print("\n-- Recent finalized markets (first 5) --")
    settled.sort(key=lambda m: m.get("close_time", ""), reverse=True)
    for m in settled[:5]:
        print(
            f"  {m['ticker']:<28} close={m.get('close_time')} "
            f"result={m.get('result'):<6} settled=${m.get('settlement_value_dollars')}"
        )

    print("\n-- Candlestick fetch: KXFEDDECISION-25DEC-C25 (Dec 2025 cut 25bp) --")
    target = "KXFEDDECISION-25DEC-C25"
    detail = c.historical_market(target)
    open_t = detail.get("open_time")
    close_t = detail.get("close_time")
    print(f"open_time={open_t}  close_time={close_t}  result={detail.get('result')}")

    start_ts = to_unix(open_t) if open_t else to_unix(datetime(2025, 11, 1, tzinfo=timezone.utc))
    end_ts = to_unix(close_t) if close_t else to_unix(datetime(2025, 12, 11, tzinfo=timezone.utc))
    candles = c.historical_candlesticks(target, start_ts, end_ts, period_interval=1440)
    print(f"daily candles returned: {len(candles)}")
    if candles:
        first, last = candles[0], candles[-1]
        for label, k in [("first", first), ("last ", last)]:
            ts = datetime.fromtimestamp(k["end_period_ts"], tz=timezone.utc)
            close = k.get("price", {}).get("close")
            yes_bid_close = k.get("yes_bid", {}).get("close")
            yes_ask_close = k.get("yes_ask", {}).get("close")
            vol = k.get("volume")
            print(
                f"  {label} {ts.date()} price.close={close} "
                f"yes_bid.close={yes_bid_close} yes_ask.close={yes_ask_close} vol={vol}"
            )


if __name__ == "__main__":
    main()
