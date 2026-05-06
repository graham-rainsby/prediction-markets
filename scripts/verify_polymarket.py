"""Smoke test: fetch a known political market on Polymarket and probe its book."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from predmkt.polymarket import PolymarketClient


def main() -> None:
    p = PolymarketClient()

    print("-- list active markets, ordered by 24h volume --")
    mkts = p.list_markets(active=True, closed=False, limit=8, order="volume24hr", ascending=False)
    for m in mkts[:5]:
        print(
            f"  id={m['id']:>7}  vol24h=${float(m.get('volume24hr', 0) or 0):>12,.0f}  "
            f"bestBid={m.get('bestBid'):<6} bestAsk={m.get('bestAsk'):<6}  "
            f"{(m.get('question') or '')[:80]}"
        )

    if not mkts:
        print("no markets returned"); return

    target = mkts[0]
    print(f"\n-- detail for top market id={target['id']} --")
    print(f"  question:  {target.get('question')}")
    print(f"  end_date:  {target.get('endDate')}")
    print(f"  outcomes:  {target.get('outcomes')}")
    print(f"  prices:    {target.get('outcomePrices')}")
    print(f"  cond_id:   {target.get('conditionId')}")

    yes_tok = PolymarketClient.yes_token(target)
    if not yes_tok:
        print("no YES token id found"); return
    print(f"  YES tok:   {yes_tok}")

    print("\n-- orderbook (top 3 of each side) --")
    book = p.orderbook(yes_tok)
    bb, ba = PolymarketClient.best_bid_ask(book)
    print(f"  best bid: {bb}   best ask: {ba}   spread: {(ba - bb) if bb and ba else 'n/a'}")
    for label, side in [("bids (buyers)", book.get("bids", [])), ("asks (sellers)", book.get("asks", []))]:
        print(f"  {label}:")
        for lvl in sorted(side, key=lambda x: float(x["price"]), reverse=(label.startswith("bids")))[:3]:
            print(f"    {float(lvl['price']):.4f} x {float(lvl['size']):,.2f}")

    print("\n-- last trade price --")
    print(p.last_trade_price(yes_tok))

    print("\n-- price history (1h interval) --")
    hist = p.price_history(yes_tok, interval="1h")
    print(f"  points: {len(hist)}")
    if hist:
        print(f"  first: {hist[0]}")
        print(f"  last : {hist[-1]}")


if __name__ == "__main__":
    main()
