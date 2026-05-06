"""Quick analytics on the observation log. Run after a few snapshots accumulate."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from predmkt.observer import load_observations


def main() -> None:
    pd.set_option("display.width", 180)
    pd.set_option("display.max_colwidth", 70)

    df = load_observations()
    if df.empty:
        print("no observations yet — run scripts/snapshot_markets.py first.")
        return

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    snaps = sorted(df["ts"].unique())
    print(f"snapshots: {len(snaps)}, span: {snaps[0]} to {snaps[-1]}")
    print(f"unique markets:")
    for v in df["venue"].unique():
        print(f"  {v}: {df[df['venue'] == v]['market_id'].nunique()}")

    if len(snaps) < 2:
        print("\nNeed >=2 snapshots for change analysis. Re-run snapshot in ~1h.")
        return

    last_two = sorted(snaps)[-2:]
    a = df[df["ts"] == last_two[0]].set_index(["venue", "market_id"])
    b = df[df["ts"] == last_two[1]].set_index(["venue", "market_id"])

    common = a.index.intersection(b.index)
    print(f"\nbetween {last_two[0]} and {last_two[1]}:")
    print(f"  markets in both snapshots: {len(common)}")

    # Largest mid-price moves
    a_mid = (a.loc[common, "yes_bid"] + a.loc[common, "yes_ask"]) / 2
    b_mid = (b.loc[common, "yes_bid"] + b.loc[common, "yes_ask"]) / 2
    delta = (b_mid - a_mid).dropna()
    delta_pts = delta * 100
    print(f"\n  mid-price moves (pts): mean={delta_pts.mean():+.2f}  std={delta_pts.std():.2f}")
    print(f"  largest |move|s:")
    biggest = delta_pts.abs().nlargest(8)
    for (venue, mid), d in biggest.items():
        title = b.loc[(venue, mid), "title"]
        signed = delta_pts.loc[(venue, mid)]
        print(f"    {signed:+6.2f}pt  {venue:<10} {mid:<25} {str(title)[:60]}")

    # Spread widening events
    spread_delta = (b.loc[common, "spread"] - a.loc[common, "spread"]).dropna() * 100
    if len(spread_delta):
        print(f"\n  spread changes (cents): mean={spread_delta.mean():+.2f}")
        widened = spread_delta.nlargest(5)
        print(f"  largest spread widenings:")
        for (venue, mid), w in widened.items():
            title = b.loc[(venue, mid), "title"]
            print(f"    +{w:5.1f}c  {venue:<10} {mid:<25} {str(title)[:60]}")


if __name__ == "__main__":
    main()
