"""Deep-dive on the PGA Championship single-name win markets — exact-title
matches across Kalshi and Polymarket, multiple observations, real liquidity.

Goal: see if the cross-venue price gap is persistent (real edge) or transient
(stale prints / market just opened).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from predmkt.observer import load_observations

# The five Kalshi <-> Polymarket pairs surfaced in full_analysis.py
PAIRS = [
    ("Scheffler", "KXPGATOUR-PGC26-SSCH", "2234555"),
    ("McIlroy",   "KXPGATOUR-PGC26-RMCI", "2234556"),
    ("Schauffele","KXPGATOUR-PGC26-XSCH", "2234559"),
    ("Rahm",      "KXPGATOUR-PGC26-JRAH", "2234558"),
]


def main() -> None:
    pd.set_option("display.width", 200)

    df = load_observations()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["mid"] = (df["yes_bid"] + df["yes_ask"]) / 2

    for name, k_id, p_id in PAIRS:
        k = df[(df["venue"] == "kalshi") & (df["market_id"] == k_id)].sort_values("ts")
        p = df[(df["venue"] == "polymarket") & (df["market_id"] == p_id)].sort_values("ts")
        print(f"\n=== {name}  Kalshi {k_id}  <->  Polymarket {p_id} ===")
        print(f"  Kalshi obs:     {len(k)}   span: {k['ts'].min()} -> {k['ts'].max()}" if len(k) else "  Kalshi obs:     0")
        print(f"  Polymarket obs: {len(p)}   span: {p['ts'].min()} -> {p['ts'].max()}" if len(p) else "  Polymarket obs: 0")

        if k.empty or p.empty:
            continue

        # Find timestamps where BOTH venues have an observation within 30 min of each other
        # (asof merge on ts, then filter to small deltas).
        k_small = k[["ts", "mid", "yes_bid", "yes_ask", "volume_24h"]].rename(
            columns={"mid": "k_mid", "yes_bid": "k_bid", "yes_ask": "k_ask", "volume_24h": "k_vol"}
        )
        p_small = p[["ts", "mid", "yes_bid", "yes_ask", "volume_24h"]].rename(
            columns={"mid": "p_mid", "yes_bid": "p_bid", "yes_ask": "p_ask", "volume_24h": "p_vol"}
        )
        merged = pd.merge_asof(
            k_small.sort_values("ts"),
            p_small.sort_values("ts"),
            on="ts",
            tolerance=pd.Timedelta("30min"),
            direction="nearest",
        ).dropna(subset=["p_mid"])

        if merged.empty:
            print("  no overlapping observations within 30 min")
            continue

        merged["gap_pt"] = (merged["p_mid"] - merged["k_mid"]) * 100
        print(f"  near-simultaneous obs: {len(merged)}")
        print(f"  gap (poly - kalshi) pts:")
        print(f"    mean: {merged['gap_pt'].mean():+.2f}")
        print(f"    median: {merged['gap_pt'].median():+.2f}")
        print(f"    min: {merged['gap_pt'].min():+.2f}  max: {merged['gap_pt'].max():+.2f}")
        print(f"    std: {merged['gap_pt'].std():.2f}")
        print(f"\n  sample observations (first 5, last 5):")
        cols = ["ts", "k_mid", "k_bid", "k_ask", "p_mid", "p_bid", "p_ask", "gap_pt"]
        sample = pd.concat([merged.head(5), merged.tail(5)]).drop_duplicates()
        print(sample[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
