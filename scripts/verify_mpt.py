"""Smoke test: load MPT and inspect probabilities for the Dec 10 2025 FOMC.

The Dec 2025 FOMC delivered a 25bp cut, taking 3.75-4.00% to 3.50-3.75%.
So:
  Prob: 350bps - 375bps   <- the realized bucket; should rise to ~1 by meeting day
  Prob: 375bps - 400bps   <- the "hold" bucket; should fall to ~0
  Prob: cut               <- should rise toward 1
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from predmkt.mpt import MPT


def main() -> None:
    mpt = MPT.load()
    print(f"rows: {len(mpt.df):,}")
    print(f"date range: {mpt.df['date'].min().date()} -> {mpt.df['date'].max().date()}")
    print(f"meetings tracked: {len(mpt.meetings)}")
    print(f"first 3: {[m.date() for m in mpt.meetings[:3]]}")
    print(f"last  3: {[m.date() for m in mpt.meetings[-3:]]}")

    dec_2025 = pd.Timestamp("2025-12-10", tz="UTC")
    if dec_2025 not in mpt.meetings:
        nearest = min(mpt.meetings, key=lambda m: abs(m - dec_2025))
        print(f"\nNote: {dec_2025.date()} not in meetings list, using nearest {nearest.date()}")
        dec_2025 = nearest

    print(f"\n-- buckets available for {dec_2025.date()} --")
    print(mpt.available_buckets(dec_2025))

    print(f"\n-- daily P(cut) for {dec_2025.date()} FOMC --")
    cut_prob = mpt.cut_prob_series(dec_2025)
    print(f"obs: {len(cut_prob)} | first: {cut_prob.head(3).to_dict()}")
    print(f"last: {cut_prob.tail(5).to_dict()}")

    print(f"\n-- realized bucket P(350-375bps) for {dec_2025.date()} FOMC --")
    s = mpt.bucket_prob_series(dec_2025, 350, 375)
    if len(s):
        print(f"first 3: {s.head(3).to_dict()}")
        print(f"last  5: {s.tail(5).to_dict()}")

    print(f"\n-- hold bucket P(375-400bps) for {dec_2025.date()} FOMC --")
    s2 = mpt.bucket_prob_series(dec_2025, 375, 400)
    if len(s2):
        print(f"first 3: {s2.head(3).to_dict()}")
        print(f"last  5: {s2.tail(5).to_dict()}")


if __name__ == "__main__":
    main()
