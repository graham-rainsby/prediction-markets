"""One-shot market snapshot.

To run periodically:
    # ad-hoc:    python3 scripts/snapshot_markets.py
    # hourly:    cron entry: 0 * * * * cd <repo> && /path/to/python3 scripts/snapshot_markets.py
    # loop:      run via Claude Code /loop 1h on this script, or an OS scheduler

Output: data/observations/<YYYY-MM-DD>.parquet (one file per UTC day, appended).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from predmkt.observer import append_snapshot, snapshot_all


def _show_top(df: pd.DataFrame, venue: str, n: int = 10) -> None:
    sub = df[df["venue"] == venue].nlargest(n, "volume_24h").copy()
    if sub.empty:
        print(f"  (no {venue} rows)"); return
    sub["vol_24h"] = sub["volume_24h"].map(lambda v: f"${v:,.0f}")
    sub["spr_c"] = (sub["spread"] * 100).round(1)
    cols = ["market_id", "title", "yes_bid", "yes_ask", "spr_c", "vol_24h"]
    print(sub[cols].to_string(index=False))


def main() -> None:
    pd.set_option("display.width", 180)
    pd.set_option("display.max_colwidth", 70)

    df = snapshot_all(kalshi_top_n=200, poly_top_n=200)
    path = append_snapshot(df)

    n_k = int((df["venue"] == "kalshi").sum())
    n_p = int((df["venue"] == "polymarket").sum())
    print(f"snapshot saved: {path}  rows: {len(df)} ({n_k} kalshi, {n_p} polymarket)")

    print(f"\n-- top 10 Kalshi markets --")
    _show_top(df, "kalshi")

    print(f"\n-- top 10 Polymarket markets --")
    _show_top(df, "polymarket")

    print(f"\n-- spread distribution (cents), per venue --")
    sp = df.assign(spr_c=df["spread"] * 100).groupby("venue")["spr_c"].describe(percentiles=[0.1, 0.5, 0.9])
    print(sp.round(2).to_string())


if __name__ == "__main__":
    main()
