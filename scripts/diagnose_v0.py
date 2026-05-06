"""Deeper diagnostic on the v0 obs table — reverse signal, time-to-meeting cuts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from predmkt.backtest import KALSHI_TAKER_RATE


def simulate(obs: pd.DataFrame, threshold_pts: float, reverse: bool = False) -> pd.DataFrame:
    """Take FIRST qualifying obs per (event, ticker). reverse=True flips bet direction."""
    edge = threshold_pts / 100.0
    o = obs.copy()
    o["gap"] = o["mpt_prob"] - o["yes_mid"]
    qual = o[o["gap"].abs() > edge].sort_values(["event", "ticker", "date"])
    qual = qual.drop_duplicates(subset=["event", "ticker"], keep="first")
    rows = []
    for _, r in qual.iterrows():
        signal_long = (r["gap"] > 0) ^ reverse  # XOR with reverse flips direction
        if signal_long:
            entry = float(r["yes_ask"]); side = "BUY_YES"
            payoff = 1.0 if r["settled_yes"] else 0.0
        else:
            entry = 1.0 - float(r["yes_bid"]); side = "BUY_NO"
            payoff = 1.0 if not r["settled_yes"] else 0.0
        fee = KALSHI_TAKER_RATE * entry * (1.0 - entry)
        rows.append({**r.to_dict(), "side": side, "entry": entry, "fee": fee,
                     "payoff": payoff, "pnl": payoff - entry - fee})
    return pd.DataFrame(rows)


def main() -> None:
    obs = pd.read_parquet(ROOT / "data" / "obs_v0.parquet")
    print(f"obs rows: {len(obs):,}")

    # add days-to-meeting and bucket-type-tail flag
    obs["days_to_mtg"] = obs.groupby(["event", "ticker"])["date"].transform(
        lambda s: (s.max() - s).dt.days
    )

    print("\n=== Strategy comparison @ 7pt threshold ===")
    for label, rev in [("FORWARD (MPT signal)", False), ("REVERSE (with Kalshi)", True)]:
        bets = simulate(obs, 7.0, reverse=rev)
        if bets.empty:
            print(f"{label}: no bets"); continue
        print(f"\n  {label}")
        print(f"    n={len(bets)}  win_rate={(bets['pnl']>0).mean():.1%}")
        print(f"    total P&L: ${bets['pnl'].sum():.2f}   mean: ${bets['pnl'].mean():+.4f}/contract")
        print(f"    median: ${bets['pnl'].median():+.4f}   std: ${bets['pnl'].std():.4f}")

    print("\n=== REVERSE strategy by days-to-meeting (threshold=7pt) ===")
    bets = simulate(obs, 7.0, reverse=True)
    if not bets.empty:
        bets["bucket"] = pd.cut(
            bets["days_to_mtg"], bins=[-1, 3, 7, 14, 30, 90],
            labels=["0-3d", "4-7d", "8-14d", "15-30d", "31-90d"],
        )
        agg = bets.groupby("bucket", observed=True).agg(
            n=("pnl", "size"), wr=("pnl", lambda s: (s > 0).mean()),
            total=("pnl", "sum"), mean=("pnl", "mean"),
        ).round(4)
        print(agg.to_string())

    print("\n=== REVERSE strategy by Kalshi action code ===")
    if not bets.empty:
        agg2 = bets.groupby("code").agg(
            n=("pnl", "size"), wr=("pnl", lambda s: (s > 0).mean()),
            total=("pnl", "sum"), mean=("pnl", "mean"),
        ).round(4)
        print(agg2.to_string())


if __name__ == "__main__":
    main()
