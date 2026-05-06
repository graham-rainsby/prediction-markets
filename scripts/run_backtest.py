"""Run the v0 Kalshi-vs-MPT backtest and print summary stats."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from predmkt.backtest import build_observation_table, simulate_bets, summarize
from predmkt.fed import discover_fomc_meetings, load_target_range_history
from predmkt.kalshi import KalshiClient
from predmkt.mpt import MPT


def main() -> None:
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)

    print("loading FOMC meetings + MPT...")
    tr = load_target_range_history()
    kalshi = KalshiClient()
    meetings = discover_fomc_meetings(kalshi=kalshi, target_range=tr)
    mpt = MPT.load()

    print(f"meetings: {len(meetings)}, MPT references: {len(mpt.meetings)}")

    print("\nbuilding observation table (this fetches Kalshi candlesticks, ~60s)...")
    obs = build_observation_table(meetings, mpt, kalshi, lookback_days=60)
    print(f"obs rows: {len(obs):,}")
    if obs.empty:
        print("no observations — abort.")
        return

    obs.to_parquet(ROOT / "data" / "obs_v0.parquet", index=False)

    print("\n-- gap distribution (mpt_prob - kalshi_yes_mid), in pts --")
    gap_pts = (obs["mpt_prob"] - obs["yes_mid"]) * 100
    desc = gap_pts.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    print(desc.round(2).to_string())

    print("\n-- abs(gap) histogram (pts) --")
    bins = [0, 1, 2, 3, 5, 7, 10, 15, 25, 50, 100]
    cuts = pd.cut(gap_pts.abs(), bins=bins, right=False)
    print(cuts.value_counts().sort_index().to_string())

    for thresh in [3, 5, 7, 10]:
        bets = simulate_bets(obs, edge_threshold_pts=thresh, one_bet_per_market=True)
        s = summarize(bets)
        print(f"\n-- edge threshold = {thresh}pt, 1 bet/market --")
        print(f"  bets:        {s.get('n_bets', 0)}")
        if s.get("n_bets", 0):
            print(f"  win rate:    {s['win_rate']:.1%}")
            print(f"  total P&L:   ${s['total_pnl']:.2f}")
            print(f"  mean P&L:    ${s['mean_pnl']:+.4f}/contract")
            print(f"  median P&L:  ${s['median_pnl']:+.4f}/contract")
            print(f"  total fees:  ${s['total_fees']:.2f}")
            print(f"  sharpe (n):  {s['sharpe']:.2f}")
            by_side = bets.groupby("side").agg(n=("pnl", "size"), mean=("pnl", "mean")).round(4)
            print("  by side:")
            print(by_side.to_string().replace("\n", "\n    "))


if __name__ == "__main__":
    main()
