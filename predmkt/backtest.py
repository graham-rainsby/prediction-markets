"""v0 backtest: pair Kalshi KXFEDDECISION markets with Atlanta Fed MPT probs.

For each cleanly-mappable FOMC:
  - Discover all outcome markets (H0, C25, H25, C26, H26)
  - For each market, derive its MPT bucket(s) from the action code
  - Walk back N days from the FOMC, pulling daily Kalshi candlesticks
  - Pair Kalshi yes-mid against MPT bucket probability on each date
  - Simulate "bet when |MPT_prob - kalshi_yes_mid| > edge_threshold"
  - Compute P&L vs the realized binary settlement, net of Kalshi taker fees

Kalshi taker fee:  fee = 0.07 * C * (1 - C) per contract  (Other/Economics tier)
"""

from __future__ import annotations

import pandas as pd

from predmkt.fed import (
    FOMCMeeting,
    cleanly_mappable_meetings,
    code_to_mpt_buckets,
    parse_kalshi_action_code,
)
from predmkt.kalshi import KalshiClient
from predmkt.mpt import MPT

KALSHI_TAKER_RATE = 0.07  # Other/Economics tier; Fed markets currently fall in this bucket


def _sum_bucket_probs(mpt: MPT, ref: pd.Timestamp, buckets: list[tuple[int, int]]) -> pd.Series:
    """Daily Series of summed bucket probabilities for a meeting (in percent)."""
    if not buckets:
        return pd.Series(dtype=float)
    parts = [mpt.bucket_prob_series(ref, lo, hi) for (lo, hi) in buckets]
    df = pd.concat(parts, axis=1)
    return df.sum(axis=1, min_count=1)


def build_observation_table(
    meetings: list[FOMCMeeting],
    mpt: MPT,
    kalshi: KalshiClient,
    lookback_days: int = 60,
) -> pd.DataFrame:
    """Long-format frame: one row per (meeting, ticker, observation_date) with
    Kalshi yes_bid/ask, MPT prob, and the realized settlement."""
    rows: list[dict] = []
    clean = cleanly_mappable_meetings(meetings, mpt.meetings)
    for meeting, mpt_ref in clean:
        markets = list(kalshi.historical_markets(event_ticker=meeting.event_ticker))
        avail = mpt.available_buckets(mpt_ref)

        start_ts = int((meeting.close_time - pd.Timedelta(days=lookback_days)).timestamp())
        end_ts = int((meeting.close_time - pd.Timedelta(hours=1)).timestamp())

        for mkt in markets:
            ticker = mkt["ticker"]
            try:
                code = parse_kalshi_action_code(ticker)
                buckets = code_to_mpt_buckets(code, meeting.pre_lo_bps, meeting.pre_hi_bps, avail)
            except ValueError:
                continue
            mpt_pct = _sum_bucket_probs(mpt, mpt_ref, buckets)
            if mpt_pct.empty:
                continue

            try:
                candles = kalshi.historical_candlesticks(ticker, start_ts, end_ts, period_interval=1440)
            except Exception as e:  # noqa: BLE001 — log + continue
                print(f"  candle fetch failed for {ticker}: {e}")
                continue

            settled_yes = mkt.get("result") == "yes"
            for c in candles:
                yes_bid = c.get("yes_bid", {}).get("close")
                yes_ask = c.get("yes_ask", {}).get("close")
                if yes_bid is None or yes_ask is None:
                    continue
                date = pd.Timestamp(c["end_period_ts"], unit="s", tz="UTC").normalize()
                if date >= meeting.close_time.normalize():
                    continue  # never observe on/after meeting day (look-ahead)
                if date not in mpt_pct.index:
                    continue
                pct = mpt_pct.loc[date]
                if pd.isna(pct):
                    continue
                rows.append(
                    {
                        "event": meeting.event_ticker,
                        "ticker": ticker,
                        "code": code,
                        "date": date,
                        "yes_bid": float(yes_bid),
                        "yes_ask": float(yes_ask),
                        "yes_mid": (float(yes_bid) + float(yes_ask)) / 2.0,
                        "mpt_prob": float(pct) / 100.0,
                        "settled_yes": bool(settled_yes),
                    }
                )
    return pd.DataFrame(rows).sort_values(["event", "ticker", "date"]).reset_index(drop=True)


def simulate_bets(
    obs: pd.DataFrame,
    edge_threshold_pts: float = 6.0,
    one_bet_per_market: bool = True,
) -> pd.DataFrame:
    """For each observation where |gap| > edge_threshold, simulate a 1-contract bet.

    one_bet_per_market: only place the FIRST qualifying bet per (event, ticker) —
    avoids inflating sample by repeating the same conviction 30 days running.
    """
    edge = edge_threshold_pts / 100.0
    obs = obs.copy()
    obs["gap"] = obs["mpt_prob"] - obs["yes_mid"]
    triggered = obs[obs["gap"].abs() > edge].copy()

    if one_bet_per_market:
        triggered = triggered.sort_values(["event", "ticker", "date"]).drop_duplicates(
            subset=["event", "ticker"], keep="first"
        )

    if triggered.empty:
        return pd.DataFrame(
            columns=[
                "event", "ticker", "code", "date", "side",
                "entry_price", "gap", "fee", "payoff", "pnl",
            ]
        )

    def _row_to_bet(r: pd.Series) -> dict:
        if r["gap"] > 0:
            entry = float(r["yes_ask"])
            side = "BUY_YES"
            payoff = 1.0 if r["settled_yes"] else 0.0
        else:
            entry = 1.0 - float(r["yes_bid"])
            side = "BUY_NO"
            payoff = 1.0 if not r["settled_yes"] else 0.0
        fee = KALSHI_TAKER_RATE * entry * (1.0 - entry)
        pnl = payoff - entry - fee
        return {
            "event": r["event"],
            "ticker": r["ticker"],
            "code": r["code"],
            "date": r["date"],
            "side": side,
            "entry_price": entry,
            "gap": float(r["gap"]),
            "fee": fee,
            "payoff": payoff,
            "pnl": pnl,
        }

    return pd.DataFrame([_row_to_bet(r) for _, r in triggered.iterrows()])


def summarize(bets: pd.DataFrame) -> dict:
    if bets.empty:
        return {"n": 0}
    n = len(bets)
    wins = (bets["pnl"] > 0).sum()
    return {
        "n_bets": n,
        "win_rate": wins / n,
        "total_pnl": float(bets["pnl"].sum()),
        "mean_pnl": float(bets["pnl"].mean()),
        "median_pnl": float(bets["pnl"].median()),
        "std_pnl": float(bets["pnl"].std()),
        "total_fees": float(bets["fee"].sum()),
        "sharpe": float(bets["pnl"].mean() / bets["pnl"].std() * (n**0.5)) if bets["pnl"].std() else float("nan"),
    }
