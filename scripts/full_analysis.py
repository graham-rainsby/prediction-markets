"""Deeper analysis of the accumulated 2-week observation log.

Looks for:
  1. Per-venue spread distribution (efficiency proxy)
  2. Markets where spread is *persistently* wide (potential edge from poor MM)
  3. Markets with the most price variation over the period (volatile = either
     informational or oscillating)
  4. Cross-venue candidate pairs surfaced by title-jaccard, with each side's
     observed price range — useful for spotting persistent divergence
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from predmkt.observer import load_observations

STOPWORDS = {
    "the","a","an","of","in","on","by","at","to","for","and","or","be","is",
    "are","was","were","will","would","could","should","this","that","these",
    "those","it","its","with","before","after","from","into","out","over",
    "under","than","then","do","does","did","any","all","no","yes",
}
_PUNCT = re.compile(r"[^\w\s]")


def tokens(s: str | None) -> set[str]:
    s = (s or "").lower()
    s = _PUNCT.sub(" ", s)
    return {t for t in s.split() if t and t not in STOPWORDS and len(t) > 2}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a and b) else 0.0


def main() -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 75)

    df = load_observations()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["mid"] = (df["yes_bid"] + df["yes_ask"]) / 2
    print(f"loaded {len(df):,} rows; {df['ts'].nunique()} snapshots; {df.groupby('venue')['market_id'].nunique().to_dict()} unique markets per venue")

    # ----- 1. Spread distribution per venue -----
    print("\n=== 1. SPREAD distribution per venue (cents) ===")
    spr = df.assign(spr_c=df["spread"] * 100).groupby("venue")["spr_c"].describe(percentiles=[0.1, 0.5, 0.9, 0.99])
    print(spr.round(2).to_string())

    # ----- 2. Markets with persistently WIDE spread -----
    # Average spread across all observations, weighted by volume so we don't
    # surface random microcap markets.
    print("\n=== 2. Markets with persistently WIDE spread (>=5 obs, weighted by mean vol) ===")
    g = df.groupby(["venue", "market_id"]).agg(
        n_obs=("ts", "count"),
        mean_spread_c=("spread", lambda s: (s.mean() or 0) * 100),
        mean_vol=("volume_24h", "mean"),
        last_title=("title", "last"),
    ).reset_index()
    g = g[g["n_obs"] >= 5]
    g["edge_score"] = g["mean_spread_c"] * (g["mean_vol"] ** 0.5)  # wide & liquid-ish
    wide = g.sort_values("edge_score", ascending=False).head(15)
    print(wide[["venue", "market_id", "n_obs", "mean_spread_c", "mean_vol", "last_title"]]
          .to_string(index=False))

    # ----- 3. Most volatile markets (range of mid) -----
    print("\n=== 3. Most VOLATILE markets (range of mid over period, >=10 obs) ===")
    v = df.dropna(subset=["mid"]).groupby(["venue", "market_id"]).agg(
        n_obs=("ts", "count"),
        mid_min=("mid", "min"),
        mid_max=("mid", "max"),
        mid_std=("mid", "std"),
        mean_vol=("volume_24h", "mean"),
        last_title=("title", "last"),
    ).reset_index()
    v["range_pt"] = (v["mid_max"] - v["mid_min"]) * 100
    v = v[(v["n_obs"] >= 10) & (v["mean_vol"] > 10_000)]
    vol = v.sort_values("range_pt", ascending=False).head(15)
    print(vol[["venue", "market_id", "n_obs", "range_pt", "mid_std", "mean_vol", "last_title"]]
          .round(3).to_string(index=False))

    # ----- 4. Cross-venue candidate pairs via title-jaccard -----
    # Use one observation per market (the LATEST) for matching, then look up
    # full price history for any promising matches.
    print("\n=== 4. Cross-venue candidate pairs (jaccard >= 0.30 on titles) ===")
    latest = df.sort_values("ts").groupby(["venue", "market_id"]).tail(1)
    latest = latest.dropna(subset=["title"])
    # Only consider markets with non-trivial volume to cut noise
    latest = latest[latest["volume_24h"] >= 5000]

    k = latest[latest["venue"] == "kalshi"][["market_id", "title", "volume_24h"]].copy()
    p = latest[latest["venue"] == "polymarket"][["market_id", "title", "volume_24h"]].copy()
    k["toks"] = k["title"].map(tokens)
    p["toks"] = p["title"].map(tokens)

    pairs = []
    for _, kr in k.iterrows():
        if not kr["toks"]:
            continue
        for _, pr in p.iterrows():
            if not pr["toks"]:
                continue
            sim = jaccard(kr["toks"], pr["toks"])
            if sim >= 0.30:
                pairs.append({
                    "sim": round(sim, 2),
                    "k_id": kr["market_id"], "k_title": kr["title"][:55],
                    "p_id": pr["market_id"], "p_title": pr["title"][:55],
                    "k_vol": int(kr["volume_24h"]), "p_vol": int(pr["volume_24h"]),
                })
    if not pairs:
        print("  no candidates with jaccard >= 0.30")
    else:
        pdf = pd.DataFrame(pairs).sort_values("sim", ascending=False).head(25)
        print(pdf.to_string(index=False))

        # For top candidates, show observed price history side-by-side
        print("\n--- price history for top 5 candidates ---")
        for _, row in pdf.head(5).iterrows():
            k_hist = df[(df["venue"] == "kalshi") & (df["market_id"] == row["k_id"])]
            p_hist = df[(df["venue"] == "polymarket") & (df["market_id"] == row["p_id"])]
            k_mid = k_hist["mid"].dropna()
            p_mid = p_hist["mid"].dropna()
            if k_mid.empty or p_mid.empty:
                continue
            print(f"\n  KALSHI [{row['k_id']}] {row['k_title']}")
            print(f"    n={len(k_mid)}  mean={k_mid.mean():.3f}  min={k_mid.min():.3f}  max={k_mid.max():.3f}")
            print(f"  POLY   [{row['p_id']}] {row['p_title']}")
            print(f"    n={len(p_mid)}  mean={p_mid.mean():.3f}  min={p_mid.min():.3f}  max={p_mid.max():.3f}")
            print(f"  mean gap (poly - kalshi): {(p_mid.mean() - k_mid.mean())*100:+.1f}pt")


if __name__ == "__main__":
    main()
