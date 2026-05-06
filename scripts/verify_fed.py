"""Smoke test: discover FOMC meetings, verify pre-meeting ranges, map codes."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from predmkt.fed import (
    code_to_mpt_buckets,
    discover_fomc_meetings,
    load_target_range_history,
    nearest_mpt_reference_start,
)
from predmkt.mpt import MPT


def main() -> None:
    print("-- FRED target range history --")
    tr = load_target_range_history()
    print(f"rows: {len(tr):,} from {tr.index.min().date()} to {tr.index.max().date()}")
    print(f"latest: lo={tr.iloc[-1]['lo_bps']}bps hi={tr.iloc[-1]['hi_bps']}bps")

    print("\n-- FOMC meetings discovered --")
    meetings = discover_fomc_meetings(target_range=tr)
    print(f"count: {len(meetings)}")
    for m in meetings[-6:]:
        print(
            f"  {m.event_ticker:<22} close={m.close_time.date()} "
            f"pre_range={m.pre_lo_bps}-{m.pre_hi_bps}bps"
        )

    print("\n-- MPT alignment per FOMC (None = contaminated, skip) --")
    mpt = MPT.load()
    for m in meetings:
        ref = nearest_mpt_reference_start(m.close_time, mpt.meetings)
        ref_s = ref.date().isoformat() if ref else "—"
        print(f"  {m.event_ticker:<22} close={m.close_time.date()} -> ref={ref_s}")

    from predmkt.fed import cleanly_mappable_meetings
    clean = cleanly_mappable_meetings(meetings, mpt.meetings)
    print(f"\n-- Cleanly mappable FOMCs: {len(clean)} of {len(meetings)} --")
    for m, ref in clean:
        print(f"  {m.event_ticker:<22} close={m.close_time.date()} -> ref={ref.date()}")

    print("\n-- Code -> MPT bucket mapping for Dec 2025 (pre 375-400bps) --")
    dec25 = next(m for m in meetings if m.event_ticker == "KXFEDDECISION-25DEC")
    ref = nearest_mpt_reference_start(dec25.close_time, mpt.meetings)
    avail = mpt.available_buckets(ref)
    for code in ["H0", "C25", "H25", "C26", "H26"]:
        buckets = code_to_mpt_buckets(code, dec25.pre_lo_bps, dec25.pre_hi_bps, avail)
        print(f"  {code:<4} -> {buckets[:5]}{'  ...' if len(buckets) > 5 else ''}  ({len(buckets)} bkts)")


if __name__ == "__main__":
    main()
