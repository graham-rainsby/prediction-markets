"""FOMC meeting + fed funds target range helpers.

Sources:
- DFEDTARU / DFEDTARL: FRED daily fed funds target upper/lower bounds.
- FOMC meeting dates: derived from Kalshi `KXFEDDECISION-YYMMM` event close_time.

Kalshi action codes (per KXFEDDECISION rules text):
    H0   = Fed maintains rate
    C25  = Cut 25bps
    H25  = Hike 25bps
    C26  = Cut >25bps    (any cut larger than 25bps)
    H26  = Hike >25bps   (any hike larger than 25bps)

Mapping each code to MPT post-meeting target-range bucket(s), given pre-meeting
target bounds (pre_lo_bps, pre_hi_bps):
    H0  -> bucket (pre_lo,        pre_hi)
    C25 -> bucket (pre_lo - 25,   pre_hi - 25)
    H25 -> bucket (pre_lo + 25,   pre_hi + 25)
    C26 -> all buckets with lo <= pre_lo - 50
    H26 -> all buckets with lo >= pre_hi + 25
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from predmkt.kalshi import KalshiClient

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFEDTARU,DFEDTARL"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FRED_CACHE = DATA_DIR / "fed_target_range.csv"


def _fetch_fred_csv() -> str:
    """FRED is fronted by Akamai and can take 10-30s on a cold request.
    Try requests first, then fall back to curl which seems to negotiate the
    bot-detection cookies more reliably from this environment.
    """
    import shutil
    import subprocess
    import tempfile

    try:
        r = requests.get(FRED_CSV, headers={"User-Agent": "Mozilla/5.0"}, timeout=90)
        r.raise_for_status()
        return r.text
    except requests.RequestException:
        pass

    if not shutil.which("curl"):
        raise RuntimeError("FRED requests fetch failed and curl unavailable")
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        out = tmp.name
    proc = subprocess.run(
        ["curl", "-sL", "--connect-timeout", "15", "--max-time", "120",
         "-A", "Mozilla/5.0", FRED_CSV, "-o", out],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"FRED curl fetch failed: rc={proc.returncode} {proc.stderr!r}")
    return Path(out).read_text()


def load_target_range_history(force_refresh: bool = False) -> pd.DataFrame:
    """Daily fed funds target lower & upper bounds in **basis points**.

    Cached to data/fed_target_range.csv. Returns DataFrame indexed by UTC midnight
    date with columns 'lo_bps', 'hi_bps'; forward-filled for non-trading days.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if force_refresh or not FRED_CACHE.exists():
        FRED_CACHE.write_text(_fetch_fred_csv())
    df = pd.read_csv(FRED_CACHE)
    df["observation_date"] = pd.to_datetime(df["observation_date"]).dt.tz_localize("UTC").dt.normalize()
    df["lo_bps"] = (pd.to_numeric(df["DFEDTARL"], errors="coerce") * 100).round().astype("Int64")
    df["hi_bps"] = (pd.to_numeric(df["DFEDTARU"], errors="coerce") * 100).round().astype("Int64")
    df = df.set_index("observation_date")[["lo_bps", "hi_bps"]].dropna()
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="D", tz="UTC")
    return df.reindex(full_idx).ffill()


@dataclass
class FOMCMeeting:
    event_ticker: str  # e.g. KXFEDDECISION-25DEC
    close_time: pd.Timestamp  # Kalshi announcement timestamp
    pre_lo_bps: int  # pre-meeting target lower bound
    pre_hi_bps: int  # pre-meeting target upper bound

    @property
    def date(self) -> pd.Timestamp:
        return self.close_time.normalize()


def discover_fomc_meetings(
    kalshi: KalshiClient | None = None,
    target_range: pd.DataFrame | None = None,
) -> list[FOMCMeeting]:
    """List historical FOMC meetings via Kalshi KXFEDDECISION + FRED rate history."""
    kalshi = kalshi or KalshiClient()
    target_range = load_target_range_history() if target_range is None else target_range

    by_event: dict[str, pd.Timestamp] = {}
    for m in kalshi.historical_markets(series_ticker="KXFEDDECISION"):
        ev = m.get("event_ticker")
        ct = m.get("close_time")
        if not ev or not ct:
            continue
        ts = pd.Timestamp(ct)
        if ev not in by_event or ts < by_event[ev]:
            by_event[ev] = ts

    meetings: list[FOMCMeeting] = []
    for ev, close_ts in sorted(by_event.items(), key=lambda kv: kv[1]):
        # Pre-meeting range = previous business day's value.
        prev_day = close_ts.normalize() - pd.Timedelta(days=1)
        if prev_day not in target_range.index:
            continue
        row = target_range.loc[prev_day]
        meetings.append(
            FOMCMeeting(
                event_ticker=ev,
                close_time=close_ts,
                pre_lo_bps=int(row["lo_bps"]),
                pre_hi_bps=int(row["hi_bps"]),
            )
        )
    return meetings


def code_to_mpt_buckets(
    code: str,
    pre_lo_bps: int,
    pre_hi_bps: int,
    available: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Return MPT (lo, hi) buckets corresponding to a Kalshi action code.

    `available` is the list of MPT buckets present for the relevant meeting
    (used to constrain the C26/H26 tail unions to actually-published buckets).
    """
    code = code.upper()
    avail = sorted(set(available))
    if code == "H0":
        return [(pre_lo_bps, pre_hi_bps)]
    if code == "C25":
        return [(pre_lo_bps - 25, pre_hi_bps - 25)]
    if code == "H25":
        return [(pre_lo_bps + 25, pre_hi_bps + 25)]
    if code == "C26":
        return [(lo, hi) for (lo, hi) in avail if lo <= pre_lo_bps - 50]
    if code == "H26":
        return [(lo, hi) for (lo, hi) in avail if lo >= pre_hi_bps + 25]
    raise ValueError(f"unknown Kalshi action code: {code}")


def parse_kalshi_action_code(ticker: str) -> str:
    """Extract action code from a KXFEDDECISION ticker.

    KXFEDDECISION-25DEC-C25 -> 'C25'
    """
    parts = ticker.rsplit("-", 1)
    if len(parts) != 2:
        raise ValueError(f"unexpected ticker format: {ticker}")
    return parts[1]


def nearest_mpt_reference_start(
    fomc_close: pd.Timestamp,
    mpt_reference_starts: list[pd.Timestamp],
    max_days: int = 14,
) -> pd.Timestamp | None:
    """MPT reference_start on or close-after the FOMC, or None if too far.

    Compares calendar dates (UTC). MPT references at 00:00 UTC on the FOMC
    day are valid — the reference represents the IORB period starting that
    day, which encodes the FOMC decision announced earlier at 18:59 UTC.
    """
    fomc_date = fomc_close.normalize()
    after = [r for r in mpt_reference_starts if r.normalize() >= fomc_date]
    if not after:
        return None
    nearest = min(after, key=lambda r: r - fomc_date)
    return nearest if (nearest.normalize() - fomc_date) <= pd.Timedelta(days=max_days) else None


def cleanly_mappable_meetings(
    meetings: list[FOMCMeeting],
    mpt_reference_starts: list[pd.Timestamp],
    max_days: int = 14,
) -> list[tuple[FOMCMeeting, pd.Timestamp]]:
    """FOMCs that have an MPT reference_start within max_days *and* whose
    previous FOMC is far enough back that no intermediate FOMC sits between
    the previous FOMC and the matched MPT reference."""
    sorted_meetings = sorted(meetings, key=lambda m: m.close_time)
    out: list[tuple[FOMCMeeting, pd.Timestamp]] = []
    for i, m in enumerate(sorted_meetings):
        ref = nearest_mpt_reference_start(m.close_time, mpt_reference_starts, max_days)
        if ref is None:
            continue
        # Confirm no other FOMC sits in (m.close_time, ref]
        if any(
            m.close_time < other.close_time <= ref for other in sorted_meetings if other is not m
        ):
            continue
        out.append((m, ref))
    return out
